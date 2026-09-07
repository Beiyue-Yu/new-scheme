#!/usr/bin/env python3
"""Extract LanguageBind VGGSound features from local Hugging Face tar shards.

The complete VGGSound archive is larger than the free space available for a
second, fully expanded copy. This driver handles one tar shard at a time:
it lists only benchmark videos in that shard, extracts those clips into a
temporary directory, invokes the resumable LanguageBind extractor in the
foreground, and removes only the temporary extracted clips afterwards.
Original tar files are never modified or deleted.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np

from extract_languagebind_vggsound import read_rows


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive_dir", type=Path,
                        default=Path("/home/wwj/文档/AVGZSL/rawVGGSound"))
    parser.add_argument("--split_dir", type=Path,
                        default=Path("avgzsl_benchmark_datasets/VGGSound/class-split/main_split"))
    parser.add_argument("--class_file", type=Path,
                        default=Path("avgzsl_benchmark_datasets/VGGSound/class-split/all_class.txt"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/languagebind_features/VGGSound/vggsound_main_segments.npz"))
    parser.add_argument(
        "--error_log", type=Path,
        default=Path("runs/languagebind_features/VGGSound/extraction_errors.log"))
    parser.add_argument(
        "--corrupt_log", type=Path,
        default=Path("runs/languagebind_features/VGGSound/corrupt_archives.log"),
        help="Append-only JSONL log for archives skipped after gzip/tar errors")
    parser.add_argument("--staging_root", type=Path,
                        default=Path("/home/wwj/文档/AVGZSL/rawVGGSound/.languagebind_staging"))
    parser.add_argument("--model_dir", type=Path,
                        default=Path("model_cache/LanguageBind_Video_FT"))
    parser.add_argument("--vendor_dir", type=Path,
                        default=Path("model_cache/LanguageBind"))
    parser.add_argument("--dependency_dir", type=Path,
                        default=Path("model_cache/languagebind_python"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num_segments", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--batch_videos", type=int, default=4)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--shard", action="append", default=[],
                        help="Only run selected shard numbers (00-19); repeatable")
    parser.add_argument("--keep_staging", action="store_true",
                        help="Keep temporary videos after each successful shard")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.num_segments, args.num_frames, args.batch_videos,
           args.checkpoint_every) <= 0:
        parser.error("segment, frame, batch and checkpoint values must be positive")
    return args


def _archives(archive_dir: Path, requested: Sequence[str]) -> list[Path]:
    all_archives = [archive_dir / f"vggsound_{index:02d}.tar.gz" for index in range(20)]
    if not requested:
        selected = all_archives
    else:
        requested_indexes = set()
        for value in requested:
            if not value.isdigit() or not 0 <= int(value) < 20:
                raise ValueError(f"Invalid --shard {value!r}; expected 00 through 19")
            requested_indexes.add(int(value))
        selected = [all_archives[index] for index in sorted(requested_indexes)]
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing VGGSound archive(s): {missing}")
    return selected


def _cached_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with np.load(path, allow_pickle=False) as cache:
        names = {str(value) for value in cache["video_names"]}
        failures = json.loads(str(cache["failures_json"]))
    names.update(str(item["filename"]) for item in failures)
    return names


def _target_members(archive: Path, target_names: set[str], completed: set[str]) -> list[str]:
    members: list[str] = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".mp4"):
                continue
            filename = PurePosixPath(member.name).name
            if filename[:-4] in target_names - completed:
                members.append(member.name)
    return members


def _record_corrupt_archive(path: Path, log_path: Path, stage: str, error: Exception) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "archive": str(path),
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"archive_skipped={path.name} stage={stage} "
        f"error={type(error).__name__}: {error} log={log_path}",
        flush=True,
    )


def _extract_members(archive: Path, members: Iterable[str], staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    requested = set(members)
    extracted = 0
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if member.name not in requested:
                continue
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read {member.name} from {archive}")
            destination = staging / PurePosixPath(member.name).name
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            extracted += 1
    if extracted != len(requested):
        raise RuntimeError(f"Extracted {extracted} videos, expected {len(requested)} from {archive}")


def _run_extractor(args: argparse.Namespace, names_file: Path, staging: Path) -> None:
    command = [
        sys.executable, "extract_languagebind_vggsound.py",
        "--split_dir", str(args.split_dir),
        "--class_file", str(args.class_file),
        "--output", str(args.output),
        "--error_log", str(args.error_log),
        "--video_root", str(staging),
        "--names_file", str(names_file),
        "--model_dir", str(args.model_dir),
        "--vendor_dir", str(args.vendor_dir),
        "--dependency_dir", str(args.dependency_dir),
        "--device", args.device,
        "--num_segments", str(args.num_segments),
        "--num_frames", str(args.num_frames),
        "--batch_videos", str(args.batch_videos),
        "--checkpoint_every", str(args.checkpoint_every),
    ]
    print(f"extract_command={' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.class_file.is_file() or not args.split_dir.is_dir():
        raise FileNotFoundError("VGGSound split directory or class file is missing")
    target_names = {row.filename for row in read_rows(args.split_dir)}
    archives = _archives(args.archive_dir, args.shard)
    args.staging_root.mkdir(parents=True, exist_ok=True)
    print(f"target_samples={len(target_names)} archives={len(archives)} output={args.output}", flush=True)

    for index, archive in enumerate(archives, start=1):
        completed = _cached_names(args.output)
        try:
            members = _target_members(archive, target_names, completed)
        except (tarfile.TarError, EOFError, OSError, ValueError, zlib.error) as error:
            _record_corrupt_archive(archive, args.corrupt_log, "listing", error)
            continue
        print(f"shard={index}/{len(archives)} archive={archive.name} "
              f"cached={len(completed)} targets_in_shard={len(members)}", flush=True)
        if not members:
            continue
        if args.dry_run:
            continue
        staging = Path(tempfile.mkdtemp(prefix=f"{archive.stem}_", dir=args.staging_root))
        names_file = staging / "selected_names.txt"
        try:
            names_file.write_text(
                "\n".join(PurePosixPath(member).name[:-4] for member in members) + "\n",
                encoding="utf-8")
            print(f"extracting_shard={archive.name} clips={len(members)} staging={staging}", flush=True)
            try:
                _extract_members(archive, members, staging)
            except (tarfile.TarError, EOFError, OSError, ValueError, zlib.error) as error:
                _record_corrupt_archive(archive, args.corrupt_log, "extraction", error)
                continue
            print(f"extracting_features={archive.name} clips={len(members)}", flush=True)
            _run_extractor(args, names_file, staging)
        finally:
            if args.keep_staging:
                print(f"staging_retained={staging}", flush=True)
            else:
                shutil.rmtree(staging, ignore_errors=True)
                print(f"staging_removed={staging}", flush=True)

    cached = _cached_names(args.output)
    missing = sorted(target_names - cached)
    print(f"complete_cached={len(cached)} target_samples={len(target_names)} missing={len(missing)}", flush=True)
    if missing:
        print(f"first_missing={missing[:10]}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
