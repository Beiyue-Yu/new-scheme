#!/usr/bin/env python3
"""Extract LanguageBind video/text features for the complete VGGSound GZSL task.

VGGSound is distributed as YouTube IDs plus temporal offsets rather than local
video files.  This extractor keeps the benchmark filename as the primary key,
downloads one short temporal clip when the source video is not already present,
and stores resumable LanguageBind segment embeddings.  No array-position join
is used: the downstream builder matches these records by ``filename``.

The script intentionally fails when ``yt-dlp`` is unavailable instead of
silently falling back to the old 512-D video features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import shutil
import tempfile
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from extract_languagebind_ucf_segments import (
    _bootstrap_languagebind,
    _encode_text,
    _encode_video_batch,
)
from src.languagebind_segments import (
    PROMPT_TEMPLATES,
    class_prompt_groups,
    ensemble_prompt_embeddings,
    preprocess_video_frames,
    segment_frame_indices,
)


SPLIT_NAMES = (
    "stage_1_train",
    "stage_1_val_seen",
    "stage_1_val_unseen",
    "stage_2_train",
    "stage_2_test_seen",
    "stage_2_test_unseen",
)


@dataclass(frozen=True)
class ManifestRow:
    filename: str
    label: str
    class_id: int
    youtube_id: str
    start_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--class_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--error_log", type=Path, default=None,
        help="Optional append-only log for per-video extraction errors.")
    parser.add_argument("--video_root", type=Path, default=None,
                        help="Optional local directory containing filename.mp4 files")
    parser.add_argument(
        "--names_file", type=Path,
        help="Optional newline-delimited benchmark filenames to process. This "
             "keeps other manifest rows pending, which is required when local "
             "videos are materialized one archive shard at a time.")
    parser.add_argument("--model_dir", type=Path,
                        default=Path("model_cache/LanguageBind_Video_FT"))
    parser.add_argument("--vendor_dir", type=Path,
                        default=Path("model_cache/LanguageBind"))
    parser.add_argument("--dependency_dir", type=Path,
                        default=Path("model_cache/languagebind_python"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num_segments", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--clip_seconds", type=float, default=10.0)
    parser.add_argument("--download_timeout", type=int, default=45,
                        help="Hard timeout per YouTube request")
    parser.add_argument("--batch_videos", type=int, default=4)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--max_new_videos", type=int, default=0,
                        help="Bounded pilot; 0 processes every pending sample")
    parser.add_argument("--retry_failures", action="store_true")
    parser.add_argument("--keep_downloads", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report local-video/dependency coverage")
    args = parser.parse_args()
    if min(args.num_segments, args.num_frames, args.batch_videos,
           args.checkpoint_every) <= 0:
        parser.error("segment, frame, batch and checkpoint values must be positive")
    if (args.clip_seconds <= 0 or args.max_new_videos < 0
            or args.download_timeout <= 0):
        parser.error("clip_seconds/timeout must be positive and max_new_videos non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    return args


def _find_local_video(video_root: Path, row: ManifestRow) -> Path:
    """Accept the common VGGSound downloader layouts without guessing IDs."""
    normalized_label = re.sub(r"[^a-z0-9]+", "_", row.label.lower()).strip("_")
    start = int(row.start_seconds)
    end = start + 10
    tool_name = f"v{row.youtube_id}_{start}_{end}_out"
    candidates = (
        video_root / row.label / f"{row.filename}.mp4",
        video_root / row.label / f"{row.filename}.avi",
        video_root / row.label / row.filename,
        video_root / "train" / normalized_label / f"{tool_name}.mkv",
        video_root / "test" / normalized_label / f"{tool_name}.mkv",
        video_root / "train" / normalized_label / f"{tool_name}.mp4",
        video_root / "test" / normalized_label / f"{tool_name}.mp4",
        video_root / f"{row.filename}.mp4",
        video_root / f"{row.filename}.avi",
        video_root / row.filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"local VGGSound clip missing for {row.filename}; checked "
        + ", ".join(str(path) for path in candidates))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(split_dir: Path) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    for split_name in SPLIT_NAMES:
        path = split_dir / f"{split_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing VGGSound manifest: {path}")
        with path.open(newline="", encoding="utf-8") as source:
            for record in csv.DictReader(source):
                rows.append(ManifestRow(
                    filename=record["filename"].strip(),
                    label=record["label"].strip(),
                    class_id=int(record["label_code"]),
                    youtube_id=record["youtube_id"].strip(),
                    start_seconds=float(record["start_seconds"]),
                ))
    if not rows:
        raise ValueError("VGGSound manifests are empty")
    by_filename: Dict[str, ManifestRow] = {}
    for row in rows:
        previous = by_filename.setdefault(row.filename, row)
        if previous != row:
            raise ValueError(f"Conflicting duplicate filename: {row.filename}")
    return list(by_filename.values())


def read_classes(path: Path, rows: Sequence[ManifestRow]) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not names:
        raise ValueError(f"Class file is empty: {path}")
    for row in rows:
        if row.class_id < 0 or row.class_id >= len(names):
            raise ValueError(f"Class id {row.class_id} is outside {path}")
        if names[row.class_id] != row.label:
            raise ValueError(
                f"Class mapping mismatch for {row.filename}: "
                f"id {row.class_id} is {names[row.class_id]!r}, row says {row.label!r}")
    return names


def read_selected_names(path: Path | None, rows: Sequence[ManifestRow]) -> set[str] | None:
    """Read a strict manifest subset without treating other rows as failures."""
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Selected-name file does not exist: {path}")
    names = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()}
    available = {row.filename for row in rows}
    unknown = sorted(names - available)
    if unknown:
        raise ValueError(
            f"Selected-name file contains {len(unknown)} names outside the VGGSound "
            f"manifest; first: {unknown[:5]}")
    if not names:
        raise ValueError(f"Selected-name file is empty: {path}")
    return names


class _DownloadTimeout(RuntimeError):
    pass


def _download_alarm(_signum, _frame):
    raise _DownloadTimeout("yt-dlp request exceeded the hard timeout")


def _download_clip(row: ManifestRow, destination: Path, clip_seconds: float,
                   timeout_seconds: int) -> Path:
    """Download one temporal clip using yt-dlp and return the materialized file."""
    try:
        import yt_dlp
        from yt_dlp.utils import download_range_func
    except ImportError as error:
        raise RuntimeError(
            "VGGSound extraction requires yt-dlp; install it in the active "
            "environment before starting full extraction") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    stem = destination.with_suffix("")
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 30,
        "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "outtmpl": str(stem) + ".%(ext)s",
        "download_ranges": download_range_func(
            None, [(row.start_seconds, row.start_seconds + clip_seconds)]),
        "force_keyframes_at_cuts": False,
        "merge_output_format": "mp4",
    }
    previous_handler = signal.signal(signal.SIGALRM, _download_alarm)
    signal.alarm(timeout_seconds)
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([f"https://www.youtube.com/watch?v={row.youtube_id}"])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    candidates = sorted(destination.parent.glob(stem.name + ".*"))
    candidates = [path for path in candidates if path.suffix not in {".part", ".ytdl"}]
    if not candidates:
        raise FileNotFoundError(f"yt-dlp produced no clip for {row.filename}")
    source = candidates[0]
    if source != destination:
        source.replace(destination)
    return destination


def _decode_video(path: Path, num_segments: int, num_frames: int):
    """Decode RGB frames with PyAV, avoiding a hard decord dependency."""
    import av
    container = av.open(str(path))
    try:
        frames = [frame.to_rgb().to_ndarray() for frame in container.decode(video=0)]
    finally:
        container.close()
    if not frames:
        raise ValueError(f"No video frames decoded from {path}")
    values = np.stack(frames)
    indices = segment_frame_indices(len(values), num_segments, num_frames)
    sampled = values[indices.reshape(-1)].reshape(
        num_segments, num_frames, *values.shape[1:])
    return preprocess_video_frames(sampled), indices


def _save_cache(path: Path, metadata: dict, class_names: Sequence[str],
                text_embeddings: torch.Tensor, records: Sequence[dict],
                failures: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dimension = int(text_embeddings.shape[1])
    if records:
        video_embeddings = np.stack([r["embedding"] for r in records]).astype(np.float16)
        frame_indices = np.stack([r["frame_indices"] for r in records]).astype(np.int32)
    else:
        video_embeddings = np.empty((0, metadata["num_segments"], dimension), dtype=np.float16)
        frame_indices = np.empty((0, metadata["num_segments"], metadata["num_frames"]), dtype=np.int32)
    payload = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "class_names": np.asarray(class_names),
        "text_embeddings": text_embeddings.numpy().astype(np.float32),
        "video_names": np.asarray([r["filename"] for r in records]),
        "labels": np.asarray([r["label"] for r in records]),
        "class_ids": np.asarray([r["class_id"] for r in records], dtype=np.int64),
        "youtube_ids": np.asarray([r["youtube_id"] for r in records]),
        "start_seconds": np.asarray([r["start_seconds"] for r in records], dtype=np.float32),
        "video_embeddings": video_embeddings,
        "frame_indices": frame_indices,
        "failures_json": np.asarray(json.dumps(list(failures), sort_keys=True)),
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _load_existing(path: Path, metadata: dict):
    if not path.exists():
        return [], []
    with np.load(path, allow_pickle=False) as cache:
        old = json.loads(str(cache["metadata_json"]))
        for key in ("manifest_sha256", "model_sha256", "num_segments", "num_frames",
                    "clip_seconds", "class_file_sha256"):
            if old.get(key) != metadata.get(key):
                raise ValueError(f"Existing cache configuration mismatch: {path}")
        records = []
        for values in zip(cache["video_names"], cache["labels"], cache["class_ids"],
                          cache["youtube_ids"], cache["start_seconds"],
                          cache["video_embeddings"], cache["frame_indices"]):
            filename, label, class_id, youtube_id, start, embedding, indices = values
            records.append({"filename": str(filename), "label": str(label),
                            "class_id": int(class_id), "youtube_id": str(youtube_id),
                            "start_seconds": float(start),
                            "embedding": embedding.astype(np.float32),
                            "frame_indices": indices.astype(np.int64)})
        failures = json.loads(str(cache["failures_json"]))
    return records, failures


def main() -> None:
    args = parse_args()
    rows = read_rows(args.split_dir)
    class_names = read_classes(args.class_file, rows)
    selected_names = read_selected_names(args.names_file, rows)
    if args.dry_run:
        local_present = 0
        if args.video_root and args.video_root.is_dir():
            for row in rows:
                try:
                    _find_local_video(args.video_root, row)
                    local_present += 1
                except FileNotFoundError:
                    pass
        try:
            import yt_dlp  # noqa: F401
            downloader_available = True
        except ImportError:
            downloader_available = False
        report = {
            "dataset": "VGGSound",
            "manifest_samples": len(rows),
            "classes": len(class_names),
            "local_video_root": str(args.video_root) if args.video_root else None,
            "local_videos_present": local_present,
            "local_video_coverage": local_present / max(len(rows), 1),
            "yt_dlp_available": downloader_available,
            "ready_for_extraction": bool(local_present == len(rows)
                                          or downloader_available),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    manifest_hash = hashlib.sha256("\n".join(
        f"{r.filename}|{r.label}|{r.class_id}|{r.youtube_id}|{r.start_seconds}"
        for r in rows).encode()).hexdigest()
    model_file = args.model_dir / "pytorch_model.bin"
    if not model_file.is_file():
        raise FileNotFoundError(f"LanguageBind checkpoint missing: {model_file}")
    metadata = {
        "format_version": 1,
        "dataset": "VGGSound",
        "source_scope": "VGGSound main_split complete GZSL manifests",
        "manifest_sha256": manifest_hash,
        "class_file_sha256": _sha256(args.class_file),
        "model": str(args.model_dir),
        "model_sha256": _sha256(model_file),
        "num_segments": args.num_segments,
        "num_frames": args.num_frames,
        "clip_seconds": args.clip_seconds,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "selected_videos": len(rows),
        "selected_classes": len(class_names),
        "selected_class_ids": list(range(len(class_names))),
    }
    # Import yt-dlp before adding the vendored LanguageBind dependency path.
    # That checkout contains an old private requests package which can shadow
    # yt-dlp's networking stack and defeat its socket timeout.
    try:
        import yt_dlp  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install yt-dlp before VGGSound extraction") from error
    LanguageBindVideo, LanguageBindVideoTokenizer = _bootstrap_languagebind(
        args.vendor_dir, args.dependency_dir)
    device = torch.device(args.device)
    print(f"startup=loading_model device={device}", flush=True)
    tokenizer = LanguageBindVideoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = LanguageBindVideo.from_pretrained(args.model_dir, local_files_only=True,
                                              low_cpu_mem_usage=True).eval().to(device)
    print("startup=model_loaded", flush=True)
    text_embeddings = _encode_text(model, tokenizer, class_names, device)
    print("startup=text_encoded", flush=True)
    records, failures = _load_existing(args.output, metadata)
    if args.retry_failures:
        failures = []
    completed = {r["filename"] for r in records}
    failed = {r["filename"] for r in failures}
    pending = [r for r in rows if r.filename not in completed and r.filename not in failed]
    if selected_names is not None:
        pending = [row for row in pending if row.filename in selected_names]
    if args.max_new_videos:
        pending = pending[:args.max_new_videos]
    print(f"selected={len(rows)} completed={len(records)} pending={len(pending)} "
          f"failures={len(failures)} classes={len(class_names)}", flush=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="vggsound_lb_"))
    error_log = args.error_log or args.output.with_name("extraction_errors.log")
    error_log.parent.mkdir(parents=True, exist_ok=True)
    decoded_batch = []
    try:
        for position, row in enumerate(pending, start=1):
            local_path = None
            try:
                if args.video_root:
                    local_path = _find_local_video(args.video_root, row)
                else:
                    local_path = _download_clip(
                        row, temporary_root / f"{row.filename}.mp4",
                        args.clip_seconds, args.download_timeout)
                values, indices = _decode_video(local_path, args.num_segments, args.num_frames)
                decoded_batch.append((row, values, indices))
                if len(decoded_batch) >= args.batch_videos:
                    embeddings = _encode_video_batch(model,
                                                     torch.stack([x[1] for x in decoded_batch]),
                                                     device).numpy()
                    for (item, embedding) in zip(decoded_batch, embeddings):
                        current, values, indices = item
                        records.append({"filename": current.filename, "label": current.label,
                                        "class_id": current.class_id,
                                        "youtube_id": current.youtube_id,
                                        "start_seconds": current.start_seconds,
                                        "embedding": embedding,
                                        "frame_indices": indices})
                    decoded_batch.clear()
            except Exception as error:
                failures.append({"filename": row.filename,
                                 "error": f"{type(error).__name__}: {error}"})
                with error_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{row.filename}\t{type(error).__name__}: {error}\n")
                print(f"extract_failed={row.filename} error={error}", flush=True)
            if position % args.checkpoint_every == 0:
                metadata["encoded_videos"] = len(records)
                metadata["failed_videos"] = len(failures)
                _save_cache(args.output, metadata, class_names, text_embeddings, records, failures)
                print(f"progress={position}/{len(pending)} encoded={len(records)} "
                      f"failures={len(failures)}", flush=True)
        if decoded_batch:
            embeddings = _encode_video_batch(model,
                                             torch.stack([x[1] for x in decoded_batch]),
                                             device).numpy()
            for (item, embedding) in zip(decoded_batch, embeddings):
                current, values, indices = item
                records.append({"filename": current.filename, "label": current.label,
                                "class_id": current.class_id,
                                "youtube_id": current.youtube_id,
                                "start_seconds": current.start_seconds,
                                "embedding": embedding,
                                "frame_indices": indices})
        metadata["encoded_videos"] = len(records)
        metadata["failed_videos"] = len(failures)
        metadata["coverage_fraction"] = len(records) / max(len(rows), 1)
        _save_cache(args.output, metadata, class_names, text_embeddings, records, failures)
        print(f"saved={args.output} encoded={len(records)} failures={len(failures)} "
              f"coverage={metadata['coverage_fraction']:.6f}", flush=True)
    finally:
        if args.keep_downloads:
            print(f"downloads={temporary_root}", flush=True)
        else:
            shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
