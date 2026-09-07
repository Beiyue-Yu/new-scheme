#!/usr/bin/env python3
"""Extract genuine UCF temporal segments with a frozen LanguageBind encoder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import importlib.machinery
import os
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.languagebind_segments import (
    PROMPT_TEMPLATES,
    class_prompt_groups,
    decode_video_segments,
    ensemble_prompt_embeddings,
    preprocess_video_frames,
)


VIDEO_GROUP = re.compile(r"_g([0-9]+)_", re.IGNORECASE)


@dataclass(frozen=True)
class ManifestRow:
    filename: str
    label: str
    class_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen LanguageBind multi-segment extraction for UCF")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("avgzsl_benchmark_datasets/UCF/class-split/main_split/"
                     "stage_1_train.csv"))
    parser.add_argument(
        "--video_root", type=Path,
        default=Path("raw_datasets/UCF101/videos"))
    parser.add_argument(
        "--model_dir", type=Path,
        default=Path("model_cache/LanguageBind_Video_FT"))
    parser.add_argument(
        "--vendor_dir", type=Path,
        default=Path("model_cache/LanguageBind"))
    parser.add_argument(
        "--dependency_dir", type=Path,
        default=Path("model_cache/languagebind_python"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source_scope", default="UCF Stage A training manifest only",
        help="Provenance label recorded verbatim in cache metadata")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num_segments", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--batch_videos", type=int, default=1)
    parser.add_argument(
        "--max_groups_per_class", type=int, default=0,
        help="Deterministic pilot subset; 0 extracts every manifest video")
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument(
        "--max_new_videos", type=int, default=0,
        help="Cleanly exit after this many new videos; 0 processes all pending")
    parser.add_argument(
        "--retry_failures", action="store_true",
        help="Retry videos recorded as failures in an existing cache")
    args = parser.parse_args()
    positive = (
        args.num_segments, args.num_frames, args.batch_videos,
        args.checkpoint_every)
    if (min(positive) <= 0 or args.max_groups_per_class < 0
            or args.max_new_videos < 0):
        parser.error("segment, frame, batch, checkpoint values must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    return args


def read_manifest(path: Path) -> List[ManifestRow]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = [ManifestRow(
            filename=row["filename"].strip(),
            label=row["label"].strip(),
            class_id=int(row["label_code"]),
        ) for row in csv.DictReader(source)]
    if not rows:
        raise ValueError(f"UCF manifest is empty: {path}")
    if len({row.filename for row in rows}) != len(rows):
        raise ValueError(f"UCF manifest contains duplicate filenames: {path}")
    label_ids: Dict[str, int] = {}
    for row in rows:
        previous = label_ids.setdefault(row.label, row.class_id)
        if previous != row.class_id:
            raise ValueError(f"Class {row.label} has inconsistent label codes")
    return rows


def _group_id(filename: str) -> int:
    match = VIDEO_GROUP.search(filename)
    if match is None:
        raise ValueError(f"Cannot parse UCF video group from {filename!r}")
    return int(match.group(1))


def select_manifest_rows(rows: Sequence[ManifestRow],
                         max_groups_per_class: int) -> List[ManifestRow]:
    if max_groups_per_class == 0:
        return list(rows)
    groups_by_class: Dict[int, set] = defaultdict(set)
    for row in rows:
        groups_by_class[row.class_id].add(_group_id(row.filename))
    selected_groups = {
        class_id: set(sorted(groups)[:max_groups_per_class])
        for class_id, groups in groups_by_class.items()
    }
    return [
        row for row in rows
        if _group_id(row.filename) in selected_groups[row.class_id]
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_path(root: Path, row: ManifestRow) -> Path:
    return root / row.label / f"{row.filename}.avi"


def validate_video_coverage(root: Path, rows: Sequence[ManifestRow]) -> None:
    missing = [str(_video_path(root, row)) for row in rows
               if not _video_path(root, row).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(rows)} manifest videos; first: {preview}")


def _bootstrap_languagebind(vendor_dir: Path, dependency_dir: Path):
    dependency_path = str(dependency_dir.resolve())
    if dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)
    # The upstream top-level package imports its obsolete pytorchvideo
    # processor even when callers only need the model. Load the official video
    # modules as a narrow package so current torchvision remains untouched.
    package_path = vendor_dir.resolve() / "languagebind"
    if "languagebind" not in sys.modules:
        package = types.ModuleType("languagebind")
        package.__path__ = [str(package_path)]
        package.__package__ = "languagebind"
        package.__spec__ = importlib.machinery.ModuleSpec(
            "languagebind", loader=None, is_package=True)
        sys.modules["languagebind"] = package
    from languagebind.video.modeling_video import LanguageBindVideo
    from languagebind.video.tokenization_video import LanguageBindVideoTokenizer
    return LanguageBindVideo, LanguageBindVideoTokenizer


@torch.inference_mode()
def _encode_text(model, tokenizer, class_names: Sequence[str],
                 device: torch.device) -> torch.Tensor:
    prompt_groups = class_prompt_groups(class_names)
    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    tokens = tokenizer(
        flat_prompts, max_length=77, padding="max_length", truncation=True,
        return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=device.type == "cuda"):
        embeddings = model.get_text_features(**tokens)
    embeddings = embeddings.reshape(
        len(class_names), len(PROMPT_TEMPLATES), -1).float()
    return ensemble_prompt_embeddings(embeddings).cpu()


@torch.inference_mode()
def _encode_video_batch(model, videos: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    batch_size, segments = videos.shape[:2]
    values = videos.flatten(0, 1).to(device, non_blocking=True)
    with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=device.type == "cuda"):
        features = model.get_image_features(pixel_values=values)
    features = F.normalize(features.float(), dim=-1)
    return features.reshape(batch_size, segments, -1).cpu()


def _save_cache(path: Path, metadata: Dict[str, object],
                class_names: Sequence[str], text_embeddings: torch.Tensor,
                records: Sequence[Dict[str, object]],
                failures: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        video_embeddings = np.stack([
            record["embedding"] for record in records]).astype(np.float16)
        frame_indices = np.stack([
            record["frame_indices"] for record in records]).astype(np.int32)
    else:
        dimension = int(text_embeddings.shape[1])
        video_embeddings = np.empty(
            (0, metadata["num_segments"], dimension), dtype=np.float16)
        frame_indices = np.empty(
            (0, metadata["num_segments"], metadata["num_frames"]),
            dtype=np.int32)
    payload = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "class_names": np.asarray(class_names),
        "text_embeddings": text_embeddings.numpy().astype(np.float32),
        "video_names": np.asarray([record["filename"] for record in records]),
        "labels": np.asarray([record["label"] for record in records]),
        "class_ids": np.asarray(
            [record["class_id"] for record in records], dtype=np.int64),
        "video_embeddings": video_embeddings,
        "frame_indices": frame_indices,
        "failures_json": np.asarray(json.dumps(list(failures), sort_keys=True)),
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, **payload)
    os.replace(temporary, path)


def _load_existing(path: Path, metadata: Dict[str, object]
                   ) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with np.load(path, allow_pickle=False) as cache:
        existing_metadata = json.loads(str(cache["metadata_json"]))
        comparable_keys = (
            "manifest_sha256", "model_sha256", "num_segments", "num_frames",
            "max_groups_per_class", "prompt_templates")
        if any(existing_metadata.get(key) != metadata.get(key)
               for key in comparable_keys):
            raise ValueError(
                f"Existing cache configuration does not match this run: {path}")
        records = [{
            "filename": str(filename),
            "label": str(label),
            "class_id": int(class_id),
            "embedding": embedding.astype(np.float32),
            "frame_indices": indices.astype(np.int64),
        } for filename, label, class_id, embedding, indices in zip(
            cache["video_names"], cache["labels"], cache["class_ids"],
            cache["video_embeddings"], cache["frame_indices"])]
        failures = json.loads(str(cache["failures_json"]))
    return records, failures


def _flush_batch(model, batch: List[Tuple[ManifestRow, torch.Tensor, np.ndarray]],
                 records: List[Dict[str, object]], device: torch.device) -> None:
    if not batch:
        return
    videos = torch.stack([item[1] for item in batch])
    embeddings = _encode_video_batch(model, videos, device).numpy()
    for (row, _, indices), embedding in zip(batch, embeddings):
        records.append({
            "filename": row.filename,
            "label": row.label,
            "class_id": row.class_id,
            "embedding": embedding,
            "frame_indices": indices,
        })
    batch.clear()


def main() -> None:
    args = parse_args()
    rows = select_manifest_rows(
        read_manifest(args.manifest), args.max_groups_per_class)
    validate_video_coverage(args.video_root, rows)
    model_file = args.model_dir / "pytorch_model.bin"
    if not model_file.is_file() or (args.model_dir / "pytorch_model.bin.aria2").exists():
        raise FileNotFoundError(f"LanguageBind checkpoint is incomplete: {model_file}")
    class_rows = sorted({(row.class_id, row.label) for row in rows})
    class_names = [label for _, label in class_rows]
    metadata = {
        "format_version": 1,
        "source_scope": args.source_scope,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "model": str(args.model_dir),
        "model_sha256": _sha256(model_file),
        "num_segments": args.num_segments,
        "num_frames": args.num_frames,
        "max_groups_per_class": args.max_groups_per_class,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "selected_videos": len(rows),
        "selected_classes": len(class_names),
        "selected_class_ids": [class_id for class_id, _ in class_rows],
    }
    LanguageBindVideo, LanguageBindVideoTokenizer = _bootstrap_languagebind(
        args.vendor_dir, args.dependency_dir)
    device = torch.device(args.device)
    print(f"startup=loading_model device={device}", flush=True)
    tokenizer = LanguageBindVideoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True)
    model = LanguageBindVideo.from_pretrained(
        args.model_dir, local_files_only=True,
        low_cpu_mem_usage=True).eval().to(device)
    print("startup=model_loaded", flush=True)
    text_embeddings = _encode_text(model, tokenizer, class_names, device)
    print("startup=text_encoded", flush=True)
    records, failures = _load_existing(args.output, metadata)
    print("startup=cache_loaded", flush=True)
    if args.retry_failures:
        failures = []
    completed = {record["filename"] for record in records}
    failed = {record["filename"] for record in failures}
    pending = [row for row in rows
               if row.filename not in completed and row.filename not in failed]
    pending_total = len(pending)
    if args.max_new_videos:
        pending = pending[:args.max_new_videos]
    print(
        f"selected={len(rows)} completed={len(records)} "
        f"pending_total={pending_total} this_run={len(pending)} "
        f"classes={len(class_names)} device={device}", flush=True)
    batch: List[Tuple[ManifestRow, torch.Tensor, np.ndarray]] = []
    since_checkpoint = 0
    for position, row in enumerate(pending, start=1):
        path = _video_path(args.video_root, row)
        try:
            frames, indices = decode_video_segments(
                path, args.num_segments, args.num_frames)
            values = preprocess_video_frames(frames)
            batch.append((row, values, indices))
        except Exception as error:
            failures.append({
                "filename": row.filename,
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"decode_failed={row.filename} error={error}", flush=True)
        if len(batch) >= args.batch_videos:
            before = len(records)
            _flush_batch(model, batch, records, device)
            since_checkpoint += len(records) - before
        if since_checkpoint >= args.checkpoint_every:
            _save_cache(
                args.output, metadata, class_names, text_embeddings,
                records, failures)
            since_checkpoint = 0
            print(
                f"progress={len(records) + len(failures)}/{len(rows)} "
                f"encoded={len(records)} failures={len(failures)}", flush=True)
    _flush_batch(model, batch, records, device)
    metadata["encoded_videos"] = len(records)
    metadata["failed_videos"] = len(failures)
    _save_cache(
        args.output, metadata, class_names, text_embeddings, records, failures)
    print(
        f"saved={args.output} encoded={len(records)} failures={len(failures)}",
        flush=True)


if __name__ == "__main__":
    main()
