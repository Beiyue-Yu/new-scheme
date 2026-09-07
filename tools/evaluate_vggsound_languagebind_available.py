#!/usr/bin/env python3
"""Evaluate the available VGGSound LanguageBind cache on the test split.

This is deliberately a diagnostic evaluator.  It performs frozen
video-to-text cosine matching and reports the test coverage explicitly; it
does not train MSTR and it never presents a partially covered split as an
official complete benchmark result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# ``python tools/<script>.py`` places ``tools`` rather than the repository
# root on ``sys.path``.  Keep project imports stable for that normal command.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_languagebind_official_val import (
    class_balanced_accuracy,
    harmonic_mean,
)
from src.languagebind_segments import temporal_class_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_rows(split_dir: Path) -> List[dict]:
    rows = []
    for split in ("stage_2_test_seen", "stage_2_test_unseen"):
        path = split_dir / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                row["protocol_split"] = split
                row["class_id"] = int(row["label_code"])
                rows.append(row)
    if not rows or len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Test manifests are empty or contain duplicate filenames")
    return rows


def _prediction_coverage(predictions: np.ndarray,
                         class_ids: np.ndarray) -> float:
    return float(np.mean([
        np.any(predictions == class_id) for class_id in class_ids
    ]))


def evaluate(cache_path: Path, split_dir: Path) -> Dict[str, object]:
    rows = _read_rows(split_dir)
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"]))
        failures = json.loads(str(cache["failures_json"]))
        names = cache["video_names"].astype(str)
        embeddings = torch.from_numpy(
            cache["video_embeddings"].astype(np.float32))
        text = torch.from_numpy(cache["text_embeddings"].astype(np.float32))
        cache_targets = cache["class_ids"].astype(np.int64)
        class_names = cache["class_names"].astype(str)

    if len(set(names.tolist())) != len(names):
        raise ValueError("LanguageBind cache contains duplicate video names")
    index = {name: i for i, name in enumerate(names)}
    available = [row for row in rows if row["filename"] in index]
    missing = [row for row in rows if row["filename"] not in index]
    if not available:
        raise ValueError("No test samples are present in the LanguageBind cache")

    indices = np.asarray([index[row["filename"]] for row in available], dtype=np.int64)
    embeddings = embeddings[indices]
    targets = np.asarray([row["class_id"] for row in available], dtype=np.int64)
    if not np.array_equal(targets, cache_targets[indices]):
        raise ValueError("Cache labels do not match the test manifest")

    class_ids = np.asarray(metadata["selected_class_ids"], dtype=np.int64)
    if len(class_ids) != len(class_names) or text.shape[0] != len(class_ids):
        raise ValueError("Cache class metadata and text embeddings are misaligned")
    seen_ids = np.asarray(sorted({row["class_id"] for row in rows
                                  if row["protocol_split"] == "stage_2_test_seen"}),
                         dtype=np.int64)
    unseen_ids = np.asarray(sorted({row["class_id"] for row in rows
                                    if row["protocol_split"] == "stage_2_test_unseen"}),
                           dtype=np.int64)
    if np.intersect1d(seen_ids, unseen_ids).size:
        raise ValueError("Seen and Unseen test class IDs overlap")

    logits = temporal_class_logits(embeddings, text)
    predictions = class_ids[logits.argmax(dim=1).numpy()]
    seen_mask = np.isin(targets, seen_ids)
    unseen_mask = np.isin(targets, unseen_ids)
    seen = class_balanced_accuracy(predictions[seen_mask], targets[seen_mask], seen_ids)
    unseen = class_balanced_accuracy(
        predictions[unseen_mask], targets[unseen_mask], unseen_ids)
    unseen_columns = np.asarray([
        int(np.flatnonzero(class_ids == class_id)[0]) for class_id in unseen_ids
    ], dtype=np.int64)
    zsl_predictions = unseen_ids[
        logits[unseen_mask][:, unseen_columns].argmax(dim=1).numpy()]
    zsl = class_balanced_accuracy(
        zsl_predictions, targets[unseen_mask], unseen_ids)

    per_class = []
    for class_id in class_ids:
        mask = targets == class_id
        if mask.any():
            per_class.append({
                "class_id": int(class_id),
                "class_name": str(class_names[class_id]),
                "split": "seen" if class_id in set(seen_ids.tolist()) else "unseen",
                "samples": int(mask.sum()),
                "gzsl_accuracy": float(np.mean(predictions[mask] == class_id)),
            })

    missing_names = "\n".join(sorted(row["filename"] for row in missing)) + "\n"
    return {
        "report_version": 1,
        "protocol": "diagnostic frozen LanguageBind video-text matching on available VGGSound Stage 2 test samples",
        "official_complete": False,
        "repository_data_fitting": False,
        "mstr_training_performed": False,
        "source_cache": str(cache_path),
        "source_metadata": metadata,
        "encoded_cache_videos": int(len(names)),
        "decode_failures": failures,
        "official_test_samples": int(len(rows)),
        "available_test_samples": int(len(available)),
        "missing_test_samples": int(len(missing)),
        "test_coverage": float(len(available) / len(rows)),
        "missing_seen_samples": int(sum(row["protocol_split"] == "stage_2_test_seen"
                                         for row in missing)),
        "missing_unseen_samples": int(sum(row["protocol_split"] == "stage_2_test_unseen"
                                           for row in missing)),
        "missing_test_sample_sha256": hashlib.sha256(
            missing_names.encode("utf-8")).hexdigest(),
        "missing_test_sample_preview": sorted(row["filename"] for row in missing)[:20],
        "seen_classes": int(len(seen_ids)),
        "unseen_classes": int(len(unseen_ids)),
        "candidate_classes": int(len(class_ids)),
        "temporal_mean": {
            "seen": seen,
            "unseen": unseen,
            "hm": harmonic_mean(seen, unseen),
            "zsl": zsl,
            "prediction_coverage": _prediction_coverage(predictions, class_ids),
        },
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args.cache, args.split_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    metrics = report["temporal_mean"]
    print(
        f"coverage={100 * report['test_coverage']:.2f}% "
        f"seen={100 * metrics['seen']:.2f} "
        f"unseen={100 * metrics['unseen']:.2f} "
        f"hm={100 * metrics['hm']:.2f} "
        f"zsl={100 * metrics['zsl']:.2f}", flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
