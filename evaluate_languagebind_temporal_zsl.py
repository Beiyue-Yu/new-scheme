#!/usr/bin/env python3
"""Evaluate frozen LanguageBind temporal features without repository-data fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.languagebind_segments import temporal_class_logits
from train_class_conditioned_evidence import make_class_folds


GATE_BALANCED_ACCURACY = 0.20
GATE_WORST_FOLD_UNSEEN = 0.15
GATE_PREDICTION_COVERAGE = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen LanguageBind UCF temporal ZSL gate")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.folds < 2 or args.seed < 0:
        parser.error("folds must be at least two and seed must be non-negative")
    return args


def class_balanced_accuracy(predictions: np.ndarray, targets: np.ndarray,
                            class_ids: Sequence[int]) -> float:
    values = []
    for class_id in class_ids:
        mask = targets == class_id
        if not mask.any():
            raise ValueError(f"Evaluation data has no samples for class {class_id}")
        values.append(float(np.mean(predictions[mask] == class_id)))
    return float(np.mean(values))


def prediction_diagnostics(predictions: np.ndarray,
                           candidate_ids: np.ndarray) -> Dict[str, float]:
    counts = np.asarray([
        np.sum(predictions == class_id) for class_id in candidate_ids
    ], dtype=np.int64)
    return {
        "prediction_coverage": float(np.mean(counts > 0)),
        "dominant_prediction_fraction": float(counts.max() / max(1, counts.sum())),
    }


def _fold_metrics(predictions: np.ndarray, targets: np.ndarray,
                  class_ids: np.ndarray, folds: int,
                  seed: int) -> Sequence[Dict[str, object]]:
    reports = []
    for fold_index, unseen_ids in enumerate(
            make_class_folds(class_ids, folds, seed), start=1):
        unseen_set = set(unseen_ids.tolist())
        seen_ids = np.asarray([
            class_id for class_id in class_ids if class_id not in unseen_set
        ], dtype=np.int64)
        seen = class_balanced_accuracy(predictions, targets, seen_ids)
        unseen = class_balanced_accuracy(predictions, targets, unseen_ids)
        hm = 0.0 if seen + unseen == 0.0 else 2.0 * seen * unseen / (seen + unseen)
        unseen_sample_predictions = predictions[np.isin(targets, unseen_ids)]
        unseen_coverage = float(np.mean([
            np.any(unseen_sample_predictions == class_id) for class_id in unseen_ids
        ]))
        reports.append({
            "fold": fold_index,
            "seen_class_ids": seen_ids.tolist(),
            "unseen_class_ids": unseen_ids.tolist(),
            "seen": seen,
            "unseen": unseen,
            "hm": hm,
            "unseen_prediction_coverage": unseen_coverage,
        })
    return reports


def evaluate_cache(cache_path: Path, folds: int = 3,
                   seed: int = 1) -> Dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"]))
        failures = json.loads(str(cache["failures_json"]))
        embeddings = torch.from_numpy(cache["video_embeddings"].astype(np.float32))
        text = torch.from_numpy(cache["text_embeddings"].astype(np.float32))
        targets = cache["class_ids"].astype(np.int64)
        class_names = cache["class_names"].astype(str)
    class_ids = np.asarray(metadata["selected_class_ids"], dtype=np.int64)
    if len(class_ids) != len(class_names) or text.shape[0] != len(class_ids):
        raise ValueError("Cache class metadata and text embeddings are not aligned")
    if embeddings.shape[0] != len(targets) or embeddings.shape[0] == 0:
        raise ValueError("Cache contains no aligned video embeddings")
    if set(np.unique(targets).tolist()) != set(class_ids.tolist()):
        raise ValueError("Cache does not cover every selected class")
    normalized_segments = F.normalize(embeddings, dim=-1)
    normalized_text = F.normalize(text, dim=-1)
    segment_logits = torch.einsum(
        "nsd,cd->nsc", normalized_segments, normalized_text)
    mean_logits = temporal_class_logits(embeddings, text)
    mean_predictions = class_ids[mean_logits.argmax(dim=1).numpy()]
    segment_reports = []
    segment_correct = []
    for segment in range(embeddings.shape[1]):
        predictions = class_ids[segment_logits[:, segment].argmax(dim=1).numpy()]
        segment_correct.append(predictions == targets)
        segment_reports.append({
            "segment": segment + 1,
            "balanced_accuracy": class_balanced_accuracy(
                predictions, targets, class_ids),
            **prediction_diagnostics(predictions, class_ids),
        })
    balanced_accuracy = class_balanced_accuracy(
        mean_predictions, targets, class_ids)
    diagnostics = prediction_diagnostics(mean_predictions, class_ids)
    fold_reports = _fold_metrics(
        mean_predictions, targets, class_ids, folds, seed)
    per_class = []
    for class_id, class_name in zip(class_ids, class_names):
        mask = targets == class_id
        per_class.append({
            "class_id": int(class_id),
            "class_name": class_name,
            "samples": int(mask.sum()),
            "accuracy": float(np.mean(mean_predictions[mask] == class_id)),
        })
    worst_fold_unseen = min(fold["unseen"] for fold in fold_reports)
    gate_checks = {
        "balanced_accuracy": balanced_accuracy >= GATE_BALANCED_ACCURACY,
        "worst_fold_unseen": worst_fold_unseen >= GATE_WORST_FOLD_UNSEEN,
        "prediction_coverage": (
            diagnostics["prediction_coverage"] >= GATE_PREDICTION_COVERAGE),
    }
    return {
        "report_version": 1,
        "protocol": "frozen Stage-A-train-only temporal gate",
        "repository_data_fitting": False,
        "official_stage_a_validation_loaded": False,
        "stage_b_loaded": False,
        "test_split_loaded": False,
        "source_cache": str(cache_path),
        "source_metadata": metadata,
        "encoded_videos": int(len(targets)),
        "decode_failures": failures,
        "candidate_classes": int(len(class_ids)),
        "chance_accuracy": float(1.0 / len(class_ids)),
        "temporal_mean": {
            "balanced_accuracy": balanced_accuracy,
            **diagnostics,
            "worst_fold_unseen": worst_fold_unseen,
            "mean_fold_hm": float(np.mean([fold["hm"] for fold in fold_reports])),
        },
        "single_segments": segment_reports,
        "oracle_any_segment_correct": float(
            np.mean(np.stack(segment_correct).any(axis=0))),
        "folds": fold_reports,
        "per_class": per_class,
        "gate_thresholds": {
            "balanced_accuracy": GATE_BALANCED_ACCURACY,
            "worst_fold_unseen": GATE_WORST_FOLD_UNSEEN,
            "prediction_coverage": GATE_PREDICTION_COVERAGE,
        },
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }


def main() -> None:
    args = parse_args()
    report = evaluate_cache(args.cache, args.folds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    result = report["temporal_mean"]
    print(
        f"balanced_accuracy={100 * result['balanced_accuracy']:.2f} "
        f"worst_fold_unseen={100 * result['worst_fold_unseen']:.2f} "
        f"coverage={100 * result['prediction_coverage']:.2f} "
        f"gate_passed={report['gate_passed']}", flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
