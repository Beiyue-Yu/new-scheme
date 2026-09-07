#!/usr/bin/env python3
"""Evaluate the frozen LanguageBind video route on official Stage A validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from evaluate_languagebind_temporal_zsl import (
    class_balanced_accuracy,
    prediction_diagnostics,
)
from extract_languagebind_ucf_segments import read_manifest
from src.languagebind_segments import temporal_class_logits


GATE_HM = 0.30
GATE_UNSEEN = 0.25
GATE_ZSL = 0.50
GATE_UNSEEN_COVERAGE = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official Stage A validation gate for frozen LanguageBind")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--seen_manifest", type=Path, required=True)
    parser.add_argument("--unseen_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation_scope",
        choices=("stage_a_validation", "stage_2_test"),
        default="stage_a_validation")
    return parser.parse_args()


def harmonic_mean(seen: float, unseen: float) -> float:
    return 0.0 if seen + unseen == 0.0 else 2.0 * seen * unseen / (seen + unseen)


def official_metrics(logits: torch.Tensor, targets: np.ndarray,
                     class_ids: np.ndarray, seen_ids: np.ndarray,
                     unseen_ids: np.ndarray) -> Dict[str, object]:
    if logits.shape != (len(targets), len(class_ids)):
        raise ValueError("Logits, targets, and candidate classes are not aligned")
    gzsl_predictions = class_ids[logits.argmax(dim=1).numpy()]
    seen_mask = np.isin(targets, seen_ids)
    unseen_mask = np.isin(targets, unseen_ids)
    if not seen_mask.any() or not unseen_mask.any():
        raise ValueError("Official validation requires both Seen and Unseen samples")
    seen = class_balanced_accuracy(
        gzsl_predictions[seen_mask], targets[seen_mask], seen_ids)
    unseen = class_balanced_accuracy(
        gzsl_predictions[unseen_mask], targets[unseen_mask], unseen_ids)
    unseen_columns = np.asarray([
        int(np.flatnonzero(class_ids == class_id)[0]) for class_id in unseen_ids
    ], dtype=np.int64)
    zsl_predictions = unseen_ids[
        logits[unseen_mask][:, unseen_columns].argmax(dim=1).numpy()]
    zsl = class_balanced_accuracy(
        zsl_predictions, targets[unseen_mask], unseen_ids)
    unseen_coverage = float(np.mean([
        np.any(gzsl_predictions[unseen_mask] == class_id)
        for class_id in unseen_ids
    ]))
    return {
        "seen": seen,
        "unseen": unseen,
        "hm": harmonic_mean(seen, unseen),
        "zsl": zsl,
        "prediction_coverage": prediction_diagnostics(
            gzsl_predictions, class_ids)["prediction_coverage"],
        "unseen_prediction_coverage": unseen_coverage,
        "unseen_to_seen_rate": float(np.mean(
            np.isin(gzsl_predictions[unseen_mask], seen_ids))),
        "seen_to_unseen_rate": float(np.mean(
            np.isin(gzsl_predictions[seen_mask], unseen_ids))),
        "gzsl_predictions": gzsl_predictions,
        "zsl_predictions": zsl_predictions,
    }


def _public_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in metrics.items()
            if key not in {"gzsl_predictions", "zsl_predictions"}}


def evaluate_cache(cache_path: Path, seen_manifest: Path,
                   unseen_manifest: Path,
                   evaluation_scope: str = "stage_a_validation") -> Dict[str, object]:
    if evaluation_scope not in {"stage_a_validation", "stage_2_test"}:
        raise ValueError(f"Unsupported evaluation scope: {evaluation_scope}")
    seen_rows = read_manifest(seen_manifest)
    unseen_rows = read_manifest(unseen_manifest)
    seen_ids = np.asarray(sorted({row.class_id for row in seen_rows}), dtype=np.int64)
    unseen_ids = np.asarray(
        sorted({row.class_id for row in unseen_rows}), dtype=np.int64)
    if np.intersect1d(seen_ids, unseen_ids).size:
        raise ValueError("Official Seen and Unseen class IDs overlap")
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"]))
        failures = json.loads(str(cache["failures_json"]))
        embeddings = torch.from_numpy(
            cache["video_embeddings"].astype(np.float32))
        text = torch.from_numpy(cache["text_embeddings"].astype(np.float32))
        targets = cache["class_ids"].astype(np.int64)
        class_names = cache["class_names"].astype(str)
        video_names = cache["video_names"].astype(str)
    class_ids = np.asarray(metadata["selected_class_ids"], dtype=np.int64)
    expected_ids = np.sort(np.concatenate((seen_ids, unseen_ids)))
    if not np.array_equal(class_ids, expected_ids):
        raise ValueError("Cache candidate classes are not the official Seen+Unseen union")
    if embeddings.shape[0] != len(seen_rows) + len(unseen_rows):
        raise ValueError("Cache does not contain every official validation sample")
    expected_names = {row.filename for row in seen_rows + unseen_rows}
    if set(video_names.tolist()) != expected_names:
        raise ValueError("Cache video names do not match official validation manifests")
    mean_logits = temporal_class_logits(embeddings, text)
    mean_metrics = official_metrics(
        mean_logits, targets, class_ids, seen_ids, unseen_ids)
    segment_reports = []
    for segment in range(embeddings.shape[1]):
        logits = temporal_class_logits(
            embeddings[:, segment:segment + 1], text)
        segment_reports.append({
            "segment": segment + 1,
            **_public_metrics(official_metrics(
                logits, targets, class_ids, seen_ids, unseen_ids)),
        })
    gzsl_predictions = mean_metrics["gzsl_predictions"]
    unseen_mask = np.isin(targets, unseen_ids)
    zsl_predictions = mean_metrics["zsl_predictions"]
    per_class = []
    for class_id, class_name in zip(class_ids, class_names):
        mask = targets == class_id
        item = {
            "class_id": int(class_id),
            "class_name": class_name,
            "split": "seen" if class_id in set(seen_ids.tolist()) else "unseen",
            "samples": int(mask.sum()),
            "gzsl_accuracy": float(np.mean(gzsl_predictions[mask] == class_id)),
        }
        if class_id in set(unseen_ids.tolist()):
            class_zsl_mask = targets[unseen_mask] == class_id
            item["zsl_accuracy"] = float(np.mean(
                zsl_predictions[class_zsl_mask] == class_id))
        per_class.append(item)
    is_test = evaluation_scope == "stage_2_test"
    gate_checks = None if is_test else {
        "hm": mean_metrics["hm"] >= GATE_HM,
        "unseen": mean_metrics["unseen"] >= GATE_UNSEEN,
        "zsl": mean_metrics["zsl"] >= GATE_ZSL,
        "unseen_prediction_coverage": (
            mean_metrics["unseen_prediction_coverage"] >= GATE_UNSEEN_COVERAGE),
    }
    return {
        "report_version": 1,
        "protocol": ("frozen official Stage 2 test" if is_test
                     else "frozen official Stage A validation"),
        "repository_data_fitting": False,
        "official_stage_a_validation_loaded": not is_test,
        "stage_b_loaded": False,
        "test_split_loaded": is_test,
        "test_labels_used_for_selection": False,
        "source_cache": str(cache_path),
        "source_metadata": metadata,
        "seen_manifest": str(seen_manifest),
        "unseen_manifest": str(unseen_manifest),
        "encoded_videos": int(len(targets)),
        "decode_failures": failures,
        "seen_classes": int(len(seen_ids)),
        "unseen_classes": int(len(unseen_ids)),
        "candidate_classes": int(len(class_ids)),
        "temporal_mean": _public_metrics(mean_metrics),
        "single_segments": segment_reports,
        "per_class": per_class,
        "gate_thresholds": None if is_test else {
            "hm": GATE_HM,
            "unseen": GATE_UNSEEN,
            "zsl": GATE_ZSL,
            "unseen_prediction_coverage": GATE_UNSEEN_COVERAGE,
        },
        "gate_checks": gate_checks,
        "gate_passed": None if is_test else all(gate_checks.values()),
    }


def main() -> None:
    args = parse_args()
    report = evaluate_cache(
        args.cache, args.seen_manifest, args.unseen_manifest,
        evaluation_scope=args.evaluation_scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    result = report["temporal_mean"]
    print(
        f"seen={100 * result['seen']:.2f} "
        f"unseen={100 * result['unseen']:.2f} "
        f"hm={100 * result['hm']:.2f} "
        f"zsl={100 * result['zsl']:.2f} "
        f"gate_passed={report['gate_passed']}", flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
