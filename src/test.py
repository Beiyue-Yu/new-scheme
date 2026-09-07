import json
import logging
from pathlib import Path

import numpy as np

from src.utils import evaluate_dataset, evaluate_dataset_baseline


def _format_performance(label, evaluation):
    output = (
        f"{label} performance: "
        f"Seen={100 * evaluation['seen']:.2f}, "
        f"Unseen={100 * evaluation['unseen']:.2f}, "
        f"HM={100 * evaluation['hm']:.2f}, "
        f"ZSL={100 * evaluation['zsl']:.2f}, "
        f"Beta={evaluation['beta']:.2f}"
    )
    if "audio_weight" in evaluation:
        output += f", AudioWeight={evaluation['audio_weight']:.2f}"
    return output


def _build_bias_report(dataset, evaluation):
    """Build a JSON-serializable per-class Seen/Unseen bias report."""
    seen_ids = set(int(value) for value in dataset.seen_class_ids)
    unseen_ids = set(int(value) for value in dataset.unseen_class_ids)
    class_ids = sorted(seen_ids | unseen_ids)
    names = np.asarray(dataset.all_class_names)
    modalities = {}
    for mode, mode_evaluation in evaluation.items():
        recall = np.asarray(
            mode_evaluation.get("gzsl_recall", mode_evaluation["recall"]),
            dtype=float)
        rows = []
        for class_id in class_ids:
            rows.append({
                "class_id": class_id,
                "class_name": str(names[class_id]),
                "split": "seen" if class_id in seen_ids else "unseen",
                "recall": float(recall[class_id]),
            })
        seen_rows = [row for row in rows if row["split"] == "seen"]
        unseen_rows = [row for row in rows if row["split"] == "unseen"]
        seen_mean = sum(row["recall"] for row in seen_rows) / max(len(seen_rows), 1)
        unseen_mean = sum(row["recall"] for row in unseen_rows) / max(len(unseen_rows), 1)
        modalities[mode] = {
            "beta": (float(mode_evaluation["beta"])
                     if "beta" in mode_evaluation else None),
            "audio_weight": (float(mode_evaluation["audio_weight"])
                             if "audio_weight" in mode_evaluation else None),
            "energy_threshold": (float(mode_evaluation["threshold"])
                                 if "threshold" in mode_evaluation else None),
            "routed_seen_rate": (float(mode_evaluation["routed_seen_rate"])
                                 if "routed_seen_rate" in mode_evaluation else None),
            "score_normalization": mode_evaluation.get("score_normalization"),
            "score_source": mode_evaluation.get("score_source"),
            "semantic_aware_calibration": mode_evaluation.get(
                "semantic_aware_calibration", False),
            "seen_penalty_scale_min": mode_evaluation.get(
                "seen_penalty_scale_min"),
            "seen_penalty_scale_max": mode_evaluation.get(
                "seen_penalty_scale_max"),
            "seen_mean": seen_mean,
            "unseen_mean": unseen_mean,
            "bias_gap_seen_minus_unseen": seen_mean - unseen_mean,
            "seen_to_unseen_ratio": (seen_mean / unseen_mean
                                      if unseen_mean > 0 else None),
            "strongest_classes": sorted(
                rows, key=lambda row: row["recall"], reverse=True)[:10],
            "weakest_classes": sorted(
                rows, key=lambda row: row["recall"])[:10],
            "classes": rows,
        }
    return {
        "dataset": str(getattr(dataset, "dataset_name", "unknown")),
        "modalities": modalities,
    }


def _write_bias_report(path, dataset, evaluation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as report_file:
        json.dump(_build_bias_report(dataset, evaluation), report_file,
                  ensure_ascii=False, indent=2)


def test(eval_name, val_dataset, test_dataset, model_A, model_B, device, distance_fn,
          args=None, new_model_attention=False, devise_model=False, apn=False,
          save_performances=False, adaptive_modality_fusion=False,
          energy_ood_routing=False,
          energy_ood_score="raw",
          bias_report_path=None):
    logger = logging.getLogger()
    model_A.eval()
    model_B.eval()

    test_evaluation = _get_test_performance(val_dataset=val_dataset, test_dataset=test_dataset, model_A=model_A,
                                            model_B=model_B, device=device, distance_fn=distance_fn,
                                            args=args,
                                            new_model_attention=new_model_attention,
                                            devise_model=devise_model,
                                            apn=apn, save_performances=save_performances,
                                            adaptive_modality_fusion=adaptive_modality_fusion,
                                            energy_ood_routing=energy_ood_routing,
                                            energy_ood_score=energy_ood_score)

    if args.dataset_name not in {"AudioSetZSL", "VGGSound", "UCF", "ActivityNet"}:
        raise NotImplementedError()

    logger.info(_format_performance("Audio", test_evaluation["audio"]))
    logger.info(_format_performance("Video", test_evaluation["video"]))
    logger.info(_format_performance("Both", test_evaluation["both"]))
    if "energy_ood" in test_evaluation:
        energy = test_evaluation["energy_ood"]
        logger.info(
            "Energy-OOD[%s,%s] performance: Seen=%.2f, Unseen=%.2f, HM=%.2f, "
            "ZSL=%.2f, Threshold=%.6f, RoutedSeen=%.2f%%",
            energy["score_normalization"], energy["score_source"],
            100 * energy["seen"], 100 * energy["unseen"],
            100 * energy["hm"], 100 * energy["zsl"],
            energy["threshold"], 100 * energy["routed_seen_rate"])
    # Keep the original combined line for spreadsheet extraction and existing
    # result parsers.
    output_string = fr"""
            Seen performance={100*test_evaluation["both"]["seen"]:.2f}, Unseen performance={100*test_evaluation["both"]["unseen"]:.2f}, GZSL performance={100*test_evaluation["both"]["hm"]:.2f}, ZSL performance={100*test_evaluation["both"]["zsl"]:.2f}
            """
    logger.info(output_string)
    if bias_report_path is not None:
        _write_bias_report(bias_report_path, test_dataset, test_evaluation)
        logger.info("Saved class bias report to %s", bias_report_path)
    return test_evaluation


def _get_test_performance(val_dataset, test_dataset, model_A, model_B, device,
                          distance_fn, args, new_model_attention, devise_model,
                          apn, save_performances=False,
                          adaptive_modality_fusion=False,
                          energy_ood_routing=False,
                          energy_ood_score="raw"):
    logger = logging.getLogger()
    if  new_model_attention or devise_model or apn:
        val_evaluation = evaluate_dataset_baseline(val_dataset, model_A, device, distance_fn,
                                                   args=args,
                                                   new_model_attention=new_model_attention,
                                                   model_devise=devise_model,
                                                   apn=apn,
                                                   adaptive_modality_fusion=adaptive_modality_fusion,
                                                   energy_ood_routing=energy_ood_routing,
                                                   energy_ood_score=energy_ood_score)
    else:
        val_evaluation = evaluate_dataset(
            val_dataset, model_A, device, distance_fn, args=args,
            adaptive_modality_fusion=adaptive_modality_fusion,
            energy_ood_routing=energy_ood_routing,
            energy_ood_score=energy_ood_score)
    # Each distance has a different scale, so calibrate each test modality
    # with the beta selected for that same modality on validation data.
    best_betas = {
        mode: val_evaluation[mode]['beta']
        for mode in ('audio', 'video', 'both')
    }
    best_fusion_weight = val_evaluation['both'].get('audio_weight')
    best_energy_threshold = (
        val_evaluation["energy_ood"]["threshold"]
        if energy_ood_routing else None)
    logger.info(
        f"Validation betas:\tAudio={val_evaluation['audio']['beta']}\tVideo={val_evaluation['video']['beta']}\tBoth={val_evaluation['both']['beta']}")
    logger.info(f"Best beta combined: {best_betas['both']}")
    if adaptive_modality_fusion:
        logger.info(f"Best combined audio weight: {best_fusion_weight}")
    if energy_ood_routing:
        logger.info(
            "Validation Energy-OOD threshold: %s (routed Seen %.2f%%)",
            best_energy_threshold,
            100 * val_evaluation["energy_ood"]["routed_seen_rate"])

    if new_model_attention or devise_model or apn:
        test_evaluation = evaluate_dataset_baseline(test_dataset, model_B, device, distance_fn, best_beta=best_betas,
                                                    args=args,
                                                    new_model_attention=new_model_attention,
                                                    model_devise=devise_model,
                                                    apn=apn, save_performances=save_performances,
                                                    best_fusion_weight=best_fusion_weight,
                                                    adaptive_modality_fusion=adaptive_modality_fusion,
                                                    energy_ood_routing=energy_ood_routing,
                                                    best_energy_threshold=best_energy_threshold,
                                                    energy_ood_score=energy_ood_score,
                                                    energy_ood_score_model=(
                                                        model_A if energy_ood_routing
                                                        else None))
    else:
        test_evaluation = evaluate_dataset(
            test_dataset, model_B, device, distance_fn,
            best_beta=best_betas, args=args,
            best_fusion_weight=best_fusion_weight,
            adaptive_modality_fusion=adaptive_modality_fusion,
            energy_ood_routing=energy_ood_routing,
            best_energy_threshold=best_energy_threshold,
            energy_ood_score=energy_ood_score)

    return test_evaluation
