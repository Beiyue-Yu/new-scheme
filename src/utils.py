import json
import logging
import pickle
import random
import socket
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.logger import PD_Stats, create_logger


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def read_features(path):
    hf = h5py.File(path, 'r')
    # keys = list(hf.keys())
    data = hf['data']
    url = [str(u, 'utf-8') for u in list(hf['video_urls'])]

    return data, url


def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_experiment(args, *stats):
    if args.exp_name == "":
        exp_name = f"runs/{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}"
    else:
        exp_name = "runs/" + str(args.exp_name)
        #exp_name = "/mnt/store_runs/" + str(args.exp_name)
    log_dir = (args.dump_path / exp_name)
    is_resume = getattr(args, "resume_checkpoint", None) is not None
    allow_existing = getattr(args, "allow_existing_run", False)
    log_dir.mkdir(parents=True, exist_ok=is_resume or allow_existing)
    (log_dir / "checkpoints").mkdir(
        exist_ok=is_resume or allow_existing)
    with (log_dir / "args.pkl").open("wb") as args_file:
        pickle.dump(args, args_file)
    train_stats = PD_Stats(log_dir / "train_stats.pkl", stats)
    val_stats = PD_Stats(log_dir / "val_stats.pkl", stats)
    logger = create_logger(log_dir / "train.log")

    logger.info(f"Start experiment {exp_name}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"The experiment will be stored in {log_dir.resolve()}\n")
    logger.info("")
    if getattr(args, "disable_tensorboard", False):
        writer = NullSummaryWriter()
    else:
        from torch.utils.tensorboard import SummaryWriter
        if args.exp_name == "":
            writer = SummaryWriter()
        else:
            writer = SummaryWriter(log_dir=exp_name)
    return logger, log_dir, writer, train_stats, val_stats


def setup_evaluation(args, *stats):
    eval_dir = args.load_path_stage_B
    assert eval_dir.exists()
    # pickle.dump(args, (eval_dir / "args.pkl").open("wb"))
    test_stats = PD_Stats(eval_dir / "test_stats.pkl", list(sorted(stats)))
    logger = create_logger(eval_dir / "eval.log")

    logger.info(f"Start evaluation {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items()))
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    return logger, eval_dir, test_stats


def save_best_model(epoch, best_metric, model, optimizer, log_dir, metric="", checkpoint=False):
    logger = logging.getLogger()
    logger.info(f"Saving model to {log_dir} with {metric} = {best_metric:.4f}")
    save_dict = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "metric": metric
    }
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if checkpoint:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}_ckpt_{epoch}.pt"
        )
    else:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}.pt"
        )


def save_training_state(epoch, model, optimizer, scheduler, log_dir,
                        best_loss, best_score):
    """Atomically save everything required to resume at the next epoch."""
    state = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_loss": best_loss,
        "best_score": best_score,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    destination = log_dir / "last.pt"
    temporary = log_dir / "last.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(destination)
    epoch_destination = log_dir / "last_epoch.txt"
    epoch_temporary = log_dir / "last_epoch.txt.tmp"
    epoch_temporary.write_text(f"{epoch + 1}\n", encoding="ascii")
    epoch_temporary.replace(epoch_destination)


def load_training_state(path, model, optimizer, scheduler):
    """Restore a complete state and return (start_epoch, best_loss, best_score)."""
    state = torch.load(path, map_location="cpu")
    load_model_parameters(model, state["model"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    logger = logging.getLogger()
    try:
        torch.set_rng_state(state["torch_rng_state"])
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        logger.warning("Skipping incompatible CPU RNG state from checkpoint: %s", error)
    if state.get("numpy_rng_state") is not None:
        np.random.set_state(state["numpy_rng_state"])
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        saved_cuda_states = state["cuda_rng_state_all"]
        current_cuda_states = torch.cuda.get_rng_state_all()
        saved_sizes = [rng_state.numel() for rng_state in saved_cuda_states]
        current_sizes = [rng_state.numel() for rng_state in current_cuda_states]
        if saved_sizes != current_sizes:
            logger.warning(
                "Skipping incompatible CUDA RNG state from checkpoint: "
                "saved sizes %s, current sizes %s", saved_sizes, current_sizes)
        else:
            try:
                torch.cuda.set_rng_state_all(saved_cuda_states)
            except (RuntimeError, TypeError, ValueError) as error:
                logger.warning(
                    "Skipping incompatible CUDA RNG state from checkpoint: %s",
                    error)
    return state["epoch"], state.get("best_loss"), state.get("best_score")


def check_best_loss(epoch, best_loss, val_loss, model, optimizer, log_dir):
    if not best_loss:
        save_best_model(epoch, val_loss, model, optimizer, log_dir, metric="loss")
        return val_loss
    if val_loss < best_loss:
        best_loss = val_loss
        save_best_model(epoch, best_loss, model, optimizer, log_dir, metric="loss")
    return best_loss


def check_best_score(epoch, best_score, hm_score, model, optimizer, log_dir):
    if not best_score:
        save_best_model(epoch, hm_score, model, optimizer, log_dir, metric="score")
        return hm_score
    if hm_score > best_score:
        best_score = hm_score
        save_best_model(epoch, best_score, model, optimizer, log_dir, metric="score")
    return best_score


def load_model_parameters(model, model_weights, strict=False):
    logger = logging.getLogger()
    loaded_state = model_weights
    self_state = model.state_dict()
    missing = set(self_state)
    unexpected = []
    mismatched = []
    for name, param in loaded_state.items():
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys():
            if self_state[name].shape != param.shape:
                logger.info("skip shape mismatch %s: model %s vs ckpt %s",
                            name, tuple(self_state[name].shape), tuple(param.shape))
                mismatched.append((name, tuple(self_state[name].shape), tuple(param.shape)))
                continue
            self_state[name].copy_(param)
            missing.discard(name)
        else:
            logger.info("didnt load %s", name)
            unexpected.append(name)

    if strict and (missing or unexpected or mismatched):
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)}")
        if mismatched:
            details.append(f"shape_mismatch={mismatched}")
        raise RuntimeError("Checkpoint is incompatible with the model: " + "; ".join(details))


def load_args(path):
    return pickle.load((path / "args.pkl").open("rb"))


def cos_dist(a, b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res


def evaluate_dataset_baseline(dataset, model, device, distance_fn, best_beta=None,
                              new_model_attention=False, model_devise=False, apn=False,
                              args=None, save_performances=False,
                              best_fusion_weight=None,
                              adaptive_modality_fusion=False,
                              energy_ood_routing=False,
                              best_energy_threshold=None,
                              energy_ood_score="raw",
                              energy_ood_score_model=None):
    data = dataset.all_data
    data_a = data["audio"].to(device)
    data_v = data["video"].to(device)
    data_t = data["text"].to(device)

    data_num = data["target"].to(device)
    if new_model_attention == True or model_devise == True or apn == True:
        all_data = (
            data_a, data_v, data_num, data_t
        )
    else:
        all_data = (
            data_a, data_v, data_t
        )
    try:
        if args.z_score_inputs:
            all_data = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in all_data])
    except AttributeError:
        print("Namespace has no fitting attribute. Continuing")

    all_targets = dataset.targets.to(device)
    model.eval()

    with torch.no_grad():
        if new_model_attention == False and model_devise == False and apn == False:
            outputs_all = model(*all_data)
        elif apn == True:
            input_features = torch.cat((all_data[1], all_data[0]), 1)
            output_final, pre_attri, attention, pre_class, attributes = model(input_features, all_data[3])
            outputs_all = (pre_attri["final"], attributes)
        elif model_devise == True:
            input_features = torch.cat((all_data[1], all_data[0]), 1)
            outputs_all, projected_features, embeddings = model(input_features, all_data[3])
            outputs_all = (projected_features, embeddings)
        elif new_model_attention == True:
            audio_emb, video_emb, emb_cls = model.get_embeddings(all_data[0], all_data[1], all_data[3])
            outputs_all = (audio_emb, video_emb, emb_cls)

    if model_devise == True or apn == True:
        a_p, t_p = outputs_all
        v_p = None
    elif new_model_attention == True:
        a_p, v_p, t_p = outputs_all
        # a_p = None

    if model_devise == True or apn == True:
        audio_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="audio", device=device,
                                               distance_fn=distance_fn, best_beta=best_beta, save_performances=save_performances, args=args)
    if new_model_attention == True:
        audio_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="audio", device=device,
                                               distance_fn=distance_fn, best_beta=best_beta, save_performances=save_performances,args=args)
        video_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="video", device=device,
                                               distance_fn=distance_fn, best_beta=best_beta, save_performances=save_performances,args=args)
        both_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="both", device=device,
                                              distance_fn=distance_fn, best_beta=best_beta,
                                              best_fusion_weight=best_fusion_weight,
                                              adaptive_modality_fusion=adaptive_modality_fusion,
                                              save_performances=save_performances,args=args)

    if  new_model_attention == True:
        evaluation = {
            "audio": audio_evaluation,
            "video": video_evaluation,
            "both": both_evaluation
        }
        if energy_ood_routing:
            score_embeddings = None
            if energy_ood_score_model is not None:
                energy_ood_score_model.eval()
                with torch.no_grad():
                    score_embeddings = energy_ood_score_model.get_embeddings(
                        all_data[0], all_data[1],
                        all_data[3] if new_model_attention else all_data[2])
            evaluation["energy_ood"] = get_energy_ood_evaluation(
                dataset, all_targets, a_p, v_p, t_p,
                distance_fn=distance_fn,
                threshold=best_energy_threshold,
                score_normalization=energy_ood_score,
                score_embeddings=score_embeddings)
        return evaluation
    elif model_devise == True or apn == True:
        return {
            "audio": audio_evaluation,
            "video": audio_evaluation,
            "both": audio_evaluation
        }



def get_best_evaluation(dataset, targets, a_p, v_p, t_p, mode, device,
                        distance_fn, best_beta=None, save_performances=False,
                        args=None, attention_weights=None,
                        best_fusion_weight=None,
                        adaptive_modality_fusion=False):
    seen_scores = []
    zsl_scores = []
    unseen_scores = []
    hm_scores = []
    per_class_recalls = []
    gzsl_per_class_recalls = []
    start = 0.0
    end = float(getattr(args, "evaluation_beta_max", 3.0))
    if end <= start:
        raise ValueError("evaluation_beta_max must be positive")
    if isinstance(best_beta, Mapping):
        best_beta = best_beta[mode]
    # The optional adaptive search uses a 0.1 beta step. The default fixed-sum
    # protocol retains the released 0.2 grid for historical comparability.
    default_beta_step = (
        0.1 if mode == "both" and adaptive_modality_fusion else 0.2)
    configured_beta_step = getattr(args, "evaluation_beta_step", None)
    beta_step = (default_beta_step if configured_beta_step is None
                 else float(configured_beta_step))
    if beta_step <= 0:
        raise ValueError("evaluation_beta_step must be positive")
    steps = int(round((end - start) / beta_step)) + 1
    betas = (torch.tensor([best_beta], dtype=torch.float, device=device)
             if best_beta is not None else torch.linspace(start, end, steps, device=device))
    seen_label_array = torch.tensor(dataset.seen_class_ids, dtype=torch.long, device=device)
    unseen_label_array = torch.tensor(dataset.unseen_class_ids, dtype=torch.long, device=device)
    seen_unseen_array = torch.tensor(np.sort(np.concatenate((dataset.seen_class_ids, dataset.unseen_class_ids))),
                                     dtype=torch.long, device=device)

    normalize_shared_embeddings = bool(
        getattr(args, "normalize_shared_embeddings", False))
    classes_embeddings = (
        F.normalize(t_p, dim=1) if normalize_shared_embeddings else t_p)
    distance_audio_embeddings = (
        F.normalize(a_p, dim=1)
        if normalize_shared_embeddings and a_p is not None else a_p)
    distance_video_embeddings = (
        F.normalize(v_p, dim=1)
        if normalize_shared_embeddings and v_p is not None else v_p)
    candidate_betas = []
    candidate_audio_weights = []
    with torch.no_grad():
        semantic_aware_calibration = bool(
            getattr(args, "semantic_aware_calibration", False))
        seen_penalty_scales = torch.ones(
            len(seen_label_array), dtype=classes_embeddings.dtype,
            device=device)
        if semantic_aware_calibration:
            # Calibrated stacking penalizes every Seen prototype equally. A
            # Seen class whose learned semantic prototype is close to an
            # Unseen prototype is more likely to absorb that class's samples,
            # so allocate more of the same mean beta budget to it. This uses
            # only the standard GZSL class dictionary, never sample labels.
            normalized_classes = F.normalize(classes_embeddings, dim=1)
            seen_positions = torch.searchsorted(
                seen_unseen_array, seen_label_array)
            unseen_positions = torch.searchsorted(
                seen_unseen_array, unseen_label_array)
            seen_to_unseen_similarity = (
                normalized_classes[seen_positions] @
                normalized_classes[unseen_positions].transpose(0, 1))
            seen_affinity = seen_to_unseen_similarity.max(dim=1).values
            mean_affinity = seen_affinity.mean().clamp_min(
                torch.finfo(seen_affinity.dtype).eps)
            seen_penalty_scales = seen_affinity / mean_affinity

        if mode in {"audio", "video", "both"}:
            audio_distance = (torch.cdist(
                distance_audio_embeddings, classes_embeddings, p=2)
                if distance_audio_embeddings is not None else None)
            video_distance = (torch.cdist(
                distance_video_embeddings, classes_embeddings, p=2)
                if distance_video_embeddings is not None else None)
            if distance_fn == "SquaredL2Loss":
                audio_distance = (audio_distance.pow(2)
                                  if audio_distance is not None else None)
                video_distance = (video_distance.pow(2)
                                  if video_distance is not None else None)
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")

        use_adaptive_fusion = (
            mode == "both" and adaptive_modality_fusion and
            not getattr(args, "cjme", False))
        if use_adaptive_fusion:
            if best_fusion_weight is None:
                # Center-first ordering makes exact ties prefer the original
                # equal fusion instead of an arbitrary single modality.
                audio_weights = torch.tensor(
                    [0.5, 0.4, 0.6, 0.3, 0.7, 0.2,
                     0.8, 0.1, 0.9, 0.0, 1.0],
                    dtype=torch.float, device=device)
            else:
                audio_weights = torch.tensor(
                    [best_fusion_weight], dtype=torch.float, device=device)
        else:
            audio_weights = torch.tensor([0.5], dtype=torch.float, device=device)

        batch_size = v_p.shape[0] if a_p is None else a_p.shape[0]
        for audio_weight in audio_weights:
            if mode == "audio":
                class_distance = audio_distance
            elif mode == "video":
                class_distance = video_distance
            elif getattr(args, "cjme", False):
                class_distance = ((1 - attention_weights) * audio_distance +
                                  attention_weights * video_distance)
            elif not adaptive_modality_fusion:
                class_distance = audio_distance + video_distance
            else:
                class_distance = (audio_weight * audio_distance +
                                  (1.0 - audio_weight) * video_distance)

            distance_mat = torch.full(
                (batch_size, len(dataset.all_class_ids)), 99999999999999.0,
                dtype=torch.float, device=device)
            distance_mat[:, seen_unseen_array] = class_distance
            zsl_mask = torch.zeros(
                len(dataset.all_class_ids), dtype=torch.float, device=device)
            zsl_mask[seen_label_array] = 99999999999999.0
            distance_mat_zsl = distance_mat + zsl_mask

            for beta in betas:
                mask = torch.full(
                    (len(dataset.all_class_ids),), 0.0,
                    dtype=torch.float, device=device)
                if semantic_aware_calibration:
                    mask[seen_label_array] = beta.item() * seen_penalty_scales
                else:
                    mask[seen_label_array] = beta.item()
                neighbor_batch = torch.argmin(distance_mat + mask, dim=1)
                match_idx = neighbor_batch.eq(targets.int()).nonzero().flatten()
                match_counts = torch.bincount(
                    neighbor_batch[match_idx], minlength=len(dataset.all_class_ids)
                )[seen_unseen_array]
                target_counts = torch.bincount(
                    targets, minlength=len(dataset.all_class_ids)
                )[seen_unseen_array]
                per_class_recall = torch.zeros(
                    len(dataset.all_class_ids), dtype=torch.float, device=device)
                per_class_recall[seen_unseen_array] = match_counts / target_counts
                seen_recall_dict = per_class_recall[seen_label_array]
                unseen_recall_dict = per_class_recall[unseen_label_array]
                s = seen_recall_dict.mean()
                u = unseen_recall_dict.mean()

                if save_performances:
                    seen_dict = {k: v for k, v in zip(
                        np.array(dataset.all_class_names)[seen_label_array.cpu().numpy()],
                        seen_recall_dict.cpu().numpy())}
                    unseen_dict = {k: v for k, v in zip(
                        np.array(dataset.all_class_names)[unseen_label_array.cpu().numpy()],
                        unseen_recall_dict.cpu().numpy())}
                    save_class_performances(
                        seen_dict, unseen_dict, dataset.dataset_name)

                hm = (2 * u * s) / ((u + s) + np.finfo(float).eps)
                gzsl_per_class_recalls.append(per_class_recall.tolist())
                neighbor_batch_zsl = torch.argmin(distance_mat_zsl, dim=1)
                match_idx = neighbor_batch_zsl.eq(targets.int()).nonzero().flatten()
                match_counts = torch.bincount(
                    neighbor_batch_zsl[match_idx],
                    minlength=len(dataset.all_class_ids))[seen_unseen_array]
                per_class_recall = torch.zeros(
                    len(dataset.all_class_ids), dtype=torch.float, device=device)
                per_class_recall[seen_unseen_array] = match_counts / target_counts
                zsl = per_class_recall[unseen_label_array].mean()

                zsl_scores.append(zsl.item())
                seen_scores.append(s.item())
                unseen_scores.append(u.item())
                hm_scores.append(hm.item())
                per_class_recalls.append(per_class_recall.tolist())
                candidate_betas.append(beta.item())
                candidate_audio_weights.append(audio_weight.item())
        argmax_hm = np.argmax(hm_scores)
        max_seen = seen_scores[argmax_hm]
        max_zsl = zsl_scores[argmax_hm]
        max_unseen = unseen_scores[argmax_hm]
        max_hm = hm_scores[argmax_hm]
        max_recall = per_class_recalls[argmax_hm]
        max_gzsl_recall = gzsl_per_class_recalls[argmax_hm]
        best_beta = candidate_betas[argmax_hm]
        best_audio_weight = candidate_audio_weights[argmax_hm]
    evaluation = {
        "seen": max_seen,
        "unseen": max_unseen,
        "hm": max_hm,
        "recall": max_recall,
        "zsl": max_zsl,
        "beta": best_beta
    }
    if semantic_aware_calibration:
        evaluation["semantic_aware_calibration"] = True
        evaluation["seen_penalty_scale_min"] = seen_penalty_scales.min().item()
        evaluation["seen_penalty_scale_max"] = seen_penalty_scales.max().item()
    # ``recall`` predates the diagnostics and is the ZSL per-class vector.
    # Expose the GZSL vector separately so Seen/Unseen bias reports use the
    # actual calibrated GZSL predictions.
    evaluation["gzsl_recall"] = max_gzsl_recall
    if mode == "both" and adaptive_modality_fusion and not getattr(args, "cjme", False):
        evaluation["audio_weight"] = best_audio_weight
    return evaluation


def get_energy_ood_evaluation(dataset, targets, a_p, v_p, t_p,
                              distance_fn, threshold=None,
                              score_normalization="raw",
                              score_embeddings=None):
    """Route samples to Seen/Unseen nearest-class experts using Seen energy."""
    if a_p is None or v_p is None:
        raise ValueError("Energy-OOD routing requires both audio and video embeddings")

    device = a_p.device
    all_label_array = torch.tensor(
        np.sort(np.concatenate((dataset.seen_class_ids,
                                dataset.unseen_class_ids))),
        dtype=torch.long, device=device)
    seen_ids = set(int(value) for value in dataset.seen_class_ids)
    seen_positions = torch.tensor(
        [index for index, value in enumerate(all_label_array.tolist())
         if value in seen_ids], dtype=torch.long, device=device)
    unseen_positions = torch.tensor(
        [index for index, value in enumerate(all_label_array.tolist())
         if value not in seen_ids], dtype=torch.long, device=device)
    seen_label_array = all_label_array[seen_positions]
    unseen_label_array = all_label_array[unseen_positions]

    with torch.no_grad():
        audio_distance = torch.cdist(a_p, t_p, p=2)
        video_distance = torch.cdist(v_p, t_p, p=2)
        if distance_fn == "SquaredL2Loss":
            audio_distance = audio_distance.pow(2)
            video_distance = video_distance.pow(2)
        class_distance = audio_distance + video_distance
        if score_embeddings is None:
            score_class_distance = class_distance
            score_source = "classification_model"
        else:
            score_a_p, score_v_p, score_t_p = score_embeddings
            score_audio_distance = torch.cdist(score_a_p, score_t_p, p=2)
            score_video_distance = torch.cdist(score_v_p, score_t_p, p=2)
            if distance_fn == "SquaredL2Loss":
                score_audio_distance = score_audio_distance.pow(2)
                score_video_distance = score_video_distance.pow(2)
            score_class_distance = score_audio_distance + score_video_distance
            score_source = "stage_a_model"
        if score_normalization == "zscore":
            distance_mean = score_class_distance.mean(dim=1, keepdim=True)
            distance_std = score_class_distance.std(
                dim=1, keepdim=True, unbiased=False).clamp_min(1e-12)
            energy_distance = (
                score_class_distance - distance_mean) / distance_std
        elif score_normalization == "raw":
            energy_distance = score_class_distance
        else:
            raise ValueError(
                f"Unknown Energy-OOD score normalization: {score_normalization}")
        seen_distance = class_distance[:, seen_positions]
        unseen_distance = class_distance[:, unseen_positions]
        seen_energy_distance = energy_distance[:, seen_positions]

        # A larger score means that the sample lies closer to the Seen-class
        # embedding set. This is the energy-only part of EZ-AVOOD.
        seen_energy = torch.logsumexp(-seen_energy_distance, dim=1)
        seen_prediction = seen_label_array[torch.argmin(seen_distance, dim=1)]
        unseen_prediction = unseen_label_array[
            torch.argmin(unseen_distance, dim=1)]

        total_classes = len(dataset.all_class_ids)
        target_counts = torch.bincount(
            targets.long(), minlength=total_classes).float()

        def recall_for(predictions):
            correct_targets = targets.long()[predictions.eq(targets.long())]
            correct_counts = torch.bincount(
                correct_targets, minlength=total_classes).float()
            recalls = torch.zeros(total_classes, dtype=torch.float,
                                  device=device)
            present = target_counts > 0
            recalls[present] = correct_counts[present] / target_counts[present]
            return recalls

        zsl_recall = recall_for(unseen_prediction)
        zsl = zsl_recall[unseen_label_array].mean().item()

        if threshold is None:
            # Quantiles cap validation cost while covering both all-Seen and
            # all-Unseen endpoints. Intermediate values are sufficient because
            # routing changes only when a sample score is crossed.
            n_quantiles = min(401, seen_energy.numel() + 1)
            quantiles = torch.linspace(0, 1, n_quantiles, device=device)
            candidates = torch.unique(torch.quantile(seen_energy, quantiles))
            scale = max(float(seen_energy.abs().max().item()), 1.0)
            margin = scale * 1e-6
            candidates = torch.cat((
                seen_energy.min().reshape(1) - margin,
                candidates,
                seen_energy.max().reshape(1) + margin,
            ))
        else:
            candidates = torch.tensor(
                [float(threshold)], dtype=seen_energy.dtype, device=device)

        best = None
        for candidate in candidates:
            route_to_seen = seen_energy >= candidate
            predictions = torch.where(
                route_to_seen, seen_prediction, unseen_prediction)
            gzsl_recall = recall_for(predictions)
            seen = gzsl_recall[seen_label_array].mean().item()
            unseen = gzsl_recall[unseen_label_array].mean().item()
            hm = 2 * seen * unseen / (seen + unseen + np.finfo(float).eps)
            result = {
                "seen": seen,
                "unseen": unseen,
                "hm": hm,
                "zsl": zsl,
                "recall": zsl_recall.tolist(),
                "gzsl_recall": gzsl_recall.tolist(),
                "threshold": candidate.item(),
                "routed_seen_rate": route_to_seen.float().mean().item(),
                "score_normalization": score_normalization,
                "score_source": score_source,
            }
            if best is None or result["hm"] > best["hm"]:
                best = result

    return best


def evaluate_dataset(dataset, model, device, distance_fn, best_beta=None,
                     args=None, best_fusion_weight=None,
                     adaptive_modality_fusion=False,
                     energy_ood_routing=False,
                     best_energy_threshold=None,
                     energy_ood_score="raw"):
    data = dataset.all_data
    data_a = data["audio"].to(device)
    data_v = data["video"].to(device)
    data_t = data["text"].to(device)
    all_data = (
        data_a, data_v, data_t
    )
    try:
        if args.z_score_inputs:
            all_data = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in all_data])
    except AttributeError:
        print("Namespace has no fitting attribute. Continuing")

    all_targets = dataset.targets.to(device)
    model.eval()
    outputs_all = model(*all_data, *all_data)
    if args.cjme==True:
        a_p, v_p, t_p, a_q, v_q, t_q, attention_weights, threshold_attention=outputs_all
    else:
        x_t_p, a_p, v_p, t_p, a_q, v_q, t_q, x_ta_p, x_tv_p, x_tt_p, x_ta_q, x_tv_q = outputs_all
        threshold_attention=None
    audio_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="audio", device=device,
                                           distance_fn=distance_fn, best_beta=best_beta, args=args)
    video_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="video", device=device,
                                           distance_fn=distance_fn, best_beta=best_beta, args=args)
    both_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="both", device=device,
                                          distance_fn=distance_fn, best_beta=best_beta,
                                          best_fusion_weight=best_fusion_weight,
                                          adaptive_modality_fusion=adaptive_modality_fusion,
                                          args=args, attention_weights=threshold_attention)
    evaluation = {
        "audio": audio_evaluation,
        "video": video_evaluation,
        "both": both_evaluation
    }
    if energy_ood_routing:
        evaluation["energy_ood"] = get_energy_ood_evaluation(
            dataset, all_targets, a_p, v_p, t_p,
            distance_fn=distance_fn,
            threshold=best_energy_threshold,
            score_normalization=energy_ood_score)
    return evaluation


def get_class_names(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("r") as f:
        classes = sorted([line.strip() for line in f])
    return classes


def load_model_weights(weights_path, model):
    logging.info(f"Loading model weights from {weights_path}")
    load_dict = torch.load(weights_path, map_location="cpu")
    model_weights = load_dict["model"]
    epoch = load_dict["epoch"]
    logging.info(f"Load from epoch: {epoch}")
    load_model_parameters(model, model_weights, strict=True)
    return epoch
    
def plot_hist_from_dict(dict):
    plt.bar(range(len(dict)), list(dict.values()), align="center")
    plt.xticks(range(len(dict)), list(dict.keys()), rotation='vertical')
    plt.tight_layout()
    plt.show()

def save_class_performances(seen_dict, unseen_dict, dataset_name):
    seen_path = Path(f"doc/cvpr2022/fig/final/class_performance_{dataset_name}_seen.pkl")
    unseen_path = Path(f"doc/cvpr2022/fig/final/class_performance_{dataset_name}_unseen.pkl")
    with seen_path.open("wb") as f:
        pickle.dump(seen_dict, f)
        logging.info(f"Saving seen class performances to {seen_path}")
    with unseen_path.open("wb") as f:
        pickle.dump(unseen_dict, f)
        logging.info(f"Saving unseen class performances to {unseen_path}")
