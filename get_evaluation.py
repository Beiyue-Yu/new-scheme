import copy
import logging

import torch

from src.args import args_eval
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, VGGSoundDataset, UCFDataset
from src.model import AVGZSLNet, DeviseModel, APN, CJME
from src.model_improvements import MSTR
from src.model_residual import ResidualMSTR
from src.model_mstr_baseline import MSTRBaseline
from src.languagebind_anchor_residual import LanguageBindAnchorResidual
from src.utils_improvements import get_model_params
from src.test import test
from src.utils import fix_seeds, load_args, load_model_parameters, setup_evaluation, load_model_weights
from pathlib import Path


def task_text_prototypes(dataset):
    """Return the current evaluation task's class texts ordered by raw id."""
    class_ids = torch.unique(torch.as_tensor(
        list(dataset.test_seen_ids) + list(dataset.test_unseen_ids),
        dtype=torch.long), sorted=True)
    text = dataset.data["text"]
    text_data = torch.as_tensor(text["data"], dtype=torch.float32)
    text_targets = torch.as_tensor(text["target"], dtype=torch.long).reshape(-1)
    if text_data.shape[0] != text_targets.numel():
        raise ValueError("text prototype data and target ids must be aligned")
    target_to_position = {
        int(class_id): position
        for position, class_id in enumerate(text_targets.tolist())}
    missing = [int(class_id) for class_id in class_ids.tolist()
               if int(class_id) not in target_to_position]
    if missing:
        raise ValueError(
            f"evaluation text prototypes are missing class ids: {missing}")
    positions = [target_to_position[int(class_id)] for class_id in class_ids]
    return text_data[positions], class_ids


def get_evaluation():
    args = args_eval()
    config = load_args(args.load_path_stage_B)
    assert config.retrain_all, f"--retrain_all flag is not set in load_path_stage_B. Are you sure this is the correct path?. {args.load_path_stage_B}"
    config.device = args.device
    config.evaluation_beta_max = args.evaluation_beta_max
    config.evaluation_beta_step = args.evaluation_beta_step
    config.normalize_shared_embeddings = args.normalize_shared_embeddings
    config.semantic_aware_calibration = args.semantic_aware_calibration
    fix_seeds(config.seed)

    logger, eval_dir, test_stats = setup_evaluation(args, config.__dict__.keys())

    dataset_name = config.dataset_name
    if dataset_name == "AudioSetZSL":
        val_all_dataset = AudioSetZSLDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode="all",
        )
        test_dataset = AudioSetZSLDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode="all",
        )
    elif dataset_name == "VGGSound":
        val_all_dataset = VGGSoundDataset(
            args=config,
            dataset_split="val",
            #dataset_split="test",
            zero_shot_mode=None,
        )
        test_dataset = VGGSoundDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif dataset_name == "UCF":
        val_all_dataset = UCFDataset(
            args=config,
            dataset_split="val",
            #dataset_split="test",
            zero_shot_mode=None,
        )
        test_dataset = UCFDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif dataset_name == "ActivityNet":
        val_all_dataset = ActivityNetDataset(
            args=config,
            dataset_split="val",
            #dataset_split="test",
            zero_shot_mode=None,
        )
        test_dataset = ActivityNetDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()

    is_mstr = bool(getattr(config, 'MSTR', False) or
                   getattr(config, 'AVCA', False))
    is_ale = bool(getattr(config, 'ale', False))
    is_sje = bool(getattr(config, 'sje', False))
    is_devise = bool(getattr(config, 'devise', False))
    is_apn = bool(getattr(config, 'apn', False))
    is_cjme = bool(getattr(config, 'cjme', False))

    if is_mstr:
        # Backward-compat: configs saved before these args existed fall back to defaults.
        snn_T = getattr(config, 'snn_T', 10)
        trl_rank = getattr(config, 'trl_rank', 400)
        snn_tau = getattr(config, 'snn_tau', 2.0)
        lkc_n_slots = getattr(config, 'lkc_n_slots', 4)
        lkc_n_heads = getattr(config, 'lkc_n_heads', 8)
        lkc_residual_scale = getattr(config, 'lkc_residual_scale', 0.2)
        tucker_rank = getattr(config, 'tucker_rank', 60)
        stft_dim = getattr(config, 'stft_dim', 512)
        model_params = get_model_params(config.lr, config.first_additional_triplet, config.second_additional_triplet, \
                                        config.reg_loss, config.additional_triplets_loss, config.embedding_dropout, \
                                        config.decoder_dropout, config.additional_dropout,
                                        config.embeddings_hidden_size, \
                                        config.decoder_hidden_size, config.depth_transformer, config.momentum,
                                        snn_T=snn_T, trl_rank=trl_rank,
                                        snn_tau=snn_tau, lkc_n_slots=lkc_n_slots,
                                        lkc_n_heads=lkc_n_heads,
                                        snn_membrane_readout_scale=getattr(
                                            config, 'snn_membrane_readout_scale', 0.0),
                                        legacy_batch_dth=getattr(
                                            config, 'legacy_batch_dth', False),
                                        lkc_residual_scale=lkc_residual_scale,
                                        ahse_standardize=getattr(
                                            config, 'ahse_standardize', False),
                                        semantic_geometry_weight=getattr(
                                            config, 'semantic_geometry_weight', 0.0),
                                        cross_modal_residual=getattr(
                                            config, 'cross_modal_residual', False),
                                        cross_modal_residual_scale=getattr(
                                            config, 'cross_modal_residual_scale', 0.2),
                                        semantic_contrastive_weight=getattr(
                                            config, 'semantic_contrastive_weight', 0.0),
                                        semantic_contrastive_temperature=getattr(
                                            config, 'semantic_contrastive_temperature', 0.1),
                                        temporal_quality_alignment_weight=getattr(
                                            config, 'temporal_quality_alignment_weight', 0.0),
                                        cross_modal_contrastive_weight=getattr(
                                            config, 'cross_modal_contrastive_weight', 0.0),
                                        cross_modal_contrastive_temperature=getattr(
                                            config, 'cross_modal_contrastive_temperature', 0.1),
                                        avla_contrastive_only=getattr(
                                            config, 'avla_contrastive_only', False),
                                        avla_temperature=getattr(
                                            config, 'avla_temperature', 0.1),
                                        global_prototype_contrastive_weight=getattr(
                                            config, 'global_prototype_contrastive_weight', 0.0),
                                        global_prototype_contrastive_temperature=getattr(
                                            config, 'global_prototype_contrastive_temperature', 0.1),
                                        semantic_hard_negative_weight=getattr(
                                            config, 'semantic_hard_negative_weight', 0.0),
                                        semantic_hard_negative_margin=getattr(
                                            config, 'semantic_hard_negative_margin', 0.1),
                                        semantic_batch_hard_weight=getattr(
                                            config, 'semantic_batch_hard_weight', 0.0),
                                        semantic_batch_hard_margin=getattr(
                                            config, 'semantic_batch_hard_margin', 0.1),
                                        semantic_batch_hard_neighbors=getattr(
                                            config, 'semantic_batch_hard_neighbors', 5),
                                        semantic_neighbor_rank_weight=getattr(
                                            config, 'semantic_neighbor_rank_weight', 0.0),
                                        semantic_neighbor_rank_margin=getattr(
                                            config, 'semantic_neighbor_rank_margin', 0.05),
                                        semantic_neighbor_rank_neighbors=getattr(
                                            config, 'semantic_neighbor_rank_neighbors', 5),
                                        semantic_mixup_weight=getattr(
                                            config, 'semantic_mixup_weight', 0.0),
                                        semantic_mixup_alpha=getattr(
                                            config, 'semantic_mixup_alpha', 1.0),
                                        feature_mixup_weight=getattr(
                                            config, 'feature_mixup_weight', 0.0),
                                        feature_mixup_alpha=getattr(
                                            config, 'feature_mixup_alpha', 0.2),
                                        feature_debias_weight=getattr(
                                            config, 'feature_debias_weight', 0.0),
                                        feature_debias_temperature=getattr(
                                            config, 'feature_debias_temperature', 0.1),
                                        text_projection_norm=getattr(
                                            config, 'text_projection_norm', 'batchnorm'),
                                        text_embedding_size=getattr(
                                            config, 'text_embedding_size', 300),
                                        tucker_rank=tucker_rank, stft_dim=stft_dim)
        model_params.update(
            fusion_mode=getattr(config, 'fusion_mode', 'stft'),
            vector_trl_rank=getattr(config, 'vector_trl_rank', 64),
            stft_vector_trl=getattr(config, 'stft_vector_trl', False),
            stft_spatial_reliability_gate=getattr(
                config, 'stft_spatial_reliability_gate', False),
            trl_gate_scale=getattr(config, 'trl_gate_scale', 0.25),
            backbone_lr_scale=getattr(config, 'backbone_lr_scale', 1.0),
            use_glp=not getattr(config, 'disable_glp', False),
            use_lkc=not getattr(config, 'disable_lkc', False),
        )

    if not any((is_ale, is_sje, is_devise, is_apn, is_cjme, is_mstr)):
        model_A = AVGZSLNet(config)
    elif is_ale or is_sje or is_devise:
        model_A=DeviseModel(config)
    elif is_apn:
        model_A=APN(config)
    elif is_cjme:
        model_A=CJME(config)
    elif is_mstr:
        fusion_mode = getattr(config, 'fusion_mode', 'stft')
        if (getattr(config, 'cross_modal_residual', False) and
                fusion_mode != 'stft'):
            raise ValueError(
                "cross_modal_residual checkpoints require "
                "fusion_mode=stft")
        if (getattr(config, 'semantic_contrastive_weight', 0.0) > 0.0 and
                fusion_mode != 'stft'):
            raise ValueError(
                "semantic contrastive checkpoints require fusion_mode=stft")
        if (getattr(config, 'cross_modal_contrastive_weight', 0.0) > 0.0 and
                fusion_mode != 'stft'):
            raise ValueError(
                "cross-modal contrastive checkpoints require fusion_mode=stft")
        if (getattr(config, 'avla_contrastive_only', False) and
                fusion_mode != 'stft'):
            raise ValueError("AV-language alignment checkpoints require fusion_mode=stft")
        if (getattr(config, 'global_prototype_contrastive_weight', 0.0) > 0.0 and
                fusion_mode != 'stft'):
            raise ValueError(
                "global prototype contrastive checkpoints require "
                "fusion_mode=stft")
        if (getattr(config, 'semantic_hard_negative_weight', 0.0) > 0.0 and
                fusion_mode != 'stft'):
            raise ValueError(
                "semantic hard-negative checkpoints require fusion_mode=stft")
        if (getattr(config, 'semantic_batch_hard_weight', 0.0) > 0.0 and
                fusion_mode != 'stft'):
            raise ValueError(
                "semantic batch-hard checkpoints require fusion_mode=stft")
        if (getattr(config, 'semantic_neighbor_rank_weight', 0.0) > 0.0 and
                fusion_mode != 'stft'):
            raise ValueError(
                "semantic neighbour-rank checkpoints require fusion_mode=stft")
        if (getattr(config, 'semantic_mixup_weight', 0.0) > 0.0 and
                fusion_mode != 'stft'):
            raise ValueError(
                "semantic mixup checkpoints require fusion_mode=stft")
        if (getattr(config, 'feature_mixup_weight', 0.0) > 0.0 and
                fusion_mode != 'stft'):
            raise ValueError(
                "feature mixup checkpoints require fusion_mode=stft")
        if (getattr(config, 'feature_debias_weight', 0.0) > 0.0 and
                fusion_mode != 'stft'):
            raise ValueError(
                "feature debias checkpoints require fusion_mode=stft")
        if fusion_mode == 'languagebind_anchor_residual':
            model_class = LanguageBindAnchorResidual
        elif fusion_mode == 'residual':
            model_class = ResidualMSTR
        elif fusion_mode in {'mstr_released', 'mstr_paper'}:
            model_class = MSTRBaseline
        else:
            model_class = MSTR
        model_A = model_class(params_model=model_params,
                              input_size_audio=config.input_size_audio,
                              input_size_video=config.input_size_video)

    logger.info(model_A)

    model_B = copy.deepcopy(model_A)

    # MSTR selects Stage B using Stage A's best epoch. The optional independent
    # Stage B selection is kept for compatibility with existing experiments.
    # Use matched selection for protocol-faithful MSTR comparisons.
    weights_path_stage_A = list(args.load_path_stage_A.glob("*_score.pt"))[0]
    epoch_A = load_model_weights(weights_path_stage_A, model_A)
    if args.stage_b_selection == "matched":
        matching = list((args.load_path_stage_B / "checkpoints").glob(
            f"*_ckpt_{epoch_A - 1}.pt"))
        if not matching:
            raise FileNotFoundError(
                f"No Stage B checkpoint found for Stage A epoch {epoch_A}; "
                "train Stage B with --save_checkpoints")
        weights_path_stage_B = matching[0]
        selection = f"matched Stage A epoch {epoch_A}"
    else:
        loss_candidates = list(args.load_path_stage_B.glob("*_loss.pt"))
        score_candidates = list(args.load_path_stage_B.glob("*_score.pt"))
        if getattr(config, "retrain_all", False) and loss_candidates:
            weights_path_stage_B = loss_candidates[0]
            selection = "independent Stage B lowest validation loss"
        elif score_candidates:
            weights_path_stage_B = score_candidates[0]
            selection = "independent Stage B best score"
        elif loss_candidates:
            weights_path_stage_B = loss_candidates[0]
            selection = "independent Stage B lowest validation loss"
        else:
            raise FileNotFoundError(
                "No Stage B score or loss checkpoint found for independent "
                "selection")
    _ = load_model_weights(weights_path_stage_B, model_B)
    logger.info(f"Stage A best-score checkpoint epoch: {epoch_A}; "
                f"Stage B checkpoint ({selection}): {weights_path_stage_B.name}")

    if args.text_bn_semantic_recalibration:
        if not (isinstance(model_A, MSTR) and isinstance(model_B, MSTR)):
            raise ValueError(
                "--text_bn_semantic_recalibration supports only STFT MSTR")
        val_texts, _ = task_text_prototypes(val_all_dataset)
        test_texts, _ = task_text_prototypes(test_dataset)
        count_a = model_A.recalibrate_text_batchnorm(
            val_texts, mix=args.text_bn_semantic_mix)
        count_b = model_B.recalibrate_text_batchnorm(
            test_texts, mix=args.text_bn_semantic_mix)
        logger.info(
            "Recalibrated %d/%d text BatchNorm layers from %d validation and "
            "%d test task class prototypes (mix %.2f).",
            count_a, count_b, val_texts.shape[0], test_texts.shape[0],
            args.text_bn_semantic_mix)

    model_A.to(args.device)
    model_B.to(args.device)

    bias_report_name = "bias_report.json"
    if args.normalize_shared_embeddings:
        bias_report_name = "bias_report_normalized_embeddings.json"
    elif args.energy_ood_routing:
        bias_report_name = f"bias_report_energy_ood_{args.energy_ood_score}.json"
    elif args.text_bn_semantic_recalibration:
        bias_report_name = "bias_report_text_bn_semantic_recalibration.json"
    elif args.adaptive_modality_fusion:
        bias_report_name = "bias_report_adaptive_fusion.json"
    elif args.semantic_aware_calibration:
        bias_report_name = "bias_report_semantic_aware_calibration.json"
    elif (args.evaluation_beta_max != 3.0 or
          args.evaluation_beta_step is not None):
        bias_report_name = "bias_report_extended_beta.json"

    test(
        eval_name=args.eval_name,
        val_dataset=val_all_dataset,
        test_dataset=test_dataset,
        model_A=model_A,
        model_B=model_B,
        device=args.device,
        distance_fn=config.distance_fn,
        devise_model=is_ale or is_sje or is_devise,
        new_model_attention=is_mstr,
        apn=is_apn,
        args=config,
        adaptive_modality_fusion=args.adaptive_modality_fusion,
        energy_ood_routing=args.energy_ood_routing,
        energy_ood_score=args.energy_ood_score,
        bias_report_path=args.load_path_stage_B / bias_report_name
    )

    logger.info("FINISHED")


if __name__ == "__main__":
    get_evaluation()
