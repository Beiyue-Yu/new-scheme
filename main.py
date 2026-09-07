import sys

import numpy as np
import torch
from torch import optim
from torch.utils import data
from ptflops import get_model_complexity_info
from src.args import args_main
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, ContrastiveDataset, VGGSoundDataset, UCFDataset
from src.loss import AVGZSLLoss, L2Loss, SquaredL2Loss, ClsContrastiveLoss, APN_Loss, CJMELoss
from src.metrics import DetailedLosses, MeanClassAccuracy, PercentOverlappingClasses, TargetDifficulty
from src.model import AVGZSLNet, DeviseModel, APN, CJME
from src.sampler import SamplerFactory
from src.model_improvements import MSTR
from src.model_residual import ResidualMSTR
from src.model_mstr_baseline import MSTRBaseline
from src.languagebind_anchor_residual import LanguageBindAnchorResidual
from src.utils_improvements import get_model_params
from src.train import train
from src.utils import (fix_seeds, load_model_parameters, load_training_state,
                       setup_experiment)
from torch.optim.lr_scheduler import ReduceLROnPlateau


def dataset_text_prototypes(dataset, class_ids):
    """Return class texts aligned to explicit raw class ids."""
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    class_ids = np.unique(class_ids)
    text = dataset.data["text"]
    text_data = torch.as_tensor(text["data"], dtype=torch.float32)
    text_targets = torch.as_tensor(text["target"], dtype=torch.long).reshape(-1)
    if text_data.shape[0] != text_targets.numel():
        raise ValueError("text prototype data and target ids must be aligned")
    target_to_position = {
        int(class_id): position
        for position, class_id in enumerate(text_targets.tolist())}
    missing = [int(class_id) for class_id in class_ids
               if int(class_id) not in target_to_position]
    if missing:
        raise ValueError(f"final task class text prototypes are missing ids: {missing}")
    positions = [target_to_position[int(class_id)] for class_id in class_ids]
    return text_data[positions], torch.as_tensor(class_ids, dtype=torch.long)


def final_task_text_prototypes(dataset):
    """Return final Seen+Unseen class texts ordered by raw class id."""
    class_ids = np.unique(np.concatenate((
        np.asarray(dataset.test_seen_ids), np.asarray(dataset.test_unseen_ids))))
    return dataset_text_prototypes(dataset, class_ids)


def validate_feature_dimensions(args, dataset):
    """Fail before model construction when cache and CLI dimensions disagree."""
    expected = {
        "audio": int(args.input_size_audio),
        "video": int(args.input_size_video),
        "text": int(args.text_embedding_size),
    }
    actual = {
        name: int(dataset.data[name]["data"].shape[-1])
        for name in expected
    }
    mismatches = [
        f"{name}: expected {expected[name]}, found {actual[name]}"
        for name in expected if actual[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "Feature dimension mismatch for "
            f"{args.dataset_name}/{args.feature_extraction_method}: "
            + "; ".join(mismatches)
            + ". Set input_size_audio/input_size_video/text_embedding_size "
              "to match the processed cache.")


def main():
    args = args_main()
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError(
            "--init_checkpoint and --resume_checkpoint cannot be used together")
    if args.input_size is not None:
        args.input_size_audio = args.input_size
        args.input_size_video = args.input_size
    fix_seeds(args.seed)
    logger, log_dir, writer, train_stats, val_stats = setup_experiment(args, "epoch", "loss", "hm")

    if args.dataset_name == "AudioSetZSL":
        train_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="seen",
        )

        val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode="seen",
        )

        train_val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode="seen",
        )

        val_all_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode="all",
        )

    elif args.dataset_name == "VGGSound":
        train_dataset = VGGSoundDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="train",
        )
        val_dataset = VGGSoundDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )

        train_val_dataset = VGGSoundDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )

        val_all_dataset = VGGSoundDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "UCF":
        train_dataset = UCFDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="train",
        )
        val_dataset = UCFDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )

        train_val_dataset = UCFDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )

        val_all_dataset = UCFDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "ActivityNet":
        train_dataset = ActivityNetDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="train",
        )
        val_dataset = ActivityNetDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )

        train_val_dataset = ActivityNetDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )

        val_all_dataset = ActivityNetDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()

    validate_feature_dimensions(args, train_dataset)

    contrastive_train_dataset = ContrastiveDataset(train_dataset)
    contrastive_val_dataset = ContrastiveDataset(val_dataset)
    contrastive_train_val_dataset = ContrastiveDataset(train_val_dataset)
    contrastive_val_all_dataset = ContrastiveDataset(val_all_dataset)

    train_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_train_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    train_val_class_ids = list(contrastive_train_val_dataset.target_to_indices)
    train_val_class_idxs = [
        contrastive_train_val_dataset.target_to_indices[class_id]
        for class_id in train_val_class_ids]
    stage_b_class_distribution = None
    if args.retrain_all and args.stage_b_new_class_fraction is not None:
        stage_a_seen_ids = {int(class_id) for class_id in train_dataset.classes}
        seen_positions = [
            index for index, class_id in enumerate(train_val_class_ids)
            if int(class_id) in stage_a_seen_ids]
        new_positions = [
            index for index, class_id in enumerate(train_val_class_ids)
            if int(class_id) not in stage_a_seen_ids]
        if not seen_positions or not new_positions:
            raise ValueError(
                "Stage B group-balanced sampling requires both Stage A seen "
                "classes and newly introduced classes")
        stage_b_class_distribution = np.zeros(
            len(train_val_class_ids), dtype=np.float64)
        new_fraction = args.stage_b_new_class_fraction
        stage_b_class_distribution[seen_positions] = (
            (1.0 - new_fraction) / len(seen_positions))
        stage_b_class_distribution[new_positions] = (
            new_fraction / len(new_positions))
        logger.info(
            "Stage B group-balanced sampling: %d Stage A seen classes receive "
            "%.3f batch mass; %d new classes receive %.3f batch mass.",
            len(seen_positions), 1.0 - new_fraction,
            len(new_positions), new_fraction)

    train_val_sampler = SamplerFactory(logger).get(
        class_idxs=train_val_class_idxs,
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random',
        target_class_distribution=stage_b_class_distribution,
    )

    val_all_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_all_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    train_loader = data.DataLoader(
        dataset=contrastive_train_dataset,
        batch_sampler=train_sampler,
        num_workers=0
    )

    val_loader = data.DataLoader(
        dataset=contrastive_val_dataset,
        batch_sampler=val_sampler,
        num_workers=0
    )

    train_val_loader = data.DataLoader(
        dataset=contrastive_train_val_dataset,
        batch_sampler=train_val_sampler,
        num_workers=0
    )

    val_all_loader = data.DataLoader(
        dataset=contrastive_val_all_dataset,
        batch_sampler=val_all_sampler,
        num_workers=0
    )

    if args.AVCA==True or args.MSTR==True:
        model_params = get_model_params(args.lr, args.first_additional_triplet, args.second_additional_triplet, \
                                        args.reg_loss, args.additional_triplets_loss, args.embedding_dropout, \
                                        args.decoder_dropout, args.additional_dropout, args.embeddings_hidden_size, \
                                        args.decoder_hidden_size, args.depth_transformer, args.momentum,
                                        snn_T=args.snn_T, trl_rank=args.trl_rank,
                                        snn_tau=args.snn_tau,
                                        snn_activity_floor_weight=args.snn_activity_floor_weight,
                                        snn_min_spike_rate=args.snn_min_spike_rate,
                                        snn_membrane_readout_scale=args.snn_membrane_readout_scale,
                                        lkc_n_slots=args.lkc_n_slots,
                                        lkc_n_heads=args.lkc_n_heads,
                                        lkc_residual_scale=args.lkc_residual_scale,
                                        tucker_rank=args.tucker_rank,
                                        stft_dim=args.stft_dim,
                                        fusion_mode=args.fusion_mode,
                                        vector_trl_rank=args.vector_trl_rank,
                                        stft_vector_trl=args.stft_vector_trl,
                                        stft_spatial_reliability_gate=args.stft_spatial_reliability_gate,
                                        trl_gate_scale=args.trl_gate_scale,
                                        backbone_lr_scale=args.backbone_lr_scale,
                                        use_glp=not args.disable_glp,
                                        use_lkc=not args.disable_lkc,
                                        legacy_batch_dth=args.legacy_batch_dth,
                                        ahse_standardize=args.ahse_standardize,
                                        semantic_geometry_weight=args.semantic_geometry_weight,
                                        cross_modal_residual=args.cross_modal_residual,
                                        cross_modal_residual_scale=args.cross_modal_residual_scale,
                                        semantic_contrastive_weight=args.semantic_contrastive_weight,
                                        semantic_contrastive_temperature=args.semantic_contrastive_temperature,
                                        pseudo_unseen_weight=args.pseudo_unseen_weight,
                                        pseudo_unseen_temperature=args.pseudo_unseen_temperature,
                                        pseudo_unseen_class_fraction=args.pseudo_unseen_class_fraction,
                                        pseudo_unseen_min_classes=args.pseudo_unseen_min_classes,
                                        snn_temporal_consistency_weight=args.snn_temporal_consistency_weight,
                                        snn_temporal_view_fraction=args.snn_temporal_view_fraction,
                                        temporal_quality_alignment_weight=args.temporal_quality_alignment_weight,
                                        cross_modal_contrastive_weight=args.cross_modal_contrastive_weight,
                                        cross_modal_contrastive_temperature=args.cross_modal_contrastive_temperature,
                                        avla_contrastive_only=args.avla_contrastive_only,
                                        avla_temperature=args.avla_temperature,
                                        global_prototype_contrastive_weight=args.global_prototype_contrastive_weight,
                                        global_prototype_contrastive_temperature=args.global_prototype_contrastive_temperature,
                                        semantic_hard_negative_weight=args.semantic_hard_negative_weight,
                                        semantic_hard_negative_margin=args.semantic_hard_negative_margin,
                                        semantic_batch_hard_weight=args.semantic_batch_hard_weight,
                                        semantic_batch_hard_margin=args.semantic_batch_hard_margin,
                                        semantic_batch_hard_neighbors=args.semantic_batch_hard_neighbors,
                                        semantic_neighbor_rank_weight=args.semantic_neighbor_rank_weight,
                                        semantic_neighbor_rank_margin=args.semantic_neighbor_rank_margin,
                                        semantic_neighbor_rank_neighbors=args.semantic_neighbor_rank_neighbors,
                                        semantic_mixup_weight=args.semantic_mixup_weight,
                                        semantic_mixup_alpha=args.semantic_mixup_alpha,
                                        feature_mixup_weight=args.feature_mixup_weight,
                                        feature_mixup_alpha=args.feature_mixup_alpha,
                                        feature_debias_weight=args.feature_debias_weight,
                                        feature_debias_temperature=args.feature_debias_temperature,
                                        text_projection_norm=args.text_projection_norm,
                                        text_embedding_size=args.text_embedding_size)


    if args.ale==True or args.devise==True or args.sje==True:
        model= DeviseModel(args)
    elif args.apn==True:
        model=APN(args)
    elif args.cjme==True:
        model=CJME(args)
    elif args.AVCA==True or args.MSTR==True:
        if args.fusion_mode == "languagebind_anchor_residual":
            model_class = LanguageBindAnchorResidual
        elif args.fusion_mode == "residual":
            model_class = ResidualMSTR
        elif args.fusion_mode in {"mstr_released", "mstr_paper"}:
            model_class = MSTRBaseline
        else:
            model_class = MSTR
        if args.ahse_standardize and model_class is not MSTR:
            raise ValueError(
                "--ahse_standardize currently supports only --fusion_mode stft")
        if args.cross_modal_residual and model_class is not MSTR:
            raise ValueError(
                "--cross_modal_residual currently supports only "
                "--fusion_mode stft")
        if args.semantic_contrastive_weight > 0.0 and model_class is not MSTR:
            raise ValueError(
                "--semantic_contrastive_weight currently supports only "
                "--fusion_mode stft")
        if args.pseudo_unseen_weight > 0.0 and model_class is not MSTR:
            raise ValueError(
                "--pseudo_unseen_weight currently supports only "
                "--fusion_mode stft")
        if (args.snn_temporal_consistency_weight > 0.0 and
                model_class is not MSTR):
            raise ValueError(
                "--snn_temporal_consistency_weight currently supports only "
                "--fusion_mode stft")
        if (args.temporal_quality_alignment_weight > 0.0 and
                model_class is not MSTR):
            raise ValueError(
                "--temporal_quality_alignment_weight currently supports only "
                "--fusion_mode stft")
        if (args.cross_modal_contrastive_weight > 0.0 and
                model_class is not MSTR):
            raise ValueError(
                "--cross_modal_contrastive_weight currently supports only "
                "--fusion_mode stft")
        if args.avla_contrastive_only and model_class is not MSTR:
            raise ValueError(
                "--avla_contrastive_only currently supports only "
                "--fusion_mode stft")
        if (args.global_prototype_contrastive_weight > 0.0 and
                model_class is not MSTR):
            raise ValueError(
                "--global_prototype_contrastive_weight currently supports "
                "only --fusion_mode stft")
        if args.semantic_hard_negative_weight > 0.0 and model_class is not MSTR:
            raise ValueError(
                "--semantic_hard_negative_weight currently supports only "
                "--fusion_mode stft")
        if args.semantic_batch_hard_weight > 0.0 and model_class is not MSTR:
            raise ValueError(
                "--semantic_batch_hard_weight currently supports only "
                "--fusion_mode stft")
        if args.semantic_neighbor_rank_weight > 0.0 and model_class is not MSTR:
            raise ValueError(
                "--semantic_neighbor_rank_weight currently supports only "
                "--fusion_mode stft")
        if args.semantic_mixup_weight > 0.0 and model_class is not MSTR:
            raise ValueError(
                "--semantic_mixup_weight currently supports only "
                "--fusion_mode stft")
        if args.feature_mixup_weight > 0.0 and model_class is not MSTR:
            raise ValueError(
                "--feature_mixup_weight currently supports only "
                "--fusion_mode stft")
        if args.feature_debias_weight > 0.0 and model_class is not MSTR:
            raise ValueError(
                "--feature_debias_weight currently supports only "
                "--fusion_mode stft")
        model = model_class(model_params, input_size_audio=args.input_size_audio,
                            input_size_video=args.input_size_video)
        if args.pseudo_unseen_weight > 0.0:
            episode_dataset = train_val_dataset if args.retrain_all else train_dataset
            episode_text, episode_ids = dataset_text_prototypes(
                episode_dataset, episode_dataset.classes)
            model.set_pseudo_unseen_text_prototypes(episode_text, episode_ids)
            logger.info(
                "Enabled train-only pseudo-Unseen episodes over %d class "
                "prototypes (weight %.4f, temperature %.4f, query fraction %.2f)",
                episode_ids.numel(), args.pseudo_unseen_weight,
                args.pseudo_unseen_temperature, args.pseudo_unseen_class_fraction)
        if args.global_prototype_contrastive_weight > 0.0:
            text_prototypes, prototype_ids = final_task_text_prototypes(
                val_all_dataset)
            model.set_global_text_prototypes(text_prototypes, prototype_ids)
            logger.info(
                "Enabled global prototype contrastive loss over %d final task "
                "class texts (weight %.4f, temperature %.4f)",
                prototype_ids.numel(), args.global_prototype_contrastive_weight,
                args.global_prototype_contrastive_temperature)
    else:
        model = AVGZSLNet(args)

    if getattr(args, "init_checkpoint", None) is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        load_model_parameters(model, checkpoint["model"], strict=False)
        logger.info("Initialized matching model parameters from %s (epoch %s)",
                    args.init_checkpoint, checkpoint.get("epoch", "unknown"))
    model.to(args.device)
    teacher_model = None
    teacher_seen_class_ids = None
    if args.stage_b_teacher_checkpoint is not None:
        if not args.retrain_all or not isinstance(model, MSTR):
            raise ValueError(
                "--stage_b_teacher_checkpoint supports only Stage B STFT MSTR training")
        if args.stage_b_seen_distill_weight <= 0.0:
            raise ValueError(
                "--stage_b_seen_distill_weight must be positive when a "
                "Stage B teacher checkpoint is provided")
        teacher_checkpoint = torch.load(
            args.stage_b_teacher_checkpoint, map_location="cpu")
        teacher_model = MSTR(
            model_params, input_size_audio=args.input_size_audio,
            input_size_video=args.input_size_video)
        load_model_parameters(teacher_model, teacher_checkpoint["model"], strict=True)
        teacher_model.to(args.device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        teacher_seen_class_ids = torch.as_tensor(
            train_dataset.classes, dtype=torch.long, device=args.device)
        if teacher_seen_class_ids.numel() == 0:
            raise ValueError("Stage B teacher has no Stage A seen classes to distill")
        logger.info(
            "Loaded frozen Stage A teacher from %s (epoch %s) for %d seen classes "
            "with distillation weight %.4f",
            args.stage_b_teacher_checkpoint,
            teacher_checkpoint.get("epoch", "unknown"),
            teacher_seen_class_ids.numel(), args.stage_b_seen_distill_weight)
    elif args.stage_b_seen_distill_weight > 0.0:
        raise ValueError(
            "--stage_b_seen_distill_weight requires --stage_b_teacher_checkpoint")
    start_epoch = 0
    resume_best_loss = None
    resume_best_score = None
    distance_fn = getattr(sys.modules[__name__], args.distance_fn)()
    if args.ale==True:
        criterion = ClsContrastiveLoss(margin=0.1, max_violation=False, topk=None, reduction="weighted")
    elif args.devise==True:
        criterion = ClsContrastiveLoss(margin=0.1, max_violation=False, topk=None, reduction="sum")
    elif args.sje==True:
        criterion = ClsContrastiveLoss(margin=0.1, max_violation=True, topk=1, reduction="sum")
    elif args.apn==True:
        criterion=APN_Loss()
    elif args.cjme==True:
        criterion=CJMELoss(margin=args.margin, distance_fn=distance_fn)
    elif args.AVCA==True or args.MSTR==True:
        # MSTR carries its own loss + internal optimizer, no external criterion needed.
        criterion=None
    else:
        criterion = AVGZSLLoss(margin=args.margin, distance_fn=distance_fn)

    # MSTR manages its optimizer/scheduler internally; other models use an external Adam.
    if args.AVCA==True or args.MSTR==True:
        optimizer = None
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, verbose=True) if (args.lr_scheduler and optimizer is not None) else None

    if args.resume_checkpoint is not None:
        active_optimizer = (getattr(model, "optimizer_gen", None)
                            if args.AVCA or args.MSTR else optimizer)
        active_scheduler = (getattr(model, "scheduler_gen", None)
                            if args.AVCA or args.MSTR else lr_scheduler)
        start_epoch, resume_best_loss, resume_best_score = load_training_state(
            args.resume_checkpoint, model, active_optimizer, active_scheduler)
        logger.info("Resumed complete training state from %s at epoch %d",
                    args.resume_checkpoint, start_epoch)

    if args.retrain_all:
        # Stage B adds Stage A's validation-Unseen classes to train_val. Any
        # GZSL score on val_all would therefore label trained classes as
        # "Unseen" and leak to HM; use held-out loss only in this stage.
        metrics = []
        logger.info(
            "Stage B retrain_all: GZSL validation metrics are disabled because "
            "the validation-Unseen classes are part of training.")
    else:
        metrics = [
            MeanClassAccuracy(
                model=model, dataset=val_all_dataset, device=args.device,
                distance_fn=distance_fn,
                model_devise=args.ale or args.sje or args.devise,
                new_model_attention=args.AVCA or args.MSTR,
                apn=args.apn, args=args)
        ]




    logger.info(model)
    logger.info(criterion)
    logger.info(optimizer)
    logger.info(lr_scheduler)
    logger.info([metric.__class__.__name__ for metric in metrics])

    if args.val_all_loss:
        v_loader = val_all_loader
    elif args.retrain_all:
        v_loader = train_val_loader
    else:
        v_loader = val_loader

    best_loss, best_score = train(
        train_loader=train_val_loader if args.retrain_all else train_loader,
        val_loader=v_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device,
        writer=writer,
        metrics=metrics,
        train_stats=train_stats,
        new_model_attention=args.AVCA or args.MSTR,
        val_stats=val_stats,
        log_dir=log_dir,
        model_devise=args.ale or args.sje or args.devise,
        apn=args.apn,
        cjme=args.cjme,
        args=args,
        start_epoch=start_epoch,
        best_loss=resume_best_loss,
        best_score=resume_best_score,
        teacher_model=teacher_model,
        teacher_seen_class_ids=teacher_seen_class_ids,
        teacher_weight=args.stage_b_seen_distill_weight,
    )

    writer.close()
    logger.info(f"FINISHED. Run is stored at {log_dir}")


if __name__ == '__main__':
    main()
