import argparse
import pathlib


def args_main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Explainable Audio Visual Low Shot Learning")

    ### Filesystem ###
    parser.add_argument(
        "--root_dir",
        help="Path to dataset directory. Expected subfolder structure: '{root_dir}/features/{feature_extraction_method}/{audio,video,text}'",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument(
        "--feature_extraction_method",
        help="Name of folder containing respective extracted features. Has to match {feature_extraction_method} in --root_dir argument.",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument(
        "--dropout_baselines",
        help="Dropout to use for baselines",
        default=0.2,
        type=float
    )
    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to use",
        choices=["AudioSetZSL", "VGGSound", "UCF", "ActivityNet"],
        default="AudioSetZSL",
        type=str
    )

    parser.add_argument(
        "--momentum",
        help="Momentum for batch norm",
        default = 0.99,
        type = float
    )


    parser.add_argument(
        "--zero_shot_split",
        help="Name of zero shot split to use.",
        choices=["", "main_split", "cls_split"],
        default=""
    )

    parser.add_argument(
        "--manual_text_word2vec",
        help="Flag to use the manual word2vec text embeddings. CARE: Need to create cache files again!",
        action="store_true"
    )

    parser.add_argument(
        "--val_all_loss",
        help="Validate loss with seen + unseen",
        action="store_true"
    )

    parser.add_argument(
        "--additional_triplets_loss",
        help="Flag for using more triplets loss",
        action="store_true"
    )

    parser.add_argument(
        "--reg_loss",
        help="Flag for setting the regularization loss",
        action="store_true"

    )

    parser.add_argument(
        "--cycle_loss",
        help="Flag for using cycle loss",
        action="store_true"
    )

    parser.add_argument(
        "--retrain_all",
        help="Retrain with all data from train and validation",
        action="store_true"
    )

    parser.add_argument(
        "--save_checkpoints",
        help="Save checkpoints of the model every epoch",
        action="store_true"
    )

    ### Development options ###
    parser.add_argument(
        "--debug",
        help="Run the program in debug mode",
        action="store_true"
    )
    parser.add_argument(
        "--verbose",
        help="Run verbosely",
        action="store_true",
    )
    parser.add_argument(
        "--debug_comment",
        help="Custom comment string for the summary writer",
        default="",
        type=str
    )
    parser.add_argument(
        "--disable_tensorboard",
        help="Disable TensorBoard event writing for long-running experiments.",
        action="store_true"
    )
    parser.add_argument(
        "--allow_existing_run",
        help="Allow the managed runner to restart an incomplete run directory.",
        action="store_true"
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs",
        default=100,
        type=int
    )

    parser.add_argument(
        "--norm_inputs",
        help="Normalize inputs before model",
        action="store_true"
    )

    parser.add_argument(
        "--z_score_inputs",
        help="Z-Score standardize inputs before model",
        action="store_true"
    )

    ### Hyperparameters ###
    parser.add_argument(
        "--lr",
        help="Learning rate",
        default=3e-4,
        type=float
    )
    parser.add_argument(
        "--bs",
        help="Batch size",
        default=256,
        type=int
    )
    parser.add_argument(
        "--n_batches",
        help="Number of batches for the balanced batch sampler",
        default=250,
        type=int
    )
    parser.add_argument(
        "--input_size",
        help="Dimension of the extracted features",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--input_size_audio",
        help="Dimension of the extracted audio features",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--input_size_video",
        help="Dimension of the extracted video features",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--text_embedding_size",
        help="Dimension of the class text embeddings used by MSTR",
        default=300,
        type=int,
    )

    parser.add_argument(
        "--embeddings_hidden_size",
        help="Hidden layer size for the embedding networks",
        default=1024,
        type=int
    )
    parser.add_argument(
        "--decoder_hidden_size",
        help="Hidden layer size for the decoder loss network",
        default=64,
        type=int
    )
    parser.add_argument(
        "--embedding_dropout",
        help="Dropout in the embedding networks",
        default=0.8,
        type=float
    )
    parser.add_argument(
        "--decoder_dropout",
        help="Dropout in the decoder loss network",
        default=0.5,
        type=float
    )
    parser.add_argument(
        "--embedding_use_bn",
        help="Use batchnorm in the embedding networks",
        action="store_true",
    )
    parser.add_argument(
        "--decoder_use_bn",
        help="Use batchnorm in the decoder network",
        action="store_true",
    )
    parser.add_argument(
        "--normalize_decoder_outputs",
        help="L2 normalize the outputs of the decoder",
        action="store_true"
    )
    parser.add_argument(
        "--margin",
        help="Margin for the contrastive loss calculation",
        default=1.,
        type=float
    )
    parser.add_argument(
        "--distance_fn",
        help="Distance function for the contrastive loss calculation",
        choices=["L2Loss", "SquaredL2Loss"],
        default="L2Loss",
        type=str
    )
    parser.add_argument(
        "--lr_scheduler",
        help="Use LR_scheduler",
        action="store_true",
    )

    # defaults
    parser.add_argument(
        "--seed",
        help="Random seed",
        default=42,
        type=int
    )
    parser.add_argument(
        "--dump_path",
        help="Path where to create experiment log dirs",
        default=pathlib.Path("."),
        type=pathlib.Path
    )
    parser.add_argument(
        "--device",
        help="Device to run on.",
        choices=["cuda", "cpu"],
        default="cuda"
    )

    parser.add_argument(
        "--baseline",
        help="Flag to use the baseline where we have two ALEs, one for each modality and we just try to push the modalities to text embeddings",
        action="store_true"
    )

    parser.add_argument(
        "--audio_baseline",
        help="Flag to use the audio baseline",
        action="store_true"
    )
    parser.add_argument(
        "--video_baseline",
        help="Flag to use the video baseline",
        action="store_true"

    )
    parser.add_argument(
        "--concatenated_baseline",
        help="Flag to use the concatenated baseline",
        action="store_true"

    )
    parser.add_argument(
        "--cjme",
        help="Flag to use the CJME baseline",
        action="store_true"
    )

    parser.add_argument(
        "--new_model",
        help="Flag to use the new model",
        action="store_true"
    )

    parser.add_argument(
        "--new_model_early_fusion",
        help="Flag to use the early fusion new model",
        action="store_true"
    )

    parser.add_argument(
        "--new_model_middle_fusion",
        help="Flag to set the middle fusion new model",
        action="store_true"
    )

    parser.add_argument(
        "--MSTR",
        help="Flag to set the attention to the new model",
        action="store_true"

    )

    parser.add_argument(
        "--AVCA",
        help="Alias of --MSTR (kept for backward compatibility with the AVCA training scripts).",
        action="store_true"
    )

    parser.add_argument(
        "--snn_T",
        help="Number of time steps for the spiking neural network branches in MSTR.",
        default=10,
        type=int
    )

    parser.add_argument(
        "--trl_rank",
        help="Rank R of the Tucker regression core used by the MSTR spatial branches.",
        default=400,
        type=int
    )

    # --- STFT upgrade hyperparameters (all have safe defaults, so the
    # existing mstr.sh keeps working without specifying them). ---
    parser.add_argument(
        "--snn_tau",
        help="Membrane time constant tau of the STFT LIF neurons (Eq. 5). "
             "Larger -> slower decay between time steps.",
        default=2.0,
        type=float
    )
    parser.add_argument(
        "--snn_activity_floor_weight",
        default=0.0,
        type=float,
        help="Weight of the one-sided SNN firing-rate floor loss; zero "
             "preserves the original objective."
    )
    parser.add_argument(
        "--snn_min_spike_rate",
        default=0.05,
        type=float,
        help="Minimum positive-branch final-layer spike rate used by the "
             "optional SNN activity-floor loss."
    )
    parser.add_argument(
        "--snn_membrane_readout_scale",
        default=0.0,
        type=float,
        help="Maximum learned residual scale for the final LIF membrane "
             "readout; zero preserves spike-only STFT output."
    )
    parser.add_argument(
        "--legacy_batch_dth",
        action="store_true",
        help="Experimental ablation: restore the historical batch-shared DTH "
             "threshold update. Disabled by default because it couples samples."
    )
    parser.add_argument(
        "--lkc_n_slots",
        help="Number of learnable latent knowledge slots in the Latent "
             "Knowledge Combiner (Eq. 1).",
        default=4,
        type=int
    )
    parser.add_argument(
        "--lkc_n_heads",
        help="Number of attention heads in the Latent Knowledge Combiner.",
        default=8,
        type=int
    )
    parser.add_argument(
        "--lkc_residual_scale",
        help="Initial scale of the normalized residual LKC contribution.",
        default=0.2,
        type=float
    )
    parser.add_argument(
        "--tucker_rank",
        help="Rank R of the Temporal-Semantic Tucker Fusion core (Eq. 12-13). "
             "STFT paper ablation finds R=60 optimal on ActivityNet.",
        default=60,
        type=int
    )
    parser.add_argument(
        "--stft_dim",
        help="Internal semantic/temporal feature dimension. The STFT paper uses 512.",
        default=512,
        type=int
    )
    parser.add_argument(
        "--fusion_mode",
        choices=["stft", "residual", "mstr_released", "mstr_paper",
                 "languagebind_anchor_residual"],
        default="stft",
        help="Select full STFT, residual MSTR, released-code-compatible MSTR, "
             "paper-parameter MSTR, or the LanguageBind anchor-residual route."
    )
    parser.add_argument(
        "--disable_glp",
        action="store_true",
        help="Disable GLP current modulation while keeping the adaptive STFT path."
    )
    parser.add_argument(
        "--disable_lkc",
        action="store_true",
        help="Bypass the Latent Knowledge Combiner in the STFT path."
    )
    parser.add_argument(
        "--ahse_standardize",
        action="store_true",
        help="Apply AHSE Stage-I per-feature Z-score standardization to "
             "projected audio-visual and text embeddings."
    )
    parser.add_argument(
        "--semantic_geometry_weight",
        default=0.0,
        type=float,
        help="Weight of the pairwise word2vec geometry preservation loss "
             "on W_proj text embeddings."
    )
    parser.add_argument(
        "--semantic_contrastive_weight",
        default=0.0,
        type=float,
        help="Weight of the supervised batch audio/video-to-text contrastive "
             "loss; zero preserves the original objective."
    )
    parser.add_argument(
        "--semantic_contrastive_temperature",
        default=0.1,
        type=float,
        help="Temperature for the supervised semantic contrastive logits."
    )
    parser.add_argument(
        "--pseudo_unseen_weight",
        default=0.0,
        type=float,
        help="Weight of the train-only pseudo-Unseen episodic semantic loss; "
             "zero preserves the original objective."
    )
    parser.add_argument(
        "--pseudo_unseen_temperature",
        default=0.15,
        type=float,
        help="Temperature for pseudo-Unseen text-prototype classification."
    )
    parser.add_argument(
        "--pseudo_unseen_class_fraction",
        default=0.5,
        type=float,
        help="Fraction of classes in a batch sampled as pseudo-Unseen query "
             "classes."
    )
    parser.add_argument(
        "--pseudo_unseen_min_classes",
        default=2,
        type=int,
        help="Minimum number of query classes required for an episodic loss."
    )
    parser.add_argument(
        "--snn_temporal_consistency_weight",
        default=0.0,
        type=float,
        help="Weight of the train-only SNN temporal-view consistency loss; "
             "zero preserves the original objective."
    )
    parser.add_argument(
        "--snn_temporal_view_fraction",
        default=0.25,
        type=float,
        help="Fraction of each training batch supervised through the pure "
             "SNN temporal view."
    )
    parser.add_argument(
        "--temporal_quality_alignment_weight",
        default=0.0,
        type=float,
        help="Weight of the train-only semantic/SNN temporal agreement loss; "
             "zero preserves the original STFT objective."
    )
    parser.add_argument(
        "--cross_modal_contrastive_weight",
        default=0.0,
        type=float,
        help="Weight of the class-level audio-video consistency contrastive "
             "loss; zero preserves the original objective."
    )
    parser.add_argument(
        "--cross_modal_contrastive_temperature",
        default=0.1,
        type=float,
        help="Temperature for class-level audio-video contrastive logits."
    )
    parser.add_argument(
        "--avla_contrastive_only",
        action="store_true",
        help="Replace the MSTR triplet/projection/reconstruction objective with "
             "a supervised joint AV-to-text contrastive objective."
    )
    parser.add_argument(
        "--avla_temperature",
        default=0.1,
        type=float,
        help="Temperature for the standalone AV-language alignment objective."
    )
    parser.add_argument(
        "--global_prototype_contrastive_weight",
        default=0.0,
        type=float,
        help="Weight of the audio/video-to-final-task-prototype contrastive "
             "loss; zero preserves the original objective."
    )
    parser.add_argument(
        "--global_prototype_contrastive_temperature",
        default=0.1,
        type=float,
        help="Temperature for the final-task-prototype contrastive logits."
    )
    parser.add_argument(
        "--semantic_hard_negative_weight",
        default=0.0,
        type=float,
        help="Weight of the semantic nearest-different-class ranking loss; "
             "zero preserves the original objective."
    )
    parser.add_argument(
        "--semantic_hard_negative_margin",
        default=0.1,
        type=float,
        help="Cosine margin against each batch item's semantic hard negative."
    )
    parser.add_argument(
        "--semantic_batch_hard_weight",
        default=0.0,
        type=float,
        help="Weight of the online semantic-neighbour batch-hard ranking loss; "
             "zero preserves the original objective."
    )
    parser.add_argument(
        "--semantic_batch_hard_margin",
        default=0.1,
        type=float,
        help="Cosine margin against the current batch's hardest semantic peer."
    )
    parser.add_argument(
        "--semantic_batch_hard_neighbors",
        default=5,
        type=int,
        help="Number of raw-word-vector semantic neighbours eligible as hard negatives."
    )
    parser.add_argument(
        "--semantic_neighbor_rank_weight",
        default=0.0,
        type=float,
        help="Weight of the train-only class-level semantic neighbourhood "
             "ranking loss; zero preserves the original objective."
    )
    parser.add_argument(
        "--semantic_neighbor_rank_margin",
        default=0.05,
        type=float,
        help="Cosine margin for semantic neighbour ordering."
    )
    parser.add_argument(
        "--semantic_neighbor_rank_neighbors",
        default=5,
        type=int,
        help="Maximum number of near and far semantic classes per anchor."
    )
    parser.add_argument(
        "--semantic_mixup_weight",
        default=0.0,
        type=float,
        help="Weight of the Seen-class semantic manifold mixup loss; zero "
             "preserves the original objective."
    )
    parser.add_argument(
        "--semantic_mixup_alpha",
        default=1.0,
        type=float,
        help="Beta distribution concentration for semantic manifold mixup."
    )
    parser.add_argument(
        "--feature_mixup_weight",
        default=0.0,
        type=float,
        help="Weight of raw audio/video/text manifold mixup through the full "
             "SNN-STFT encoder; zero preserves the original objective."
    )
    parser.add_argument(
        "--feature_mixup_alpha",
        default=0.2,
        type=float,
        help="Beta distribution concentration for raw feature manifold mixup."
    )
    parser.add_argument(
        "--feature_debias_weight",
        default=0.0,
        type=float,
        help="Weight of the feature-level visual semantic/residual debiasing "
             "loss; zero preserves the original STFT path."
    )
    parser.add_argument(
        "--feature_debias_temperature",
        default=0.1,
        type=float,
        help="Temperature of the residual-to-text adversarial probe."
    )
    parser.add_argument(
        "--text_projection_norm",
        choices=["batchnorm", "layernorm"],
        default="batchnorm",
        help="Normalization used only inside W_proj. LayerNorm avoids text "
             "BatchNorm running-statistics drift across zero-shot class splits."
    )
    parser.add_argument(
        "--cross_modal_residual",
        action="store_true",
        help="Enable sample-adaptive cross-modal complementary residual "
             "gating after STFT Tucker fusion."
    )
    parser.add_argument(
        "--cross_modal_residual_scale",
        default=0.2,
        type=float,
        help="Scale of the gated cross-modal residual contribution."
    )
    parser.add_argument(
        "--ceo_optimize_text",
        action="store_true",
        help="Optimize the full class-name dictionary with frozen CEO ranking "
             "and separation before MSTR training."
    )
    parser.add_argument("--ceo_alpha", default=0.5, type=float)
    parser.add_argument("--ceo_margin", default=1.0, type=float)
    parser.add_argument("--ceo_steps", default=500, type=int)
    parser.add_argument("--ceo_lr", default=0.05, type=float)
    parser.add_argument("--ceo_triplets", default=8192, type=int)
    parser.add_argument(
        "--vector_trl_rank",
        default=64,
        type=int,
        help="Rank of a vector-specialized TRL."
    )
    parser.add_argument(
        "--stft_vector_trl",
        action="store_true",
        help="Replace only STFT's singleton-mode spatial TRL with its stable "
             "two-factor vector equivalent."
    )
    parser.add_argument(
        "--stft_spatial_reliability_gate",
        action="store_true",
        help="Bound the VectorTRL spatial-fusion contribution with a shared "
             "sample-level semantic agreement gate."
    )
    parser.add_argument(
        "--trl_gate_scale",
        default=0.25,
        type=float,
        help="Maximum absolute contribution of each residual TRL branch."
    )
    parser.add_argument(
        "--backbone_lr_scale",
        default=1.0,
        type=float,
        help="Learning-rate multiplier for the pretrained MSTR backbone."
    )
    parser.add_argument(
        "--init_checkpoint",
        type=pathlib.Path,
        help="Optional MSTR checkpoint used to initialize matching parameters."
    )
    parser.add_argument(
        "--stage_b_teacher_checkpoint",
        type=pathlib.Path,
        help="Frozen Stage A MSTR checkpoint used only for masked Stage B "
             "seen-class distillation."
    )
    parser.add_argument(
        "--stage_b_seen_distill_weight",
        default=0.0,
        type=float,
        help="Weight of the masked Stage B audio/video teacher consistency "
             "loss; zero disables it."
    )
    parser.add_argument(
        "--stage_b_new_class_fraction",
        default=None,
        type=float,
        help="Optional fraction of each Stage B batch allocated to classes "
             "newly introduced after Stage A; unset preserves class-uniform sampling."
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=pathlib.Path,
        help="Resume a complete training state saved after an epoch."
    )

    parser.add_argument(
        "--new_model_attention_both_heads",
        help="Flag to set if attention should provide output from both branches",
        action="store_true"

    )

    parser.add_argument(
        "--depth_transformer",
        help="Flag to se the number of layers of the transformer",
        default=1,
        type=int
    )

    parser.add_argument(
        "--exp_name",
        help="Flag to set the name of the experiment",
        default="",
        type=str
    )
    parser.add_argument(
        "--ale",
        help="Flag to set the ale",
        action="store_true"
    )
    parser.add_argument(
        "--devise",
        help="Flag to set the devise model",
        action="store_true"
    )
    parser.add_argument(
        "--sje",
        help="Flag to set the sje model",
        action="store_true"
    )

    parser.add_argument(
        "--apn",
        help="Flag to set the apn model",
        action="store_true"
    )

    parser.add_argument(
        "--first_additional_triplet",
        help="flag to set the first pair of additional triplets",
        default=1,
        type=int
    )

    parser.add_argument(
        "--second_additional_triplet",
        help="flag to set the second pair of additional triplets",
        default=1,
        type=int

    )

    parser.add_argument(
        "--third_additional_triplet",
        help="flag to set the third pair of additional triplets",
        default=1,
        type=int
    )
    parser.add_argument(
        "--additional_dropout",
        help="flag to set the additional dropouts",
        default=0,
        type=float

    )
    args = parser.parse_args(*args, **kwargs)
    # --AVCA and --MSTR are aliases: either flag selects the MSTR model.
    # This keeps the old run_scripts/avca.sh working while the README uses --MSTR.
    args.AVCA = bool(args.AVCA) or bool(args.MSTR)
    args.MSTR = bool(args.MSTR) or bool(args.AVCA)
    if args.stage_b_seen_distill_weight < 0.0:
        parser.error("--stage_b_seen_distill_weight must be non-negative")
    if args.vector_trl_rank <= 0:
        parser.error("--vector_trl_rank must be positive")
    if args.stft_spatial_reliability_gate and not args.stft_vector_trl:
        parser.error("--stft_spatial_reliability_gate requires --stft_vector_trl")
    if args.snn_activity_floor_weight < 0.0:
        parser.error("--snn_activity_floor_weight must be non-negative")
    if not 0.0 < args.snn_min_spike_rate < 1.0:
        parser.error("--snn_min_spike_rate must be strictly between 0 and 1")
    if args.snn_membrane_readout_scale < 0.0:
        parser.error("--snn_membrane_readout_scale must be non-negative")
    if args.semantic_mixup_weight < 0.0:
        parser.error("--semantic_mixup_weight must be non-negative")
    if args.semantic_mixup_alpha <= 0.0:
        parser.error("--semantic_mixup_alpha must be positive")
    if args.global_prototype_contrastive_weight < 0.0:
        parser.error("--global_prototype_contrastive_weight must be non-negative")
    if args.global_prototype_contrastive_temperature <= 0.0:
        parser.error("--global_prototype_contrastive_temperature must be positive")
    if args.avla_temperature <= 0.0:
        parser.error("--avla_temperature must be positive")
    if args.cross_modal_contrastive_weight < 0.0:
        parser.error("--cross_modal_contrastive_weight must be non-negative")
    if args.cross_modal_contrastive_temperature <= 0.0:
        parser.error("--cross_modal_contrastive_temperature must be positive")
    if args.semantic_batch_hard_weight < 0.0:
        parser.error("--semantic_batch_hard_weight must be non-negative")
    if args.semantic_batch_hard_margin < 0.0:
        parser.error("--semantic_batch_hard_margin must be non-negative")
    if args.semantic_batch_hard_neighbors <= 0:
        parser.error("--semantic_batch_hard_neighbors must be positive")
    if args.semantic_neighbor_rank_weight < 0.0:
        parser.error("--semantic_neighbor_rank_weight must be non-negative")
    if args.semantic_neighbor_rank_margin < 0.0:
        parser.error("--semantic_neighbor_rank_margin must be non-negative")
    if args.semantic_neighbor_rank_neighbors <= 0:
        parser.error("--semantic_neighbor_rank_neighbors must be positive")
    if args.pseudo_unseen_weight < 0.0:
        parser.error("--pseudo_unseen_weight must be non-negative")
    if args.pseudo_unseen_temperature <= 0.0:
        parser.error("--pseudo_unseen_temperature must be positive")
    if not 0.0 < args.pseudo_unseen_class_fraction < 1.0:
        parser.error("--pseudo_unseen_class_fraction must be in (0, 1)")
    if args.pseudo_unseen_min_classes < 2:
        parser.error("--pseudo_unseen_min_classes must be at least 2")
    if args.snn_temporal_consistency_weight < 0.0:
        parser.error("--snn_temporal_consistency_weight must be non-negative")
    if not 0.0 < args.snn_temporal_view_fraction <= 1.0:
        parser.error("--snn_temporal_view_fraction must be in (0, 1]")
    if args.temporal_quality_alignment_weight < 0.0:
        parser.error("--temporal_quality_alignment_weight must be non-negative")
    if args.feature_mixup_weight < 0.0:
        parser.error("--feature_mixup_weight must be non-negative")
    if args.feature_mixup_alpha <= 0.0:
        parser.error("--feature_mixup_alpha must be positive")
    if (args.stage_b_new_class_fraction is not None and
            not 0.0 < args.stage_b_new_class_fraction < 1.0):
        parser.error("--stage_b_new_class_fraction must be strictly between 0 and 1")
    return args


def args_eval():
    parser = argparse.ArgumentParser(description="Explainable Audio Visual Low Shot Learning [Evaluation]")
    parser.add_argument(
        "--load_path_stage_A",
        help="Path to experiment log folder of stage A",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument(
        "--root_dir",
        help="Path to dataset directory. Expected subfolder structure: '{root_dir}/features/{feature_extraction_method}/{audio,video,text}'",
        type=pathlib.Path
    )

    parser.add_argument(
        "--load_path_stage_B",
        help="Path to experiment log folder of stage B",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument(
        "--stage_b_selection",
        help="Select the Stage B checkpoint independently or match Stage A's best epoch.",
        choices=["best", "matched"],
        default="matched"
    )

    """
    parser.add_argument(
        "--weights_path",
        help="Path to trained model weights. If not stated, random weights will be used!",
        type=pathlib.Path
    )

    parser.add_argument(
        "--weights_path_stage_A",
        help="Path to trained model weights from stage A. If not stated, random weights will be used!",
        type=pathlib.Path
    )

    parser.add_argument(
        "--weights_path_stage_B",
        help="Path to trained model weights from stage B. If not stated, random weights will be used!",
        type=pathlib.Path
    )
    """
    parser.add_argument(
        "--eval_name",
        help="Evaluation name to be displayed in the final output string",
        type=str,
    )

    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to use",
        choices=["AudioSetZSL", "VGGSound", "UCF", "ActivityNet"],
        default="AudioSetZSL",
        type=str
    )

    parser.add_argument(
        "--bs",
        help="Batch size",
        default=256,
        type=int
    )
    parser.add_argument(
        "--num_workers",
        help="Number of dataloader workers",
        default=8,
        type=int
    )
    parser.add_argument(
        "--pin_memory",
        help="Flag for pin_memory in dataloader",
        default=True,
        type=bool
    )
    parser.add_argument(
        "--drop_last",
        help="Drop last batch in dataloader",
        default=True,
        type=bool
    )
    parser.add_argument(
        "--device",
        help="Device to run on.",
        choices=["cuda", "cpu"],
        default="cuda"
    )

    parser.add_argument(
        "--baseline",
        help="Flag for setting baseline",
        action="store_true"

    )

    parser.add_argument(
        "--audio_baseline",
        help="Flag to use the audio baseline",
        action="store_true"
    )
    parser.add_argument(
        "--video_baseline",
        help="Flag to use the video baseline",
        action="store_true"

    )

    parser.add_argument(
        "--concatenated_baseline",
        help="Flag to use the concatenated baseline",
        action="store_true"

    )

    parser.add_argument(
        "--new_model",
        help="Flag to use the new model",
        action="store_true"
    )

    parser.add_argument(
        "--new_model_early_fusion",
        help="Flag to use the early fusion new model",
        action="store_true"
    )

    parser.add_argument(
        "--new_model_middle_fusion",
        help="Flag to set the middle fusion new model",
        action="store_true"
    )

    parser.add_argument(
        "--MSTR",
        help="Flag to set the attention to the new model",
        action="store_true"

    )

    parser.add_argument(
        "--AVCA",
        help="Alias of --MSTR (kept for backward compatibility with the AVCA training scripts).",
        action="store_true"
    )

    parser.add_argument(
        "--snn_T",
        help="Number of time steps for the spiking neural network branches in MSTR.",
        default=10,
        type=int
    )

    parser.add_argument(
        "--trl_rank",
        help="Rank R of the Tucker regression core used by the MSTR spatial branches.",
        default=400,
        type=int
    )

    # --- STFT upgrade hyperparameters (mirror args_main; defaults match the
    # STFT paper so eval works without extra flags). ---
    parser.add_argument(
        "--snn_tau",
        help="Membrane time constant tau of the STFT LIF neurons (Eq. 5).",
        default=2.0,
        type=float
    )
    parser.add_argument("--snn_membrane_readout_scale", default=0.0, type=float)
    parser.add_argument("--legacy_batch_dth", action="store_true")
    parser.add_argument(
        "--lkc_n_slots",
        help="Number of learnable latent knowledge slots in the LKC (Eq. 1).",
        default=4,
        type=int
    )
    parser.add_argument(
        "--lkc_n_heads",
        help="Number of attention heads in the Latent Knowledge Combiner.",
        default=8,
        type=int
    )
    parser.add_argument(
        "--lkc_residual_scale",
        help="Scale of the normalized residual LKC contribution used at eval.",
        default=0.2,
        type=float
    )
    parser.add_argument(
        "--tucker_rank",
        help="Rank R of the Temporal-Semantic Tucker Fusion core (Eq. 12-13).",
        default=60,
        type=int
    )
    parser.add_argument(
        "--stft_dim",
        help="Internal semantic/temporal feature dimension (paper default: 512).",
        default=512,
        type=int
    )
    parser.add_argument(
        "--fusion_mode",
        choices=["stft", "residual", "mstr_released", "mstr_paper",
                 "languagebind_anchor_residual"],
        default="stft"
    )
    parser.add_argument("--disable_glp", action="store_true")
    parser.add_argument("--disable_lkc", action="store_true")
    parser.add_argument("--ahse_standardize", action="store_true")
    parser.add_argument("--semantic_geometry_weight", default=0.0, type=float)
    parser.add_argument("--semantic_contrastive_weight", default=0.0, type=float)
    parser.add_argument("--semantic_contrastive_temperature", default=0.1, type=float)
    parser.add_argument("--cross_modal_contrastive_weight", default=0.0, type=float)
    parser.add_argument("--cross_modal_contrastive_temperature", default=0.1, type=float)
    parser.add_argument("--avla_contrastive_only", action="store_true")
    parser.add_argument("--avla_temperature", default=0.1, type=float)
    parser.add_argument("--global_prototype_contrastive_weight", default=0.0, type=float)
    parser.add_argument("--global_prototype_contrastive_temperature", default=0.1, type=float)
    parser.add_argument("--semantic_hard_negative_weight", default=0.0, type=float)
    parser.add_argument("--semantic_hard_negative_margin", default=0.1, type=float)
    parser.add_argument("--semantic_batch_hard_weight", default=0.0, type=float)
    parser.add_argument("--semantic_batch_hard_margin", default=0.1, type=float)
    parser.add_argument("--semantic_batch_hard_neighbors", default=5, type=int)
    parser.add_argument("--semantic_neighbor_rank_weight", default=0.0, type=float)
    parser.add_argument("--semantic_neighbor_rank_margin", default=0.05, type=float)
    parser.add_argument("--semantic_neighbor_rank_neighbors", default=5, type=int)
    parser.add_argument("--feature_debias_weight", default=0.0, type=float)
    parser.add_argument("--feature_debias_temperature", default=0.1, type=float)
    parser.add_argument(
        "--text_projection_norm",
        choices=["batchnorm", "layernorm"],
        default="batchnorm",
        help="Normalization used only inside W_proj. LayerNorm avoids text "
             "BatchNorm running-statistics drift across zero-shot class splits.")
    parser.add_argument("--text_bn_semantic_recalibration", action="store_true")
    parser.add_argument("--text_bn_semantic_mix", default=1.0, type=float)
    parser.add_argument("--cross_modal_residual", action="store_true")
    parser.add_argument("--cross_modal_residual_scale", default=0.2, type=float)
    parser.add_argument("--vector_trl_rank", default=64, type=int)
    parser.add_argument("--stft_vector_trl", action="store_true")
    parser.add_argument("--stft_spatial_reliability_gate", action="store_true")
    parser.add_argument("--trl_gate_scale", default=0.25, type=float)
    parser.add_argument("--backbone_lr_scale", default=1.0, type=float)
    parser.add_argument(
        "--cjme",
        help="Flag to use the CJME baseline",
        action="store_true"
    )

    parser.add_argument(
        "--ale",
        help="Flag to set the ale",
        action="store_true"
    )
    parser.add_argument(
        "--devise",
        help="Flag to set the devise model",
        action="store_true"
    )
    parser.add_argument(
        "--sje",
        help="Flag to se the sje model",
        action="store_true"
    )
    parser.add_argument(
        "--apn",
        help="flag to set apn model",
        action="store_true"
    )

    parser.add_argument(
        "--save_performances",
        help="Save class performances to disk",
        action="store_true"
    )
    parser.add_argument(
        "--adaptive_modality_fusion",
        help="Select a global audio/video distance weight on validation data. "
             "Diagnostic only; fixed distance summation remains the default.",
        action="store_true"
    )
    parser.add_argument(
        "--evaluation_beta_max",
        default=3.0,
        type=float,
        help="Maximum validation beta used for GZSL calibration. The "
             "historical default is 3.0."
    )
    parser.add_argument(
        "--evaluation_beta_step",
        default=None,
        type=float,
        help="Optional beta search step. Defaults to 0.2, or 0.1 for the "
             "adaptive fusion diagnostic."
    )
    parser.add_argument(
        "--normalize_shared_embeddings",
        action="store_true",
        help="L2-normalize projected audio, video, and text embeddings before "
             "distance-based evaluation. Diagnostic only."
    )
    parser.add_argument(
        "--semantic_aware_calibration",
        action="store_true",
        help="Allocate the validation-selected Seen calibration beta by each "
             "Seen prototype's learned semantic proximity to Unseen prototypes."
    )
    parser.add_argument(
        "--energy_ood_routing",
        help="Select a Seen-energy routing threshold on validation data and "
             "reuse it on test data. Diagnostic only.",
        action="store_true"
    )
    parser.add_argument(
        "--energy_ood_score",
        choices=["raw", "zscore"],
        default="raw",
        help="Energy score computed from raw distances or per-sample "
             "Z-standardized class distances."
    )

    args = parser.parse_args()
    if args.vector_trl_rank <= 0:
        parser.error("--vector_trl_rank must be positive")
    if args.stft_spatial_reliability_gate and not args.stft_vector_trl:
        parser.error("--stft_spatial_reliability_gate requires --stft_vector_trl")
    if not 0.0 <= args.text_bn_semantic_mix <= 1.0:
        parser.error("--text_bn_semantic_mix must be in [0, 1]")
    if args.avla_temperature <= 0.0:
        parser.error("--avla_temperature must be positive")
    if args.cross_modal_contrastive_weight < 0.0:
        parser.error("--cross_modal_contrastive_weight must be non-negative")
    if args.cross_modal_contrastive_temperature <= 0.0:
        parser.error("--cross_modal_contrastive_temperature must be positive")
    if args.semantic_batch_hard_weight < 0.0:
        parser.error("--semantic_batch_hard_weight must be non-negative")
    if args.semantic_batch_hard_margin < 0.0:
        parser.error("--semantic_batch_hard_margin must be non-negative")
    if args.semantic_batch_hard_neighbors <= 0:
        parser.error("--semantic_batch_hard_neighbors must be positive")
    if args.semantic_neighbor_rank_weight < 0.0:
        parser.error("--semantic_neighbor_rank_weight must be non-negative")
    if args.semantic_neighbor_rank_margin < 0.0:
        parser.error("--semantic_neighbor_rank_margin must be non-negative")
    if args.semantic_neighbor_rank_neighbors <= 0:
        parser.error("--semantic_neighbor_rank_neighbors must be positive")
    # --AVCA and --MSTR are aliases: either flag selects the MSTR model.
    args.AVCA = bool(args.AVCA) or bool(args.MSTR)
    args.MSTR = bool(args.MSTR) or bool(args.AVCA)
    return args
