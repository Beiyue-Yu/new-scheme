#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

: "${DATASET:?DATASET is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${RUN_PREFIX:?RUN_PREFIX is required}"
: "${ENC_DROPOUT:?ENC_DROPOUT is required}"
: "${DEC_DROPOUT:?DEC_DROPOUT is required}"
: "${ADD_DROPOUT:?ADD_DROPOUT is required}"

PYTHON=${PYTHON:-/home/wwj/anaconda3/envs/MSTR-torch24/bin/python}
MODEL_MODE=${MODEL_MODE:-mstr_paper}
ABLATION=${ABLATION:-full}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
SEED=${SEED:-42}
SKIP_STAGE_A=${SKIP_STAGE_A:-0}
STOP_AFTER_STAGE_A=${STOP_AFTER_STAGE_A:-0}
STAGE_A_RESUME=${STAGE_A_RESUME:-}
STAGE_B_RESUME=${STAGE_B_RESUME:-}
EPOCH_CHUNK=${EPOCH_CHUNK:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}
SNN_T=${SNN_T:-10}
SNN_TAU=${SNN_TAU:-2.0}
TEXT_EMBEDDING_SIZE=${TEXT_EMBEDDING_SIZE:-}
FEATURE_EXTRACTION_METHOD=${FEATURE_EXTRACTION_METHOD:-main_features}
# LanguageBind stores native 768-D video/text embeddings.  Keep the legacy
# 512-D/300-D defaults for the repository's original feature routes, but infer
# the correct dimensions when the LanguageBind cache is selected.  Explicit
# INPUT_SIZE_* or TEXT_EMBEDDING_SIZE values still take precedence.
case "$FEATURE_EXTRACTION_METHOD" in
  languagebind_mstr|languagebind*)
    INPUT_SIZE_VIDEO=${INPUT_SIZE_VIDEO:-768}
    TEXT_EMBEDDING_SIZE=${TEXT_EMBEDDING_SIZE:-768}
    ;;
  *)
    INPUT_SIZE_VIDEO=${INPUT_SIZE_VIDEO:-512}
    TEXT_EMBEDDING_SIZE=${TEXT_EMBEDDING_SIZE:-300}
    ;;
esac
INPUT_SIZE_AUDIO=${INPUT_SIZE_AUDIO:-512}
STAGE_B_LR_SCALE=${STAGE_B_LR_SCALE:-1.0}
STAGE_A_DIR=${STAGE_A_DIR:-}
STAGE_A_CHECKPOINT=${STAGE_A_CHECKPOINT:-}
MAX_STAGE_RETRIES=${MAX_STAGE_RETRIES:-5}
RETRY_DELAY=${RETRY_DELAY:-2}
DISABLE_TENSORBOARD=${DISABLE_TENSORBOARD:-1}
SKIP_EVALUATION=${SKIP_EVALUATION:-0}
SAVE_STAGE_A_CHECKPOINTS=${SAVE_STAGE_A_CHECKPOINTS:-0}
ALLOW_UNSAFE_NVIDIA=${ALLOW_UNSAFE_NVIDIA:-0}
if (( EPOCH_CHUNK <= 0 || TOTAL_EPOCHS <= 0 || MAX_STAGE_RETRIES < 0 || RETRY_DELAY < 0 )); then
  echo "EPOCH_CHUNK and TOTAL_EPOCHS must be positive; retry values must be non-negative" >&2
  exit 2
fi
if [[ "$DISABLE_TENSORBOARD" != "0" && "$DISABLE_TENSORBOARD" != "1" ]]; then
  echo "DISABLE_TENSORBOARD must be 0 or 1" >&2
  exit 2
fi
if [[ "$STOP_AFTER_STAGE_A" != "0" && "$STOP_AFTER_STAGE_A" != "1" ]]; then
  echo "STOP_AFTER_STAGE_A must be 0 or 1" >&2
  exit 2
fi
if [[ "$SKIP_EVALUATION" != "0" && "$SKIP_EVALUATION" != "1" ]]; then
  echo "SKIP_EVALUATION must be 0 or 1" >&2
  exit 2
fi
if [[ "$SAVE_STAGE_A_CHECKPOINTS" != "0" && "$SAVE_STAGE_A_CHECKPOINTS" != "1" ]]; then
  echo "SAVE_STAGE_A_CHECKPOINTS must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$SNN_T" =~ ^[1-9][0-9]*$ ]]; then
  echo "SNN_T must be a positive integer" >&2
  exit 2
fi
if ! awk -v value="$SNN_TAU" 'BEGIN { exit !(value + 0 > 0) }'; then
  echo "SNN_TAU must be positive" >&2
  exit 2
fi
if ! awk -v value="$STAGE_B_LR_SCALE" 'BEGIN { exit !(value + 0 > 0) }'; then
  echo "STAGE_B_LR_SCALE must be positive" >&2
  exit 2
fi
for dimension_name in INPUT_SIZE_AUDIO INPUT_SIZE_VIDEO TEXT_EMBEDDING_SIZE; do
  dimension_value=${!dimension_name}
  if ! [[ "$dimension_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$dimension_name must be a positive integer (got $dimension_value)" >&2
    exit 2
  fi
done
if [[ "$ALLOW_UNSAFE_NVIDIA" != "0" && "$ALLOW_UNSAFE_NVIDIA" != "1" ]]; then
  echo "ALLOW_UNSAFE_NVIDIA must be 0 or 1" >&2
  exit 2
fi

check_runtime_safety() {
  local kernel_release nvidia_version nvidia_license
  kernel_release=$(uname -r)
  nvidia_version=$(modinfo -F version nvidia 2>/dev/null || true)
  nvidia_license=$(modinfo -F license nvidia 2>/dev/null || true)

  if [[ "$ALLOW_UNSAFE_NVIDIA" != "1" \
        && "$kernel_release" == 6.17.* \
        && "$nvidia_version" == "580.126.09" \
        && "$nvidia_license" == "NVIDIA" ]]; then
    cat >&2 <<EOF
Refusing to start GPU training on a known-unsafe local runtime:
  kernel=$kernel_release
  NVIDIA=$nvidia_version ($nvidia_license kernel module)

This machine has already recorded multi-CPU hard lockups when PyTorch CUDA
processes exit on this runtime. Install the NVIDIA open kernel module or boot a
stable alternate kernel, reboot, and verify the module before training again.
EOF
    return 1
  fi
}

check_runtime_safety

mkdir -p runs
exec 9>runs/.mstr_gpu.lock
if ! flock -n 9; then
  echo "Another two-stage GPU experiment is already running in this repository." >&2
  exit 2
fi

case "$MODEL_MODE" in
  languagebind_anchor_residual)
    # Keep LanguageBind video/text in their native shared space.  This route
    # does not use the STFT MSTR ablation switches below.
    RUN_KIND="languagebind_anchor_residual"
    EXTRA_ARGS=()
    ;;
  mstr_paper|mstr_released)
    if [[ "$ABLATION" != "full" ]]; then
      echo "ABLATION is only valid with MODEL_MODE=stft" >&2
      exit 2
    fi
    RUN_KIND=$MODEL_MODE
    EXTRA_ARGS=()
    ;;
  stft)
    RUN_KIND="stft_${ABLATION}"
    case "$ABLATION" in
      full) EXTRA_ARGS=() ;;
      adaptive_only) EXTRA_ARGS=(--disable_glp --disable_lkc) ;;
      adaptive_glp) EXTRA_ARGS=(--disable_lkc) ;;
      adaptive_lkc) EXTRA_ARGS=(--disable_glp) ;;
      adaptive_lkc_residual) EXTRA_ARGS=(--disable_glp) ;;
      adaptive_lkc_residual_vector_trl)
        EXTRA_ARGS=(--disable_glp --stft_vector_trl --vector_trl_rank 64)
        ;;
      adaptive_lkc_residual_vector_trl_reliability_gate)
        EXTRA_ARGS=(--disable_glp --stft_vector_trl --vector_trl_rank 64 \
                    --stft_spatial_reliability_gate)
        ;;
      adaptive_lkc_residual_snn_activity_floor)
        EXTRA_ARGS=(--disable_glp --snn_activity_floor_weight 0.02 \
                    --snn_min_spike_rate 0.05)
        ;;
      adaptive_lkc_residual_membrane_readout)
        EXTRA_ARGS=(--disable_glp --snn_membrane_readout_scale 0.2)
        ;;
      adaptive_lkc_residual_text_layernorm)
        # Text BatchNorm sees only training-class word vectors, while the
        # validation and test zero-shot class dictionaries are disjoint.
        EXTRA_ARGS=(--disable_glp --text_projection_norm layernorm)
        ;;
      adaptive_lkc_residual_stageb_init)
        # Stage B starts from the Stage A selected weights, but receives a
        # fresh optimizer.  This is assigned after MSTR_score.pt is known.
        EXTRA_ARGS=(--disable_glp)
        ;;
      adaptive_lkc_residual_stageb_seen_distill)
        EXTRA_ARGS=(--disable_glp)
        ;;
      adaptive_lkc_residual_stageb_init_seen_distill)
        EXTRA_ARGS=(--disable_glp)
        ;;
      adaptive_lkc_residual_stageb_group_balanced)
        EXTRA_ARGS=(--disable_glp)
        ;;
      adaptive_lkc_residual_legacy_batch_dth)
        EXTRA_ARGS=(--disable_glp --legacy_batch_dth)
        ;;
      adaptive_lkc_residual_ahse_es)
        EXTRA_ARGS=(--disable_glp --ahse_standardize)
        ;;
      adaptive_lkc_residual_semantic_geometry)
        EXTRA_ARGS=(--disable_glp --semantic_geometry_weight 0.1)
        ;;
      adaptive_lkc_residual_ceo_text)
        EXTRA_ARGS=(--disable_glp --ceo_optimize_text)
        ;;
      adaptive_lkc_residual_cross_residual)
        EXTRA_ARGS=(--disable_glp --cross_modal_residual)
        ;;
      adaptive_lkc_residual_semantic_contrastive)
        EXTRA_ARGS=(--disable_glp --semantic_contrastive_weight 0.02)
        ;;
      adaptive_lkc_residual_ceo_semantic_contrastive)
        # Frozen class-name optimization plus the complete AV/text objective.
        # Keep MSTR's reconstruction and triplet losses: replacing them caused
        # the earlier standalone AV-language control to collapse on UCF.
        EXTRA_ARGS=(--disable_glp --ceo_optimize_text \
                    --semantic_contrastive_weight 0.02 \
                    --semantic_contrastive_temperature 0.1)
        ;;
      adaptive_lkc_residual_pseudo_unseen)
        EXTRA_ARGS=(--disable_glp --pseudo_unseen_weight 0.05 \
                    --pseudo_unseen_temperature 0.15 \
                    --pseudo_unseen_class_fraction 0.5 \
                    --pseudo_unseen_min_classes 2)
        ;;
      adaptive_lkc_residual_snn_temporal_consistency)
        EXTRA_ARGS=(--disable_glp --snn_temporal_consistency_weight 0.01 \
                    --snn_temporal_view_fraction 0.25)
        ;;
      adaptive_lkc_residual_temporal_quality_alignment)
        EXTRA_ARGS=(--disable_glp --temporal_quality_alignment_weight 0.02)
        ;;
      adaptive_lkc_residual_cross_modal_contrastive)
        EXTRA_ARGS=(--disable_glp --cross_modal_contrastive_weight 0.005 \
                    --cross_modal_contrastive_temperature 0.1)
        ;;
      adaptive_lkc_residual_avla_contrastive)
        EXTRA_ARGS=(--disable_glp --ceo_optimize_text --avla_contrastive_only \
                    --avla_temperature 0.1)
        ;;
      adaptive_lkc_residual_global_prototype_contrastive)
        EXTRA_ARGS=(--disable_glp --global_prototype_contrastive_weight 0.01 \
                    --global_prototype_contrastive_temperature 0.1)
        ;;
      adaptive_lkc_residual_semantic_hard_negative)
        EXTRA_ARGS=(--disable_glp --semantic_hard_negative_weight 0.05)
        ;;
      adaptive_lkc_residual_semantic_batch_hard)
        EXTRA_ARGS=(--disable_glp --semantic_batch_hard_weight 0.02 \
                    --semantic_batch_hard_margin 0.1 \
                    --semantic_batch_hard_neighbors 5)
        ;;
      adaptive_lkc_residual_semantic_neighbor_rank)
        EXTRA_ARGS=(--disable_glp --semantic_neighbor_rank_weight 0.02 \
                    --semantic_neighbor_rank_margin 0.05 \
                    --semantic_neighbor_rank_neighbors 5)
        ;;
      adaptive_lkc_residual_semantic_mixup)
        EXTRA_ARGS=(--disable_glp --semantic_mixup_weight 0.02 \
                    --semantic_mixup_alpha 1.0)
        ;;
      adaptive_lkc_residual_feature_mixup)
        EXTRA_ARGS=(--disable_glp --feature_mixup_weight 0.01 \
                    --feature_mixup_alpha 0.2)
        ;;
      adaptive_lkc_residual_feature_debias)
        EXTRA_ARGS=(--disable_glp --feature_debias_weight 0.05)
        ;;
      *)
        echo "Unknown ABLATION=$ABLATION" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unknown MODEL_MODE=$MODEL_MODE" >&2
    exit 2
    ;;
esac

RUNTIME_ARGS=()
if [[ "$DISABLE_TENSORBOARD" == "1" ]]; then
  RUNTIME_ARGS+=(--disable_tensorboard)
fi
RUNTIME_ARGS+=(--allow_existing_run)

STAGE_A="${RUN_PREFIX}_${RUN_KIND}_val_${RUN_TAG}"
STAGE_B="${RUN_PREFIX}_${RUN_KIND}_all_${RUN_TAG}"
if [[ -z "$STAGE_A_DIR" ]]; then
  STAGE_A_DIR="runs/$STAGE_A"
fi
COMMON=(
  --root_dir "$DATA_ROOT"
  --feature_extraction_method "$FEATURE_EXTRACTION_METHOD"
  --input_size_audio "$INPUT_SIZE_AUDIO" \
  --input_size_video "$INPUT_SIZE_VIDEO" \
  --text_embedding_size "$TEXT_EMBEDDING_SIZE"
  --dataset_name "$DATASET" --zero_shot_split main_split --MSTR
  --seed "$SEED"
  --lr 1e-3 --n_batches 500 --bs 256
  --embeddings_hidden_size 512 --decoder_hidden_size 512
  --embedding_dropout "$ENC_DROPOUT"
  --decoder_dropout "$DEC_DROPOUT"
  --additional_dropout "$ADD_DROPOUT"
  --depth_transformer 1 --additional_triplets_loss
  --first_additional_triplet 1 --second_additional_triplet 1
  --momentum 0.1 --reg_loss --lr_scheduler
  --snn_T "$SNN_T" --trl_rank 400 --snn_tau "$SNN_TAU"
  --lkc_n_slots 4 --lkc_n_heads 8 --tucker_rank 60 --stft_dim 512
  --fusion_mode "$MODEL_MODE"
  "${EXTRA_ARGS[@]}"
  "${RUNTIME_ARGS[@]}"
)

run_python() {
  env -u PYTHONPATH PYTHONNOUSERSITE=1 \
    CUDA_MODULE_LOADING=LAZY MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 "$PYTHON" "$@"
}

validate_processed_dimensions() {
  local processed_file="$DATA_ROOT/_features_processed/$FEATURE_EXTRACTION_METHOD/trainingmain_split.pkl"
  if [[ ! -f "$processed_file" ]]; then
    echo "preflight=skipped processed_file_missing=$processed_file" >&2
    return 0
  fi
  run_python - "$processed_file" "$INPUT_SIZE_AUDIO" "$INPUT_SIZE_VIDEO" \
    "$TEXT_EMBEDDING_SIZE" <<'PY'
import pickle
import sys

path, expected_audio, expected_video, expected_text = sys.argv[1:]
expected = {
    "audio": int(expected_audio),
    "video": int(expected_video),
    "text": int(expected_text),
}
with open(path, "rb") as handle:
    data = pickle.load(handle)
actual = {
    name: int(data[name]["data"].shape[-1])
    for name in expected
}
print("preflight=feature_dimensions", " ".join(
    f"{name}={actual[name]} expected={expected[name]}"
    for name in ("audio", "video", "text")), flush=True)
errors = [
    f"{name}: expected {expected[name]}, found {actual[name]}"
    for name in expected if actual[name] != expected[name]
]
if errors:
    raise SystemExit("Feature dimension mismatch in " + path + ": " + "; ".join(errors))
PY
}

checkpoint_epoch() {
  local checkpoint=$1
  local epoch_file
  epoch_file="$(dirname "$checkpoint")/last_epoch.txt"
  # `last_epoch.txt` describes the latest resumable state, not necessarily a
  # named best checkpoint such as MSTR_score.pt.  It is only valid for last.pt.
  if [[ "$(basename "$checkpoint")" == "last.pt" && -f "$epoch_file" ]]; then
    local epoch
    epoch=$(<"$epoch_file")
    if [[ "$epoch" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$epoch"
      return 0
    fi
    echo "Invalid checkpoint epoch file: $epoch_file" >&2
    return 1
  fi
  run_python -c \
    'import os, sys, torch; epoch = int(torch.load(sys.argv[1], map_location="cpu")["epoch"]); sys.stdout.write(f"{epoch}\n"); sys.stdout.flush(); os._exit(0)' \
    "$checkpoint"
}

run_chunked_stage() {
  local stage_name=$1
  local resume_checkpoint=$2
  local target_epochs=$3
  local init_checkpoint=$4
  shift 4
  local stage_args=("$@")
  local start_epoch=0
  local consecutive_failures=0

  if [[ -n "$resume_checkpoint" ]]; then
    start_epoch=$(checkpoint_epoch "$resume_checkpoint")
  fi
  while (( start_epoch < target_epochs )); do
    local end_epoch=$(( (start_epoch / EPOCH_CHUNK + 1) * EPOCH_CHUNK ))
    if (( end_epoch > target_epochs )); then
      end_epoch=$target_epochs
    fi
    local chunk_args=(--epochs "$end_epoch" --exp_name "$stage_name")
    if [[ -n "$resume_checkpoint" ]]; then
      chunk_args+=(--resume_checkpoint "$resume_checkpoint")
    elif [[ -n "$init_checkpoint" ]]; then
      chunk_args+=(--init_checkpoint "$init_checkpoint")
    fi
    # A CUDA/PyTorch shutdown can occasionally return a non-zero status after
    # the epoch checkpoint has been committed.  Run it as an `if` condition so
    # `set -e` cannot terminate the orchestration before the checkpoint check.
    local process_status=0
    if run_python main.py "${COMMON[@]}" "${stage_args[@]}" "${chunk_args[@]}"; then
      process_status=0
    else
      process_status=$?
    fi

    resume_checkpoint="runs/$stage_name/last.pt"
    if [[ ! -f "$resume_checkpoint" ]]; then
      if (( process_status == 0 )); then
        echo "Training process exited successfully but did not save $resume_checkpoint" >&2
        return 1
      fi
      consecutive_failures=$((consecutive_failures + 1))
      if (( consecutive_failures > MAX_STAGE_RETRIES )); then
        echo "Training failed $consecutive_failures times before creating its first checkpoint" >&2
        return "$process_status"
      fi
      echo "Warning: process exited with status $process_status before creating its first checkpoint; " \
           "restarting epoch 0 (retry $consecutive_failures/$MAX_STAGE_RETRIES)" >&2
      resume_checkpoint=""
      sleep "$RETRY_DELAY"
      continue
    fi
    local saved_epoch
    saved_epoch=$(checkpoint_epoch "$resume_checkpoint")
    if (( saved_epoch < end_epoch )); then
      if (( process_status == 0 )); then
        echo "Training process exited successfully before epoch $end_epoch; checkpoint is only epoch $saved_epoch" >&2
        return 1
      fi
      if (( saved_epoch < start_epoch )); then
        echo "Checkpoint regressed from epoch $start_epoch to $saved_epoch" >&2
        return 1
      fi
      if (( saved_epoch == start_epoch )); then
        consecutive_failures=$((consecutive_failures + 1))
      else
        consecutive_failures=0
      fi
      if (( consecutive_failures > MAX_STAGE_RETRIES )); then
        echo "Training failed $consecutive_failures times without advancing beyond epoch $start_epoch" >&2
        return "$process_status"
      fi
      echo "Warning: process exited with status $process_status before epoch $end_epoch; " \
           "resuming checkpoint epoch $saved_epoch (retry $consecutive_failures/$MAX_STAGE_RETRIES)" >&2
      start_epoch=$saved_epoch
      sleep "$RETRY_DELAY"
      continue
    fi
    if (( process_status != 0 )); then
      echo "Warning: process exited with status $process_status after atomically saving epoch $saved_epoch; continuing with a fresh process" >&2
    fi
    if (( saved_epoch <= start_epoch )); then
      echo "Checkpoint did not advance beyond epoch $start_epoch" >&2
      return 1
    fi
    consecutive_failures=0
    start_epoch=$saved_epoch
  done
}

STAGE_A_TRAIN_ARGS=()
validate_processed_dimensions
if [[ "$SAVE_STAGE_A_CHECKPOINTS" == "1" ]]; then
  STAGE_A_TRAIN_ARGS+=(--save_checkpoints)
fi

if [[ "$SKIP_STAGE_A" != "1" ]]; then
  run_chunked_stage "$STAGE_A" "$STAGE_A_RESUME" "$TOTAL_EPOCHS" "" \
    "${STAGE_A_TRAIN_ARGS[@]}"
fi

if [[ "$STOP_AFTER_STAGE_A" == "1" ]]; then
  echo "Stage A is complete; stopping before Stage B so validation-only selection can be frozen."
  exit 0
fi

STAGE_A_SCORE=${STAGE_A_CHECKPOINT:-"$STAGE_A_DIR/MSTR_score.pt"}
if [[ ! -f "$STAGE_A_SCORE" ]]; then
  echo "Matched Stage B selection requires $STAGE_A_SCORE" >&2
  exit 1
fi
STAGE_B_EPOCHS=$(checkpoint_epoch "$STAGE_A_SCORE")
if (( STAGE_B_EPOCHS <= 0 || STAGE_B_EPOCHS > TOTAL_EPOCHS )); then
  echo "Stage A selected invalid epoch $STAGE_B_EPOCHS for TOTAL_EPOCHS=$TOTAL_EPOCHS" >&2
  exit 1
fi
echo "Stage A selected epoch $STAGE_B_EPOCHS; training Stage B only through its matched checkpoint."
STAGE_B_LR=$(awk -v scale="$STAGE_B_LR_SCALE" 'BEGIN { printf "%.12g", 1e-3 * scale }')
if [[ "$STAGE_B_LR_SCALE" != "1.0" && "$STAGE_B_LR_SCALE" != "1" ]]; then
  echo "Stage B learning rate is scaled to $STAGE_B_LR (scale $STAGE_B_LR_SCALE)."
fi

STAGE_B_INIT=""
if [[ ( "$ABLATION" == "adaptive_lkc_residual_stageb_init" ||
        "$ABLATION" == "adaptive_lkc_residual_stageb_init_seen_distill" ) &&
      -z "$STAGE_B_RESUME" ]]; then
  STAGE_B_INIT="$STAGE_A_SCORE"
  echo "Stage B will initialize model weights from $STAGE_B_INIT with a fresh optimizer."
fi

STAGE_B_DISTILL_ARGS=()
if [[ "$ABLATION" == "adaptive_lkc_residual_stageb_seen_distill" ||
      "$ABLATION" == "adaptive_lkc_residual_stageb_init_seen_distill" ]]; then
  STAGE_B_SEEN_DISTILL_WEIGHT=${STAGE_B_SEEN_DISTILL_WEIGHT:-0.02}
  STAGE_B_DISTILL_ARGS=(
    --stage_b_teacher_checkpoint "$STAGE_A_SCORE"
    --stage_b_seen_distill_weight "$STAGE_B_SEEN_DISTILL_WEIGHT"
  )
  echo "Stage B will preserve Stage A seen-class embeddings with teacher weight $STAGE_B_SEEN_DISTILL_WEIGHT."
fi

STAGE_B_SAMPLER_ARGS=()
if [[ "$ABLATION" == "adaptive_lkc_residual_stageb_group_balanced" ]]; then
  STAGE_B_NEW_CLASS_FRACTION=${STAGE_B_NEW_CLASS_FRACTION:-0.5}
  STAGE_B_SAMPLER_ARGS=(
    --stage_b_new_class_fraction "$STAGE_B_NEW_CLASS_FRACTION"
  )
  echo "Stage B will allocate $STAGE_B_NEW_CLASS_FRACTION batch mass to newly introduced classes."
fi

run_chunked_stage "$STAGE_B" "$STAGE_B_RESUME" "$STAGE_B_EPOCHS" "$STAGE_B_INIT" \
  --retrain_all --save_checkpoints --lr "$STAGE_B_LR" \
  "${STAGE_B_DISTILL_ARGS[@]}" "${STAGE_B_SAMPLER_ARGS[@]}"
if [[ "$SKIP_EVALUATION" == "0" ]]; then
  run_python get_evaluation.py \
    --load_path_stage_A "$STAGE_A_DIR" \
    --load_path_stage_B "runs/$STAGE_B" \
    --stage_b_selection matched
else
  echo "Skipping automatic evaluation; an explicit frozen post-hoc evaluator is required."
fi

if [[ "$ABLATION" == "adaptive_lkc_residual_semantic_geometry" \
      || "$ABLATION" == "adaptive_lkc_residual_ceo_text" ]]; then
  run_python analyze_semantic_geometry.py \
    --load_path_stage_A "runs/$STAGE_A" \
    --load_path_stage_B "runs/$STAGE_B"
fi
