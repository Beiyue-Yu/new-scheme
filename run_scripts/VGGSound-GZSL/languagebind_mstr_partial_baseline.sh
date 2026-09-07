#!/usr/bin/env bash
set -euo pipefail

# Frozen 2026-09-07 baseline: LanguageBind video/text + original SeLaVi audio
# with the existing fixed-step STFT MSTR route.  The feature directory is
# intentionally suffixed with `partial_20260907`: missing VGGSound shard 08
# samples are zero-filled by the documented preprocessing step.

DATASET=VGGSound \
DATA_ROOT=avgzsl_benchmark_datasets/VGGSound \
RUN_PREFIX=vggsound_languagebind_partial \
FEATURE_EXTRACTION_METHOD=languagebind_mstr_partial_20260907 \
INPUT_SIZE_AUDIO=512 \
INPUT_SIZE_VIDEO=768 \
TEXT_EMBEDDING_SIZE=768 \
ENC_DROPOUT=0.2 \
DEC_DROPOUT=0.2 \
ADD_DROPOUT=0.1 \
MODEL_MODE=stft \
ABLATION=adaptive_lkc_residual \
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50} \
EPOCH_CHUNK=${EPOCH_CHUNK:-5} \
SNN_T=${SNN_T:-10} \
RUN_TAG=${RUN_TAG:-20260907_1427} \
MAX_STAGE_RETRIES=${MAX_STAGE_RETRIES:-5} \
DISABLE_TENSORBOARD=${DISABLE_TENSORBOARD:-1} \
bash run_scripts/run_mstr_two_stage.sh
