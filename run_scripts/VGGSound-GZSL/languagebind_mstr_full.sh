#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATASET=VGGSound \
DATA_ROOT=avgzsl_benchmark_datasets/VGGSound \
RUN_PREFIX=vggsound \
FEATURE_EXTRACTION_METHOD=languagebind_mstr \
INPUT_SIZE_AUDIO=512 \
INPUT_SIZE_VIDEO=768 \
TEXT_EMBEDDING_SIZE=768 \
ENC_DROPOUT=0.2 \
DEC_DROPOUT=0.2 \
ADD_DROPOUT=0.1 \
MODEL_MODE=${MODEL_MODE:-stft} \
ABLATION=${ABLATION:-adaptive_lkc_residual} \
"$SCRIPT_DIR/../run_mstr_two_stage.sh"
