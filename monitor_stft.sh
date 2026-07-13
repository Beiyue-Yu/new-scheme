#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MSTR_PYTHON:-/home/wwj/anaconda3/envs/MSTR/bin/python}"
LOG_DIR="$ROOT_DIR/logs/stft_monitor"
SAMPLE_INTERVAL="${MONITOR_INTERVAL:-10}"

usage() {
  echo "Usage: $0 <a|b> <ucf|vgg|activity> [experiment_name]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage
stage="${1,,}"
dataset="${2,,}"
[[ "$stage" == "a" || "$stage" == "b" ]] || usage

case "$dataset" in
  ucf)
    dataset_tag="ucf"
    dataset_name="UCF"
    data_root="avgzsl_benchmark_datasets/UCF"
    epochs=50
    batch_size=256
    embedding_dropout=0.2
    decoder_dropout=0.3
    additional_dropout=0.5
    ;;
  vgg|vggsound)
    dataset_tag="vgg"
    dataset_name="VGGSound"
    data_root="avgzsl_benchmark_datasets/VGGSound/"
    epochs=50
    batch_size=64
    embedding_dropout=0.0
    decoder_dropout=0.0
    additional_dropout=0.1
    ;;
  activity|activitynet)
    dataset_tag="activity"
    dataset_name="ActivityNet"
    data_root="avgzsl_benchmark_datasets/ActivityNet/"
    epochs=60
    batch_size=64
    embedding_dropout=0.2
    decoder_dropout=0.25
    additional_dropout=0.1
    ;;
  *) usage ;;
esac

if [[ "$stage" == "a" ]]; then
  split_tag="val"
else
  split_tag="all"
fi

epochs="${MSTR_EPOCHS:-$epochs}"
batch_size="${MSTR_BATCH_SIZE:-$batch_size}"
n_batches="${MSTR_N_BATCHES:-500}"
timestamp="$(date +%Y%m%d_%H%M%S)"
exp_name="${3:-stft_${dataset_tag}_${split_tag}_main_monitored_${timestamp}}"
exp_dir="$ROOT_DIR/runs/$exp_name"
resource_log="$LOG_DIR/${exp_name}_${timestamp}_resources.log"
kernel_log="$LOG_DIR/${exp_name}_${timestamp}_kernel.log"
start_time="$(date '+%Y-%m-%d %H:%M:%S')"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/main.py" || ! -f "$ROOT_DIR/monitor_train.py" ]]; then
  echo "Training or monitor entry point is missing under: $ROOT_DIR" >&2
  exit 1
fi
if [[ -e "$exp_dir" ]]; then
  echo "Experiment directory already exists: $exp_dir" >&2
  echo "Pass a new experiment name as the third argument." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONFAULTHANDLER=1

train_command=(
  "$PYTHON_BIN"
  main.py
  --root_dir "$data_root"
  --feature_extraction_method main_features
  --input_size_audio 512
  --input_size_video 512
  --epochs "$epochs"
  --lr_scheduler
  --dataset_name "$dataset_name"
  --zero_shot_split main_split
  --MSTR
  --lr 1e-3
  --n_batches "$n_batches"
  --bs "$batch_size"
  --embeddings_hidden_size 512
  --decoder_hidden_size 512
  --embedding_dropout "$embedding_dropout"
  --decoder_dropout "$decoder_dropout"
  --additional_dropout "$additional_dropout"
  --depth_transformer 1
  --additional_triplets_loss
  --first_additional_triplet 1
  --second_additional_triplet 1
  --momentum 0.1
  --reg_loss
  --snn_T 10
  --snn_tau 2.0
  --lkc_n_slots 4
  --lkc_n_heads 8
  --tucker_rank 60
  --stft_dim 512
  --trl_rank 400
  --exp_name "$exp_name"
)
if [[ "$stage" == "b" ]]; then
  train_command+=(--retrain_all --save_checkpoints)
fi

sample_resources() {
  while true; do
    {
      echo "=== $(date --iso-8601=seconds) ==="
      free -h | sed -n '1,3p'
      nvidia-smi \
        --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.free,power.draw \
        --format=csv,noheader 2>&1 || true
      ps -eo pid,ppid,stat,%cpu,%mem,rss,vsz,etime,cmd \
        | awk 'NR == 1 || /monitor_train[.]py|[p]ython.*main[.]py/' \
        || true
      echo
    } >> "$resource_log"
    sleep "$SAMPLE_INTERVAL"
  done
}

cleanup() {
  if [[ -n "${sampler_pid:-}" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[stft-monitor] Stage: ${stage^^}"
echo "[stft-monitor] Dataset: $dataset_name"
echo "[stft-monitor] Experiment: $exp_name"
echo "[stft-monitor] Epochs/batches/bs: $epochs/$n_batches/$batch_size"
echo "[stft-monitor] Resource log: $resource_log"
echo "[stft-monitor] Kernel log: $kernel_log"

cd "$ROOT_DIR"
sample_resources &
sampler_pid=$!

set +e
"$PYTHON_BIN" monitor_train.py \
  --log_dir "$LOG_DIR" \
  --exp_name "$exp_name" \
  -- "${train_command[@]}"
status=$?
set -e

cleanup
sampler_pid=""

journalctl -k --since "$start_time" --no-pager 2>&1 \
  | awk 'BEGIN { IGNORECASE=1 }
         /segfault|general protection|invalid opcode|NVRM|Xid|out of memory|oom-kill|fpregs/' \
  > "$kernel_log" || true

echo "[stft-monitor] Exit code: $status"
echo "[stft-monitor] Resource log: $resource_log"
echo "[stft-monitor] Kernel log: $kernel_log"

if [[ -s "$kernel_log" ]]; then
  echo "[stft-monitor] Kernel events detected:"
  tail -n 20 "$kernel_log"
fi

if (( status == 139 )); then
  echo "[stft-monitor] Training terminated by SIGSEGV." >&2
elif (( status != 0 )); then
  echo "[stft-monitor] Training failed with exit code $status." >&2
fi

exit "$status"
