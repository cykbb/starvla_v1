#!/bin/bash
#SBATCH -J libero_v3_train       # 作业名
#SBATCH -p testqueue              # 队列/partition
#SBATCH -A aicloud-testgroup      # 项目号/Account
#SBATCH -t 0-18                   # 18 小时
#SBATCH -N 1                      # 1 个节点
#SBATCH --gres=gpu:4              # 4 张 GPU
#SBATCH --signal=B:USR1@600
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

# ──────────────────────────────────────────────────────────────────────────────
# PaDTPI v3 retrain (Phase 6 + 7 + 8 of perf optimization plan)
#
# Changes vs v2:
#   - Decoder hidden_size 2048 -> 1280 (Phase 6, ViT-aligned)
#   - Decoder num_heads 8 -> 16, intermediate 4x -> 2.67x (3420)
#   - high_res_proj disabled (Phase 6, use ViT pre-merger raw 1280-d)
#   - VRT tokens moved out of vocab (Phase 7, lm_head cat path, TODO)
#   - Loss weights vrt: 0.1->0.5, bbox: 0.25->0.5, mask: 0.25->0.5 (Phase 8)
#   - per_device_batch_size 6 -> 12 (Phase 1-4 enables this)
#
# Trains from scratch (is_resume: false, v2 ckpt is shape-incompatible).
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

module purge
module load miniforge/24.11.3-2
module load cuda/12.4.1
eval "$(conda shell.bash hook)"
conda activate /home/users/astar/i2r/chengzy/.conda/envs/starVLA

cd /home/users/astar/i2r/chengzy/starVLA_origin

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export WANDB_MODE=offline

# =============================================================================
Framework_name=QwenPaDTPI
base_vlm=/home/users/astar/i2r/chengzy/starVLA_origin/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
config_yaml=./starVLA/config/training/starvla_padtpi_libero_v3.yaml
libero_data_root=/home/users/astar/i2r/chengzy/starVLA_origin/playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=qwen_padtpi_libero_v3

noisy_teacher_probability=0
use_sampled_branch=false
sampled_branch_weight=0.0
# =============================================================================

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${config_yaml}" "${output_dir}/" || true

num_processes="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "${num_processes}" || ! "${num_processes}" =~ ^[0-9]+$ ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    num_processes="${#visible_gpus[@]}"
  else
    num_processes=4
  fi
fi

echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : $(hostname)"
echo "GPUs           : ${num_processes}"
echo "Output dir     : ${output_dir}"
echo "Run ID         : ${run_id}"
echo "Config         : ${config_yaml}"
echo ""

train_pid=""
forward_preemption_signal() {
  echo "[preemption] Received SLURM signal; forwarding SIGUSR1 to accelerate workers..."
  if [[ -n "${train_pid}" ]]; then
    worker_pids="$(pgrep -P "${train_pid}" 2>/dev/null || true)"
    if [[ -n "${worker_pids}" ]]; then
      kill -USR1 ${worker_pids} 2>/dev/null || true
    else
      kill -USR1 "${train_pid}" 2>/dev/null || true
    fi
    wait "${train_pid}" || true
  fi
  exit 143
}
trap forward_preemption_signal USR1 TERM

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.padt_use_segmentation_source true \
  --datasets.vla_data.padt_task_meta_required true \
  --framework.padt.noisy_teacher_probability "${noisy_teacher_probability}" \
  --framework.padt.use_sampled_branch "${use_sampled_branch}" \
  --framework.padt.sampled_branch_weight "${sampled_branch_weight}" \
  --trainer.save_interval 5000 \
  --trainer.eval_interval 100 \
  --trainer.logging_frequency 100 \
  --trainer.visualize_padt_samples 4 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_PaDTPI \
  --wandb_entity bykkk-nanyang-technological-university-singapore &

train_pid=$!
wait "${train_pid}"
