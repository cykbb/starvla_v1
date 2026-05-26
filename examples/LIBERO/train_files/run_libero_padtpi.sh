#!/bin/bash
#SBATCH -J libero_train          # 作业名
#SBATCH -p testqueue              # 队列/partition
#SBATCH -A aicloud-testgroup          # 项目号/Account
#SBATCH -t 0-1                  # 运行时间：0-24 = 24小时
#SBATCH -N 1                      # 1 个节点
#SBATCH --gres=gpu:4              # 申请 4 张 GPU
#SBATCH --signal=B:USR1@600        # 超时前 10 分钟通知 batch shell 保存 checkpoint
#SBATCH -o slurm_%x_%j.out        # 标准输出
#SBATCH -e slurm_%x_%j.err        # 错误输出

set -euo pipefail

# ── 环境加载 ──────────────────────────────────────────────────────────────────
module purge
module load miniforge/24.11.3-2
module load cuda/12.4.1
eval "$(conda shell.bash hook)"
conda activate /home/users/astar/i2r/chengzy/.conda/envs/starVLA

# 切换到项目根目录（所有相对路径均基于此）
cd /home/users/astar/i2r/chengzy/starVLA_origin

# ── NCCL / 通信配置 ──────────────────────────────────────────────────────────
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

# ── W&B 离线模式（集群通常无法直接访问外网）────────────────────────────────
export WANDB_MODE=offline

# =============================================================================
# 训练配置 — 按需修改
# =============================================================================
Framework_name=QwenPaDTPI
base_vlm=/home/users/astar/i2r/chengzy/starVLA_origin/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
config_yaml=./starVLA/config/training/starvla_padtpi_libero.yaml
libero_data_root=/home/users/astar/i2r/chengzy/starVLA_origin/playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=qwen_padtpi_libero_dualvrt_v2_4

# --- Dual-view PaDT alignment run --------------------------------------------
# Architecture changed: view tokens, 5 VRTs per object per view, token-level
# decoder input, PaDT-style bbox/mask/score losses. Start from a fresh run.
noisy_teacher_probability=0
use_sampled_branch=false
sampled_branch_weight=0.0
# =============================================================================

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# ── GPU 数量自动检测（与 run_libero_train.sh 保持一致）─────────────────────
num_processes="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "${num_processes}" || ! "${num_processes}" =~ ^[0-9]+$ ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    num_processes="${#visible_gpus[@]}"
  else
    num_processes=4   # 与 #SBATCH --gres=gpu:4 对应的默认值
  fi
fi

echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "GPUs        : ${num_processes}"
echo "Output dir  : ${output_dir}"
echo "Run ID      : ${run_id}"
echo "Checkpoint dir: ${output_dir}/checkpoints"
echo ""

train_pid=""
forward_preemption_signal() {
  echo "[preemption] Received SLURM signal; forwarding SIGUSR1 to accelerate workers for checkpoint save..."
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
  --datasets.vla_data.per_device_batch_size 12 \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.padt_use_segmentation_source true \
  --datasets.vla_data.padt_task_meta_required true \
  --framework.padt.noisy_teacher_probability "${noisy_teacher_probability}" \
  --framework.padt.use_sampled_branch "${use_sampled_branch}" \
  --framework.padt.sampled_branch_weight "${sampled_branch_weight}" \
  --trainer.max_train_steps 200000 \
  --trainer.save_interval 20000 \
  --trainer.eval_interval 100 \
  --trainer.logging_frequency 100 \
  --trainer.visualize_padt_samples 4 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_PaDTPI \
  --wandb_entity bykkk-nanyang-technological-university-singapore &

train_pid=$!
wait "${train_pid}"

# =============================================================================
# 三阶段训练计划（修改上方变量后重新提交）
#
# Stage A — dual-view PaDT warmup（当前建议从头跑）
#   noisy_teacher_probability=0.0
#   use_sampled_branch=false
#   5 VRT / object / view，decoder 与 mask/bbox/score loss 已和 PaDT 对齐
#
# Stage B — noisy teacher joint training
#   noisy_teacher_probability=0.05 ~ 0.15
#   use_sampled_branch=false
#
# Stage C — sampled branch（最终阶段，可选）
#   noisy_teacher_probability=0.15
#   use_sampled_branch=true
#   sampled_branch_weight=0.25
# =============================================================================
