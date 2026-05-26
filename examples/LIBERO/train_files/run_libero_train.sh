#!/bin/bash
#SBATCH -J base_libero
#SBATCH -p testqueue              # 队列/partition
#SBATCH -A aicloud-testgroup          # 项目号/Account
#SBATCH -t 0-18                   # 运行时间：0-24 = 24小时
#SBATCH -N 1                      # 1 个节点
#SBATCH --gres=gpu:4              # 申请 1 张 GPU
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

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI
freeze_module_list=''
base_vlm=/home/users/astar/i2r/chengzy/starVLA_origin/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_libero_qwen.yaml
# libero_data_root=/home/users/astar/i2r/lishijie/grasping_challenge/scratch/yk/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA
libero_data_root=/home/users/astar/i2r/chengzy/starVLA_origin/playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=qwenpi_libero_baseline_sameconfig
# === End of environment variable configuration ===
###########################################################################################


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
echo "Expected resume checkpoint dir: ${output_dir}/checkpoints"
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
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_entity bykkk-nanyang-technological-university-singapore \
  --wandb_project starVLA_Libero &

train_pid=$!
wait "${train_pid}"

  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
