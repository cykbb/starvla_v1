#!/bin/bash
#SBATCH -J padtpi_validate       # 作业名
#SBATCH -p testqueue              # 队列/partition
#SBATCH -A aicloud-testgroup      # 项目号/Account
#SBATCH -t 0-01                   # 1 小时足够(实际只需 5-10 分钟)
#SBATCH -N 1                      # 1 个节点
#SBATCH --gres=gpu:1              # 1 张 GPU 够了(validation 不需要多卡)
#SBATCH -o slurm_%x_%j.out        # 标准输出
#SBATCH -e slurm_%x_%j.err        # 错误输出

# ── 用途 ─────────────────────────────────────────────────────────────────────
# Phase 1-4 性能优化的快速验证(行为保持检查 + 单步性能测量)。
# 跑 5-10 分钟,产出:
#   - ViT call count == 1(Phase 1)
#   - LLM GC enabled == True(Phase 3)
#   - decoder _use_grad_ckpt == True(Phase 4)
#   - action_loss / vrt_loss / bbox_loss / mask_loss / score_loss 数值健康
#   - per-step wall-clock + peak memory(跟 v2 baseline 对比)
#
# 提交方法:
#   sbatch scripts/sbatch_validate_padtpi.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── 环境加载(与 run_libero_padtpi.sh 一致)─────────────────────────────────
module purge
module load miniforge/24.11.3-2
module load cuda/12.4.1
eval "$(conda shell.bash hook)"
conda activate /home/users/astar/i2r/chengzy/.conda/envs/starVLA

cd /home/users/astar/i2r/chengzy/starVLA_origin

# 不需要 NCCL / DeepSpeed,单 GPU 直接跑
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# ── 跑验证 ────────────────────────────────────────────────────────────────────
python scripts/validate_padtpi_phase1_4.py \
    --config starVLA/config/training/starvla_padtpi_libero.yaml \
    --batch_size 2 \
    --n_steps 3

echo ""
echo "=============================================================================="
echo "Validation done. Key things to check in the output above:"
echo "  ✓ [Phase 1] ViT call count for 1 training forward: 1     (1 = pass, 2 = fail)"
echo "  ✓ [Phase 3] LLM gradient checkpointing enabled: True"
echo "  ✓ [Phase 4] padt_decoder._use_grad_ckpt: True"
echo "  ✓ action_loss + all aux losses finite, no NaN/Inf"
echo "  ✓ per-step wall-clock and peak memory (compare to your v2 baseline)"
echo "=============================================================================="
