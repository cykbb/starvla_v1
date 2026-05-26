#!/bin/bash
#SBATCH -J autoeval_libero
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 0-18
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

set -euo pipefail

DEFAULT_REPO_ROOT=/home/users/astar/i2r/chengzy/starVLA_origin
REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}
if [ ! -d "${REPO_ROOT}" ]; then
    echo "Repository root does not exist: ${REPO_ROOT}"
    exit 1
fi
cd "${REPO_ROOT}"

SCRIPT_PATH="${REPO_ROOT}/examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh"
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "Evaluation launcher does not exist: ${SCRIPT_PATH}"
    exit 1
fi

# ── Eval parameters ──────────────────────────────────────────────────────────
# Edit defaults below, or override at submission time:
#   sbatch --export=ALL,CKPT_PATH=/other/path.pt auto_eval_libero.sh
CKPT_PATH=${CKPT_PATH:-/home/users/astar/i2r/chengzy/starVLA_origin/results/Checkpoints/qwenpi_libero_baseline/checkpoints/steps_100000_pytorch_model.pt}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-libero_goal}
EVAL_MODE=${EVAL_MODE:-standard}
RUN_INDEX=${RUN_INDEX:-4}
TRAIN_LOG_PATH=${TRAIN_LOG_PATH:-}
TASK_IDS=${TASK_IDS:-0}
MAX_EPISODES_PER_TASK=${MAX_EPISODES_PER_TASK:-}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
EPISODE_START_INDEX=${EPISODE_START_INDEX:-0}
# These must be exported so eval_libero_parall.sh can read them from the environment.
export SAVE_VIDEO_VALUE=${SAVE_VIDEO_VALUE:-false}
export TRACE_ENV_STEPS_VALUE=${TRACE_ENV_STEPS_VALUE:-true}
export SAVE_PATCH_VIS_VALUE=${SAVE_PATCH_VIS_VALUE:-true}
export SAVE_RAW_DEBUG_FRAMES_VALUE=${SAVE_RAW_DEBUG_FRAMES_VALUE:-false}
export SAVE_DEBUG_JSONL_VALUE=${SAVE_DEBUG_JSONL_VALUE:-false}
export SAVE_ACTIONS_VALUE=${SAVE_ACTIONS_VALUE:-false}
export SAVE_EPISODE_SUMMARY_VALUE=${SAVE_EPISODE_SUMMARY_VALUE:-false}
export MUJOCO_GL_VALUE=${MUJOCO_GL_VALUE:-${MUJOCO_GL:-egl}}
export PYOPENGL_PLATFORM_VALUE=${PYOPENGL_PLATFORM_VALUE:-${PYOPENGL_PLATFORM:-egl}}
export MUJOCO_EGL_DEVICE_ID_VALUE=${MUJOCO_EGL_DEVICE_ID_VALUE:-${MUJOCO_EGL_DEVICE_ID:-0}}
export EPISODES_PER_EVAL_PROCESS_VALUE=${EPISODES_PER_EVAL_PROCESS_VALUE:-${EPISODES_PER_EVAL_PROCESS:-1}}
# PaDTPI eval task-metadata injection (auto/true/false). auto = on iff framework.name=QwenPaDTPI.
export INJECT_TASK_META_VALUE=${INJECT_TASK_META_VALUE:-auto}
# Optional override for padt_task_specs.jsonl path; leave empty to auto-resolve from checkpoint config.
export TASK_META_PATH_VALUE=${TASK_META_PATH_VALUE:-}
# ─────────────────────────────────────────────────────────────────────────────

echo "Repository root:  ${REPO_ROOT}"
echo "Checkpoint:       ${CKPT_PATH}"
echo "Task suite:       ${TASK_SUITE_NAME}"
echo "Eval mode:        ${EVAL_MODE}"
echo "RUN_INDEX:        ${RUN_INDEX}"
echo "Task IDs:         ${TASK_IDS:-<all>}"
echo "Num trials:       ${NUM_TRIALS_PER_TASK}  max_episodes: ${MAX_EPISODES_PER_TASK:-<unlimited>}  start: ${EPISODE_START_INDEX}"
echo "Save video:       ${SAVE_VIDEO_VALUE}"
echo "Trace env steps:  ${TRACE_ENV_STEPS_VALUE}"
echo "Save overlays:    ${SAVE_PATCH_VIS_VALUE}"
echo "Save raw frames:  ${SAVE_RAW_DEBUG_FRAMES_VALUE}"
echo "Save steps jsonl: ${SAVE_DEBUG_JSONL_VALUE}"
echo "Save actions:     ${SAVE_ACTIONS_VALUE}"
echo "Save ep summary:  ${SAVE_EPISODE_SUMMARY_VALUE}"
echo "MuJoCo GL:        ${MUJOCO_GL_VALUE}"
echo "PyOpenGL:         ${PYOPENGL_PLATFORM_VALUE}"
echo "EGL device:       ${MUJOCO_EGL_DEVICE_ID_VALUE}"
echo "Eval chunk size:  ${EPISODES_PER_EVAL_PROCESS_VALUE}"
echo "Inject task meta: ${INJECT_TASK_META_VALUE}"
if [ -n "${TASK_META_PATH_VALUE}" ]; then
    echo "Task meta path:   ${TASK_META_PATH_VALUE}"
else
    echo "Task meta path:   <auto-resolved from checkpoint config>"
fi
if [ -n "${TRAIN_LOG_PATH}" ]; then
    echo "Train log:        ${TRAIN_LOG_PATH}"
else
    echo "Train log:        <not provided; report generation will be skipped>"
fi

bash "${SCRIPT_PATH}" \
    "${CKPT_PATH}" \
    "${TASK_SUITE_NAME}" \
    "${RUN_INDEX}" \
    "${EVAL_MODE}" \
    "${TRAIN_LOG_PATH}" \
    "${TASK_IDS}" \
    "${MAX_EPISODES_PER_TASK}" \
    "${NUM_TRIALS_PER_TASK}" \
    "${EPISODE_START_INDEX}"
