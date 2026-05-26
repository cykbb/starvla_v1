#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
cd "${REPO_ROOT}"

export LIBERO_HOME=${LIBERO_HOME:-/home/users/astar/i2r/chengzy/LIBERO}
export LIBERO_python=${LIBERO_python:-/home/users/astar/i2r/chengzy/.conda/envs/libero/bin/python}
export starVLA_python=${starVLA_python:-/home/users/astar/i2r/chengzy/.conda/envs/starVLA/bin/python}
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}"

LD_PRELOAD_VALUE=${LD_PRELOAD_VALUE:-/home/users/astar/i2r/chengzy/.conda/envs/libero/lib/libstdc++.so.6}
SAVE_VIDEO_VALUE=${SAVE_VIDEO_VALUE:-true}
SAVE_PATCH_VIS_VALUE=${SAVE_PATCH_VIS_VALUE:-true}
SAVE_RAW_DEBUG_FRAMES_VALUE=${SAVE_RAW_DEBUG_FRAMES_VALUE:-false}
SAVE_DEBUG_JSONL_VALUE=${SAVE_DEBUG_JSONL_VALUE:-false}
SAVE_ACTIONS_VALUE=${SAVE_ACTIONS_VALUE:-false}
SAVE_EPISODE_SUMMARY_VALUE=${SAVE_EPISODE_SUMMARY_VALUE:-false}
MUJOCO_GL_VALUE=${MUJOCO_GL_VALUE:-${MUJOCO_GL:-egl}}
PYOPENGL_PLATFORM_VALUE=${PYOPENGL_PLATFORM_VALUE:-${PYOPENGL_PLATFORM:-egl}}
MUJOCO_EGL_DEVICE_ID_VALUE=${MUJOCO_EGL_DEVICE_ID_VALUE:-${MUJOCO_EGL_DEVICE_ID:-}}
EPISODES_PER_EVAL_PROCESS_VALUE=${EPISODES_PER_EVAL_PROCESS_VALUE:-${EPISODES_PER_EVAL_PROCESS:-1}}
HOST=${HOST:-127.0.0.1}
INJECT_TASK_META_VALUE=${INJECT_TASK_META_VALUE:-auto}
TASK_META_PATH_VALUE=${TASK_META_PATH_VALUE:-}

your_ckpt=${1:?Usage: eval_libero_parall.sh <ckpt> <task_suite> <run_index> [mode] [train_log] [task_ids_csv] [max_episodes] [num_trials] [episode_start_index]>}
task_suite_name=${2:?}
run_index=${3:-0}
eval_mode=${4:-diagnostic}
train_log_path=${5:-}
task_ids_csv=${6:-}
max_episodes_per_task=${7:-}
num_trials_per_task=${8:-50}
episode_start_index=${9:-${EPISODE_START_INDEX:-0}}

num_gpus=${NUM_GPUS:-1}
gpu_id=$((run_index % num_gpus))
base_port=${BASE_PORT:-$((6450 + run_index))}
if [ -z "${MUJOCO_EGL_DEVICE_ID_VALUE}" ]; then
    MUJOCO_EGL_DEVICE_ID_VALUE="${gpu_id}"
fi

model_root=$(echo "${your_ckpt}" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "${your_ckpt}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
log_root="${model_root}/logs/${task_suite_name}"
video_root="${model_root}/videos/${task_suite_name}/${folder_name}"
diag_root="${model_root}/diagnostics/${task_suite_name}/${folder_name}"
mkdir -p "${log_root}" "${video_root}" "${diag_root}"

target_step=$(basename "${your_ckpt}" | sed -n 's/^steps_\([0-9][0-9]*\)_.*$/\1/p')
target_step=${TARGET_STEP:-${target_step:-30000}}

# Resolve INJECT_TASK_META_VALUE (auto/true/false) into a strict true/false.
# auto: read framework.name from the checkpoint's sibling config.yaml; turn on for QwenPaDTPI.
resolve_inject_task_meta() {
    local mode="$1"
    local ckpt_path="$2"
    local lower
    lower=$(printf '%s' "${mode}" | tr '[:upper:]' '[:lower:]')
    case "${lower}" in
        true|1|yes|on)
            echo "true"
            return 0
            ;;
        false|0|no|off)
            echo "false"
            return 0
            ;;
        auto|"")
            local config_yaml="${model_root}/config.yaml"
            if [ ! -f "${config_yaml}" ]; then
                echo "false"
                return 0
            fi
            local framework_name
            framework_name=$("${starVLA_python}" - "${config_yaml}" <<'PY'
import sys
import yaml

try:
    with open(sys.argv[1], "r") as f:
        cfg = yaml.safe_load(f) or {}
    print(str(cfg.get("framework", {}).get("name", "")).strip())
except Exception:
    print("")
PY
            )
            if [ "${framework_name}" = "QwenPaDTPI" ]; then
                echo "true"
            else
                echo "false"
            fi
            return 0
            ;;
        *)
            echo "Unknown INJECT_TASK_META_VALUE: ${mode} (expected auto/true/false)" >&2
            return 1
            ;;
    esac
}

resolved_inject_task_meta=$(resolve_inject_task_meta "${INJECT_TASK_META_VALUE}" "${your_ckpt}")
echo "INJECT_TASK_META_VALUE=${INJECT_TASK_META_VALUE} → resolved_inject_task_meta=${resolved_inject_task_meta}"
if [ -n "${TASK_META_PATH_VALUE}" ]; then
    echo "TASK_META_PATH_VALUE=${TASK_META_PATH_VALUE}"
fi
echo "MuJoCo render env: MUJOCO_GL=${MUJOCO_GL_VALUE} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM_VALUE} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID_VALUE}"
echo "EPISODES_PER_EVAL_PROCESS_VALUE=${EPISODES_PER_EVAL_PROCESS_VALUE}"

server_pid=""

cleanup() {
    if [ -n "${server_pid}" ] && kill -0 "${server_pid}" 2>/dev/null; then
        echo "Stopping policy server PID ${server_pid}"
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}

trap cleanup EXIT

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout_sec="${3:-120}"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        if python - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
        then
            return 0
        fi
        if [ $(( $(date +%s) - start_ts )) -ge "${timeout_sec}" ]; then
            echo "Timed out waiting for ${host}:${port}"
            return 1
        fi
        sleep 2
    done
}

start_server() {
    local server_log_path="$1"
    echo "Starting policy server on port ${base_port} using GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        "${starVLA_python}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "${your_ckpt}" \
        --port "${base_port}" \
        --use_bf16 >"${server_log_path}" 2>&1 &
    server_pid=$!
    wait_for_port "${HOST}" "${base_port}" 180
}

append_task_id_args() {
    local -n out_ref=$1
    local task_ids_arg="${2:-${task_ids_csv}}"
    if [ -n "${task_ids_arg}" ]; then
        IFS=',' read -r -a task_id_items <<< "${task_ids_arg}"
        if [ "${#task_id_items[@]}" -gt 0 ]; then
            out_ref+=(--args.task-ids)
            for item in "${task_id_items[@]}"; do
                if [ -n "${item}" ]; then
                    out_ref+=("${item}")
                fi
            done
        fi
    fi
}

run_eval_once() {
    local mode_name="$1"
    local inject_task_meta="$2"
    local eval_log_path="${log_root}/${folder_name}_${mode_name}.log"
    local current_video_root="${video_root}/${mode_name}"
    mkdir -p "${current_video_root}"
    : >"${eval_log_path}"

    local base_cmd=(
        "${LIBERO_python}" "${REPO_ROOT}/examples/LIBERO/eval_files/eval_libero.py"
        --args.pretrained-path "${your_ckpt}"
        --args.host "${HOST}"
        --args.port "${base_port}"
        --args.task-suite-name "${task_suite_name}"
        --args.num-trials-per-task "${num_trials_per_task}"
        --args.video-out-path "${current_video_root}"
    )

    if [ "${eval_mode}" = "diagnostic" ] || [ "${eval_mode}" = "diagnostic_current" ] || [ "${eval_mode}" = "diagnostic_meta" ]; then
        base_cmd+=(--args.debug-dump-dir "${diag_root}/${mode_name}")
    fi

    if [ "${inject_task_meta}" = "true" ]; then
        base_cmd+=(--args.inject-task-meta)
        if [ -n "${TASK_META_PATH_VALUE}" ]; then
            base_cmd+=(--args.task-meta-path "${TASK_META_PATH_VALUE}")
        fi
    fi

    run_eval_process() {
        local chunk_start="$1"
        local chunk_count="$2"
        local task_ids_arg="${3:-${task_ids_csv}}"
        local cmd=("${base_cmd[@]}" --args.episode-start-index "${chunk_start}")
        append_task_id_args cmd "${task_ids_arg}"
        if [ -n "${chunk_count}" ]; then
            cmd+=(--args.max-episodes-per-task "${chunk_count}")
        elif [ -n "${max_episodes_per_task}" ]; then
            cmd+=(--args.max-episodes-per-task "${max_episodes_per_task}")
        fi

        echo "Running ${mode_name}: task_ids=${task_ids_arg:-<all>} episode_start=${chunk_start} max_episodes=${chunk_count:-${max_episodes_per_task:-<default>}}" | tee -a "${eval_log_path}"
        env \
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            MUJOCO_GL="${MUJOCO_GL_VALUE}" \
            PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}" \
            MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID_VALUE}" \
            SAVE_VIDEO="${SAVE_VIDEO_VALUE}" \
            SAVE_PATCH_VIS="${SAVE_PATCH_VIS_VALUE}" \
            SAVE_RAW_DEBUG_FRAMES="${SAVE_RAW_DEBUG_FRAMES_VALUE}" \
            SAVE_DEBUG_JSONL="${SAVE_DEBUG_JSONL_VALUE}" \
            SAVE_ACTIONS="${SAVE_ACTIONS_VALUE}" \
            SAVE_EPISODE_SUMMARY="${SAVE_EPISODE_SUMMARY_VALUE}" \
            TRACE_ENV_STEPS="${TRACE_ENV_STEPS_VALUE:-false}" \
            PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER_VALUE:-1}" \
            PYTHONUNBUFFERED=1 \
            LD_PRELOAD="${LD_PRELOAD_VALUE}" \
            DEBUG= \
            "${cmd[@]}" 2>&1 | tee -a "${eval_log_path}"
    }

    local chunk_size="${EPISODES_PER_EVAL_PROCESS_VALUE}"
    if [ "${chunk_size}" -gt 0 ]; then
        local total_requested="${max_episodes_per_task:-${num_trials_per_task}}"
        local split_task_ids=("")
        if [ -n "${task_ids_csv}" ]; then
            IFS=',' read -r -a split_task_ids <<< "${task_ids_csv}"
        fi

        for task_id_item in "${split_task_ids[@]}"; do
            if [ -n "${task_ids_csv}" ] && [ -z "${task_id_item}" ]; then
                continue
            fi
            local current_start="${episode_start_index}"
            local remaining="${total_requested}"
            while [ "${remaining}" -gt 0 ]; do
                local chunk_count="${chunk_size}"
                if [ "${chunk_count}" -gt "${remaining}" ]; then
                    chunk_count="${remaining}"
                fi
                run_eval_process "${current_start}" "${chunk_count}" "${task_id_item}"
                current_start=$((current_start + chunk_count))
                remaining=$((remaining - chunk_count))
            done
        done
        return 0
    fi

    echo "Running ${mode_name}..."
    run_eval_process "${episode_start_index}" ""
}

run_analysis() {
    if [ ! -f "${train_log_path}" ]; then
        echo "TRAIN_LOG_PATH not provided or file missing, skipping report generation."
        return 0
    fi

    local report_dir="${diag_root}/report"
    mkdir -p "${report_dir}"
    local analysis_log_path="${log_root}/${folder_name}_diagnostic_report.log"
    echo "Generating diagnostic report into ${report_dir}"
    "${LIBERO_python}" "${REPO_ROOT}/examples/LIBERO/eval_files/analyze_padt_libero_diagnostics.py" \
        --train-log "${train_log_path}" \
        --run-dir "${model_root}" \
        --eval-debug-dir "${diag_root}" \
        --output-dir "${report_dir}" \
        --target-step "${target_step}" \
        --save-overlay-frames 2>&1 | tee "${analysis_log_path}"
}

start_server "${log_root}/${folder_name}_server.log"

case "${eval_mode}" in
    standard)
        run_eval_once "standard_eval" "${resolved_inject_task_meta}"
        ;;
    diagnostic)
        run_eval_once "current_eval" "false"
        run_eval_once "meta_aligned_eval" "true"
        run_analysis
        ;;
    diagnostic_current)
        run_eval_once "current_eval" "false"
        run_analysis
        ;;
    diagnostic_meta)
        run_eval_once "meta_aligned_eval" "true"
        run_analysis
        ;;
    *)
        echo "Unknown eval mode: ${eval_mode}"
        exit 1
        ;;
esac

echo "Finished ${eval_mode} run."
echo "Logs: ${log_root}"
if [ "${eval_mode}" = "diagnostic" ] || [ "${eval_mode}" = "diagnostic_current" ] || [ "${eval_mode}" = "diagnostic_meta" ]; then
    echo "Diagnostics root: ${diag_root}"
fi
