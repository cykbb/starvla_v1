import dataclasses
import gc
import json
import logging
import math
import os
import pathlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image, ImageDraw, ImageFont

from examples.LIBERO.eval_files.model2libero_interface import ModelClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
OVERLAY_PALETTE = [
    (255, 99, 71),
    (30, 144, 255),
    (60, 179, 113),
    (255, 165, 0),
    (186, 85, 211),
    (255, 215, 0),
]
TASK_SUITE_TO_DATASET_DIR = {
    "libero_goal": "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object": "libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial": "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10": "libero_10_no_noops_1.0.0_lerobot",
}


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclass
class TaskMetaRecord:
    task_index: int
    task: str
    objects: list[str]
    task_objects: list[str]
    object_role: dict[str, str]


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size: list[int] | None = None

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task_ids: list[int] | None = None
    max_episodes_per_task: int | None = None
    episode_start_index: int = 0

    #################################################################################################################
    # Utils / diagnostics
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    debug_dump_dir: str | None = None
    save_actions: bool = False
    save_debug_jsonl: bool = False
    task_meta_path: str | None = None
    inject_task_meta: bool = False

    seed: int = 7  # Random Seed (for reproducibility)
    pretrained_path: str = ""

    post_process_action: bool = True
    rotate_input_180: bool = False
    state_history_includes_current: bool = False

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    video_out_root = Path(args.video_out_path)
    video_out_root.mkdir(parents=True, exist_ok=True)
    save_video = _env_flag("SAVE_VIDEO", default=True)
    trace_env_steps = _env_flag("TRACE_ENV_STEPS", default=False)
    requested_patch_vis = _env_flag("SAVE_PATCH_VIS", default=False)
    save_episode_summary = _env_flag("SAVE_EPISODE_SUMMARY", default=True)
    logging.info(f"Save videos: {save_video}")

    debug_root = _resolve_debug_root(args, video_out_root=video_out_root)
    save_patch_vis = bool(debug_root is not None and requested_patch_vis)
    save_raw_debug_frames = _env_flag(
        "SAVE_RAW_DEBUG_FRAMES",
        default=bool(debug_root is not None and not save_patch_vis),
    )
    save_debug_jsonl = bool(
        args.save_debug_jsonl
        or _env_flag("SAVE_DEBUG_JSONL", default=bool(debug_root is not None and not save_patch_vis))
    )
    save_actions = bool(
        args.save_actions
        or _env_flag("SAVE_ACTIONS", default=bool(debug_root is not None and not save_patch_vis))
    )
    if debug_root is not None:
        debug_root.mkdir(parents=True, exist_ok=True)
        logging.info(f"Debug dump dir: {debug_root}")
        logging.info(
            "Debug outputs: overlay=%s raw_frames=%s steps_jsonl=%s actions=%s episode_summary=%s",
            save_patch_vis,
            save_raw_debug_frames,
            save_debug_jsonl,
            save_actions,
            save_episode_summary,
        )

    max_steps = _get_max_steps(args.task_suite_name)
    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        state_history_includes_current=args.state_history_includes_current,
    )

    framework_name = (
        client_model.model_config.get("framework", {}).get("name", "<unknown>")
        if isinstance(client_model.model_config, dict)
        else "<unknown>"
    )

    resolved_task_meta_path, attempted_meta_paths = _resolve_task_meta_path(
        args=args,
        client_model=client_model,
        task_suite_name=args.task_suite_name,
    )
    task_meta_lookup = _load_task_meta_lookup(resolved_task_meta_path) if resolved_task_meta_path is not None else {}
    if args.inject_task_meta:
        if resolved_task_meta_path is None:
            attempted_str = ", ".join(attempted_meta_paths) if attempted_meta_paths else "<none>"
            raise RuntimeError(
                "inject_task_meta=True but no padt_task_specs.jsonl was found. "
                f"Framework={framework_name}, task_suite={args.task_suite_name}. "
                f"Attempted paths: [{attempted_str}]. "
                "Provide --args.task-meta-path explicitly, or place the jsonl under "
                "<data_root_dir>/<dataset_dir>/meta/padt_task_specs.jsonl."
            )
        logging.info(
            f"inject_task_meta=true framework={framework_name} task_meta_path={resolved_task_meta_path}"
        )
    else:
        logging.info(f"inject_task_meta=false framework={framework_name}")

    selected_task_ids = _select_task_ids(
        args=args,
        num_tasks_in_suite=num_tasks_in_suite,
        debug_root=debug_root,
    )
    logging.info(f"Using task ids: {selected_task_ids}")

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode_name": debug_root.name if debug_root is not None else "default_eval",
        "task_suite_name": args.task_suite_name,
        "pretrained_path": args.pretrained_path,
        "inject_task_meta": bool(args.inject_task_meta and resolved_task_meta_path is not None),
        "task_meta_path": str(resolved_task_meta_path) if resolved_task_meta_path is not None else None,
        "task_ids": selected_task_ids,
        "episode_start_index": int(args.episode_start_index),
        "max_episodes_per_task": _default_episode_limit(args, debug_root),
        "num_trials_per_task": int(args.num_trials_per_task),
        "debug_dump_dir": str(debug_root) if debug_root is not None else None,
        "save_actions": save_actions,
        "save_debug_jsonl": save_debug_jsonl,
        "save_patch_vis": save_patch_vis,
        "save_raw_debug_frames": save_raw_debug_frames,
        "save_episode_summary": save_episode_summary,
        "save_video": save_video,
        "trace_env_steps": trace_env_steps,
        "video_out_path": str(video_out_root),
        "run_id": Path(args.pretrained_path).stem if args.pretrained_path else "unknown",
    }
    if debug_root is not None:
        _write_json(debug_root / "run_manifest.json", manifest)

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(selected_task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        episode_start = _episode_start_index(args, num_initial_states=len(initial_states))
        episode_limit = min(len(initial_states), episode_start + args.num_trials_per_task)
        short_episode_limit = _default_episode_limit(args, debug_root)
        if short_episode_limit is not None:
            episode_limit = min(episode_limit, episode_start + short_episode_limit)

        task_meta_record, task_meta_source = _resolve_task_meta_record(
            task_meta_lookup, task_id=task_id, task_description=task_description
        )
        if args.inject_task_meta and resolved_task_meta_path is not None and task_meta_record is None:
            raise RuntimeError(
                f"inject_task_meta=True but no matching record for task_id={task_id} "
                f"task=\"{task_description}\" in {resolved_task_meta_path}. "
                "Check that the jsonl contains this task string (by_task match) or that "
                "task_index matches LIBERO benchmark order."
            )
        if args.inject_task_meta and task_meta_record is not None:
            logging.info(
                "task_meta task_id=%s source=%s task=\"%s\" task_objects=%s object_role=%s",
                task_id,
                task_meta_source,
                task_description,
                list(task_meta_record.task_objects),
                dict(task_meta_record.object_role),
            )

        task_episodes, task_successes = 0, 0
        task_dir = debug_root / f"task_{task_id:03d}_{_slugify(task_description)}" if debug_root is not None else None
        if task_dir is not None:
            task_dir.mkdir(parents=True, exist_ok=True)

        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        for episode_idx in tqdm.tqdm(range(episode_start, episode_limit)):
            logging.info(f"\nTask: {task_description}")
            client_model.reset(task_description=task_description)
            logging.info(f"Resetting environment for episode index {episode_idx}...")
            env.reset()
            logging.info(f"Applying initial state for episode index {episode_idx}...")
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            step = 0
            done = False
            replay_images: list[np.ndarray] = []
            full_actions: list[np.ndarray] = []
            step_records: list[dict[str, Any]] = []
            episode_start_time = time.time()

            episode_dir = task_dir / f"episode_{episode_idx:03d}" if task_dir is not None else None
            if episode_dir is not None:
                episode_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    if trace_env_steps:
                        logging.info(
                            f"Stabilization step {t + 1}/{args.num_steps_wait} for episode index {episode_idx}..."
                        )
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                if args.rotate_input_180:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                else:
                    img = np.ascontiguousarray(obs["agentview_image"])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])

                if save_video:
                    replay_images.append(img)
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                example_dict: dict[str, Any] = {
                    "image": [img, wrist_img],
                    "lang": str(task_description),
                    "state": state.astype(np.float32),
                }
                task_meta_applied = False
                if args.inject_task_meta and task_meta_record is not None:
                    _inject_task_meta(example_dict, task_meta_record)
                    task_meta_applied = True

                frame_paths = (
                    _save_debug_frames(
                        episode_dir=episode_dir,
                        step=step,
                        agentview=img,
                        wrist=wrist_img,
                    )
                    if save_raw_debug_frames
                    else None
                )

                start_time = time.time()
                response = client_model.step(
                    example=example_dict,
                    step=step,
                    return_debug=bool(debug_root is not None or save_debug_jsonl or save_patch_vis),
                )
                inference_time = time.time() - start_time

                raw_action = response["raw_action"]
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        "Unexpected action sizes: wv=%s, rot=%s, grip=%s. Falling back to LIBERO_DUMMY_ACTION.",
                        world_vector_delta.shape,
                        rotation_delta.shape,
                        gripper.shape,
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )

                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0).astype(np.float32)
                full_actions.append(delta_action)

                if (save_debug_jsonl or save_patch_vis) and episode_dir is not None:
                    step_record: dict[str, Any] = {
                        "task_suite_name": args.task_suite_name,
                        "task_id": int(task_id),
                        "task_description": str(task_description),
                        "episode_idx": int(episode_idx),
                        "step": int(step),
                        "env_step": int(t),
                        "task_meta_applied": bool(task_meta_applied),
                        "agentview_path": str(frame_paths["agentview"]) if frame_paths else None,
                        "wrist_path": str(frame_paths["wrist"]) if frame_paths else None,
                        "delta_action": delta_action,
                        "policy_latency_sec": float(inference_time),
                    }
                    if "debug" in response:
                        step_record.update(_jsonable(response["debug"]))
                    if save_patch_vis:
                        overlay_paths = _save_prediction_overlays(
                            episode_dir=episode_dir,
                            step=step,
                            agentview=img,
                            wrist=wrist_img,
                            record=step_record,
                        )
                        step_record["overlay_paths"] = overlay_paths
                    if save_debug_jsonl:
                        step_records.append(step_record)

                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = _slugify(task_description)
            video_path = video_out_root / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4"
            if save_video:
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

            actions_array = (
                np.stack(full_actions).astype(np.float32)
                if full_actions
                else np.zeros((0, 7), dtype=np.float32)
            )

            action_paths = _save_actions(
                episode_dir=episode_dir,
                actions=actions_array,
                enabled=bool(save_actions and episode_dir is not None),
            )
            if save_debug_jsonl and step_records and episode_dir is not None:
                _write_jsonl(episode_dir / "steps.jsonl", step_records)

            episode_summary = {
                "task_suite_name": args.task_suite_name,
                "task_id": int(task_id),
                "task_description": str(task_description),
                "episode_idx": int(episode_idx),
                "success": bool(done),
                "task_meta_applied": bool(args.inject_task_meta and task_meta_record is not None),
                "video_path": str(video_path) if save_video else None,
                "steps_jsonl_path": str(episode_dir / "steps.jsonl") if episode_dir is not None and step_records else None,
                "num_policy_steps": int(actions_array.shape[0]),
                "duration_sec": float(time.time() - episode_start_time),
                "action_chunk_size": int(client_model.action_chunk_size),
                "action_norm_mean": _safe_array_stat(actions_array, lambda x: np.linalg.norm(x[:, :6], axis=1).mean()),
                "action_norm_max": _safe_array_stat(actions_array, lambda x: np.linalg.norm(x[:, :6], axis=1).max()),
                "max_abs_action": _safe_array_stat(actions_array, lambda x: np.abs(x).max()),
                "gripper_flips": _count_gripper_flips(actions_array[:, 6]) if actions_array.size else 0,
                "actions_npy_path": action_paths.get("npy"),
                "actions_csv_path": action_paths.get("csv"),
            }
            if save_episode_summary and episode_dir is not None:
                _write_json(episode_dir / "episode_summary.json", episode_summary)

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        env.close()
        del env
        gc.collect()
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if debug_root is not None:
        manifest.update(
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_episodes": int(total_episodes),
                "total_successes": int(total_successes),
                "total_success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
            }
        )
        _write_json(debug_root / "run_manifest.json", manifest)

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _resolve_debug_root(args: Args, *, video_out_root: Path) -> Path | None:
    if args.debug_dump_dir:
        return Path(args.debug_dump_dir)
    if args.save_actions or args.save_debug_jsonl:
        return video_out_root / "debug"
    return None


def _default_episode_limit(args: Args, debug_root: Path | None) -> int | None:
    if args.max_episodes_per_task is not None:
        return int(args.max_episodes_per_task)
    if debug_root is not None:
        return 5
    return None


def _episode_start_index(args: Args, *, num_initial_states: int) -> int:
    episode_start = int(args.episode_start_index)
    if episode_start < 0 or episode_start >= num_initial_states:
        raise ValueError(
            f"episode_start_index={episode_start} is out of range for "
            f"{num_initial_states} initial states"
        )
    return episode_start


def _select_task_ids(args: Args, *, num_tasks_in_suite: int, debug_root: Path | None) -> list[int]:
    if args.task_ids is None:
        if debug_root is not None:
            return [0]
        return list(range(num_tasks_in_suite))

    task_ids = [int(task_id) for task_id in args.task_ids]
    invalid = [task_id for task_id in task_ids if task_id < 0 or task_id >= num_tasks_in_suite]
    if invalid:
        raise ValueError(f"Task ids out of range for {args.task_suite_name}: {invalid}")
    return task_ids


def _normalize_task_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _resolve_task_meta_path(
    args: Args, *, client_model: ModelClient, task_suite_name: str
) -> tuple[Path | None, list[str]]:
    attempted: list[str] = []

    if args.task_meta_path:
        path = Path(args.task_meta_path)
        attempted.append(str(path))
        return (path if path.exists() else None), attempted

    dataset_dir = TASK_SUITE_TO_DATASET_DIR.get(task_suite_name, None)
    if dataset_dir is None:
        attempted.append(f"<no dataset_dir mapping for task_suite={task_suite_name}>")
        return None, attempted

    dataset_root = (
        client_model.model_config.get("datasets", {})
        .get("vla_data", {})
        .get("data_root_dir", None)
    )
    if dataset_root in (None, "", "null"):
        attempted.append("<datasets.vla_data.data_root_dir missing from checkpoint config>")
        return None, attempted

    candidate = Path(dataset_root) / dataset_dir / "meta" / "padt_task_specs.jsonl"
    attempted.append(str(candidate))
    return (candidate if candidate.exists() else None), attempted


def _load_task_meta_lookup(task_meta_path: Path | None) -> dict[str, dict[Any, TaskMetaRecord]]:
    if task_meta_path is None or not task_meta_path.exists():
        return {"by_index": {}, "by_task": {}}

    by_index: dict[int, TaskMetaRecord] = {}
    by_task: dict[str, TaskMetaRecord] = {}
    with open(task_meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_index = int(record["task_index"])
            task = str(record.get("task", "")).strip()
            task_objects = [str(x).strip() for x in record.get("task_objects", []) if str(x).strip()]
            objects = [str(x).strip() for x in record.get("objects", task_objects) if str(x).strip()]
            object_role = {
                str(k).strip(): str(v).strip()
                for k, v in dict(record.get("object_role", {})).items()
            }
            meta_record = TaskMetaRecord(
                task_index=task_index,
                task=task,
                objects=objects if objects else list(task_objects),
                task_objects=task_objects,
                object_role=object_role,
            )
            by_index[task_index] = meta_record
            if task:
                by_task[_normalize_task_text(task)] = meta_record
    return {"by_index": by_index, "by_task": by_task}


def _resolve_task_meta_record(
    task_meta_lookup: dict[str, dict[Any, TaskMetaRecord]],
    *,
    task_id: int,
    task_description: str,
) -> tuple[TaskMetaRecord | None, str]:
    """Resolve metadata record. Prefers task-string match because jsonl task_index
    order is not guaranteed to align with LIBERO benchmark.get_task(i) order.
    Returns (record, source) where source is "by_task", "by_index", or "missing"."""
    by_index = task_meta_lookup.get("by_index", {})
    by_task = task_meta_lookup.get("by_task", {})

    by_task_hit = by_task.get(_normalize_task_text(task_description))
    if by_task_hit is not None:
        return by_task_hit, "by_task"

    by_index_hit = by_index.get(task_id)
    if by_index_hit is not None:
        if _normalize_task_text(by_index_hit.task) != _normalize_task_text(task_description):
            raise RuntimeError(
                f"Task meta lookup mismatch: jsonl task_index={task_id} has task=\"{by_index_hit.task}\" "
                f"but LIBERO benchmark task_id={task_id} is task=\"{task_description}\". "
                "jsonl task_index order does not match the benchmark; fix the jsonl so task strings "
                "are present (by_task lookup will then find them) or align task_index with benchmark order."
            )
        return by_index_hit, "by_index"

    return None, "missing"


def _inject_task_meta(example_dict: dict[str, Any], task_meta: TaskMetaRecord) -> None:
    role_map = dict(task_meta.object_role)
    task_objects = []
    for obj_idx, object_id in enumerate(task_meta.task_objects):
        task_objects.append(
            {
                "object_id": object_id,
                "label": object_id,
                "object_role": role_map.get(object_id, f"slot_{obj_idx + 1}"),
            }
        )

    objects = []
    for obj_idx, object_id in enumerate(task_meta.objects or task_meta.task_objects):
        objects.append(
            {
                "object_id": object_id,
                "label": object_id,
                "object_role": role_map.get(object_id, f"slot_{obj_idx + 1}"),
            }
        )

    example_dict["task_index"] = int(task_meta.task_index)
    example_dict["task_name"] = task_meta.task
    example_dict["objects"] = objects
    example_dict["task_objects"] = task_objects
    example_dict["object_role"] = role_map


def _save_prediction_overlays(
    *,
    episode_dir: Path,
    step: int,
    agentview: np.ndarray,
    wrist: np.ndarray,
    record: dict[str, Any],
) -> dict[str, str]:
    overlays_dir = episode_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    agent_overlay = _overlay_prediction_array(
        agentview,
        record,
        view_idx=0,
        header_text=f"step {step} | agentview",
    )
    wrist_overlay = _overlay_prediction_array(
        wrist,
        record,
        view_idx=1,
        header_text=f"step {step} | wrist",
    )
    pair = _stitch_overlay_pair(
        [agent_overlay, wrist_overlay],
        title=f"task {record.get('task_id', '?')} ep {record.get('episode_idx', '?')} step {step}",
        labels=["agentview", "wrist"],
    )

    agent_path = overlays_dir / f"step_{step:04d}_agentview_overlay.jpg"
    wrist_path = overlays_dir / f"step_{step:04d}_wrist_overlay.jpg"
    pair_path = overlays_dir / f"step_{step:04d}_pair.jpg"
    agent_overlay.save(agent_path, quality=95)
    wrist_overlay.save(wrist_path, quality=95)
    pair.save(pair_path, quality=95)
    return {
        "agentview_overlay": str(agent_path),
        "wrist_overlay": str(wrist_path),
        "pair": str(pair_path),
    }


def _overlay_prediction_array(
    image_array: np.ndarray,
    record: dict[str, Any],
    *,
    view_idx: int,
    header_text: str,
) -> Image.Image:
    image = Image.fromarray(np.asarray(image_array, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()

    width, height = image.size
    bbox_by_view = np.asarray(record.get("bbox_by_view", []), dtype=np.float32)
    patch_mask_by_view = np.asarray(record.get("patch_mask_by_view", []), dtype=np.float32)
    predicted_patch_ids = np.asarray(record.get("predicted_patch_ids", []), dtype=np.int64)
    object_presence_mask = np.asarray(record.get("object_presence_mask", []), dtype=bool)
    score_logits = np.asarray(record.get("score_logits", []), dtype=np.float32).reshape(-1)
    visibility_logits = np.asarray(record.get("visibility_logits", []), dtype=np.float32)
    task_object_roles = list(record.get("task_object_roles", []))

    if bbox_by_view.ndim == 4:
        bbox_by_view = bbox_by_view[0]
    if patch_mask_by_view.ndim == 4:
        patch_mask_by_view = patch_mask_by_view[0]
    if predicted_patch_ids.ndim == 4:
        predicted_patch_ids = predicted_patch_ids[0]
    if object_presence_mask.ndim == 2:
        object_presence_mask = object_presence_mask[0]
    if visibility_logits.ndim == 3:
        visibility_logits = visibility_logits[0]
    if task_object_roles and isinstance(task_object_roles[0], list):
        task_object_roles = list(task_object_roles[0])

    score_probs = _sigmoid_np(score_logits) if score_logits.size else np.zeros((0,), dtype=np.float32)
    visibility_probs = _sigmoid_np(visibility_logits) if visibility_logits.size else np.zeros((0, 2), dtype=np.float32)

    num_slots = min(4, bbox_by_view.shape[0] if bbox_by_view.ndim >= 3 else 0)
    for slot_idx in range(num_slots):
        is_present = bool(object_presence_mask[slot_idx]) if slot_idx < object_presence_mask.shape[0] else False
        if not is_present:
            continue

        color = OVERLAY_PALETTE[slot_idx % len(OVERLAY_PALETTE)]
        score_prob = float(score_probs[slot_idx]) if slot_idx < len(score_probs) else 0.0
        view_prob = (
            float(visibility_probs[slot_idx, view_idx])
            if visibility_probs.ndim == 2
            and slot_idx < visibility_probs.shape[0]
            and view_idx < visibility_probs.shape[1]
            else 0.0
        )
        bbox = bbox_by_view[slot_idx, view_idx]
        patch_mask = (
            patch_mask_by_view[slot_idx, view_idx]
            if patch_mask_by_view.ndim == 3
            else np.zeros((0,), dtype=np.float32)
        )
        role = task_object_roles[slot_idx] if slot_idx < len(task_object_roles) else f"slot_{slot_idx + 1}"

        if predicted_patch_ids.ndim == 3 and slot_idx < predicted_patch_ids.shape[0] and view_idx < predicted_patch_ids.shape[1]:
            slot_patch_ids = predicted_patch_ids[slot_idx, view_idx]
        elif predicted_patch_ids.ndim == 2 and slot_idx < predicted_patch_ids.shape[0]:
            slot_patch_ids = predicted_patch_ids[slot_idx]
        else:
            slot_patch_ids = np.asarray([], dtype=np.int64)
        view_patch_ids = [
            int(idx)
            for idx in np.asarray(slot_patch_ids).reshape(-1).tolist()
            if int(idx) >= 0
        ]

        if patch_mask.size:
            grid_size = int(round(math.sqrt(patch_mask.size)))
            if grid_size * grid_size == patch_mask.size:
                patch_grid = patch_mask.reshape(grid_size, grid_size)
                flat_probs = patch_grid.reshape(-1)
                selected_patch_ids = set(view_patch_ids)
                top_k = min(6, flat_probs.size)
                selected_patch_ids.update(
                    int(idx)
                    for idx in np.argsort(flat_probs)[-top_k:]
                    if flat_probs[int(idx)] >= 0.45
                )

                for patch_id in sorted(selected_patch_ids):
                    if not (0 <= patch_id < flat_probs.size):
                        continue
                    row_idx = patch_id // grid_size
                    col_idx = patch_id % grid_size
                    prob = float(flat_probs[patch_id])
                    x0 = int(col_idx * width / grid_size)
                    y0 = int(row_idx * height / grid_size)
                    x1 = int((col_idx + 1) * width / grid_size)
                    y1 = int((row_idx + 1) * height / grid_size)

                    alpha = int(min(110, 30 + 110 * prob))
                    draw.rectangle([x0, y0, x1, y1], fill=(*color, alpha))

                    outline_width = 4 if patch_id in view_patch_ids else 2
                    outline_color = (*color, 255) if patch_id in view_patch_ids else (*color, 180)
                    draw.rectangle([x0, y0, x1, y1], outline=outline_color, width=outline_width)

                    if patch_id in view_patch_ids:
                        patch_label = f"p{patch_id}"
                        patch_text_bbox = draw.textbbox((x0 + 2, y0 + 2), patch_label, font=font)
                        draw.rectangle(patch_text_bbox, fill=(0, 0, 0, 180))
                        draw.text((x0 + 2, y0 + 2), patch_label, fill=(255, 255, 255, 255), font=font)

        if bbox.shape[-1] != 4:
            continue
        x0, y0, x1, y1 = [float(x) for x in bbox.tolist()]
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
            x0 *= width
            x1 *= width
            y0 *= height
            y1 *= height
        if not np.all(np.isfinite([x0, y0, x1, y1])):
            continue
        x0, x1 = sorted((max(0.0, min(width - 1.0, x0)), max(0.0, min(width - 1.0, x1))))
        y0, y1 = sorted((max(0.0, min(height - 1.0, y0)), max(0.0, min(height - 1.0, y1))))
        if x1 <= x0 or y1 <= y0:
            continue

        draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=3)
        patch_text = ",".join(str(idx) for idx in view_patch_ids[:5]) if view_patch_ids else "-"
        label = f"{role} s={score_prob:.2f} v={view_prob:.2f} vrt={patch_text}"
        text_bbox = draw.textbbox((int(x0) + 4, int(y0) + 4), label, font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
        draw.text((int(x0) + 4, int(y0) + 4), label, fill=(*color, 255), font=font)

    composed = Image.alpha_composite(image, overlay).convert("RGB")
    return _draw_overlay_header(composed, header_text)


def _draw_overlay_header(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((8, 8), text, font=font)
    padded_bbox = (
        text_bbox[0] - 4,
        text_bbox[1] - 2,
        text_bbox[2] + 4,
        text_bbox[3] + 2,
    )
    draw.rectangle(padded_bbox, fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return image


def _stitch_overlay_pair(images: list[Image.Image], *, title: str, labels: list[str]) -> Image.Image:
    if not images:
        return Image.new("RGB", (512, 256), (20, 20, 20))
    font = ImageFont.load_default()
    title_height = 28
    label_height = 18
    gap = 8
    width = sum(image.width for image in images) + gap * max(0, len(images) - 1)
    height = title_height + label_height + max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    x = 0
    for label, image in zip(labels, images):
        draw.text((x + 6, title_height + 3), label, fill=(255, 255, 255), font=font)
        canvas.paste(image, (x, title_height + label_height))
        x += image.width + gap
    return canvas


def _sigmoid_np(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-value))


def _save_debug_frames(
    *,
    episode_dir: Path | None,
    step: int,
    agentview: np.ndarray,
    wrist: np.ndarray,
) -> dict[str, Path] | None:
    if episode_dir is None:
        return None

    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    agentview_path = frames_dir / f"step_{step:04d}_agentview.jpg"
    wrist_path = frames_dir / f"step_{step:04d}_wrist.jpg"
    imageio.imwrite(agentview_path, np.asarray(agentview, dtype=np.uint8))
    imageio.imwrite(wrist_path, np.asarray(wrist, dtype=np.uint8))
    return {"agentview": agentview_path, "wrist": wrist_path}


def _save_actions(*, episode_dir: Path | None, actions: np.ndarray, enabled: bool) -> dict[str, str | None]:
    if not enabled or episode_dir is None:
        return {"npy": None, "csv": None}

    actions_npy = episode_dir / "actions.npy"
    actions_csv = episode_dir / "actions.csv"
    np.save(actions_npy, actions)
    header = "x,y,z,roll,pitch,yaw,gripper"
    np.savetxt(actions_csv, actions, delimiter=",", header=header, comments="")
    return {"npy": str(actions_npy), "csv": str(actions_csv)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_array_stat(actions: np.ndarray, fn) -> float:
    if actions.size == 0:
        return 0.0
    return float(fn(actions))


def _count_gripper_flips(gripper_values: np.ndarray) -> int:
    if gripper_values.size <= 1:
        return 0
    signs = np.sign(np.asarray(gripper_values, dtype=np.float32))
    return int(np.sum(signs[1:] != signs[:-1]))


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return normalized.lower() or "task"


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        # hard_reset=False: env.reset() only calls sim.reset() (physics data only),
        # avoiding EGL context recreation on every episode which causes mjr_readPixels SIGABRT.
        "hard_reset": False,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


if __name__ == "__main__":
    if _env_flag("DEBUG"):
        start_debugpy_once()
    tyro.cli(eval_libero)
