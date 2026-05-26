#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import io
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SEGMENTATION_TO_VIDEO_KEY = {
    "agentview_bbox_mask": "observation.images.image",
    "wrist_bbox_mask": "observation.images.wrist_image",
}

DEFAULT_DATASET_NAMES = [
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
]

PALETTE = [
    (255, 99, 71),
    (30, 144, 255),
    (60, 179, 113),
    (255, 165, 0),
    (186, 85, 211),
    (255, 215, 0),
    (70, 130, 180),
    (220, 20, 60),
    (0, 191, 255),
    (154, 205, 50),
]


def get_ffmpeg_exe() -> str:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class EpisodeRecord:
    dataset_root: Path
    episode_index: int
    task_text: str
    length: int
    data_chunk_index: int
    data_file_index: int
    video_chunk_indices: dict[str, int]
    video_file_indices: dict[str, int]
    from_timestamps: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize bbox/mask/label overlays for LeRobot LIBERO segmentation."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "playground" / "Datasets" / "LEROBOT_LIBERO_DATA",
        help="Root directory that contains libero_*_lerobot datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASET_NAMES,
        help="Dataset directory names under --data-root.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Single episode index to visualize for every dataset. Ignored when --episode-indices is provided.",
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of episode indices to visualize.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of leading episodes to export when no explicit episode index list is provided.",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=0,
        help="Step index inside the chosen episode. Negative values count from the end. Used as the single step when --step-stride is unset, or as the start step when --step-stride is set.",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=None,
        help="If set, export the whole episode by sampling one frame every N steps.",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="ffmpeg",
        choices=["ffmpeg", "pyav", "opencv"],
        help="Backend used to decode a frame from the source video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "segmentation_debug",
        help="Directory to save visualization outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_segmentation_payload(payload) -> dict[str, dict]:
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported segmentation payload type: {type(payload)}")


def color_for_obj_id(obj_id: str) -> tuple[int, int, int]:
    try:
        idx = int(obj_id)
    except ValueError:
        idx = abs(hash(obj_id))
    return PALETTE[idx % len(PALETTE)]


def decode_mask(mask_info: dict, output_size: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(mask_info, dict):
        return None
    counts = mask_info.get("counts")
    size = mask_info.get("size")
    if counts is None or size is None:
        return None

    rle = {"counts": counts.encode("utf-8") if isinstance(counts, str) else counts, "size": size}
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    decoded = decoded.astype(np.uint8)

    mask_image = Image.fromarray(decoded * 255, mode="L")
    if mask_image.size != output_size:
        mask_image = mask_image.resize(output_size, resample=Image.Resampling.NEAREST)
    return np.array(mask_image) > 0


def denormalize_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    if max(bbox) <= 1.5:
        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height
    return (
        int(round(x0)),
        int(round(y0)),
        int(round(x1)),
        int(round(y1)),
    )


def draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int], font) -> None:
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    padding = 2
    draw.rectangle(
        [left - padding, top - padding, right + padding, bottom + padding],
        fill=(0, 0, 0, 180),
    )
    draw.text((x, y), text, fill=color, font=font)


def overlay_segmentation(frame: np.ndarray, seg: dict[str, dict], title: str) -> Image.Image:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()

    width, height = image.size
    for obj_id, obj in sorted(seg.items(), key=lambda item: int(item[0])):
        mask_info = obj.get("mask", {}) if isinstance(obj, dict) else {}
        label = mask_info.get("label", "unknown")
        color = color_for_obj_id(obj_id)
        rgba = (*color, 80)

        mask = decode_mask(mask_info, (width, height))
        if mask is not None:
            mask_array = np.zeros((height, width, 4), dtype=np.uint8)
            mask_array[mask] = rgba
            mask_image = Image.fromarray(mask_array, mode="RGBA")
            overlay = Image.alpha_composite(overlay, mask_image)
            draw = ImageDraw.Draw(overlay, "RGBA")

        bbox = obj.get("bbox")
        if bbox:
            x0, y0, x1, y1 = denormalize_bbox(bbox, width, height)
            draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=3)
            draw_label(draw, x0 + 4, max(2, y0 + 4), f"Obj_{obj_id}: {label}", color, font)

    composed = Image.alpha_composite(image, overlay).convert("RGB")
    if not title:
        return composed

    title_height = 24
    canvas = Image.new("RGB", (composed.width, composed.height + title_height), (18, 18, 18))
    canvas.paste(composed, (0, title_height))
    title_draw = ImageDraw.Draw(canvas)
    title_draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    return canvas


def build_combined_panel(title: str, images_by_view: list[tuple[str, Image.Image]]) -> Image.Image:
    font = ImageFont.load_default()
    panel_gap = 8
    header_height = 30
    widths = [img.width for _, img in images_by_view]
    heights = [img.height for _, img in images_by_view]
    total_width = sum(widths) + panel_gap * (len(images_by_view) - 1)
    total_height = max(heights) + header_height

    canvas = Image.new("RGB", (total_width, total_height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(255, 255, 255), font=font)

    x = 0
    for view_name, image in images_by_view:
        canvas.paste(image, (x, header_height))
        draw.text((x + 8, header_height + 8), view_name, fill=(255, 255, 255), font=font)
        x += image.width + panel_gap
    return canvas


def load_frame_at_timestamp(video_path: Path, timestamp: float, backend: str) -> np.ndarray:
    if backend == "ffmpeg":
        ffmpeg_exe = get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path.as_posix(),
            "-ss",
            f"{timestamp:.6f}",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0 or not result.stdout:
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()
            raise ValueError(
                f"ffmpeg failed for {video_path} at {timestamp:.4f}s"
                + (f": {stderr}" if stderr else "")
            )
        with Image.open(io.BytesIO(result.stdout)) as image:
            return np.array(image.convert("RGB"))

    if backend == "pyav":
        container = None
        try:
            container = av.open(video_path.as_posix())
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            target_pts = int(timestamp / time_base)
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)

            closest_frame = None
            closest_diff = float("inf")
            for frame in container.decode(video=0):
                current_ts = float(frame.pts * time_base)
                current_diff = abs(current_ts - timestamp)
                if current_diff < closest_diff:
                    closest_diff = current_diff
                    closest_frame = frame
                if current_ts > timestamp and current_diff > closest_diff:
                    break
                if current_ts > timestamp + 1.0:
                    break

            if closest_frame is None:
                raise ValueError(f"Unable to decode frame near {timestamp:.4f}s from {video_path}")
            return closest_frame.to_ndarray(format="rgb24")
        except Exception as exc:
            print(
                f"[WARN] pyav failed for {video_path.name} at {timestamp:.4f}s ({exc}); retrying with ffmpeg.",
                flush=True,
            )
            return load_frame_at_timestamp(video_path, timestamp, "ffmpeg")
        finally:
            if container is not None:
                container.close()
            gc.collect()

    if backend == "opencv":
        cap = cv2.VideoCapture(video_path.as_posix())
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_index = int(round(timestamp * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                print(
                    f"[WARN] opencv failed for {video_path.name} at {timestamp:.4f}s; retrying with ffmpeg.",
                    flush=True,
                )
                return load_frame_at_timestamp(video_path, timestamp, "ffmpeg")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    raise ValueError(f"Unsupported video backend: {backend}")


def resolve_episode_table(dataset_root: Path) -> tuple[list[EpisodeRecord], dict]:
    info = load_json(dataset_root / "meta" / "info.json")
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    episodes: list[EpisodeRecord] = []
    for file_path in episode_files:
        df = pd.read_parquet(file_path)
        timestamp_cols = [
            c for c in df.columns if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
        ]
        chunk_cols = {
            str(c)[len("videos/") : -len("/chunk_index")]: c
            for c in df.columns
            if str(c).startswith("videos/") and str(c).endswith("/chunk_index")
        }
        file_cols = {
            str(c)[len("videos/") : -len("/file_index")]: c
            for c in df.columns
            if str(c).startswith("videos/") and str(c).endswith("/file_index")
        }
        for _, row in df.iterrows():
            from_timestamps = {}
            for col in timestamp_cols:
                value = row[col]
                if pd.isna(value):
                    continue
                video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                from_timestamps[video_key] = float(value)

            video_chunk_indices = {}
            for key, col in chunk_cols.items():
                value = row[col]
                if not pd.isna(value):
                    video_chunk_indices[key] = int(value)

            video_file_indices = {}
            for key, col in file_cols.items():
                value = row[col]
                if not pd.isna(value):
                    video_file_indices[key] = int(value)

            task_value = row.get("tasks", "")
            if isinstance(task_value, list):
                task_text = task_value[0] if task_value else ""
            else:
                task_text = str(task_value)

            episodes.append(
                EpisodeRecord(
                    dataset_root=dataset_root,
                    episode_index=int(row["episode_index"]),
                    task_text=task_text,
                    length=int(row["length"]),
                    data_chunk_index=int(row["data/chunk_index"]),
                    data_file_index=int(row["data/file_index"]),
                    video_chunk_indices=video_chunk_indices,
                    video_file_indices=video_file_indices,
                    from_timestamps=from_timestamps,
                )
            )
    return episodes, info


def choose_episode(episodes: list[EpisodeRecord], episode_index: int | None) -> EpisodeRecord:
    if not episodes:
        raise ValueError("No episodes found.")
    if episode_index is None:
        return episodes[0]
    for episode in episodes:
        if episode.episode_index == episode_index:
            return episode
    raise ValueError(f"Episode index {episode_index} not found.")


def choose_episodes(
    episodes: list[EpisodeRecord],
    episode_indices: list[int] | None,
    episode_index: int | None,
    num_episodes: int,
) -> list[EpisodeRecord]:
    if not episodes:
        raise ValueError("No episodes found.")
    if episode_indices:
        wanted = set(episode_indices)
        selected = [episode for episode in episodes if episode.episode_index in wanted]
        missing = sorted(wanted - {episode.episode_index for episode in selected})
        if missing:
            raise ValueError(f"Episode indices not found: {missing}")
        return selected
    if episode_index is not None:
        return [choose_episode(episodes, episode_index)]
    return episodes[: max(1, num_episodes)]


def resolve_step_index(step_index: int, episode_length: int) -> int:
    if step_index < 0:
        step_index = episode_length + step_index
    if not (0 <= step_index < episode_length):
        raise IndexError(f"step_index {step_index} is out of range for episode length {episode_length}.")
    return step_index


def resolve_step_indices(step_index: int, step_stride: int | None, episode_length: int) -> list[int]:
    if step_stride is None:
        return [resolve_step_index(step_index, episode_length)]
    if step_stride <= 0:
        raise ValueError(f"step_stride must be positive, got {step_stride}")

    start_step = step_index
    if start_step < 0:
        start_step = episode_length + start_step
    if start_step < 0 or start_step >= episode_length:
        raise IndexError(f"step_index {step_index} is out of range for episode length {episode_length}.")

    step_indices = list(range(start_step, episode_length, step_stride))
    last_step = episode_length - 1
    if step_indices[-1] != last_step:
        step_indices.append(last_step)
    return step_indices


def get_episode_rows(dataset_root: Path, info: dict, episode: EpisodeRecord) -> pd.DataFrame:
    data_path = dataset_root / info["data_path"].format(
        chunk_index=episode.data_chunk_index,
        file_index=episode.data_file_index,
    )
    df = pd.read_parquet(data_path)
    return df.loc[df["episode_index"] == episode.episode_index].copy()


def resolve_video_path(dataset_root: Path, info: dict, episode: EpisodeRecord, video_key: str) -> Path:
    chunk_index = episode.video_chunk_indices.get(video_key, episode.data_chunk_index)
    file_index = episode.video_file_indices.get(video_key, episode.data_file_index)
    return dataset_root / info["video_path"].format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def render_dataset_episode(
    dataset_root: Path,
    episode: EpisodeRecord,
    step_indices: list[int],
    video_backend: str,
    output_dir: Path,
) -> None:
    _, info = resolve_episode_table(dataset_root)
    rows = get_episode_rows(dataset_root, info, episode)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_dir = output_dir / dataset_root.name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    for step in step_indices:
        row = rows.iloc[step]
        saved_images: list[tuple[str, Image.Image]] = []
        summary = {
            "dataset": dataset_root.name,
            "episode_index": episode.episode_index,
            "task": episode.task_text,
            "step_index": step,
            "timestamp": float(row["timestamp"]),
            "views": {},
        }

        for seg_column, video_key in SEGMENTATION_TO_VIDEO_KEY.items():
            full_seg_column = f"segmentation.{seg_column}"
            if full_seg_column not in row.index:
                continue
            seg_payload = row[full_seg_column]
            if pd.isna(seg_payload) if not isinstance(seg_payload, str) else False:
                continue
            if not isinstance(seg_payload, (str, dict)):
                continue

            seg = parse_segmentation_payload(seg_payload)
            if not seg:
                continue

            video_path = resolve_video_path(dataset_root, info, episode, video_key)
            video_timestamp = float(row["timestamp"]) + float(episode.from_timestamps.get(video_key, 0.0))
            frame = load_frame_at_timestamp(video_path, video_timestamp, video_backend)
            if frame.ndim != 3:
                raise ValueError(f"Unexpected frame shape for {video_path}: {frame.shape}")

            view_title = f"{seg_column} | ts={video_timestamp:.3f}s"
            rendered = overlay_segmentation(frame, seg, view_title)
            image_path = dataset_output_dir / f"episode_{episode.episode_index:04d}_step_{step:04d}_{seg_column}.png"
            rendered.save(image_path)
            saved_images.append((seg_column, rendered))
            summary["views"][seg_column] = {
                "video_key": video_key,
                "video_path": str(video_path),
                "image_path": str(image_path),
                "video_timestamp": video_timestamp,
                "objects": {
                    obj_id: {
                        "label": obj.get("mask", {}).get("label", "unknown"),
                        "bbox": obj.get("bbox"),
                    }
                    for obj_id, obj in seg.items()
                },
            }

        if not saved_images:
            raise RuntimeError(
                f"No segmentation visualizations were produced for {dataset_root.name} episode {episode.episode_index} step {step}."
            )

        combined_title = (
            f"{dataset_root.name} | episode={episode.episode_index} | step={step} | task={episode.task_text}"
        )
        combined = build_combined_panel(combined_title, saved_images)
        combined_path = dataset_output_dir / f"episode_{episode.episode_index:04d}_step_{step:04d}_combined.png"
        combined.save(combined_path)
        summary["combined_image_path"] = str(combined_path)

        summary_path = dataset_output_dir / f"episode_{episode.episode_index:04d}_step_{step:04d}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[OK] {dataset_root.name}: saved {combined_path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        dataset_root = (args.data_root / dataset_name).resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        episodes, _ = resolve_episode_table(dataset_root)
        selected_episodes = choose_episodes(
            episodes=episodes,
            episode_indices=args.episode_indices,
            episode_index=args.episode_index,
            num_episodes=args.num_episodes,
        )
        for episode in selected_episodes:
            step_indices = resolve_step_indices(
                step_index=args.step_index,
                step_stride=args.step_stride,
                episode_length=episode.length,
            )
            render_dataset_episode(
                dataset_root=dataset_root,
                episode=episode,
                step_indices=step_indices,
                video_backend=args.video_backend,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
