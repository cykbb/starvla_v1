from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_VIEW_FIELD_MAP = {
    "agentview": "segmentation.agentview_bbox_mask",
    "wrist": "segmentation.wrist_bbox_mask",
}


@dataclass
class PaDTTaskSpec:
    task_index: int
    task: str
    objects: list[str]
    task_objects: list[str]
    object_role: dict[str, str]


class PaDTSegmentationSourceAdapter:
    """Build PaDT supervision directly from step-level segmentation columns.

    This adapter is designed for LeRobot-style parquet rows where each step already
    stores segmentation JSON strings such as:
        segmentation.agentview_bbox_mask
        segmentation.wrist_bbox_mask

    Dataset-level task metadata is loaded once from a compact JSONL sidecar stored
    under `meta/`, so repeated per-step fields such as task_objects / object_role /
    objects do not need to be duplicated into every parquet row.
    """

    def __init__(
        self,
        dataset_path: Path,
        tasks_df: pd.DataFrame | None,
        data_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        cfg = dict(data_cfg or {})
        self.dataset_path = Path(dataset_path)
        self.enabled = bool(cfg.get("padt_use_segmentation_source", False))
        self.required = bool(cfg.get("padt_task_meta_required", self.enabled))
        self.view_field_map = _normalize_view_field_map(cfg.get("padt_segmentation_fields", None))
        self.patch_grid_size = int(cfg.get("padt_patch_grid_size", 16))
        self.valid_threshold = float(cfg.get("padt_valid_patch_threshold", 0.30))
        self.num_core_patches = int(cfg.get("padt_num_core_patches", 3))
        self.agentview_name = str(cfg.get("padt_agentview_name", "agentview"))
        self.prefer_task_objects = bool(cfg.get("padt_prefer_task_objects", True))

        task_meta_path = cfg.get("padt_task_meta_path", None)
        if task_meta_path in (None, "", "null"):
            task_meta_path = self.dataset_path / cfg.get("padt_task_meta_filename", "meta/padt_task_specs.jsonl")
        self.task_meta_path = Path(task_meta_path)

        self.task_text_by_index = _build_task_text_lookup(tasks_df)
        self.task_specs_by_index = self._load_task_specs(self.task_meta_path)

    def _load_task_specs(self, path: Path) -> dict[int, PaDTTaskSpec]:
        if not path.exists():
            if self.required:
                raise FileNotFoundError(
                    f"PaDT task meta file not found: {path}. "
                    "Add one compact task-level sidecar JSONL under meta/ and keep step-level "
                    "segmentation inside parquet."
                )
            return {}

        records: dict[int, PaDTTaskSpec] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task_index" not in record:
                    raise KeyError(f"{path}:{line_idx} missing required field `task_index`")
                task_index = int(record["task_index"])
                task_text = str(record.get("task", self.task_text_by_index.get(task_index, "")))
                task_objects = [str(x).strip() for x in _ensure_list(record.get("task_objects", [])) if str(x).strip()]
                objects = [str(x).strip() for x in _ensure_list(record.get("objects", [])) if str(x).strip()]
                role_map = _normalize_role_map(record.get("object_role", {}), task_objects)
                if not objects:
                    objects = list(task_objects)
                if not task_objects and self.required:
                    raise ValueError(f"{path}:{line_idx} has empty `task_objects` for task_index={task_index}")
                records[task_index] = PaDTTaskSpec(
                    task_index=task_index,
                    task=task_text,
                    objects=objects,
                    task_objects=task_objects,
                    object_role=role_map,
                )
        return records

    def enrich_sample(self, sample: dict[str, Any], step_row: Any) -> dict[str, Any]:
        if not self.enabled or step_row is None:
            return sample

        task_index = _extract_task_index(step_row)
        task_spec = self.task_specs_by_index.get(task_index, None)
        if task_spec is None:
            if self.required:
                raise KeyError(
                    f"Missing task-level PaDT spec for task_index={task_index} in {self.task_meta_path}"
                )
            return sample

        parsed_by_view = {
            view_name: _parse_segmentation_payload(_row_get(step_row, field_name))
            for view_name, field_name in self.view_field_map.items()
        }

        visible_labels = sorted({label for payload in parsed_by_view.values() for label in payload.keys()})
        ordered_objects = list(dict.fromkeys(task_spec.objects if task_spec.objects else visible_labels))
        if self.prefer_task_objects:
            ordered_objects = list(dict.fromkeys(task_spec.task_objects + [x for x in ordered_objects if x not in task_spec.task_objects]))

        objects: list[dict[str, Any]] = []
        for label in ordered_objects:
            object_record: dict[str, Any] = {
                "object_id": label,
                "label": label,
                "bbox_by_view": {},
                "patch_mask_by_view": {},
                "visible_by_view": {},
                "valid_patch_ids": [],
                "core_patch_ids": [],
                "valid_patch_ids_by_view": {},
                "core_patch_ids_by_view": {},
            }

            agent_mask = None
            for view_name in self.view_field_map:
                view_entry = parsed_by_view.get(view_name, {}).get(label, None)
                visible = view_entry is not None
                object_record["visible_by_view"][view_name] = bool(visible)
                if view_entry is None:
                    object_record["bbox_by_view"][view_name] = [0.0, 0.0, 0.0, 0.0]
                    object_record["patch_mask_by_view"][view_name] = [0.0] * (self.patch_grid_size * self.patch_grid_size)
                    object_record["valid_patch_ids_by_view"][view_name] = []
                    object_record["core_patch_ids_by_view"][view_name] = []
                    continue

                object_record["bbox_by_view"][view_name] = [float(x) for x in view_entry["bbox"]]
                dense_mask = view_entry.get("mask_array", None)
                coverage = _mask_to_patch_coverage(dense_mask, self.patch_grid_size)
                object_record["patch_mask_by_view"][view_name] = coverage.reshape(-1).astype(np.float32).tolist()
                valid_ids, core_ids = _compute_patch_ids(
                    dense_mask,
                    patch_grid_size=self.patch_grid_size,
                    valid_threshold=self.valid_threshold,
                    num_core_patches=self.num_core_patches,
                )
                object_record["valid_patch_ids_by_view"][view_name] = valid_ids
                object_record["core_patch_ids_by_view"][view_name] = core_ids
                if view_name == self.agentview_name:
                    agent_mask = dense_mask

            valid_ids, core_ids = _compute_patch_ids(
                agent_mask,
                patch_grid_size=self.patch_grid_size,
                valid_threshold=self.valid_threshold,
                num_core_patches=self.num_core_patches,
            )
            object_record["valid_patch_ids"] = valid_ids
            object_record["core_patch_ids"] = core_ids
            objects.append(object_record)

        role_map = dict(task_spec.object_role)
        task_objects: list[dict[str, Any]] = []
        for obj_label in task_spec.task_objects:
            task_objects.append(
                {
                    "object_id": obj_label,
                    "label": obj_label,
                    "object_role": role_map.get(obj_label, "unknown"),
                }
            )

        sample["objects"] = objects
        sample["task_objects"] = task_objects
        sample["object_role"] = role_map
        sample["task_index"] = task_index
        sample["task_name"] = task_spec.task or self.task_text_by_index.get(task_index, "")
        sample["__padt_source__"] = "segmentation_meta"
        return sample


def _normalize_view_field_map(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping) and value:
        return {str(k): str(v) for k, v in value.items()}
    return dict(DEFAULT_VIEW_FIELD_MAP)


def _build_task_text_lookup(tasks_df: pd.DataFrame | None) -> dict[int, str]:
    if tasks_df is None or len(tasks_df) == 0:
        return {}

    df = tasks_df.copy()
    if "task_index" not in df.columns:
        if df.index.name == "task_index":
            df = df.reset_index()
        elif "index" in df.columns:
            df = df.rename(columns={"index": "task_index"})
        else:
            df = df.reset_index().rename(columns={df.reset_index().columns[0]: "task_index"})

    if "task" not in df.columns:
        candidate_columns = [col for col in df.columns if col != "task_index"]
        if candidate_columns:
            df = df.rename(columns={candidate_columns[0]: "task"})
        else:
            return {}

    lookup: dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            lookup[int(row["task_index"])] = str(row["task"])
        except Exception:
            continue
    return lookup


def _normalize_role_map(value: Any, task_objects: Sequence[str]) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(k).strip(): str(v).strip() for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        role_map: dict[str, str] = {}
        for object_name, role_name in zip(task_objects, value):
            role_map[str(object_name).strip()] = str(role_name).strip()
        return role_map
    return {str(obj).strip(): f"slot_{idx + 1}" for idx, obj in enumerate(task_objects)}


def _extract_task_index(step_row: Any) -> int:
    candidates = [
        _row_get(step_row, "task_index"),
        _row_get(step_row, "task"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (int, np.integer)):
            return int(candidate)
        if isinstance(candidate, float) and not math.isnan(candidate):
            return int(candidate)
        if isinstance(candidate, str) and candidate.strip().isdigit():
            return int(candidate.strip())
    raise KeyError("Unable to infer task_index from step parquet row")


def _row_get(step_row: Any, key: str, default: Any = None) -> Any:
    if isinstance(step_row, pd.Series):
        return step_row.get(key, default)
    if isinstance(step_row, Mapping):
        return step_row.get(key, default)
    return getattr(step_row, key, default)


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _parse_segmentation_payload(raw_payload: Any) -> dict[str, dict[str, Any]]:
    if raw_payload is None:
        return {}
    if isinstance(raw_payload, float) and math.isnan(raw_payload):
        return {}
    if isinstance(raw_payload, str):
        raw_payload = raw_payload.strip()
        if raw_payload in {"", "{}", "null", "None"}:
            return {}
        payload = json.loads(raw_payload)
    elif isinstance(raw_payload, Mapping):
        payload = dict(raw_payload)
    else:
        return {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, value in payload.items():
        if not isinstance(value, Mapping):
            continue
        # Label may be at top level OR nested inside value["mask"]["label"]
        label = str(value.get("label", "")).strip()
        if not label:
            label = str(value.get("mask", {}).get("label", "") if isinstance(value.get("mask"), Mapping) else "").strip()
        if not label:
            continue
        grouped.setdefault(label, []).append(dict(value))

    aggregated: dict[str, dict[str, Any]] = {}
    for label, entries in grouped.items():
        aggregated[label] = _aggregate_label_entries(entries, label)
    return aggregated


def _aggregate_label_entries(entries: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    dense_masks: list[np.ndarray] = []
    bbox_union: list[float] | None = None

    for entry in entries:
        bbox = entry.get("bbox", None)
        if bbox is not None:
            bbox = [float(x) for x in list(bbox)[:4]]
            if bbox_union is None:
                bbox_union = bbox
            else:
                bbox_union = [
                    min(bbox_union[0], bbox[0]),
                    min(bbox_union[1], bbox[1]),
                    max(bbox_union[2], bbox[2]),
                    max(bbox_union[3], bbox[3]),
                ]
        dense = _decode_mask(entry.get("mask", None))
        if dense is not None:
            dense_masks.append(dense.astype(bool))

    union_mask: np.ndarray | None = None
    if dense_masks:
        union_mask = np.logical_or.reduce(dense_masks).astype(np.uint8)
        if bbox_union is None:
            bbox_union = _bbox_from_dense_mask(union_mask)

    if bbox_union is None:
        bbox_union = [0.0, 0.0, 0.0, 0.0]

    return {
        "label": label,
        "bbox": bbox_union,
        "mask_array": union_mask,
    }


def _bbox_from_dense_mask(mask: np.ndarray) -> list[float]:
    if mask is None or mask.sum() <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    return [
        float(xs.min()) / float(w),
        float(ys.min()) / float(h),
        float(xs.max() + 1) / float(w),
        float(ys.max() + 1) / float(h),
    ]


def _decode_mask(mask_payload: Any) -> np.ndarray | None:
    if mask_payload is None:
        return None
    if isinstance(mask_payload, np.ndarray):
        return mask_payload.astype(np.uint8)
    if isinstance(mask_payload, list):
        arr = np.asarray(mask_payload, dtype=np.uint8)
        if arr.ndim == 2:
            return arr
        return None
    if not isinstance(mask_payload, Mapping):
        return None

    size = mask_payload.get("size", None)
    counts = mask_payload.get("counts", None)
    if size is None or counts is None:
        return None
    height, width = int(size[0]), int(size[1])

    if isinstance(counts, str):
        counts = _decode_coco_rle_counts(counts)
    else:
        counts = [int(x) for x in counts]

    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    value = 0
    for run in counts:
        run = int(run)
        if run <= 0:
            continue
        if idx >= total:
            break
        end = min(idx + run, total)
        if value == 1:
            flat[idx:end] = 1
        idx = end
        value = 1 - value
    return flat.reshape((height, width), order="F")


def _decode_coco_rle_counts(encoded: str) -> list[int]:
    counts: list[int] = []
    pos = 0
    idx = 0
    while pos < len(encoded):
        shift = 0
        value = 0
        more = 1
        while more:
            char_code = ord(encoded[pos]) - 48
            value |= (char_code & 0x1F) << (5 * shift)
            more = char_code & 0x20
            pos += 1
            shift += 1
            if not more and (char_code & 0x10):
                value |= -1 << (5 * shift)
        if idx > 2:
            value += counts[idx - 2]
        counts.append(int(value))
        idx += 1
    return counts


def _mask_to_patch_coverage(mask: np.ndarray | None, patch_grid_size: int) -> np.ndarray:
    coverage = np.zeros((patch_grid_size, patch_grid_size), dtype=np.float32)
    if mask is None:
        return coverage

    dense = np.asarray(mask, dtype=np.float32)
    if dense.ndim != 2 or dense.size == 0:
        return coverage

    height, width = dense.shape
    y_edges = np.linspace(0, height, patch_grid_size + 1, dtype=int)
    x_edges = np.linspace(0, width, patch_grid_size + 1, dtype=int)

    for row in range(patch_grid_size):
        for col in range(patch_grid_size):
            patch = dense[y_edges[row] : y_edges[row + 1], x_edges[col] : x_edges[col + 1]]
            if patch.size > 0:
                coverage[row, col] = float(patch.mean())
    return coverage


def _compute_patch_ids(
    mask: np.ndarray | None,
    patch_grid_size: int,
    valid_threshold: float,
    num_core_patches: int,
) -> tuple[list[int], list[int]]:
    coverage = _mask_to_patch_coverage(mask, patch_grid_size)
    valid = coverage >= float(valid_threshold)
    if not valid.any() and coverage.max() > 0:
        valid[np.unravel_index(int(np.argmax(coverage)), coverage.shape)] = True

    valid_coords = np.argwhere(valid)
    valid_ids = [int(r * patch_grid_size + c) for r, c in valid_coords]
    if not valid_ids:
        return [], []

    eroded = _binary_erode(valid)
    core_region = eroded if eroded.any() else valid
    core_coords = np.argwhere(core_region)
    core_ids = _farthest_point_sample(core_coords, coverage, patch_grid_size, num_core_patches)
    if len(core_ids) < num_core_patches:
        fallback_ids = _farthest_point_sample(valid_coords, coverage, patch_grid_size, num_core_patches)
        for patch_id in fallback_ids:
            if patch_id not in core_ids:
                core_ids.append(patch_id)
            if len(core_ids) >= num_core_patches:
                break
    if core_ids and len(core_ids) < num_core_patches:
        core_ids.extend([core_ids[-1]] * (num_core_patches - len(core_ids)))
    return valid_ids, core_ids[:num_core_patches]


def _binary_erode(binary_mask: np.ndarray) -> np.ndarray:
    if binary_mask.ndim != 2:
        return binary_mask
    height, width = binary_mask.shape
    eroded = np.zeros_like(binary_mask, dtype=bool)
    for row in range(height):
        for col in range(width):
            if not binary_mask[row, col]:
                continue
            row0 = max(0, row - 1)
            row1 = min(height, row + 2)
            col0 = max(0, col - 1)
            col1 = min(width, col + 2)
            neighborhood = binary_mask[row0:row1, col0:col1]
            if neighborhood.shape == (3, 3) and neighborhood.all():
                eroded[row, col] = True
    return eroded


def _farthest_point_sample(
    coords: np.ndarray,
    coverage: np.ndarray,
    patch_grid_size: int,
    num_samples: int,
) -> list[int]:
    if coords.size == 0:
        return []
    points = np.asarray(coords, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return []

    center = points.mean(axis=0)
    best_start = None
    best_score = -float("inf")
    for idx, (row, col) in enumerate(points.astype(int)):
        score = float(coverage[row, col]) - 0.01 * float(np.linalg.norm(points[idx] - center))
        if score > best_score:
            best_start = idx
            best_score = score

    assert best_start is not None
    chosen = [best_start]
    min_dist = np.full(len(points), np.inf, dtype=np.float32)
    for _ in range(1, min(num_samples, len(points))):
        last_point = points[chosen[-1]]
        dist = np.sum((points - last_point) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[chosen] = -1.0
        next_idx = int(np.argmax(min_dist))
        if min_dist[next_idx] < 0:
            break
        chosen.append(next_idx)

    return [int(points[idx][0] * patch_grid_size + points[idx][1]) for idx in chosen]


def build_task_meta_scaffold(tasks_parquet: Path | str, output_jsonl: Path | str) -> Path:
    """Create a minimal task-level JSONL scaffold from meta/tasks.parquet.

    This utility is used by the packaging script so each dataset only stores
    objects / task_objects / object_role once under meta/.
    """
    tasks_df = pd.read_parquet(tasks_parquet)
    task_lookup = _build_task_text_lookup(tasks_df)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for task_index in sorted(task_lookup.keys()):
            record = {
                "task_index": int(task_index),
                "task": task_lookup[task_index],
                "objects": [],
                "task_objects": [],
                "object_role": {},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path
