# Copyright 2026 OpenAI.
# Minimal PaDT-QwenPI integration utilities for starVLA.
#
# This file intentionally keeps all PaDT-specific raw-dict normalization in one
# place so the baseline QwenPI / dataloader code paths stay untouched.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch


PADT_VIEW_NAMES = ("agentview", "wrist")
PADT_DEFAULT_NUM_VRT = 256
PADT_DEFAULT_NUM_CORE = 5
PADT_DEFAULT_MAX_TASK_OBJECTS = 4
PADT_IGNORE_INDEX = -100


@dataclass
class PaDTRawBatch:
    """Canonicalized batch used by QwenPaDTPI.

    Notes:
        - `target_valid_patch_mask` is per view and sized to the VRT bank.
        - `target_patch_masks_by_view` keeps decoder supervision for agentview+wrist.
    """

    images: List[List[Any]]
    instructions: List[str]
    actions: Optional[torch.Tensor]
    state: Optional[torch.Tensor]
    task_object_ids: List[List[str]]
    task_object_roles: List[List[str]]
    objects: List[List[Dict[str, Any]]]
    object_lookup: List[Dict[str, Dict[str, Any]]]
    target_core_patch_ids: torch.Tensor
    target_valid_patch_mask: torch.Tensor
    target_patch_masks_by_view: torch.Tensor
    target_boxes_by_view: torch.Tensor
    target_visible_by_view: torch.Tensor
    object_presence_mask: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def _get_cfg_value(config: Any, path: str, default: Any) -> Any:
    cur = config
    for key in path.split("."):
        if cur is None:
            return default
        if hasattr(cur, "get"):
            cur = cur.get(key, default if key == path.split(".")[-1] else None)
        else:
            cur = getattr(cur, key, default if key == path.split(".")[-1] else None)
        if cur is default:
            return default
    return cur


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_tensor(x: Any, *, dtype: torch.dtype, device: Optional[torch.device] = None) -> torch.Tensor:
    if x is None:
        raise ValueError("Cannot convert None to tensor")
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype) if device is not None else x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _bool_from_any(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y", "t"}
    return bool(x)


def _normalize_view_images(image_field: Any, view_names: Sequence[str]) -> Tuple[List[Any], Dict[str, Any]]:
    """Normalize raw image field to a fixed [agentview, wrist] list.

    Accepted input shapes:
        1) list/tuple in the expected order
        2) dict keyed by view name
    """
    ordered: List[Any] = []
    view_map: Dict[str, Any] = {}

    if isinstance(image_field, Mapping):
        view_map = dict(image_field)
        for view_name in view_names:
            ordered.append(view_map.get(view_name, None))
        return ordered, view_map

    image_list = _ensure_list(image_field)
    for idx, view_name in enumerate(view_names):
        value = image_list[idx] if idx < len(image_list) else None
        ordered.append(value)
        view_map[view_name] = value
    return ordered, view_map


def _normalize_object_id(obj: Any, fallback_index: int) -> str:
    if isinstance(obj, Mapping):
        for key in ("object_id", "id", "name", "label"):
            if key in obj and obj[key] is not None:
                return str(obj[key])
    if obj is None:
        return f"object_{fallback_index}"
    return str(obj)


def _lookup_object(sample: Mapping[str, Any], object_id: str) -> Dict[str, Any]:
    objects = _ensure_list(sample.get("objects", []))
    lookup: Dict[str, Dict[str, Any]] = {}
    for obj_idx, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            continue
        lookup[_normalize_object_id(obj, obj_idx)] = dict(obj)
    return lookup.get(object_id, {})


def _sample_level_lookup(sample: Mapping[str, Any], field_name: str, object_id: str, default: Any) -> Any:
    field = sample.get(field_name, None)
    if isinstance(field, Mapping):
        if object_id in field:
            return field[object_id]
        # also support integer-style keys serialized as strings
        for key in (str(object_id), int(object_id) if str(object_id).isdigit() else None):
            if key is not None and key in field:
                return field[key]
    return default


def _object_field(sample: Mapping[str, Any], object_id: str, field_name: str, default: Any) -> Any:
    object_dict = _lookup_object(sample, object_id)
    if field_name in object_dict:
        return object_dict[field_name]
    return _sample_level_lookup(sample, field_name, object_id, default)


def patch_ids_to_mask(patch_ids: Any, num_patches: int = PADT_DEFAULT_NUM_VRT) -> torch.Tensor:
    mask = torch.zeros(num_patches, dtype=torch.float32)
    if patch_ids is None:
        return mask
    patch_ids = _ensure_list(patch_ids)
    for patch_id in patch_ids:
        if patch_id is None:
            continue
        patch_id_int = int(patch_id)
        if 0 <= patch_id_int < num_patches:
            mask[patch_id_int] = 1.0
    return mask


def _patch_mask_to_tensor(mask: Any, num_patches: int) -> torch.Tensor:
    if mask is None:
        return torch.zeros(num_patches, dtype=torch.float32)
    if torch.is_tensor(mask):
        flat = mask.float().reshape(-1)
    else:
        flat = torch.as_tensor(np.asarray(mask), dtype=torch.float32).reshape(-1)
    if flat.numel() == num_patches:
        return flat
    # Common case: [16, 16] grid for 256 merged patches.
    if int(np.sqrt(num_patches)) ** 2 == num_patches and flat.numel() == num_patches:
        return flat
    if flat.numel() > num_patches:
        return flat[:num_patches]
    padded = torch.zeros(num_patches, dtype=torch.float32)
    padded[: flat.numel()] = flat
    return padded


def _bbox_to_tensor(bbox: Any) -> torch.Tensor:
    if bbox is None:
        return torch.zeros(4, dtype=torch.float32)
    bbox_arr = torch.as_tensor(np.asarray(bbox), dtype=torch.float32).reshape(-1)
    if bbox_arr.numel() < 4:
        out = torch.zeros(4, dtype=torch.float32)
        out[: bbox_arr.numel()] = bbox_arr
        return out
    return bbox_arr[:4]


def _visible_value(visible: Any, bbox_tensor: torch.Tensor, patch_mask: torch.Tensor) -> bool:
    if visible is not None:
        return _bool_from_any(visible)
    if patch_mask.sum().item() > 0:
        return True
    return bbox_tensor.abs().sum().item() > 0


def _resolve_view_payload(view_payload: Any, view_name: str, default: Any) -> Any:
    if isinstance(view_payload, Mapping):
        if view_name in view_payload:
            return view_payload[view_name]
        # common fallback aliases
        alias_map = {
            "agentview": ("agent", "front", "image_0", "view0"),
            "wrist": ("hand", "eye_in_hand", "image_1", "view1"),
        }
        for alias in alias_map.get(view_name, ()):
            if alias in view_payload:
                return view_payload[alias]
    elif isinstance(view_payload, (list, tuple)):
        view_idx = PADT_VIEW_NAMES.index(view_name) if view_name in PADT_VIEW_NAMES else 0
        if view_idx < len(view_payload):
            return view_payload[view_idx]
    return default


def preprocess_raw_dict(
    examples: List[Mapping[str, Any]],
    config: Any,
    device: Optional[torch.device] = None,
) -> PaDTRawBatch:
    """Convert starVLA raw dict samples to the canonical PaDT batch.

    Required raw-dict additions for this experiment:
        image, lang, action, state(optional), objects, task_objects, object_role,
        bbox_by_view, patch_mask_by_view (or mask_by_view convertible to patches),
        valid_patch_ids, core_patch_ids, visible_by_view.
    """

    if not examples:
        raise ValueError("QwenPaDTPI received an empty batch")

    padt_cfg = _get_cfg_value(config, "framework.padt", None)
    view_names = tuple(_get_cfg_value(padt_cfg, "view_names", PADT_VIEW_NAMES))
    num_vrt_tokens = int(_get_cfg_value(padt_cfg, "num_vrt_tokens", PADT_DEFAULT_NUM_VRT))
    num_core_tokens = int(_get_cfg_value(padt_cfg, "num_core_vrt_tokens", PADT_DEFAULT_NUM_CORE))
    max_task_objects = int(_get_cfg_value(padt_cfg, "max_task_objects", PADT_DEFAULT_MAX_TASK_OBJECTS))

    images: List[List[Any]] = []
    instructions: List[str] = []
    actions_list: List[torch.Tensor] = []
    state_list: List[torch.Tensor] = []
    task_object_ids: List[List[str]] = []
    task_object_roles: List[List[str]] = []
    objects: List[List[Dict[str, Any]]] = []
    object_lookup: List[Dict[str, Dict[str, Any]]] = []

    batch_size = len(examples)
    num_views = len(view_names)

    target_core_patch_ids = torch.full(
        (batch_size, max_task_objects, num_views, num_core_tokens),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )
    target_valid_patch_mask = torch.zeros(
        (batch_size, max_task_objects, num_views, num_vrt_tokens),
        dtype=torch.float32,
        device=device,
    )
    target_patch_masks_by_view = torch.zeros(
        (batch_size, max_task_objects, num_views, num_vrt_tokens),
        dtype=torch.float32,
        device=device,
    )
    target_boxes_by_view = torch.zeros(
        (batch_size, max_task_objects, num_views, 4),
        dtype=torch.float32,
        device=device,
    )
    target_visible_by_view = torch.zeros(
        (batch_size, max_task_objects, num_views),
        dtype=torch.bool,
        device=device,
    )
    object_presence_mask = torch.zeros(
        (batch_size, max_task_objects),
        dtype=torch.bool,
        device=device,
    )

    for batch_idx, sample in enumerate(examples):
        if "image" not in sample:
            raise KeyError("QwenPaDTPI expects raw dict field `image`")
        if "lang" not in sample:
            raise KeyError("QwenPaDTPI expects raw dict field `lang`")
        ordered_images, _ = _normalize_view_images(sample["image"], view_names)
        images.append(ordered_images)
        instructions.append(str(sample["lang"]))
        if "action" in sample and sample["action"] is not None:
            actions_list.append(_as_tensor(sample["action"], dtype=torch.float32, device=device))

        if "state" in sample and sample["state"] is not None:
            state_list.append(_as_tensor(sample["state"], dtype=torch.float32, device=device))

        object_entries = [dict(obj) if isinstance(obj, Mapping) else {"object_id": str(obj)} for obj in _ensure_list(sample.get("objects", []))]
        objects.append(object_entries)
        lookup = {_normalize_object_id(obj, obj_idx): obj for obj_idx, obj in enumerate(object_entries)}
        object_lookup.append(lookup)

        raw_task_objects = _ensure_list(sample.get("task_objects", []))
        if not raw_task_objects:
            # Inference mode: no segmentation annotations available.
            # Activate all slots with default roles so the decoder can process
            # whatever VRT tokens the model generates autoregressively.
            raw_task_objects = [
                {"object_id": f"slot_{i+1}", "object_role": f"slot_{i+1}"}
                for i in range(max_task_objects)
            ]

        sample_object_ids: List[str] = []
        sample_roles: List[str] = []
        for obj_slot, raw_obj in enumerate(raw_task_objects[:max_task_objects]):
            object_id = _normalize_object_id(raw_obj, obj_slot)
            sample_object_ids.append(object_id)

            role_value = None
            if isinstance(raw_obj, Mapping):
                role_value = raw_obj.get("object_role", raw_obj.get("role", None))
            if role_value is None:
                role_map = sample.get("object_role", {})
                if isinstance(role_map, Mapping):
                    role_value = role_map.get(object_id, role_map.get(str(obj_slot), None))
            sample_roles.append(str(role_value if role_value is not None else f"slot_{obj_slot+1}"))

            object_presence_mask[batch_idx, obj_slot] = True

            valid_patch_ids_by_view = _object_field(sample, object_id, "valid_patch_ids_by_view", None)
            core_patch_ids_by_view = _object_field(sample, object_id, "core_patch_ids_by_view", None)
            legacy_valid_patch_ids = _object_field(sample, object_id, "valid_patch_ids", [])
            legacy_core_patch_ids = _object_field(sample, object_id, "core_patch_ids", [])

            for view_idx, view_name in enumerate(view_names):
                valid_patch_ids = _resolve_view_payload(valid_patch_ids_by_view, view_name, legacy_valid_patch_ids)
                target_valid_patch_mask[batch_idx, obj_slot, view_idx] = patch_ids_to_mask(
                    valid_patch_ids, num_vrt_tokens
                ).to(device=device, dtype=torch.float32)

                core_patch_ids = _resolve_view_payload(core_patch_ids_by_view, view_name, legacy_core_patch_ids)
                core_patch_ids = [int(x) for x in _ensure_list(core_patch_ids)[:num_core_tokens]]
                if core_patch_ids and len(core_patch_ids) < num_core_tokens:
                    core_patch_ids.extend([core_patch_ids[-1]] * (num_core_tokens - len(core_patch_ids)))
                for core_idx, core_patch_id in enumerate(core_patch_ids[:num_core_tokens]):
                    if 0 <= core_patch_id < num_vrt_tokens:
                        target_core_patch_ids[batch_idx, obj_slot, view_idx, core_idx] = core_patch_id

            bbox_by_view = _object_field(sample, object_id, "bbox_by_view", {})
            patch_mask_by_view = _object_field(sample, object_id, "patch_mask_by_view", None)
            if patch_mask_by_view is None:
                # raw mask-by-view is allowed if already converted to patch grid upstream.
                patch_mask_by_view = _object_field(sample, object_id, "mask_by_view", {})
            visible_by_view = _object_field(sample, object_id, "visible_by_view", {})

            for view_idx, view_name in enumerate(view_names):
                bbox_tensor = _bbox_to_tensor(_resolve_view_payload(bbox_by_view, view_name, None)).to(device=device)
                patch_mask_tensor = _patch_mask_to_tensor(
                    _resolve_view_payload(patch_mask_by_view, view_name, None),
                    num_patches=num_vrt_tokens,
                ).to(device=device)
                visible_val = _resolve_view_payload(visible_by_view, view_name, None)
                visible_bool = _visible_value(visible_val, bbox_tensor, patch_mask_tensor)

                target_boxes_by_view[batch_idx, obj_slot, view_idx] = bbox_tensor
                target_patch_masks_by_view[batch_idx, obj_slot, view_idx] = patch_mask_tensor
                target_visible_by_view[batch_idx, obj_slot, view_idx] = bool(visible_bool)

        task_object_ids.append(sample_object_ids)
        task_object_roles.append(sample_roles)

    # Pad action/state tensors to stack cleanly.
    actions = torch.stack(actions_list, dim=0) if actions_list else None
    state = torch.stack(state_list, dim=0) if len(state_list) == len(examples) and state_list else None

    return PaDTRawBatch(
        images=images,
        instructions=instructions,
        actions=actions,
        state=state,
        task_object_ids=task_object_ids,
        task_object_roles=task_object_roles,
        objects=objects,
        object_lookup=object_lookup,
        target_core_patch_ids=target_core_patch_ids,
        target_valid_patch_mask=target_valid_patch_mask,
        target_patch_masks_by_view=target_patch_masks_by_view,
        target_boxes_by_view=target_boxes_by_view,
        target_visible_by_view=target_visible_by_view,
        object_presence_mask=object_presence_mask,
    )


def _repeat_valid_mask_per_core(valid_mask: torch.Tensor, num_core_tokens: int) -> torch.Tensor:
    # valid_mask: [B, O, V, N] -> [B, O*V*K, N]
    return valid_mask.unsqueeze(3).repeat(1, 1, 1, num_core_tokens, 1).flatten(1, 3)


def sample_noisy_teacher_core_ids(
    core_patch_ids: torch.Tensor,
    valid_patch_mask: torch.Tensor,
    replace_probability: float,
) -> torch.Tensor:
    """Resample teacher core ids from the same object's valid patch set.

    Minimal starVLA variant of the original PaDT "pick patches from the valid
    set every round" idea:
        - sampling happens per object, not per token
        - when an object is selected for noisy-teacher replacement, all active
          core slots for that object are resampled together
        - sampling prefers without-replacement, and only falls back to
          replacement when the valid set is smaller than the number of active
          teacher slots

    The external interface stays the same so existing configs keep working:
    `replace_probability` now means "probability of resampling this object's
    full teacher patch set".
    """
    if replace_probability <= 0.0:
        return core_patch_ids

    if core_patch_ids.dim() == 3:
        core_patch_ids = core_patch_ids.unsqueeze(2)
        valid_patch_mask = valid_patch_mask.unsqueeze(2)
        squeeze_view = True
    else:
        squeeze_view = False
    device = core_patch_ids.device
    noisy = core_patch_ids.clone()
    B, O, V, K = core_patch_ids.shape
    rand = torch.rand((B, O, V), device=device)
    for b in range(B):
        for o in range(O):
            for v in range(V):
                valid_ids = torch.nonzero(valid_patch_mask[b, o, v] > 0, as_tuple=False).flatten()
                if valid_ids.numel() == 0:
                    continue
                active_core_mask = noisy[b, o, v] >= 0
                active_core_count = int(active_core_mask.sum().item())
                if active_core_count == 0:
                    continue
                if rand[b, o, v].item() >= replace_probability:
                    continue

                if valid_ids.numel() >= active_core_count:
                    sampled_ids = valid_ids[torch.randperm(valid_ids.numel(), device=device)[:active_core_count]]
                else:
                    sampled_ids = valid_ids[torch.randperm(valid_ids.numel(), device=device)]
                    extra_count = active_core_count - int(sampled_ids.numel())
                    extra_ids = valid_ids[torch.randint(low=0, high=valid_ids.numel(), size=(extra_count,), device=device)]
                    sampled_ids = torch.cat((sampled_ids, extra_ids), dim=0)

                noisy[b, o, v, active_core_mask] = sampled_ids.to(dtype=noisy.dtype)
    return noisy.squeeze(2) if squeeze_view else noisy


@dataclass
class TeacherSequenceBatch:
    solutions: List[str]
    core_patch_ids: torch.Tensor
    valid_patch_mask_per_token: torch.Tensor


def build_structured_teacher_seq(
    batch: PaDTRawBatch,
    token_table: Mapping[str, Any],
    noisy_teacher_probability: float = 0.0,
) -> TeacherSequenceBatch:
    """Build fixed-slot assistant answers for teacher forcing.

    Schema example:
        <|padt_begin|>
        <|obj1|> <|view_agentview|> <|padt_vrt_003|> ... <|view_wrist|> <|padt_vrt_011|> ...
        <|obj2|> ...
        <|padt_end|>
    """

    noisy_core_patch_ids = sample_noisy_teacher_core_ids(
        batch.target_core_patch_ids,
        batch.target_valid_patch_mask,
        replace_probability=noisy_teacher_probability,
    )

    vrt_tokens: List[str] = token_table["vrt_tokens"]
    obj_tokens: List[str] = token_table["obj_tokens"]
    view_tokens: List[str] = token_table.get("view_tokens", [])
    padt_begin = token_table["padt_begin"]
    padt_end = token_table["padt_end"]

    solutions: List[str] = []
    for batch_idx, object_ids in enumerate(batch.task_object_ids):
        lines: List[str] = [padt_begin]
        for obj_idx, _object_id in enumerate(object_ids):
            slot_tokens = [obj_tokens[obj_idx]]
            per_view_core = noisy_core_patch_ids[batch_idx, obj_idx]
            if per_view_core.dim() == 1:
                per_view_core = per_view_core.unsqueeze(0)
            for view_idx, view_core_ids in enumerate(per_view_core):
                if view_idx < len(view_tokens):
                    slot_tokens.append(view_tokens[view_idx])
                for core_patch_id in view_core_ids.tolist():
                    if core_patch_id < 0:
                        continue
                    slot_tokens.append(vrt_tokens[int(core_patch_id)])
            lines.append(" ".join(slot_tokens))
        lines.append(padt_end)
        solutions.append("\n".join(lines))

    valid_patch_mask_per_token = _repeat_valid_mask_per_core(
        batch.target_valid_patch_mask,
        noisy_core_patch_ids.shape[-1],
    )

    return TeacherSequenceBatch(
        solutions=solutions,
        core_patch_ids=noisy_core_patch_ids,
        valid_patch_mask_per_token=valid_patch_mask_per_token,
    )


@dataclass
class GroupedVRTHidden:
    vrt_token_sequences: torch.Tensor
    vrt_hidden: torch.Tensor
    predicted_patch_ids: torch.Tensor


def group_vrt_hidden_by_slots(
    final_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    obj_token_ids: Sequence[int],
    vrt_start_id: int,
    vrt_end_id: int,
    max_task_objects: int,
    num_core_tokens: int,
    num_views: int = 2,
    view_token_ids: Optional[Sequence[int]] = None,
    object_presence_mask: Optional[torch.Tensor] = None,
) -> GroupedVRTHidden:
    """Group fixed-slot VRT hidden states into per-object queries.

    Assumption:
        Every object slot is serialized as:
            <|obj_i|> <|view_agentview|> <vrt_1> ... <|view_wrist|> <vrt_1> ...
    """
    B, _, D = final_hidden.shape
    device = final_hidden.device

    vrt_hidden = torch.zeros(
        (B, max_task_objects, num_views, num_core_tokens, D),
        dtype=final_hidden.dtype,
        device=device,
    )
    predicted_patch_ids = torch.full(
        (B, max_task_objects, num_views, num_core_tokens),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )

    obj_id_set = set(int(x) for x in obj_token_ids)
    view_ids = list(view_token_ids or [])
    for b in range(B):
        seq_ids = input_ids[b].tolist()
        hidden = final_hidden[b]
        seq_len = len(seq_ids)
        for pos, token_id in enumerate(seq_ids):
            if token_id not in obj_id_set:
                continue
            slot_idx = list(obj_token_ids).index(token_id)
            if slot_idx >= max_task_objects:
                continue
            if object_presence_mask is not None and not bool(object_presence_mask[b, slot_idx]):
                continue
            cursor = pos + 1
            for view_idx in range(num_views):
                if cursor < seq_len and view_idx < len(view_ids) and int(seq_ids[cursor]) == int(view_ids[view_idx]):
                    cursor += 1
                for k in range(num_core_tokens):
                    if cursor >= seq_len:
                        break
                    patch_token_id = int(seq_ids[cursor])
                    if vrt_start_id <= patch_token_id < vrt_end_id:
                        vrt_hidden[b, slot_idx, view_idx, k] = hidden[cursor]
                        predicted_patch_ids[b, slot_idx, view_idx, k] = patch_token_id - vrt_start_id
                        cursor += 1
                    else:
                        break

    return GroupedVRTHidden(
        vrt_token_sequences=vrt_hidden,
        vrt_hidden=vrt_hidden,
        predicted_patch_ids=predicted_patch_ids,
    )
