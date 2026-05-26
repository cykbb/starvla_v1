# Copyright 2026 OpenAI.
# Minimal PaDT-style dynamic-VRT wrapper built on top of starVLA's Qwen2.5-VL interface.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info

from .QWen2_5 import _QWen_VL_Interface, IGNORE_INDEX


logger = logging.getLogger(__name__)


@dataclass
class DynamicForwardOutput:
    logits: torch.Tensor
    final_hidden: torch.Tensor
    hidden_states: Sequence[torch.Tensor]
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    lang_summary: torch.Tensor
    vrt_loss: torch.Tensor
    vrt_diagnostics: Dict[str, torch.Tensor]
    prompt_lengths: torch.Tensor


@dataclass
class DynamicDecodeOutput:
    input_ids: torch.Tensor
    final_hidden: torch.Tensor
    attention_mask: torch.Tensor
    lang_summary: torch.Tensor
    predicted_patch_ids: torch.Tensor


@dataclass
class VRTLossOutput:
    loss: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


class _QWen_PaDT_VL_Interface(_QWen_VL_Interface):
    """Qwen2.5-VL interface with dynamic PaDT VRT input/output overriding."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.padt_cfg = config.framework.get("padt", {})
        self.hidden_dim = int(self.model.config.hidden_size)
        self.num_vrt_tokens = int(self.padt_cfg.get("num_vrt_tokens", 256))
        self.num_core_vrt_tokens = int(self.padt_cfg.get("num_core_vrt_tokens", 5))
        self.max_task_objects = int(self.padt_cfg.get("max_task_objects", 4))
        self.view_names = list(self.padt_cfg.get("view_names", ["agentview", "wrist"]))
        self.num_vrt_views = int(self.padt_cfg.get("decoder_num_views", len(self.view_names)))
        self.spatial_merge_size = int(self.padt_cfg.get("spatial_merge_size", 2))
        self.high_res_tokens_per_view = int(
            self.padt_cfg.get("high_res_tokens_per_view", self.num_vrt_tokens * self.spatial_merge_size * self.spatial_merge_size)
        )
        self.debug_vrt_alignment = bool(self.padt_cfg.get("debug_vrt_alignment", False))
        self.debug_vrt_metrics = bool(self.padt_cfg.get("debug_vrt_metrics", False))
        self._did_log_vrt_alignment = False
        vision_config = getattr(self.model.config, "vision_config", None)
        self.vision_hidden_dim = int(getattr(vision_config, "hidden_size", self.hidden_dim))
        self.prototype_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # Phase 6: high_res_proj is optional. In v2 it projects ViT pre-merger
        # (vision_hidden_dim, typically 1280) up to LLM hidden (2048). In v3 we
        # set decoder hidden_size = 1280, so high-res features can flow into the
        # decoder at native ViT dim with no projection — matches original PaDT
        # (padt.py:101 -> padt_decoder cu_high_res_feats stays at 1280).
        # When disabled, extract_patch_features returns high_res at
        # vision_hidden_dim (1280) instead of hidden_dim (2048).
        self.use_high_res_proj = bool(self.padt_cfg.get("use_high_res_proj", True))
        if self.use_high_res_proj:
            self.high_res_proj = nn.Sequential(
                nn.LayerNorm(self.vision_hidden_dim),
                nn.Linear(self.vision_hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.high_res_proj = None
        self.lang_summary_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self._init_padt_tokens()

    # ---------------------------------------------------------------------
    # token setup
    # ---------------------------------------------------------------------
    def _init_padt_tokens(self) -> None:
        tokenizer = self.processor.tokenizer
        padt_begin = self.padt_cfg.get("padt_begin_token", "<|padt_begin|>")
        padt_end = self.padt_cfg.get("padt_end_token", "<|padt_end|>")
        reason_begin = self.padt_cfg.get("reason_begin_token", "<|reason_begin|>")
        reason_end = self.padt_cfg.get("reason_end_token", "<|reason_end|>")
        padt_null = self.padt_cfg.get("padt_null_token", "<|padt_null|>")
        obj_tokens = [f"<|obj{i+1}|>" for i in range(self.max_task_objects)]
        view_tokens = [f"<|view_{view_name}|>" for view_name in self.view_names[: self.num_vrt_views]]
        vrt_tokens = [f"<|padt_vrt_{idx:03d}|>" for idx in range(self.num_vrt_tokens)]

        additional_special_tokens = [
            padt_begin,
            padt_end,
            reason_begin,
            reason_end,
            padt_null,
            *obj_tokens,
            *view_tokens,
            *vrt_tokens,
        ]
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        if num_added > 0:
            self.model.resize_token_embeddings(len(tokenizer))

        self.token_table = {
            "padt_begin": padt_begin,
            "padt_end": padt_end,
            "reason_begin": reason_begin,
            "reason_end": reason_end,
            "padt_null": padt_null,
            "obj_tokens": obj_tokens,
            "view_tokens": view_tokens,
            "vrt_tokens": vrt_tokens,
        }
        self.token_ids = {
            "padt_begin": tokenizer.convert_tokens_to_ids(padt_begin),
            "padt_end": tokenizer.convert_tokens_to_ids(padt_end),
            "reason_begin": tokenizer.convert_tokens_to_ids(reason_begin),
            "reason_end": tokenizer.convert_tokens_to_ids(reason_end),
            "padt_null": tokenizer.convert_tokens_to_ids(padt_null),
            "obj_tokens": [tokenizer.convert_tokens_to_ids(tok) for tok in obj_tokens],
            "view_tokens": [tokenizer.convert_tokens_to_ids(tok) for tok in view_tokens],
            "vrt_tokens": [tokenizer.convert_tokens_to_ids(tok) for tok in vrt_tokens],
        }
        self.vrt_start_id = int(self.token_ids["vrt_tokens"][0])
        self.vrt_end_id = self.vrt_start_id + self.num_vrt_tokens
        self.obj_token_ids = list(self.token_ids["obj_tokens"])
        self.view_token_ids = list(self.token_ids["view_tokens"])

    # ---------------------------------------------------------------------
    # prompt / input building
    # ---------------------------------------------------------------------
    def _build_padt_prompt(self, instruction: str, object_roles: Optional[Sequence[str]] = None) -> str:
        role_text = ""
        if object_roles:
            role_pairs = [f"obj{i+1}={role}" for i, role in enumerate(object_roles)]
            role_text = "\nOrdered object slots: " + "; ".join(role_pairs)
        return f"Task: {instruction}{role_text}\nReturn fixed slots only."

    @staticmethod
    def _coerce_image_for_qwen(image: Any) -> Any:
        """Normalize in-memory arrays/tensors to the PIL inputs expected by qwen_vl_utils."""
        if image is None or isinstance(image, (str, Image.Image)):
            return image

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                image = image.permute(1, 2, 0)
            image = image.numpy()

        if isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating) and array.size:
                    min_val = float(np.nanmin(array))
                    max_val = float(np.nanmax(array))
                    if 0.0 <= min_val and max_val <= 1.0:
                        array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)

            array = np.ascontiguousarray(array)
            if array.ndim == 2:
                return Image.fromarray(array, mode="L")
            if array.ndim == 3 and array.shape[-1] == 1:
                return Image.fromarray(array[..., 0], mode="L")
            if array.ndim == 3 and array.shape[-1] == 3:
                return Image.fromarray(array, mode="RGB")
            if array.ndim == 3 and array.shape[-1] == 4:
                return Image.fromarray(array, mode="RGBA")

        return image

    def build_padt_inputs(
        self,
        images: List[List[Any]],
        instructions: List[str],
        *,
        object_roles: Optional[List[List[str]]] = None,
        solutions: Optional[List[str]] = None,
    ) -> BatchFeature:
        messages = []
        prompt_texts: List[str] = []
        full_texts: List[str] = []
        for batch_idx, (imgs, instruction) in enumerate(zip(images, instructions)):
            prompt = self._build_padt_prompt(
                instruction=instruction,
                object_roles=object_roles[batch_idx] if object_roles is not None else None,
            )
            content = []
            for img in imgs:
                if img is not None:
                    content.append({"type": "image", "image": self._coerce_image_for_qwen(img)})
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            messages.append(msg)
            prompt_text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(prompt_text)
            full_texts.append(prompt_text + (solutions[batch_idx] if solutions is not None else ""))

        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        batch_input["prompt_lengths"] = prompt_batch["attention_mask"].sum(dim=-1)
        if solutions is not None:
            labels = batch_input["input_ids"].clone()
            full_lengths = batch_input["attention_mask"].sum(dim=-1)
            prompt_lengths = batch_input["prompt_lengths"]
            total_seq_len = labels.size(1)
            pad_token_id = self.processor.tokenizer.pad_token_id
            for batch_idx in range(labels.size(0)):
                full_len = int(full_lengths[batch_idx].item())
                prompt_len = int(prompt_lengths[batch_idx].item())
                start_idx = total_seq_len - full_len
                prompt_end = start_idx + prompt_len
                labels[batch_idx, :prompt_end] = IGNORE_INDEX
            labels[labels == pad_token_id] = IGNORE_INDEX
            batch_input["labels"] = labels
        return batch_input.to(self.model.device)

    # ---------------------------------------------------------------------
    # patch features / prototypes
    # ---------------------------------------------------------------------
    def _normalize_visual_token_count(self, features: torch.Tensor) -> torch.Tensor:
        return self._normalize_token_count(features, self.num_vrt_tokens)

    @staticmethod
    def _normalize_token_count(features: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if features.shape[0] == target_tokens:
            return features
        if features.shape[0] > target_tokens:
            return features[:target_tokens]
        padded = torch.zeros(
            (target_tokens, features.shape[-1]),
            dtype=features.dtype,
            device=features.device,
        )
        padded[: features.shape[0]] = features
        return padded

    def _visual_forward_dual_res(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen2.5-VL visual tower and keep both pre-merger and merged tokens."""
        visual = self.model.visual
        hidden_states = visual.patch_embed(pixel_values)
        rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
        window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        spatial_merge_unit = int(getattr(visual, "spatial_merge_unit", self.spatial_merge_size**2))
        hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, block in enumerate(visual.blocks):
            if layer_num in visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if visual.gradient_checkpointing and visual.training:
                hidden_states = visual._gradient_checkpointing_func(
                    block.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = block(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        high_res_hidden_states = hidden_states
        low_res_hidden_states = visual.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        low_res_hidden_states = low_res_hidden_states[reverse_indices, :]
        high_res_hidden_states = high_res_hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        high_res_hidden_states = high_res_hidden_states[reverse_indices, :, :].reshape(seq_len, -1)
        return low_res_hidden_states, high_res_hidden_states

    def extract_patch_features(
        self,
        images: List[List[Any]],
        instructions: List[str],
        *,
        object_roles: Optional[List[List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract per-view patch features from the Qwen visual tower.

        Returns exact visual-token features from `model.visual(...)`, then slices / pads
        them into per-view VRT banks and the multi-view decoder banks.
        """
        qwen_inputs = self.build_padt_inputs(images=images, instructions=instructions, object_roles=object_roles)
        pixel_values = qwen_inputs.get("pixel_values", None)
        image_grid_thw = qwen_inputs.get("image_grid_thw", None)
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("QwenPaDTPI requires pixel_values and image_grid_thw to extract patch features")

        visual_dtype = getattr(self.model.visual, "dtype", None)
        if visual_dtype is None:
            try:
                visual_dtype = next(self.model.visual.parameters()).dtype
            except StopIteration:
                visual_dtype = pixel_values.dtype

        image_embeds, high_res_embeds = self._visual_forward_dual_res(pixel_values.type(visual_dtype), image_grid_thw)
        merge_unit = self.spatial_merge_size * self.spatial_merge_size
        per_image_counts = (image_grid_thw.prod(dim=-1) // merge_unit).tolist()
        per_image_high_counts = image_grid_thw.prod(dim=-1).tolist()
        flat_images = [img for sample_imgs in images for img in sample_imgs if img is not None]
        if len(per_image_counts) != len(flat_images):
            raise ValueError(
                f"Mismatch between processed image count ({len(per_image_counts)}) and raw images ({len(flat_images)})."
            )

        per_image_features: List[torch.Tensor] = []
        per_image_high_features: List[torch.Tensor] = []
        cursor = 0
        for count in per_image_counts:
            count_int = int(count)
            per_image_features.append(self._normalize_visual_token_count(image_embeds[cursor : cursor + count_int]))
            cursor += count_int
        cursor = 0
        # Phase 6: when high_res_proj is disabled (v3 config), high_res tokens
        # stay at vision_hidden_dim (1280, ViT pre-merger), matching original
        # PaDT. Decoder is expected to consume them at that dim directly.
        high_res_target_dim = self.hidden_dim if self.use_high_res_proj else self.vision_hidden_dim
        if self.use_high_res_proj:
            high_proj_param = next(self.high_res_proj.parameters())
        for count in per_image_high_counts:
            count_int = int(count)
            high_res = high_res_embeds[cursor : cursor + count_int]
            if self.use_high_res_proj:
                high_res = self.high_res_proj(high_res.to(device=high_proj_param.device, dtype=high_proj_param.dtype))
                high_res = high_res.to(device=image_embeds.device, dtype=image_embeds.dtype)
            else:
                # Keep ViT pre-merger features at vision_hidden_dim, but move
                # to same device/dtype as image_embeds for downstream stacking.
                high_res = high_res.to(device=image_embeds.device, dtype=image_embeds.dtype)
            per_image_high_features.append(self._normalize_token_count(high_res, self.high_res_tokens_per_view))
            cursor += count_int

        B = len(images)
        zero_view = torch.zeros(
            (self.num_vrt_tokens, self.hidden_dim),
            dtype=image_embeds.dtype,
            device=image_embeds.device,
        )
        zero_high_view = torch.zeros(
            (self.high_res_tokens_per_view, high_res_target_dim),
            dtype=image_embeds.dtype,
            device=image_embeds.device,
        )
        sample_features: List[List[torch.Tensor]] = []
        sample_high_features: List[List[torch.Tensor]] = []
        feature_cursor = 0
        high_feature_cursor = 0
        for sample_imgs in images:
            sample_view_features: List[torch.Tensor] = []
            sample_view_high_features: List[torch.Tensor] = []
            for img in sample_imgs:
                if img is None:
                    sample_view_features.append(zero_view)
                    sample_view_high_features.append(zero_high_view)
                else:
                    sample_view_features.append(per_image_features[feature_cursor])
                    sample_view_high_features.append(per_image_high_features[high_feature_cursor])
                    feature_cursor += 1
                    high_feature_cursor += 1
            while len(sample_view_features) < self.num_vrt_views:
                sample_view_features.append(zero_view)
                sample_view_high_features.append(zero_high_view)
            sample_features.append(sample_view_features[: self.num_vrt_views])
            sample_high_features.append(sample_view_high_features[: self.num_vrt_views])

        agentview = torch.stack([sample_view_features[0] for sample_view_features in sample_features], dim=0)
        wrist = torch.stack([sample_view_features[1] for sample_view_features in sample_features], dim=0)
        all_views = torch.cat((agentview, wrist), dim=1)
        high_agentview = torch.stack([sample_view_features[0] for sample_view_features in sample_high_features], dim=0)
        high_wrist = torch.stack([sample_view_features[1] for sample_view_features in sample_high_features], dim=0)
        high_all_views = torch.cat((high_agentview, high_wrist), dim=1)
        vrt_bank = torch.stack((agentview, wrist), dim=1)
        return {
            "agentview": agentview,
            "wrist": wrist,
            "all": all_views,
            "high_res_agentview": high_agentview,
            "high_res_wrist": high_wrist,
            "high_res_all": high_all_views,
            "vrt_bank": vrt_bank,
            # Raw merged visual tokens — exactly what `self.model.visual(...)` would
            # return in `_replace_image_embeddings`. Cached here so downstream
            # `forward_dynamic` / `custom_vrt_decode` can pass it back via
            # `precomputed_image_embeds` and avoid re-running the ViT. Equivalent to
            # original PaDT's `past_image_embeds` mechanism (padt.py:222).
            "_raw_visual_tokens": image_embeds,
        }

    def build_prototypes(self, patch_features: torch.Tensor) -> torch.Tensor:
        return self.prototype_proj(patch_features)

    # ---------------------------------------------------------------------
    # dynamic embeddings / logits
    # ---------------------------------------------------------------------
    def _replace_image_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        precomputed_image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replace <image> placeholder embeddings with visual-tower outputs.

        If `precomputed_image_embeds` is provided (shape `[total_visual_tokens, hidden]`,
        produced by an upstream `_visual_forward_dual_res` / `extract_patch_features`
        call), reuse it instead of running the ViT again. This is the in-step
        equivalent of original PaDT's `past_image_embeds` caching (padt.py:222).
        """
        if pixel_values is None or image_grid_thw is None:
            return inputs_embeds

        if precomputed_image_embeds is not None:
            image_embeds = precomputed_image_embeds
        else:
            visual_dtype = getattr(self.model.visual, "dtype", None)
            if visual_dtype is None:
                try:
                    visual_dtype = next(self.model.visual.parameters()).dtype
                except StopIteration:
                    visual_dtype = pixel_values.dtype
            image_embeds = self.model.visual(pixel_values.type(visual_dtype), grid_thw=image_grid_thw)

        image_mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        n_image_tokens = int((input_ids == self.model.config.image_token_id).sum().item())
        if n_image_tokens != int(image_embeds.shape[0]):
            raise ValueError(
                f"Image features and image tokens do not match: tokens={n_image_tokens}, features={image_embeds.shape[0]}"
            )
        return inputs_embeds.masked_scatter(
            image_mask.to(inputs_embeds.device),
            image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
        )

    def _infer_vrt_view_ids(self, input_ids: torch.Tensor, default_view_idx: int = 0) -> torch.Tensor:
        view_ids = torch.full_like(input_ids, fill_value=int(default_view_idx))
        view_token_to_idx = {int(token_id): idx for idx, token_id in enumerate(self.view_token_ids)}
        for batch_idx in range(input_ids.shape[0]):
            current_view = int(default_view_idx)
            for pos, token_id in enumerate(input_ids[batch_idx].tolist()):
                token_id = int(token_id)
                if token_id in view_token_to_idx:
                    current_view = int(view_token_to_idx[token_id])
                view_ids[batch_idx, pos] = current_view
        return view_ids

    def _select_p_ref(self, P_ref: torch.Tensor, view_idx: int) -> torch.Tensor:
        if P_ref.dim() == 4:
            return P_ref[:, int(view_idx)]
        return P_ref

    def _replace_vrt_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        P_ref: torch.Tensor,
        *,
        default_view_idx: int = 0,
    ) -> torch.Tensor:
        vrt_mask = (input_ids >= self.vrt_start_id) & (input_ids < self.vrt_end_id)
        if not vrt_mask.any():
            return inputs_embeds
        local_patch_ids = (input_ids - self.vrt_start_id).clamp(min=0, max=self.num_vrt_tokens - 1)
        local_view_ids = self._infer_vrt_view_ids(input_ids, default_view_idx=default_view_idx).clamp(
            min=0, max=max(0, self.num_vrt_views - 1)
        )
        for batch_idx in range(input_ids.shape[0]):
            batch_mask = vrt_mask[batch_idx]
            if not batch_mask.any():
                continue
            patch_ids = local_patch_ids[batch_idx, batch_mask]
            if P_ref.dim() == 4:
                view_ids = local_view_ids[batch_idx, batch_mask]
                inputs_embeds[batch_idx, batch_mask] = P_ref[batch_idx, view_ids, patch_ids]
            else:
                inputs_embeds[batch_idx, batch_mask] = P_ref[batch_idx, patch_ids]
        return inputs_embeds

    def build_inputs_embeds(
        self,
        qwen_inputs: BatchFeature,
        P_ref: torch.Tensor,
        *,
        default_view_idx: int = 0,
        precomputed_image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = qwen_inputs["input_ids"]
        embed_fn = self.model.get_input_embeddings()
        inputs_embeds = embed_fn(input_ids)
        inputs_embeds = self._replace_image_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=qwen_inputs.get("pixel_values", None),
            image_grid_thw=qwen_inputs.get("image_grid_thw", None),
            precomputed_image_embeds=precomputed_image_embeds,
        )
        inputs_embeds = self._replace_vrt_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            P_ref=P_ref,
            default_view_idx=default_view_idx,
        )
        return inputs_embeds

    def compute_dynamic_logits(
        self,
        hidden: torch.Tensor,
        static_logits: torch.Tensor,
        P_ref: torch.Tensor,
        *,
        view_idx: int = 0,
    ) -> torch.Tensor:
        """Overwrite the VRT-range columns of `static_logits` with dynamic
        prototype-based logits, in place.

        Phase 0 audit confirmed `outputs.logits` is only ever consumed here
        (QWen2_5_PaDT.py:728, :928), so in-place mutation is safe and saves
        the ~100 MB/step clone of the full [B, T, vocab≈152k] tensor in the
        autograd graph.
        """
        P_view = self._select_p_ref(P_ref, view_idx)
        if hidden.dim() == 2:
            dynamic_vrt_logits = torch.einsum("bd,bnd->bn", hidden, P_view)
            static_logits[..., self.vrt_start_id : self.vrt_end_id] = dynamic_vrt_logits
            return static_logits
        dynamic_vrt_logits = torch.einsum("bld,bnd->bln", hidden, P_view)
        static_logits[..., self.vrt_start_id : self.vrt_end_id] = dynamic_vrt_logits
        return static_logits

    def _compute_lang_summary(
        self,
        final_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        prompt_mask = attention_mask.bool() & (labels == IGNORE_INDEX)
        prompt_mask = prompt_mask & (input_ids != self.model.config.image_token_id)
        prompt_mask = prompt_mask & ~((input_ids >= self.vrt_start_id) & (input_ids < self.vrt_end_id))
        prompt_mask = prompt_mask.unsqueeze(-1)
        denom = prompt_mask.sum(dim=1).clamp_min(1)
        pooled = (final_hidden * prompt_mask).sum(dim=1) / denom
        return self.lang_summary_proj(pooled)

    def _compute_vrt_loss(
        self,
        logits: torch.Tensor,
        final_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        teacher_core_patch_ids: torch.Tensor,
        valid_patch_mask_per_token: torch.Tensor,
        P_ref: torch.Tensor,
    ) -> VRTLossOutput:
        loss_terms: List[torch.Tensor] = []
        collect_diagnostics = self.debug_vrt_metrics
        top1_correct_terms: List[torch.Tensor] = []
        hidden_norm_terms: List[torch.Tensor] = []
        proto_norm_terms: List[torch.Tensor] = []
        target_logit_terms: List[torch.Tensor] = []
        top1_margin_terms: List[torch.Tensor] = []
        logit_abs_max_terms: List[torch.Tensor] = []
        device = logits.device
        metric_dtype = torch.float32
        zero = torch.zeros((), device=device, dtype=metric_dtype)
        B = logits.shape[0]
        for batch_idx in range(B):
            positions = torch.nonzero(
                (labels[batch_idx] != IGNORE_INDEX)
                & (input_ids[batch_idx] >= self.vrt_start_id)
                & (input_ids[batch_idx] < self.vrt_end_id),
                as_tuple=False,
            ).flatten()
            flat_targets = teacher_core_patch_ids[batch_idx].flatten()
            flat_valid_masks = valid_patch_mask_per_token[batch_idx]
            target_cursor = 0
            for target_idx, target_patch_id in enumerate(flat_targets.tolist()):
                if target_patch_id < 0:
                    continue
                if target_cursor >= positions.numel():
                    break
                label_pos = int(positions[target_cursor].item())
                if label_pos <= 0:
                    target_cursor += 1
                    continue

                # Causal LM logits at position t predict token t+1, so the target
                # VRT token at `label_pos` must be supervised using logits from
                # `label_pos - 1`.
                logit_pos = label_pos - 1
                view_idx = (target_idx // self.num_core_vrt_tokens) % max(1, self.num_vrt_views)
                if P_ref.dim() == 4:
                    proto_bank = P_ref[batch_idx, view_idx]
                    raw_logits = torch.einsum("d,nd->n", final_hidden[batch_idx, logit_pos], proto_bank).clone()
                else:
                    raw_logits = logits[batch_idx, logit_pos, self.vrt_start_id : self.vrt_end_id].clone()
                valid_mask = flat_valid_masks[target_idx] > 0
                # Mask out valid-but-not-core patches so they don't compete with the
                # selected core patch in the softmax denominator.  This mirrors the
                # original PaDT logit-masking design: the model is only asked to
                # distinguish the core patch from non-valid patches, not from other
                # equally-valid patches of the same object.
                if valid_mask.any():
                    valid_but_not_core = valid_mask.clone()
                    valid_but_not_core[target_patch_id] = False
                    raw_logits[valid_but_not_core] = float("-inf")

                if self.debug_vrt_alignment and not self._did_log_vrt_alignment:
                    supervised_token_id = int(input_ids[batch_idx, label_pos].item())
                    prev_token_id = int(input_ids[batch_idx, logit_pos].item())
                    supervised_token = self.processor.tokenizer.convert_ids_to_tokens(supervised_token_id)
                    prev_token = self.processor.tokenizer.convert_ids_to_tokens(prev_token_id)
                    expected_token_id = self.vrt_start_id + int(target_patch_id)
                    expected_token = self.processor.tokenizer.convert_ids_to_tokens(expected_token_id)
                    logger.warning(
                        "[PaDT VRT sanity] batch=%d label_pos=%d logit_pos=%d prev_token=%s supervised_token=%s expected_target=%s",
                        batch_idx,
                        label_pos,
                        logit_pos,
                        prev_token,
                        supervised_token,
                        expected_token,
                    )
                    self._did_log_vrt_alignment = True

                token_log_probs = F.log_softmax(raw_logits, dim=-1)
                nll_core = -token_log_probs[target_patch_id]
                term = nll_core
                loss_terms.append(term)

                if collect_diagnostics:
                    teacher_hidden = final_hidden[batch_idx, logit_pos].detach().float()
                    if P_ref.dim() == 4:
                        teacher_proto = P_ref[batch_idx, view_idx, target_patch_id].detach().float()
                    else:
                        teacher_proto = P_ref[batch_idx, target_patch_id].detach().float()
                    target_logit = raw_logits[target_patch_id].detach().float()
                    pred_patch_id = int(raw_logits.argmax(dim=-1).item())
                    top1_logit = raw_logits[pred_patch_id].detach().float()
                    finite_logits = raw_logits[torch.isfinite(raw_logits)].detach().float()

                    top1_correct_terms.append(
                        torch.tensor(float(pred_patch_id == target_patch_id), device=device, dtype=metric_dtype)
                    )
                    hidden_norm_terms.append(torch.linalg.vector_norm(teacher_hidden, dim=-1).to(device=device))
                    proto_norm_terms.append(torch.linalg.vector_norm(teacher_proto, dim=-1).to(device=device))
                    target_logit_terms.append(target_logit.to(device=device))
                    top1_margin_terms.append((top1_logit - target_logit).to(device=device))
                    if finite_logits.numel() > 0:
                        logit_abs_max_terms.append(finite_logits.abs().max().to(device=device))
                target_cursor += 1

        def _mean_or_zero(values: List[torch.Tensor]) -> torch.Tensor:
            if not values:
                return zero
            return torch.stack(values).mean()

        diagnostics: Dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            diagnostics = {
                "vrt_supervised_tokens": torch.tensor(float(len(loss_terms)), device=device, dtype=metric_dtype),
                "vrt_teacher_top1_acc": _mean_or_zero(top1_correct_terms),
                "vrt_teacher_hidden_norm": _mean_or_zero(hidden_norm_terms),
                "vrt_teacher_proto_norm": _mean_or_zero(proto_norm_terms),
                "vrt_teacher_target_logit": _mean_or_zero(target_logit_terms),
                "vrt_teacher_top1_margin": _mean_or_zero(top1_margin_terms),
                "vrt_teacher_logit_abs_max": _mean_or_zero(logit_abs_max_terms),
            }
        if not loss_terms:
            return VRTLossOutput(loss=logits.sum() * 0.0, diagnostics=diagnostics)
        return VRTLossOutput(loss=torch.stack(loss_terms).mean(), diagnostics=diagnostics)

    # ---------------------------------------------------------------------
    # training forward
    # ---------------------------------------------------------------------
    def forward_dynamic(
        self,
        *,
        images: List[List[Any]],
        instructions: List[str],
        object_roles: List[List[str]],
        solutions: List[str],
        P_ref: torch.Tensor,
        teacher_core_patch_ids: torch.Tensor,
        valid_patch_mask_per_token: torch.Tensor,
        precomputed_image_embeds: Optional[torch.Tensor] = None,
    ) -> DynamicForwardOutput:
        qwen_inputs = self.build_padt_inputs(
            images=images,
            instructions=instructions,
            object_roles=object_roles,
            solutions=solutions,
        )
        inputs_embeds = self.build_inputs_embeds(
            qwen_inputs=qwen_inputs,
            P_ref=P_ref,
            precomputed_image_embeds=precomputed_image_embeds,
        )
        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=qwen_inputs["attention_mask"],
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        combined_logits = self.compute_dynamic_logits(
            hidden=outputs.hidden_states[-1],
            static_logits=outputs.logits,
            P_ref=P_ref,
        )
        vrt_loss_output = self._compute_vrt_loss(
            logits=combined_logits,
            final_hidden=outputs.hidden_states[-1],
            input_ids=qwen_inputs["input_ids"],
            labels=qwen_inputs["labels"],
            teacher_core_patch_ids=teacher_core_patch_ids,
            valid_patch_mask_per_token=valid_patch_mask_per_token,
            P_ref=P_ref,
        )
        vrt_diagnostics = dict(vrt_loss_output.diagnostics)
        if self.debug_vrt_metrics:
            proto_norms = torch.linalg.vector_norm(P_ref.detach().float(), dim=-1)
            vrt_diagnostics["vrt_proto_global_norm"] = proto_norms.mean()
            vrt_diagnostics["vrt_proto_global_norm_max"] = proto_norms.max()
        lang_summary = self._compute_lang_summary(
            final_hidden=outputs.hidden_states[-1],
            input_ids=qwen_inputs["input_ids"],
            labels=qwen_inputs["labels"],
            attention_mask=qwen_inputs["attention_mask"],
        )
        return DynamicForwardOutput(
            logits=combined_logits,
            final_hidden=outputs.hidden_states[-1],
            hidden_states=outputs.hidden_states,
            input_ids=qwen_inputs["input_ids"],
            labels=qwen_inputs["labels"],
            attention_mask=qwen_inputs["attention_mask"],
            lang_summary=lang_summary,
            vrt_loss=vrt_loss_output.loss,
            vrt_diagnostics=vrt_diagnostics,
            prompt_lengths=qwen_inputs["prompt_lengths"],
        )

    # ---------------------------------------------------------------------
    # custom cached decode (no second full VLM forward)
    # ---------------------------------------------------------------------
    def _decode_token_step(
        self,
        *,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        P_ref: torch.Tensor,
        view_idx: int = 0,
    ):
        use_dynamic_embed = ((token_ids >= self.vrt_start_id) & (token_ids < self.vrt_end_id)).any()
        if use_dynamic_embed:
            inputs_embeds = self.model.get_input_embeddings()(token_ids)
            inputs_embeds = self._replace_vrt_embeddings(token_ids, inputs_embeds, P_ref, default_view_idx=view_idx)
            outputs = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = self.forward(
                input_ids=token_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs

    def custom_vrt_decode(
        self,
        *,
        images: List[List[Any]],
        instructions: List[str],
        object_roles: List[List[str]],
        object_presence_mask: torch.Tensor,
        P_ref: torch.Tensor,
        precomputed_image_embeds: Optional[torch.Tensor] = None,
    ) -> DynamicDecodeOutput:
        """Custom cached decode.

        Important: this v1 path forces control/object-slot tokens and only samples the
        dynamic VRT tokens. That is deliberate: the schema is fixed-slot, so there is no
        benefit in letting the model free-run over control markers.
        """
        prompt_inputs = self.build_padt_inputs(
            images=images,
            instructions=instructions,
            object_roles=object_roles,
            solutions=None,
        )
        prompt_inputs_embeds = self.build_inputs_embeds(
            qwen_inputs=prompt_inputs,
            P_ref=P_ref,
            precomputed_image_embeds=precomputed_image_embeds,
        )
        prompt_outputs = self.forward(
            inputs_embeds=prompt_inputs_embeds,
            attention_mask=prompt_inputs["attention_mask"],
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        # Mirror training-side _compute_lang_summary: mean-pool over prompt text tokens
        # (excluding image and VRT positions). Under the causal mask the hidden states
        # at prompt positions are identical between prompt-only and prompt+solution
        # forwards, so this is a strict alignment with the training computation.
        prompt_input_ids = prompt_inputs["input_ids"]
        prompt_final_hidden = prompt_outputs.hidden_states[-1]
        prompt_lang_mask = prompt_inputs["attention_mask"].bool()
        prompt_lang_mask = prompt_lang_mask & (prompt_input_ids != self.model.config.image_token_id)
        prompt_lang_mask = prompt_lang_mask & ~(
            (prompt_input_ids >= self.vrt_start_id) & (prompt_input_ids < self.vrt_end_id)
        )
        prompt_lang_mask = prompt_lang_mask.unsqueeze(-1)
        prompt_denom = prompt_lang_mask.sum(dim=1).clamp_min(1)
        prompt_pooled = (prompt_final_hidden * prompt_lang_mask).sum(dim=1) / prompt_denom
        lang_summary = self.lang_summary_proj(prompt_pooled)

        batch_size = prompt_inputs["input_ids"].shape[0]
        device = prompt_inputs["input_ids"].device
        running_attention_mask = prompt_inputs["attention_mask"]
        past_key_values = prompt_outputs.past_key_values

        generated_token_steps: List[torch.Tensor] = []
        generated_hidden_steps: List[torch.Tensor] = []
        predicted_patch_ids = torch.full(
            (batch_size, self.max_task_objects, self.num_vrt_views, self.num_core_vrt_tokens),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        current_outputs = prompt_outputs
        for forced_token_id in [self.token_ids["padt_begin"]]:
            forced_token = torch.full((batch_size, 1), forced_token_id, dtype=torch.long, device=device)
            running_attention_mask = torch.cat(
                (running_attention_mask, torch.ones((batch_size, 1), dtype=running_attention_mask.dtype, device=device)),
                dim=1,
            )
            current_outputs = self._decode_token_step(
                token_ids=forced_token,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
                P_ref=P_ref,
            )
            past_key_values = current_outputs.past_key_values
            generated_token_steps.append(forced_token)
            generated_hidden_steps.append(current_outputs.hidden_states[-1])

        for obj_idx in range(self.max_task_objects):
            if object_presence_mask is None:
                slot_presence = torch.ones((batch_size,), dtype=torch.bool, device=device)
            else:
                slot_presence = object_presence_mask[:, obj_idx].to(device=device, dtype=torch.bool)
            forced_obj_token = torch.where(
                slot_presence.unsqueeze(-1),
                torch.full((batch_size, 1), self.obj_token_ids[obj_idx], dtype=torch.long, device=device),
                torch.full((batch_size, 1), self.token_ids["padt_null"], dtype=torch.long, device=device),
            )
            running_attention_mask = torch.cat(
                (running_attention_mask, torch.ones((batch_size, 1), dtype=running_attention_mask.dtype, device=device)),
                dim=1,
            )
            current_outputs = self._decode_token_step(
                token_ids=forced_obj_token,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
                P_ref=P_ref,
            )
            past_key_values = current_outputs.past_key_values
            generated_token_steps.append(forced_obj_token)
            generated_hidden_steps.append(current_outputs.hidden_states[-1])

            for view_idx in range(self.num_vrt_views):
                view_token_id = (
                    self.view_token_ids[view_idx]
                    if view_idx < len(self.view_token_ids)
                    else self.token_ids["padt_null"]
                )
                forced_view_token = torch.where(
                    slot_presence.unsqueeze(-1),
                    torch.full((batch_size, 1), view_token_id, dtype=torch.long, device=device),
                    torch.full((batch_size, 1), self.token_ids["padt_null"], dtype=torch.long, device=device),
                )
                running_attention_mask = torch.cat(
                    (running_attention_mask, torch.ones((batch_size, 1), dtype=running_attention_mask.dtype, device=device)),
                    dim=1,
                )
                current_outputs = self._decode_token_step(
                    token_ids=forced_view_token,
                    attention_mask=running_attention_mask,
                    past_key_values=past_key_values,
                    P_ref=P_ref,
                    view_idx=view_idx,
                )
                past_key_values = current_outputs.past_key_values
                generated_token_steps.append(forced_view_token)
                generated_hidden_steps.append(current_outputs.hidden_states[-1])

                for core_idx in range(self.num_core_vrt_tokens):
                    next_token_logits = self.compute_dynamic_logits(
                        hidden=current_outputs.hidden_states[-1][:, -1, :],
                        static_logits=current_outputs.logits[:, -1, :],
                        P_ref=P_ref,
                        view_idx=view_idx,
                    )
                    patch_logits = next_token_logits[:, self.vrt_start_id : self.vrt_end_id]
                    patch_ids = torch.argmax(patch_logits, dim=-1)
                    predicted_patch_ids[slot_presence, obj_idx, view_idx, core_idx] = patch_ids[slot_presence]
                    sampled_token = torch.where(
                        slot_presence.unsqueeze(-1),
                        (patch_ids + self.vrt_start_id).unsqueeze(-1),
                        torch.full((batch_size, 1), self.token_ids["padt_null"], dtype=torch.long, device=device),
                    )

                    running_attention_mask = torch.cat(
                        (running_attention_mask, torch.ones((batch_size, 1), dtype=running_attention_mask.dtype, device=device)),
                        dim=1,
                    )
                    current_outputs = self._decode_token_step(
                        token_ids=sampled_token,
                        attention_mask=running_attention_mask,
                        past_key_values=past_key_values,
                        P_ref=P_ref,
                        view_idx=view_idx,
                    )
                    past_key_values = current_outputs.past_key_values
                    generated_token_steps.append(sampled_token)
                    generated_hidden_steps.append(current_outputs.hidden_states[-1])

        forced_end_token = torch.full((batch_size, 1), self.token_ids["padt_end"], dtype=torch.long, device=device)
        running_attention_mask = torch.cat(
            (running_attention_mask, torch.ones((batch_size, 1), dtype=running_attention_mask.dtype, device=device)),
            dim=1,
        )
        current_outputs = self._decode_token_step(
            token_ids=forced_end_token,
            attention_mask=running_attention_mask,
            past_key_values=past_key_values,
            P_ref=P_ref,
        )
        generated_token_steps.append(forced_end_token)
        generated_hidden_steps.append(current_outputs.hidden_states[-1])

        generated_input_ids = torch.cat(generated_token_steps, dim=1)
        generated_hidden = torch.cat(generated_hidden_steps, dim=1)
        full_input_ids = torch.cat((prompt_inputs["input_ids"], generated_input_ids), dim=1)
        full_hidden = torch.cat((prompt_outputs.hidden_states[-1], generated_hidden), dim=1)
        full_attention_mask = running_attention_mask

        return DynamicDecodeOutput(
            input_ids=full_input_ids,
            final_hidden=full_hidden,
            attention_mask=full_attention_mask,
            lang_summary=lang_summary,
            predicted_patch_ids=predicted_patch_ids,
        )
