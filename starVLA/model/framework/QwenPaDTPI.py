# Copyright 2026 OpenAI.
# QwenPaDTPI framework for starVLA.
#
# Important architectural notes:
#   1. This file is a new framework entrypoint; it does not modify QwenPI.
#   2. VRT autoregression is per-view: agentview and wrist each produce fixed VRT slots.
#   3. The decoder keeps token-level object-view VRT evidence rather than mean-pooling it.
#   4. The action-conditioning bridge packs language, object, and object-view
#      memory into the per-layer token list consumed by the PI head.

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.padt_data_utils import (
    GroupedVRTHidden,
    TeacherSequenceBatch,
    build_structured_teacher_seq,
    group_vrt_hidden_by_slots,
    preprocess_raw_dict,
)
from starVLA.model.modules.vlm.padt_object_decoder import PaDTObjectDecoder
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.action_model.PaDTConditionBridge import PaDTConditionBridge
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenPaDTPI")
class QwenPaDTPI(baseframework):
    """Minimal PaDT-style QwenPI variant for starVLA.

    Closed-loop path:
        preprocess_raw_dict
        -> extract_patch_features
        -> build_prototypes
        -> build_structured_teacher_seq
        -> forward_dynamic / custom_vrt_decode
        -> padt_decoder
        -> build_action_condition
        -> action_model.forward / predict_action
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Keep the Qwen hidden config explicit because the PI FM head reads it.
        self.config.framework.qwenvl.vl_hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        if not hasattr(self.config.framework.qwenvl, "num_vl_layers"):
            self.config.framework.qwenvl.num_vl_layers = int(self.qwen_vl_interface.model.config.num_hidden_layers)

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.padt_decoder = PaDTObjectDecoder(config=self.config)
        self.condition_bridge = PaDTConditionBridge(config=self.config)

        self.padt_cfg = self.config.framework.get("padt", {})
        self.future_action_window_size = int(self.config.framework.action_model.future_action_window_size)
        self.past_action_window_size = int(self.config.framework.action_model.past_action_window_size)
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.max_task_objects = int(self.padt_cfg.get("max_task_objects", 4))
        self.num_core_vrt_tokens = int(self.padt_cfg.get("num_core_vrt_tokens", 5))
        default_view_count = len(self.padt_cfg.get("view_names", ["agentview", "wrist"]))
        self.num_vrt_views = int(self.padt_cfg.get("decoder_num_views", default_view_count))

        # Simple role embedding to preserve the fixed-slot / object_role supervision signal.
        role_names = list(self.padt_cfg.get(
            "role_vocab",
            ["slot_1", "slot_2", "slot_3", "slot_4", "primary", "secondary", "receptacle", "tool"],
        ))
        self.role_to_idx = {name: idx for idx, name in enumerate(role_names)}
        self.role_embedding = nn.Embedding(len(self.role_to_idx) + 1, self.config.framework.qwenvl.vl_hidden_dim)

        # Phase 3: honor trainer.enable_gradient_checkpointing flag. Without this
        # the LLM and ViT run with no activation recomputation, which is the main
        # reason PaDTPI's effective batch size is far below QwenPI's. Aligns with
        # the original PaDT trainer (padt_sft_trainer.py:217-236).
        self._maybe_enable_gradient_checkpointing()

    def _maybe_enable_gradient_checkpointing(self) -> None:
        trainer_cfg = getattr(self.config, "trainer", None)
        if trainer_cfg is None:
            return
        if not bool(getattr(trainer_cfg, "enable_gradient_checkpointing", False)):
            return
        # GC and KV cache are mutually exclusive; checkpoint requires no_cache.
        # Inference paths run under @torch.inference_mode() with model.training=False,
        # so this does not affect eval speed.
        try:
            self.qwen_vl_interface.model.config.use_cache = False
            self.qwen_vl_interface.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            # Qwen2.5-VL's visual tower needs the GC flag set separately.
            if hasattr(self.qwen_vl_interface.model, "visual"):
                self.qwen_vl_interface.model.visual.gradient_checkpointing = True
            # Phase 4: same flag also enables PaDT object decoder checkpointing.
            # Currently most useful in the 2048-hidden config; after Phase 6
            # (1280 decoder) this can be turned off via a separate yaml knob if
            # decoder activation is no longer the bottleneck.
            if hasattr(self, "padt_decoder"):
                self.padt_decoder._use_grad_ckpt = True
        except Exception as e:  # noqa: BLE001
            # If the underlying HF model does not expose the standard interface,
            # fall back silently — we'd rather lose checkpointing than crash.
            import logging
            logging.getLogger(__name__).warning(
                "QwenPaDTPI: gradient_checkpointing_enable failed (%s); continuing without GC.",
                e,
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _role_ids_from_strings(self, task_object_roles: List[List[str]], device: torch.device) -> torch.Tensor:
        role_ids = torch.zeros((len(task_object_roles), self.max_task_objects), dtype=torch.long, device=device)
        unk_id = len(self.role_to_idx)
        for batch_idx, roles in enumerate(task_object_roles):
            for obj_idx, role_name in enumerate(roles[: self.max_task_objects]):
                role_ids[batch_idx, obj_idx] = self.role_to_idx.get(str(role_name), unk_id)
        return role_ids

    def _build_action_condition(
        self,
        *,
        lang_summary: torch.Tensor,
        object_memory: torch.Tensor,
        state: Optional[torch.Tensor],
        object_view_memory: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Pack PaDT decoder memory into the PI head's per-layer condition tokens."""
        return self.condition_bridge(
            lang_summary=lang_summary,
            object_memory=object_memory,
            object_view_memory=object_view_memory,
            state=state,
        )

    def _prepare_action_targets(self, actions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return actions[:, -(self.future_action_window_size + 1) :, :].to(dtype=dtype)

    @staticmethod
    def _vector_norm_per_sample(value: torch.Tensor) -> np.ndarray:
        return torch.linalg.vector_norm(value.detach().float(), dim=-1).cpu().numpy()

    @staticmethod
    def _token_norm_per_sample(value: torch.Tensor) -> np.ndarray:
        token_norm = torch.linalg.vector_norm(value.detach().float(), dim=-1)
        if token_norm.ndim == 1:
            return token_norm.cpu().numpy()
        return token_norm.mean(dim=-1).cpu().numpy()

    def _condition_norm_per_sample(self, action_condition: List[torch.Tensor] | torch.Tensor) -> np.ndarray:
        if isinstance(action_condition, list):
            if not action_condition:
                return np.zeros((0,), dtype=np.float32)
            per_layer = [torch.from_numpy(self._token_norm_per_sample(layer)).to(dtype=torch.float32) for layer in action_condition]
            return torch.stack(per_layer, dim=0).mean(dim=0).cpu().numpy()
        return self._token_norm_per_sample(action_condition)

    def _compute_total_loss(
        self,
        *,
        action_loss: torch.Tensor,
        vrt_loss: torch.Tensor,
        loss_bbox: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_score: torch.Tensor,
    ) -> torch.Tensor:
        loss_weights = self.padt_cfg.get("loss_weights", {})
        lambda_act = float(loss_weights.get("act", 1.0))
        lambda_vrt = float(loss_weights.get("vrt", 0.5))
        lambda_bbox = float(loss_weights.get("bbox", 0.25))
        lambda_mask = float(loss_weights.get("mask", 0.25))
        lambda_score = float(loss_weights.get("score", 0.1))
        total = (
            lambda_act * action_loss
            + lambda_vrt * vrt_loss
            + lambda_bbox * loss_bbox
            + lambda_mask * loss_mask
            + lambda_score * loss_score
        )
        return total

    def _build_vrt_token_sequences(
        self,
        *,
        final_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        task_object_roles: List[List[str]],
        object_presence_mask: torch.Tensor,
    ) -> GroupedVRTHidden:
        grouped = group_vrt_hidden_by_slots(
            final_hidden=final_hidden,
            input_ids=input_ids,
            obj_token_ids=self.qwen_vl_interface.obj_token_ids,
            view_token_ids=getattr(self.qwen_vl_interface, "view_token_ids", None),
            vrt_start_id=self.qwen_vl_interface.vrt_start_id,
            vrt_end_id=self.qwen_vl_interface.vrt_end_id,
            max_task_objects=self.max_task_objects,
            num_core_tokens=self.num_core_vrt_tokens,
            num_views=self.num_vrt_views,
            object_presence_mask=object_presence_mask,
        )
        role_ids = self._role_ids_from_strings(
            task_object_roles=task_object_roles,
            device=grouped.vrt_token_sequences.device,
        )
        grouped.vrt_token_sequences = grouped.vrt_token_sequences + self.role_embedding(role_ids)[:, :, None, None, :]
        return grouped

    def _maybe_sampled_action_branch(
        self,
        *,
        batch,
        patch_features: Dict[str, torch.Tensor],
        P_ref: torch.Tensor,
        state: Optional[torch.Tensor],
        action_target: torch.Tensor,
        base_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not bool(self.padt_cfg.get("use_sampled_branch", False)):
            return action_target.sum() * 0.0
        sampled_weight = float(self.padt_cfg.get("sampled_branch_weight", 0.0))
        if sampled_weight <= 0.0:
            return action_target.sum() * 0.0

        sampled_decode = self.qwen_vl_interface.custom_vrt_decode(
            images=batch.images,
            instructions=batch.instructions,
            object_roles=batch.task_object_roles,
            object_presence_mask=batch.object_presence_mask,
            P_ref=P_ref,
            # Reuse visual tokens from extract_patch_features (Phase 1).
            precomputed_image_embeds=patch_features.get("_raw_visual_tokens"),
        )
        sampled_queries = self._build_vrt_token_sequences(
            final_hidden=sampled_decode.final_hidden,
            input_ids=sampled_decode.input_ids,
            task_object_roles=batch.task_object_roles,
            object_presence_mask=batch.object_presence_mask,
        )
        sampled_dec = self.padt_decoder(
            sampled_queries.vrt_token_sequences,
            patch_features["all"],
            patch_features.get("high_res_all", None),
            object_presence_mask=batch.object_presence_mask,
        )
        sampled_cond = self._build_action_condition(
            lang_summary=sampled_decode.lang_summary,
            object_memory=sampled_dec.object_memory,
            state=state,
            object_view_memory=sampled_dec.object_view_memory,
        )
        sampled_action_loss = self.action_model(sampled_cond, action_target, state)
        return sampled_weight * sampled_action_loss.to(dtype=base_dtype)

    # ------------------------------------------------------------------
    # train / infer
    # ------------------------------------------------------------------
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        if examples and any("action" not in s or s["action"] is None for s in examples):
            raise ValueError("QwenPaDTPI.forward (training) requires `action` in every example.")
        batch = preprocess_raw_dict(examples=examples, config=self.config)
        patch_features = self.qwen_vl_interface.extract_patch_features(
            images=batch.images,
            instructions=batch.instructions,
            object_roles=batch.task_object_roles,
        )
        P_ref = self.qwen_vl_interface.build_prototypes(patch_features["vrt_bank"])

        noisy_teacher_probability = 0.0
        if bool(self.padt_cfg.get("use_noisy_teacher_branch", True)):
            noisy_teacher_probability = float(self.padt_cfg.get("noisy_teacher_probability", 0.0))
        teacher: TeacherSequenceBatch = build_structured_teacher_seq(
            batch=batch,
            token_table=self.qwen_vl_interface.token_table,
            noisy_teacher_probability=noisy_teacher_probability,
        )

        dynamic_outputs = self.qwen_vl_interface.forward_dynamic(
            images=batch.images,
            instructions=batch.instructions,
            object_roles=batch.task_object_roles,
            solutions=teacher.solutions,
            P_ref=P_ref,
            teacher_core_patch_ids=teacher.core_patch_ids,
            valid_patch_mask_per_token=teacher.valid_patch_mask_per_token.to(P_ref.device),
            # Reuse the visual tokens already produced by extract_patch_features to
            # avoid a redundant ViT forward (Phase 1 in perf optimization plan).
            precomputed_image_embeds=patch_features.get("_raw_visual_tokens"),
        )
        grouped = self._build_vrt_token_sequences(
            final_hidden=dynamic_outputs.final_hidden,
            input_ids=dynamic_outputs.input_ids,
            task_object_roles=batch.task_object_roles,
            object_presence_mask=batch.object_presence_mask.to(dynamic_outputs.final_hidden.device),
        )
        decoder_outputs = self.padt_decoder(
            grouped.vrt_token_sequences,
            patch_features["all"],
            patch_features.get("high_res_all", None),
            target_boxes_by_view=batch.target_boxes_by_view.to(
                dynamic_outputs.final_hidden.device,
                dtype=patch_features["all"].dtype,
            ),
            target_patch_masks_by_view=batch.target_patch_masks_by_view.to(
                dynamic_outputs.final_hidden.device,
                dtype=patch_features["all"].dtype,
            ),
            target_visible_by_view=batch.target_visible_by_view.to(dynamic_outputs.final_hidden.device),
            object_presence_mask=batch.object_presence_mask.to(dynamic_outputs.final_hidden.device),
        )

        state = batch.state.to(dynamic_outputs.final_hidden.device, dtype=patch_features["all"].dtype) if batch.state is not None else None
        action_target = self._prepare_action_targets(batch.actions.to(dynamic_outputs.final_hidden.device), dtype=patch_features["all"].dtype)
        action_condition = self._build_action_condition(
            lang_summary=dynamic_outputs.lang_summary,
            object_memory=decoder_outputs.object_memory,
            state=state,
            object_view_memory=decoder_outputs.object_view_memory,
        )
        action_loss = self.action_model(action_condition, action_target, state)

        sampled_action_loss = self._maybe_sampled_action_branch(
            batch=batch,
            patch_features=patch_features,
            P_ref=P_ref,
            state=state,
            action_target=action_target,
            base_dtype=action_loss.dtype,
        )
        total_loss = self._compute_total_loss(
            action_loss=action_loss + sampled_action_loss,
            vrt_loss=dynamic_outputs.vrt_loss,
            loss_bbox=decoder_outputs.loss_bbox,
            loss_mask=decoder_outputs.loss_mask,
            loss_score=decoder_outputs.loss_score,
        )

        return {
            # Trainer currently reads only `action_loss`, so total loss is aggregated here.
            "action_loss": total_loss,
            "loss_action_fm": action_loss.detach(),
            "loss_action_sampled": sampled_action_loss.detach(),
            "loss_vrt": dynamic_outputs.vrt_loss.detach(),
            "loss_bbox": decoder_outputs.loss_bbox.detach(),
            "loss_patch_mask": decoder_outputs.loss_mask.detach(),
            "loss_score": decoder_outputs.loss_score.detach(),
            **{name: value.detach() for name, value in dynamic_outputs.vrt_diagnostics.items()},
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, return_debug: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        if type(examples) is not list:
            examples = [examples]
        batch = preprocess_raw_dict(examples=examples, config=self.config)
        patch_features = self.qwen_vl_interface.extract_patch_features(
            images=batch.images,
            instructions=batch.instructions,
            object_roles=batch.task_object_roles,
        )
        P_ref = self.qwen_vl_interface.build_prototypes(patch_features["vrt_bank"])

        decoded = self.qwen_vl_interface.custom_vrt_decode(
            images=batch.images,
            instructions=batch.instructions,
            object_roles=batch.task_object_roles,
            object_presence_mask=batch.object_presence_mask.to(P_ref.device),
            P_ref=P_ref,
            # Reuse visual tokens from extract_patch_features (Phase 1).
            precomputed_image_embeds=patch_features.get("_raw_visual_tokens"),
        )
        grouped = self._build_vrt_token_sequences(
            final_hidden=decoded.final_hidden,
            input_ids=decoded.input_ids,
            task_object_roles=batch.task_object_roles,
            object_presence_mask=batch.object_presence_mask.to(decoded.final_hidden.device),
        )
        # Align with training: training passes target_visible_by_view so absent slots
        # are masked out of object_memory aggregation (zero norm). Without it the
        # decoder falls back to query.mean(dim=(2,3)) and absent slots leak a large
        # non-zero memory into the action condition, which the action head was
        # never exposed to. We use object_presence_mask broadcast across views as
        # the inference visibility proxy: every present slot is visible in every
        # view, absent slots are masked. This zeroes object_memory for absent slots
        # exactly as in training; the minor per-view weighting drift for partially
        # occluded objects is a deliberate compromise to avoid a second decoder pass.
        presence_device = decoded.final_hidden.device
        presence_dtype = patch_features["all"].dtype
        visibility_proxy = (
            batch.object_presence_mask
            .to(device=presence_device, dtype=presence_dtype)
            .unsqueeze(-1)
            .expand(-1, -1, self.num_vrt_views)
        )
        decoder_outputs = self.padt_decoder(
            grouped.vrt_token_sequences,
            patch_features["all"],
            patch_features.get("high_res_all", None),
            target_visible_by_view=visibility_proxy,
            object_presence_mask=batch.object_presence_mask.to(presence_device),
        )
        state = batch.state.to(decoded.final_hidden.device, dtype=patch_features["all"].dtype) if batch.state is not None else None
        action_condition = self._build_action_condition(
            lang_summary=decoded.lang_summary,
            object_memory=decoder_outputs.object_memory,
            state=state,
            object_view_memory=decoder_outputs.object_view_memory,
        )
        pred_actions = self.action_model.predict_action(action_condition, state)
        normalized_actions = pred_actions.detach().float().cpu().numpy()
        output: Dict[str, Any] = {"normalized_actions": normalized_actions}
        if return_debug:
            output["debug"] = {
                "predicted_patch_ids": decoded.predicted_patch_ids.detach().cpu().numpy(),
                "object_presence_mask": batch.object_presence_mask.detach().cpu().numpy(),
                "task_object_roles": [list(roles) for roles in batch.task_object_roles],
                "bbox_by_view": decoder_outputs.bbox_by_view.detach().float().cpu().numpy(),
                "patch_mask_by_view": torch.sigmoid(decoder_outputs.patch_mask_by_view.detach().float()).cpu().numpy(),
                "score_logits": decoder_outputs.score_logits.detach().float().cpu().numpy(),
                "visibility_logits": decoder_outputs.visibility_logits.detach().float().cpu().numpy(),
                "lang_summary_norm": self._vector_norm_per_sample(decoded.lang_summary),
                "object_memory_norm": self._vector_norm_per_sample(decoder_outputs.object_memory),
                "action_condition_norm": self._condition_norm_per_sample(action_condition),
                "normalized_actions": normalized_actions,
            }
        return output
