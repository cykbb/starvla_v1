# Copyright 2026 OpenAI.
# PaDT-style multi-view decoder for QwenPaDTPI.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    return torch.stack(((x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0).clamp_min(0.0), (y1 - y0).clamp_min(0.0)), dim=-1)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0.0)
    return wh[..., 0] * wh[..., 1]


def _pairwise_aligned_giou(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    rb = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    union = _box_area(pred_xyxy) + _box_area(target_xyxy) - inter
    iou = inter / union.clamp_min(1e-6)

    enc_lt = torch.minimum(pred_xyxy[..., :2], target_xyxy[..., :2])
    enc_rb = torch.maximum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    enc_area = _box_area(torch.cat((enc_lt, enc_rb), dim=-1)).clamp_min(1e-6)
    return iou - (enc_area - union) / enc_area


def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    loss_mask = loss_mask.flatten(1)
    numerator = 2.0 * (inputs * targets * loss_mask).sum(dim=-1)
    denominator = (inputs * loss_mask).sum(dim=-1) + (targets * loss_mask).sum(dim=-1)
    loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    valid = (loss_mask.sum(dim=-1) > 0).float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * ce_loss * ((1.0 - p_t) ** gamma)
    loss = (loss * loss_mask).flatten(1)
    denom = loss_mask.flatten(1).sum(dim=-1).clamp_min(1e-5)
    valid = (loss_mask.flatten(1).sum(dim=-1) > 0).float()
    return ((loss.sum(dim=-1) / denom) * valid).sum() / valid.sum().clamp_min(1.0)


@dataclass
class PaDTDecoderOutput:
    bbox_by_view: torch.Tensor
    patch_mask_by_view: torch.Tensor
    score_logits: torch.Tensor
    visibility_logits: torch.Tensor
    object_memory: torch.Tensor
    loss_bbox: torch.Tensor
    loss_mask: torch.Tensor
    loss_score: torch.Tensor
    object_view_memory: Optional[torch.Tensor] = None


class _PaDTStyleBlock(nn.Module):
    """PaDT-style query/image block scoped to one object in one view."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, mlp_ratio: int = 4, update_memory: bool = True):
        super().__init__()
        self.update_memory = update_memory
        self.query_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.query_to_image = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.image_to_query = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) if update_memory else None
        self.q_norm1 = nn.LayerNorm(hidden_dim)
        self.q_norm2 = nn.LayerNorm(hidden_dim)
        self.q_norm3 = nn.LayerNorm(hidden_dim)
        self.m_norm1 = nn.LayerNorm(hidden_dim)
        self.m_norm2 = nn.LayerNorm(hidden_dim) if update_memory else None
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_norm1(query)
        query = query + self.query_self_attn(q, q, q, need_weights=False)[0]

        q = self.q_norm2(query)
        m = self.m_norm1(memory)
        query = query + self.query_to_image(q, m, m, need_weights=False)[0]
        query = query + self.mlp(self.q_norm3(query))

        if self.update_memory and self.image_to_query is not None and self.m_norm2 is not None:
            m = self.m_norm2(memory)
            q = self.q_norm3(query)
            memory = memory + self.image_to_query(m, q, q, need_weights=False)[0]
        return query, memory


class PaDTObjectDecoder(nn.Module):
    """PaDT-aligned two-resolution decoder with token-level, per-view VRT input.

    Inputs:
        vrt_token_sequences: [B, O, V, K, D]
        low_res_features: [B, V*N, D]
        high_res_features: [B, V*HN, D], where HN is usually 4*N
    """

    def __init__(self, config: Any):
        super().__init__()
        padt_cfg = config.framework.get("padt", {})
        hidden_dim = int(config.framework.qwenvl.vl_hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_views = int(padt_cfg.get("decoder_num_views", 2))
        self.num_patch_tokens_per_view = int(padt_cfg.get("num_vrt_tokens", 256))
        self.num_core_tokens = int(padt_cfg.get("num_core_vrt_tokens", 5))
        self.num_heads = int(padt_cfg.get("decoder_num_heads", 8))
        self.spatial_merge_size = int(padt_cfg.get("spatial_merge_size", 2))
        self.high_res_tokens_per_view = int(
            padt_cfg.get(
                "high_res_tokens_per_view",
                self.num_patch_tokens_per_view * self.spatial_merge_size * self.spatial_merge_size,
            )
        )

        self.view_embedding = nn.Embedding(self.num_views, hidden_dim)
        self.vrt_embedding = nn.Embedding(1, hidden_dim)
        self.task_tokens = nn.Embedding(3, hidden_dim)  # bbox, score, mask
        self.low_xy_embedding = nn.Linear(2, hidden_dim)
        self.high_xy_embedding = nn.Linear(2, hidden_dim)

        self.low_res_block = _PaDTStyleBlock(hidden_dim, self.num_heads, update_memory=True)
        self.high_res_block1 = _PaDTStyleBlock(hidden_dim, self.num_heads, update_memory=True)
        self.high_res_block2 = _PaDTStyleBlock(hidden_dim, self.num_heads, update_memory=True)
        self.high_res_norm = nn.LayerNorm(hidden_dim)

        self.bbox_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.mask_output_upscaling1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mask_output_upscaling2 = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.GELU(),
        )
        self.mask_output_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 16),
        )
        self.object_memory_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _add_patch_context(self, features: torch.Tensor, per_view_tokens: int, xy_embedding: nn.Linear) -> torch.Tensor:
        B, total_tokens, D = features.shape
        expected_total = self.num_views * per_view_tokens
        if total_tokens < expected_total:
            pad = torch.zeros(
                (B, expected_total - total_tokens, D),
                dtype=features.dtype,
                device=features.device,
            )
            features = torch.cat((features, pad), dim=1)
        features = features[:, :expected_total].view(B, self.num_views, per_view_tokens, D)

        grid_h = int(per_view_tokens ** 0.5)
        grid_w = max(1, per_view_tokens // max(1, grid_h))
        if grid_h * grid_w != per_view_tokens:
            grid_h, grid_w = per_view_tokens, 1
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid_h, device=features.device, dtype=features.dtype),
            torch.linspace(-1.0, 1.0, grid_w, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        pos = torch.stack((xs, ys), dim=-1).view(1, 1, per_view_tokens, 2)
        view_ids = torch.arange(self.num_views, device=features.device)
        view_embed = self.view_embedding(view_ids).view(1, self.num_views, 1, D)
        return features + xy_embedding(pos) + view_embed

    def _prepare_queries(self, vrt_token_sequences: torch.Tensor) -> torch.Tensor:
        if vrt_token_sequences.dim() == 4:
            vrt_token_sequences = vrt_token_sequences.unsqueeze(2).expand(-1, -1, self.num_views, -1, -1)
        if vrt_token_sequences.dim() != 5:
            raise ValueError(f"Expected VRT sequences [B,O,V,K,D], got {tuple(vrt_token_sequences.shape)}")
        B, O, V, K, D = vrt_token_sequences.shape
        if V != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {V}")
        if K < self.num_core_tokens:
            pad = torch.zeros(
                (B, O, V, self.num_core_tokens - K, D),
                dtype=vrt_token_sequences.dtype,
                device=vrt_token_sequences.device,
            )
            vrt_token_sequences = torch.cat((vrt_token_sequences, pad), dim=3)
        vrt = vrt_token_sequences[:, :, :, : self.num_core_tokens] + self.vrt_embedding.weight.view(1, 1, 1, 1, D)
        task = self.task_tokens.weight.to(dtype=vrt.dtype, device=vrt.device).view(1, 1, 1, 3, D)
        task = task.expand(B, O, V, -1, -1)
        return torch.cat((task, vrt), dim=3)

    def _bbox_loss(self, pred_cxcywh: torch.Tensor, target_xyxy: torch.Tensor, visible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visible_f = visible.float()
        if visible_f.sum() == 0:
            zero = pred_cxcywh.sum() * 0.0
            return zero, torch.zeros_like(visible_f)
        target_cxcywh = _xyxy_to_cxcywh(target_xyxy)
        l1 = F.l1_loss(pred_cxcywh, target_cxcywh, reduction="none").sum(dim=-1)
        pred_xyxy = _cxcywh_to_xyxy(pred_cxcywh).clamp(0.0, 1.0)
        target_xyxy = target_xyxy.clamp(0.0, 1.0)
        giou = _pairwise_aligned_giou(pred_xyxy, target_xyxy)
        loss = (l1 + (1.0 - giou)) * visible_f
        return loss.sum() / visible_f.sum().clamp_min(1.0), giou.detach()

    def _decode_high_res_masks(self, mask_token: torch.Tensor, high_memory: torch.Tensor) -> torch.Tensor:
        B, O, V, HN, D = high_memory.shape
        side = int(HN ** 0.5)
        mask_query = self.mask_output_mlp(mask_token)
        if side * side != HN:
            return torch.einsum("bovd,bovnd->bovn", mask_query, high_memory)

        up1 = self.mask_output_upscaling1(high_memory).view(B, O, V, side, side, 2, 2, D // 4)
        up1 = up1.permute(0, 1, 2, 3, 5, 4, 6, 7).reshape(B, O, V, side * 2, side * 2, D // 4)
        up2 = self.mask_output_upscaling2(up1).view(B, O, V, side * 2, side * 2, 2, 2, D // 16)
        up2 = up2.permute(0, 1, 2, 3, 5, 4, 6, 7).reshape(B, O, V, side * 4, side * 4, D // 16)
        return torch.einsum("bovd,bovhwd->bovhw", mask_query, up2)

    def _downsample_masks_to_low_res(self, mask_logits: torch.Tensor) -> torch.Tensor:
        if mask_logits.dim() == 4:
            return mask_logits[:, :, :, : self.num_patch_tokens_per_view]
        B, O, V, H, W = mask_logits.shape
        side = int(self.num_patch_tokens_per_view ** 0.5)
        if side * side != self.num_patch_tokens_per_view:
            return mask_logits.reshape(B, O, V, -1)[:, :, :, : self.num_patch_tokens_per_view]
        pooled = F.adaptive_avg_pool2d(mask_logits.reshape(B * O * V, 1, H, W), (side, side))
        return pooled.reshape(B, O, V, self.num_patch_tokens_per_view)

    def _mask_loss(self, mask_logits: torch.Tensor, target_patch_mask: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        B, O, V = target_patch_mask.shape[:3]
        if mask_logits.dim() == 5:
            _, _, _, H, W = mask_logits.shape
            target_side = int(target_patch_mask.shape[-1] ** 0.5)
            if target_side * target_side == target_patch_mask.shape[-1]:
                target = target_patch_mask.reshape(B * O * V, 1, target_side, target_side)
                target = F.interpolate(target, size=(H, W), mode="nearest").reshape(B, O, V, H, W)
            else:
                target = target_patch_mask.reshape(B, O, V, 1, -1).expand(-1, -1, -1, H, -1)
                target = target[:, :, :, :, :W]
            loss_mask = visible.to(dtype=mask_logits.dtype).view(B, O, V, 1, 1).expand_as(mask_logits)
            flat_logits = mask_logits.reshape(B * O * V, H, W)
            flat_target = target.to(dtype=mask_logits.dtype).reshape(B * O * V, H, W)
            flat_mask = loss_mask.reshape(B * O * V, H, W)
        else:
            HN = mask_logits.shape[-1]
            repeat = max(1, HN // max(1, target_patch_mask.shape[-1]))
            target = target_patch_mask.unsqueeze(-1).expand(-1, -1, -1, -1, repeat).reshape(B, O, V, -1)
            target = target[:, :, :, :HN].to(dtype=mask_logits.dtype)
            loss_mask = visible.to(dtype=mask_logits.dtype).unsqueeze(-1).expand_as(mask_logits)
            flat_logits = mask_logits.reshape(B * O * V, HN).unsqueeze(1)
            flat_target = target.reshape(B * O * V, HN).unsqueeze(1)
            flat_mask = loss_mask.reshape(B * O * V, HN).unsqueeze(1)
        return _dice_loss(flat_logits, flat_target, flat_mask) + _sigmoid_focal_loss(flat_logits, flat_target, flat_mask)

    def _score_loss(
        self,
        score_logits: torch.Tensor,
        giou: torch.Tensor,
        visible: torch.Tensor,
        object_presence_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if object_presence_mask is None:
            loss_mask = torch.ones_like(score_logits, dtype=score_logits.dtype)
        else:
            loss_mask = object_presence_mask.to(dtype=score_logits.dtype).unsqueeze(-1).expand_as(score_logits)
        if loss_mask.sum() == 0:
            return score_logits.sum() * 0.0
        pred_score = score_logits.sigmoid() * 2.0 - 1.0
        target_score = torch.where(visible.bool(), giou.detach(), torch.full_like(giou, -1.0))
        return (((pred_score - target_score) ** 2) * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    def forward(
        self,
        vrt_token_sequences: torch.Tensor,
        low_res_features: torch.Tensor,
        high_res_features: Optional[torch.Tensor] = None,
        *,
        target_boxes_by_view: Optional[torch.Tensor] = None,
        target_patch_masks_by_view: Optional[torch.Tensor] = None,
        target_visible_by_view: Optional[torch.Tensor] = None,
        object_presence_mask: Optional[torch.Tensor] = None,
    ) -> PaDTDecoderOutput:
        if high_res_features is None:
            repeat = self.high_res_tokens_per_view // self.num_patch_tokens_per_view
            high_res_features = low_res_features.view(
                low_res_features.shape[0], self.num_views, self.num_patch_tokens_per_view, self.hidden_dim
            ).repeat_interleave(repeat, dim=2).view(
                low_res_features.shape[0],
                self.num_views * self.high_res_tokens_per_view,
                self.hidden_dim,
            )

        low_memory = self._add_patch_context(low_res_features, self.num_patch_tokens_per_view, self.low_xy_embedding)
        high_memory = self._add_patch_context(high_res_features, self.high_res_tokens_per_view, self.high_xy_embedding)
        queries = self._prepare_queries(vrt_token_sequences)

        B, O, V, L, D = queries.shape
        query = queries.reshape(B * O * V, L, D)
        view_ids = torch.arange(V, device=queries.device).view(1, 1, V).expand(B, O, V).reshape(-1)

        # Select only the image memory from the matching view for each object-view item.
        low_by_view = low_memory[:, :, :].reshape(B, V, self.num_patch_tokens_per_view, D)
        high_by_view = high_memory[:, :, :].reshape(B, V, self.high_res_tokens_per_view, D)
        batch_ids = torch.arange(B, device=queries.device).view(B, 1, 1).expand(B, O, V).reshape(-1)
        low = low_by_view[batch_ids, view_ids]
        high = high_by_view[batch_ids, view_ids]

        query, low = self.low_res_block(query, low)
        high_repeat = max(1, self.high_res_tokens_per_view // self.num_patch_tokens_per_view)
        high = self.high_res_norm(
            high + low.repeat_interleave(high_repeat, dim=1)[:, : self.high_res_tokens_per_view]
        )
        query, high = self.high_res_block1(query, high)
        query, high = self.high_res_block2(query, high)

        query = query.view(B, O, V, L, D)
        high = high.view(B, O, V, self.high_res_tokens_per_view, D)
        bbox_token = query[:, :, :, 0]
        score_token = query[:, :, :, 1]
        mask_token = query[:, :, :, 2]

        bbox_cxcywh = self.bbox_head(bbox_token)
        bbox_by_view = _cxcywh_to_xyxy(bbox_cxcywh).clamp(0.0, 1.0)
        score_logits_by_view = self.score_head(score_token).squeeze(-1)
        visibility_logits = score_logits_by_view
        high_res_mask = self._decode_high_res_masks(mask_token, high)
        patch_mask_by_view = self._downsample_masks_to_low_res(high_res_mask)

        if target_visible_by_view is not None:
            visible_f = target_visible_by_view.to(dtype=query.dtype)
            object_memory = (query.mean(dim=3) * visible_f.unsqueeze(-1)).sum(dim=2)
            object_memory = object_memory / visible_f.sum(dim=2, keepdim=True).clamp_min(1.0)
            score_logits = (score_logits_by_view * visible_f).sum(dim=2) / visible_f.sum(dim=2).clamp_min(1.0)
        else:
            object_memory = query.mean(dim=(2, 3))
            score_logits = score_logits_by_view.mean(dim=2)
        object_memory = self.object_memory_proj(object_memory)
        object_view_memory = self.object_memory_proj(query.mean(dim=3))

        loss_bbox = bbox_by_view.sum() * 0.0
        loss_mask = patch_mask_by_view.sum() * 0.0
        loss_score = score_logits.sum() * 0.0
        giou = torch.zeros_like(score_logits_by_view)
        if target_boxes_by_view is not None and target_visible_by_view is not None:
            loss_bbox, giou = self._bbox_loss(bbox_cxcywh, target_boxes_by_view, target_visible_by_view)
        if target_patch_masks_by_view is not None and target_visible_by_view is not None:
            loss_mask = self._mask_loss(high_res_mask, target_patch_masks_by_view, target_visible_by_view)
        if target_visible_by_view is not None:
            loss_score = self._score_loss(score_logits_by_view, giou, target_visible_by_view, object_presence_mask)

        return PaDTDecoderOutput(
            bbox_by_view=bbox_by_view,
            patch_mask_by_view=patch_mask_by_view,
            score_logits=score_logits,
            visibility_logits=visibility_logits,
            object_memory=object_memory,
            object_view_memory=object_view_memory,
            loss_bbox=loss_bbox,
            loss_mask=loss_mask,
            loss_score=loss_score,
        )
