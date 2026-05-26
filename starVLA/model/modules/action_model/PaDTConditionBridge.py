# Copyright 2026 OpenAI.
# Bridge for feeding PaDT condition tokens into the PI FM head.

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn


class PaDTConditionBridge(nn.Module):
    """Project PaDT language/object memories into PI per-layer condition tokens."""

    def __init__(self, config: Any):
        super().__init__()
        padt_cfg = config.framework.get("padt", {})
        hidden_dim = int(config.framework.qwenvl.vl_hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_vl_layers = int(config.framework.qwenvl.num_vl_layers)
        self.max_object_tokens = int(padt_cfg.get("bridge_max_object_tokens", 4))
        self.max_view_tokens = int(padt_cfg.get("bridge_max_view_tokens", 8))
        self.max_condition_tokens = int(padt_cfg.get("bridge_max_condition_tokens", 16))

        self.lang_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.object_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.view_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_embedding = nn.Embedding(self.max_condition_tokens, hidden_dim)

    def _compose_tokens(
        self,
        lang_summary: torch.Tensor,
        object_memory: torch.Tensor,
        object_view_memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, O, D = object_memory.shape
        lang_token = self.lang_proj(lang_summary).unsqueeze(1)
        object_tokens = self.object_proj(object_memory[:, : self.max_object_tokens])
        token_parts = [lang_token, object_tokens]
        if object_view_memory is not None:
            view_tokens = object_view_memory[:, : self.max_object_tokens].reshape(B, -1, D)
            view_tokens = self.view_proj(view_tokens[:, : self.max_view_tokens])
            token_parts.append(view_tokens)
        tokens = torch.cat(token_parts, dim=1)
        if tokens.shape[1] < self.max_condition_tokens:
            pad = torch.zeros(
                (B, self.max_condition_tokens - tokens.shape[1], D),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens = torch.cat((tokens, pad), dim=1)
        else:
            tokens = tokens[:, : self.max_condition_tokens]
        slot_ids = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.slot_embedding(slot_ids).unsqueeze(0)
        return tokens

    def forward(
        self,
        lang_summary: torch.Tensor,
        object_memory: torch.Tensor,
        object_view_memory: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        del state  # state remains a separate explicit input to the PI action head.
        cond_tokens = self._compose_tokens(
            lang_summary=lang_summary,
            object_memory=object_memory,
            object_view_memory=object_view_memory,
        )
        return [cond_tokens for _ in range(self.num_vl_layers)]
