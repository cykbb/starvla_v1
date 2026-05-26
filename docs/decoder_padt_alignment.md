# Plan: Align starVLA Decoder with Original PaDT Architecture

## Context

The current starVLA QwenPaDTPI decoder deviates from the original PaDT in several ways
that cause VRT predictions to be ineffective:
1. Mean pooling of K VRT tokens into 1 vector destroys patch identity information
2. Cross-object self-attention (unlike original's within-object isolation)
3. Missing Sigmoid on bbox head — outputs unbounded values
4. No two-level (low-res + high-res) visual memory hierarchy

This refactoring eliminates the mean-pooling bottleneck and restructures the decoder
to match the original PaDT architecture, making VRT predictions consequential for
downstream bbox/mask/action quality.

## Files to Modify

| File | Scope |
|------|-------|
| `starVLA/model/modules/vlm/padt_object_decoder.py` | Complete rewrite |
| `starVLA/model/modules/vlm/QWen2_5_PaDT.py` | Add dual-resolution feature extraction |
| `starVLA/model/modules/vlm/padt_data_utils.py` | Remove mean pooling |
| `starVLA/model/framework/QwenPaDTPI.py` | Update orchestration to new interfaces |

## Step 1: Rewrite `padt_object_decoder.py` (core change)

### 1.1 Replace `_TwoWayBlock` with `PaDTDecoderBlock`

New block structure (aligned with original):
- **self_attn**: within-object only, via block-diagonal attention mask built from cu_seqlens
- **cross_attn (Q→image)**: object tokens attend to full image memory
- **FFN**: standard MLP
- **reverse_cross_attn (image→Q)** (when update_memory=True): image memory updated by query

Self-attention scope: bbox_token of obj0 can see vrt_1 of obj0, but NOT vrt_1 of obj1.
Cross-object information flows indirectly through the shared image memory.

### 1.2 Replace `PaDTObjectDecoder` with new architecture

**Input**:
- `vrt_token_sequences`: [B, O, K, D] — individual VRT hidden states (no mean pool)
- `low_res_features`: [B, V*N, D] — merged patch features (current patch_features["all"])
- `high_res_features`: [B, V*4N, D] — pre-merger patch features (projected)

**Internal per-object token sequence**:
```
[bbox_token] [score_token] [vis_token] [vrt_1] [vrt_2] [vrt_3]
 ← 3 learnable task tokens →   ← K VRT tokens from LLM →
```
Total per object: L = 3 + K, flattened to [B, O*L, D].

**Decoder blocks**:
- Block 0 (low-res): process with low_res_features, update_memory=False
- Block 1 (high-res): process with high_res_features, update_memory=True
- Block 2 (high-res): process with high_res_features, update_memory=True

**Output heads** (dedicated per token type):
- `bbox_head(task_tokens[:, 0])` → [B, O, V, 4] — **with Sigmoid**
- `score_head(task_tokens[:, 1])` → [B, O]
- `visibility_head(task_tokens[:, 2])` → [B, O, V]
- `mask_logits` via einsum(mask_output, patch_tokens) — unchanged approach
- `object_memory_proj(mean(all tokens))` → [B, O, D] for action bridge

### 1.3 `PaDTDecoderOutput` dataclass

Keep existing fields: `bbox_by_view`, `patch_mask_by_view`, `score_logits`,
`visibility_logits`, `object_memory`, `loss_bbox`, `loss_mask`, `loss_score`.

## Step 2: Add dual-resolution visual feature extraction

File: `QWen2_5_PaDT.py`

### 2.1 Add `_extract_dual_res_visual_features` method

Replicates the Qwen2.5-VL visual encoder forward but returns both resolutions:
1. `patch_embed` → `rot_pos_emb` → `get_window_index`
2. Window reorder → all transformer blocks
3. **Save pre-merger** hidden states, reverse window ordering
4. Apply merger → reverse window ordering on merged
5. Return `(merged_features, pre_merger_features, position_embeddings)`

Pre-merger features are at 4× the token count of merged features.
Dimension: pre-merger = `vit_hidden_size` (1280), merged = `llm_hidden_size` (2048).

### 2.2 Fix `per_image_counts` bug

Current line 279 uses `grid_thw.prod(-1)` for merged features — wrong.
Fix to divide by `spatial_merge_size**2`.

### 2.3 Update `extract_patch_features` return

Add to returned dict:
- `"high_res_agentview"`: [B, 4N, D_vit]
- `"high_res_wrist"`: [B, 4N, D_vit]
- `"high_res_all"`: [B, 8N, D_vit] — concatenated, projected to decoder dim

Add `self.high_res_proj` (LayerNorm → Linear → GELU → Linear) to project
vit_hidden_size → decoder hidden_dim.

### 2.4 Build prototypes unchanged

`build_prototypes` still uses merged agentview features → P_ref.
VRT autoregression works at merged resolution (agentview-only, unchanged).

## Step 3: Remove mean pooling in `padt_data_utils.py`

### 3.1 `GroupedVRTHidden` dataclass

Replace `object_queries: torch.Tensor` [B, O, D] with:
`vrt_token_sequences: torch.Tensor` [B, O, K, D]

### 3.2 `group_vrt_hidden_by_slots`

Remove mean pool lines (577-580). Return raw VRT hidden states unchanged.

## Step 4: Update `QwenPaDTPI.py` orchestration

### 4.1 `_build_object_queries`

Remove role embedding addition (moved to decoder task tokens).
Return `grouped.vrt_token_sequences` instead of `grouped.object_queries`.

### 4.2 Decoder calls (forward + predict_action + sampled branch)

Old: `self.padt_decoder(object_queries, patch_features["all"], ...)`
New: `self.padt_decoder(vrt_seq, patch_features["all"], patch_features["high_res_all"], ...)`

### 4.3 Role embedding

Remove `self.role_embedding` and `_role_ids_from_strings`.
Role information can be injected via decoder task tokens if needed later.

### 4.4 Sampled branch

Update to pass both low-res and high-res features.
Also pass target supervision (bbox, mask, visibility) so sampled decoder
gets training signal — fixes the issue identified earlier.

## Verification

1. **Forward pass**: Run a single forward step with dummy batch to verify no shape errors
2. **Loss computation**: Verify all loss components (action, bbox, mask, score, VRT) are non-zero
3. **Inference**: Run `predict_action` and check debug output shapes
4. **Overlay images**: Verify bbox values are in [0,1] range (Sigmoid) and diagnostic
   visualization no longer needs sorted()/clamp() workarounds
5. **Training**: Run a few training steps to confirm loss decreases
