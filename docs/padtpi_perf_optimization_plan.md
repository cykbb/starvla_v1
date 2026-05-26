# PaDTPI 性能优化 + 重训对齐 Plan

> **目标**:
> 1. **工程优化阶段(Phase 1-5)**:训练单步加速 1.5-2×、单卡 BS 从 6 → 12+、推理同步加速,**前向数值在 bf16 噪声容差内不变**,可在现有 ckpt 上直接受益。
> 2. **结构对齐阶段(Phase 6-8)**:decoder hidden_size 从 2048 改为 1280、VRT token 从词表内改到词表外,**完全对齐原始 PaDT**。需要重训一次,但训练成本已被 Phase 1-5 显著降低。
>
> **隔离边界**:所有改动只触碰 `QwenPaDTPI` / `_QWen_PaDT_VL_Interface` / `PaDTObjectDecoder` / `PaDTConditionBridge` 路径,**不影响 `QwenPI` baseline 的任何文件或权重**。
>
> **决策已确认**(2026-05-26):用户接受重训,Phase 6 直接 commit 1280,不做 A/B。Phase 7(VRT 词表外)顺带改,跟 Phase 6 共用一次重训。

---

## 背景:PaDTPI 当前的性能差距

| 指标 | QwenPI baseline | QwenPaDTPI 当前 | 差距 |
|---|---|---|---|
| 单卡 BS 上限(LIBERO 448×448 双视图) | 16 | 6 | ~2.7× |
| 训练单步耗时 | 1× | ~4× | 4× |
| Eval 单帧延迟 | 1× | ~4× | 4× |

差距来源分解:

1. **ViT 跑两遍**(`extract_patch_features` 一次 + `forward_dynamic`/`custom_vrt_decode` 里 `_replace_image_embeddings` 又一次)
2. **`compute_dynamic_logits` 的 `logits.clone()`** — 每 forward 多保留一份 `[B, T, vocab≈152k]` 张量(~100 MB/step at B=6, T=600)
3. **`trainer.enable_gradient_checkpointing: true` 标志可能未真正接到 HF**(待 Phase 0 audit 确认)
4. **PaDT object decoder 在 `vl_hidden_dim=2048` 上跑 + `× O × V = 8×` batch 放大** — decoder 三个 attention block 的 activation 占 600-800 MB
5. **结构选择**:decoder 对齐 LLM 空间(2048)而非 ViT 空间(1280),`high_res_proj` 随机初始化导致检测头收敛慢;VRT token 进 tokenizer 留下 256 行"墓碑"权重

---

## 整体路线图

```
Phase 0   Audit                                          (no code change)
  ↓
Phase 1   Single ViT forward                              ┐
Phase 2   Remove logits.clone                             │ 不改效果
Phase 3   Wire up LLM/ViT gradient checkpointing          │ 不重训
Phase 4   PaDT decoder gradient checkpointing             │ 可在现有 ckpt 上验证
Phase 5   Integration validation                          ┘
  ↓
Phase 6   Decoder 重构 → hidden_size 1280                 ┐
Phase 7   VRT token 移到词表外                            │ 改结构
Phase 8   重训 + 最终验证                                  │ 需要重训一次
  ↓
完成:训练 / eval 速度恢复到与 QwenPI 同量级,结构完全对齐原始 PaDT
```

---

## Phase 0 — Audit(动代码前先盘点)

| 待确认项 | 位置 / 命令 | 结论用途 |
|---|---|---|
| ① `trainer.enable_gradient_checkpointing` 是否真接到 HF | `grep -rn "enable_gradient_checkpointing\|gradient_checkpointing_enable" starVLA/training/` | 决定 Phase 3 是"新增 wiring"还是"已 wired,跳过" |
| ② `extract_patch_features` 在所有调用点 | `QwenPaDTPI.py:234, 324`、`_maybe_sampled_action_branch` | 决定 Phase 1 接口透传要覆盖哪些路径 |
| ③ `forward_dynamic` 返回的 `outputs.logits` 除了 `compute_dynamic_logits` 外是否有其他消费者 | `_compute_vrt_loss` 入参 | 决定 Phase 2 用方案 A(原位写)还是方案 B(单独 VRT logits) |
| ④ 是否有外部代码 hook `model.visual` 的输出形状 | `grep -n "model.visual" starVLA/` | 防止 Phase 1 改 `_replace_image_embeddings` 签名漏改 |
| ⑤ `padt_object_decoder.py` 是否有其他下游消费者(非 condition_bridge) | `grep -rn "PaDTObjectDecoder\|object_memory" starVLA/` | 决定 Phase 6 重构 decoder 维度时接口改动范围 |
| ⑥ `tie_word_embeddings` 配置(Qwen2.5-VL-3B 是否绑定) | `grep -n "tie_word_embeddings" starVLA/` + Qwen 原始 config.json | 决定 Phase 7 移除 VRT 词表行时是否需要同时改 embed_tokens 和 lm_head |
| ⑦ `tokenizer.added_tokens.json` 当前内容 | 查 ckpt 目录下 tokenizer 文件 | Phase 7 要把 VRT 字符串从 tokenizer 删掉,需要知道现状 |

### Audit 结果(填表)

```
① GC flag wiring:                    [pending]
② extract_patch_features call sites: [pending]
③ outputs.logits consumers:          [pending]
④ model.visual hookers:              [pending]
⑤ PaDTObjectDecoder downstream:      [pending]
⑥ tie_word_embeddings:               [pending]
⑦ tokenizer.added_tokens content:    [pending]
```

---

## Phase 1 — 单次 ViT 前向(对齐原始 PaDT 的 `past_image_embeds`)

### 现状

ViT 在每次 forward 跑两次:
```
extract_patch_features        → _visual_forward_dual_res()        ← ViT pass 1
forward_dynamic / decode      → build_inputs_embeds()
                              → _replace_image_embeddings()
                              → self.model.visual(...)             ← ViT pass 2
```
第二次输出的 `image_embeds` 跟第一次 `low_res_hidden_states` bit-for-bit 相同。

### 改动

**`QWen2_5_PaDT.py`**

1. `extract_patch_features` 返回 dict 新增 `"_raw_visual_tokens": image_embeds`(`_visual_forward_dual_res` 已算好的 merged token,扁平 `[total_visual_tokens, hidden]`,**无新计算**)。

2. `_replace_image_embeddings` 签名增加 `precomputed_image_embeds=None`:
   ```python
   if precomputed_image_embeds is not None:
       image_embeds = precomputed_image_embeds
   else:
       image_embeds = self.model.visual(pixel_values.type(visual_dtype),
                                         grid_thw=image_grid_thw)
   ```

3. `build_inputs_embeds`、`forward_dynamic`、`custom_vrt_decode` 全部透传 `precomputed_image_embeds`,默认 None。

**`QwenPaDTPI.py`**

4. `forward()`、`predict_action()`、`_maybe_sampled_action_branch()` 把 `patch_features["_raw_visual_tokens"]` 透传给下游。

### 对齐原始 PaDT

✅ **完全对齐**。语义等价于 `past_image_embeds` 缓存。

### 验证

- 数值:1 step max-abs-diff < 1e-3
- 内存:训练阶段下降 200-400 MB
- 单步耗时:下降 = ViT 占总时长的 1/2

---

## Phase 2 — 去掉 `compute_dynamic_logits` 的全词表 clone

### 改动

依 Phase 0 ③ 结论分支:

**方案 A(优先,改动小):原位写**
```python
def compute_dynamic_logits(self, hidden, static_logits, P_ref, *, view_idx=0):
    P_view = self._select_p_ref(P_ref, view_idx)
    if hidden.dim() == 2:
        dynamic_vrt_logits = torch.einsum("bd,bnd->bn",  hidden, P_view)
    else:
        dynamic_vrt_logits = torch.einsum("bld,bnd->bln", hidden, P_view)
    static_logits[..., self.vrt_start_id : self.vrt_end_id] = dynamic_vrt_logits
    return static_logits
```

**方案 B(更对齐原始 PaDT):不构造全词表 logits**
- `compute_dynamic_logits` 只返回 `[B, T, 256]` 的 VRT 段
- `_compute_vrt_loss` 改用相对索引在 256 维 CE
- `custom_vrt_decode:926` 同步改

⚠️ **如果 Phase 7(VRT 词表外)要做**,**直接走方案 B 更顺**——Phase 7 也是"VRT 段单独算 logits"的结构,Phase 2 走 B 是 Phase 7 的前置铺路。

### 验证

- 数值:VRT CE 数值对齐到 bf16 噪声
- 内存:forward 段下降 ~100 MB(B=6, T=600)

---

## Phase 3 — 真正接上 LLM + ViT 的 gradient checkpointing

### 改动

依 Phase 0 ① 结论:

- **若已接上**:跳过
- **若未接上**:在 `QwenPaDTPI.__init__` 末尾追加
   ```python
   def _maybe_enable_gradient_checkpointing(self):
       if not getattr(self.config.trainer, "enable_gradient_checkpointing", False):
           return
       self.qwen_vl_interface.model.config.use_cache = False
       self.qwen_vl_interface.model.gradient_checkpointing_enable(
           gradient_checkpointing_kwargs={"use_reentrant": False},
       )
       self.qwen_vl_interface.model.visual.gradient_checkpointing = True
   ```

### 对齐原始 PaDT

✅ 完全对齐(`padt_sft_trainer.py:225-236` 同款)

### 验证

- LLM activation 下降 60-70%
- 单步耗时 +20-35%(BS 翻倍后摊薄)

---

## Phase 4 — PaDT object decoder 加 gradient checkpoint

### 范围注意

本 Phase 在 **decoder hidden_size 仍为 2048**(Phase 6 前)的状态下做最有价值。等 Phase 6 改成 1280 后 decoder 重量级 ↓ 60%,**Phase 4 的 checkpoint 不一定还必要**。

**所以本 Phase 的实施策略**:
- 实现时设计成**可配置开关**(`padt._use_grad_ckpt`),不写死
- Phase 6 完成后基于实测决定是否保留

### 改动

**`padt_object_decoder.py:forward`**:用 `torch.utils.checkpoint.checkpoint` 包住三个 `_PaDTStyleBlock` 和 `_decode_high_res_masks`。

**`QwenPaDTPI.__init__`**:`self.padt_decoder._use_grad_ckpt = config.trainer.enable_gradient_checkpointing`。

### 对齐原始 PaDT

❌ 原始 PaDT decoder 不需要 GC(它的 decoder 小 5-10×)。本 Phase 是 starVLA 现状特有的补丁,Phase 6 完成后可能直接退役。

### 验证

- decoder activation 下降 ~500 MB
- 单步耗时 +< 5%

---

## Phase 5 — 集成验证(Phase 1-4 整体)

### 验证矩阵

| 测试 | 期望 |
|---|---|
| 1 step 数值对齐 | 全 Phase 1-4 启用 vs 全禁用:loss 各项 max-abs-diff < 5e-3 |
| 50 step loss 曲线 | 与 baseline 中位数差异 < 5% |
| eval rollout(2 LIBERO task × 50 episode) | success rate 与 baseline 差异 < 2% |
| `max_memory_allocated` | 下降 ~40-50% |
| 单步训练耗时 | 下降 30-50% |
| Eval 单帧延迟 | 下降 25-35% |

### 阶段性总结表(模板)

| 状态 | 单步耗时 | BS 上限 | 单步显存 | 数值对齐 |
|---|---|---|---|---|
| Baseline | 1.00× | 6 | 100% | ✅ |
| + Phase 1 | — | — | — | — |
| + Phase 2 | — | — | — | — |
| + Phase 3 | — | — | — | — |
| + Phase 4 | — | — | — | — |

### 提交策略

Phase 1-5 **每个 Phase 一个独立 PR**,每个 PR 必须附带:
1. 改动前后同 seed 1 step 的 loss 数值对比
2. `max_memory_allocated` before/after
3. 单步 wall-clock before/after

**Phase 5 通过 → 开始 Phase 6-7-8 重训准备。**

---

## Phase 6 — PaDT decoder 重构:hidden_size 2048 → 1280

### 决策依据(已确认)

详见 `qwen_padtpi_current_architecture.md` 的对比分析。1280 在 LIBERO 场景下相比 2048 的优势:

| 维度 | 1280 优势 |
|---|---|
| 注意力风格 | 16 heads × 80 head_dim,DETR/检测头标准配方 |
| FFN 配方 | intermediate=3420(2.67× 扩张),原始 PaDT 在 REC/OVD 上验证过 |
| 高分支特征 | 直接吃 ViT 预训练 1280-d 原生特征,无需训 `high_res_proj` |
| 过拟合风险 | 41M 参数 vs 138M,匹配 LIBERO 数据规模 |
| 超参可迁移 | 直接抄原始 PaDT 食谱,无需自调 |
| bf16 稳定性 | 长训更稳 |

唯一让位的是"LLM 端语义压缩 ~37%"(`input_projection: 2048→1280`)。给定视觉信息上界是 ViT 1280-d,这个压缩对视觉信号是无损的;对语言信号有压力,但 256 prototype 任务的语言信号容量需求不会超过 1280。

### 改动

**`padt_object_decoder.py`** — 全面重构维度:

1. **新增配置项**:
   ```python
   self.hidden_dim         = int(config.framework.padt.get("decoder_hidden_size",   1280))
   self.num_heads          = int(config.framework.padt.get("decoder_num_heads",     16))
   self.intermediate_size  = int(config.framework.padt.get("decoder_intermediate",  3420))
   self.llm_hidden_dim     = int(config.framework.qwenvl.vl_hidden_dim)  # 2048
   ```

2. **新增入口投影**(对齐原始 PaDT 的 `input_projection`):
   ```python
   self.input_projection = nn.Sequential(
       nn.LayerNorm(self.llm_hidden_dim),       # 2048
       nn.Linear(self.llm_hidden_dim, self.hidden_dim),  # 2048 → 1280
       nn.GELU(),
       nn.Linear(self.hidden_dim, self.hidden_dim),
   )
   ```
   在 `_prepare_queries` 之前把 LLM hidden(2048)的 VRT query 投影下来。

3. **修改 `_PaDTStyleBlock` 的 FFN**:用 `intermediate_size=3420` 代替当前的 `hidden_dim * 4`。所有 attention/FFN 都在 `hidden_dim=1280` 上跑。

4. **修改各 embedding/head 维度**(全跟 hidden_dim):
   - `view_embedding`, `vrt_embedding`, `task_tokens`, `low_xy_embedding`, `high_xy_embedding`
   - `bbox_head` / `score_head` / `mask_*_upscaling` 入参全部 1280

5. **出口投影回 LLM 空间**(`object_memory_proj`):
   - 现状:`LayerNorm(2048) → 2048 → GELU → 2048`
   - 改为:`LayerNorm(1280) → 1280 → GELU → 2048`(给 action head 用)

6. **删除 `high_res_proj`(在 `QWen2_5_PaDT.py:82-87`)**:
   - decoder 改成 1280 后,高分特征用 ViT 原生 1280-d,**不需要上投到 2048**
   - `extract_patch_features` 返回的 `high_res_all` 改成 ViT 原生维度(1280)
   - 注意接口变更:`padt_decoder.forward` 的 `high_res_features` 参数从 2048 变成 1280

7. **`role_embedding`**(在 `QwenPaDTPI.py:81`,LLM 空间)**保持 2048**:因为它加到 query 之前(query 仍在 LLM 空间),`input_projection` 之后才进入 decoder 1280 空间。

8. **`prototype_proj`**(`QWen2_5_PaDT.py:76-81`)**保持 2048**:P_ref 必须进 LLM input embedding,只能是 2048。

### 配置文件

新增 `starVLA/config/training/starvla_padtpi_libero_v3.yaml`(不覆盖现有 v2),关键差异:

```yaml
framework:
  padt:
    decoder_hidden_size: 1280       # 新增
    decoder_num_heads:   16         # 新增,override 当前默认 8
    decoder_intermediate: 3420      # 新增
    high_res_tokens_per_view: 1024  # 不变
    # high_res_proj 自动失效(代码层判断 hidden_dim != llm_hidden_dim 时跳过)
```

### 对齐原始 PaDT

✅ **完全对齐**(decoder 内部结构 = 原始 PaDT 的 `PaDTDecoder` 1:1)。唯一差异是出口多了 `object_memory_proj` 把 1280 投回 2048——这是 starVLA 必需的(action head 在 LLM 空间),原始 PaDT 不需要(直接接 bbox/mask head)。

### 验证

无法在现有 ckpt 上验证(权重 shape 改了),验证放到 Phase 8(重训)里做。

---

## Phase 7 — VRT token 从词表内移到词表外

### 决策依据

数学上跟现状等价(详见对话记录),不改效果。**单独做**收益是清掉 ~2 MB ckpt + 0.17% lm_head FLOPs,**不值**;**搭 Phase 6 重训顺带做**收益是代码结构与原始 PaDT 完全对齐 + 跨 VLM 可移植性 + 维护更直接。

### 改动

**`QWen2_5_PaDT.py`**

1. **`_init_padt_tokens`** — 把 VRT 字符串从 `add_special_tokens` 列表中移除,保留 begin/end/obj/view 等控制 token:
   ```python
   additional_special_tokens = [
       padt_begin, padt_end, reason_begin, reason_end, padt_null,
       *obj_tokens, *view_tokens,
       # 注意:vrt_tokens 不再加入 tokenizer
   ]
   ```
   不再调 `resize_token_embeddings` 给 VRT token 留行。

2. **`vrt_start_id` 定义改为**:`vocab_size`(LLM 当前 vocab size),`vrt_end_id = vrt_start_id + num_vrt_tokens`。这两个 ID 是"虚拟扩展"的边界,不存在于 tokenizer 内。

3. **`build_padt_inputs`** — tokenizer 不认识 VRT 字符串,所以 teacher solution 的 VRT 位置需要**手动构造 input_ids**(在 tokenizer 输出之后插入 `[vrt_start_id, vrt_end_id)` 范围的 ID),不能依赖 `processor(text=full_texts)` 一把过。

4. **`build_inputs_embeds`** — 新增 prototype 拼接路径(对齐 `padt.py:194`):
   ```python
   def build_inputs_embeds(self, qwen_inputs, P_ref, *, precomputed_image_embeds=None):
       input_ids = qwen_inputs["input_ids"]
       embed_tokens = self.model.get_input_embeddings().weight   # [V, hidden]
       # 把 prototype 拼到末尾,形成 extended_embed_tokens
       prototype_table = P_ref.reshape(-1, P_ref.shape[-1])      # [B*V*N, hidden]
       # 注意 P_ref 是 per-batch,需要 per-sample index;实际实现中按 batch dim 处理
       # ... (per-sample 索引,见下方)
       inputs_embeds = ...  # 根据 input_ids ≥ vrt_start_id 与否走不同索引
       # 后续 _replace_image_embeddings 走原路径
       return inputs_embeds
   ```
   
   ⚠️ **关键复杂度**:原始 PaDT 的 `image_prototypes` 是 batch-aware 的 `[B, num_proto, hidden]`,starVLA 的 `P_ref` 是 `[B, num_views, num_vrt, hidden]`(多视图)。需要根据 input_ids 当前所在的 view context 决定从哪个 view 的 prototype 段索引——**这部分已经在 `_replace_vrt_embeddings` 里有逻辑**,可以复用。

5. **`compute_dynamic_logits`** — 改为 `cat` 路径:
   ```python
   def compute_dynamic_logits(self, hidden, P_ref, *, view_idx=0):
       # 不再接收 static_logits 入参
       P_view = self._select_p_ref(P_ref, view_idx)
       lm_head_weight = self.model.lm_head.weight                # [V, hidden]
       extended_lm = torch.cat([lm_head_weight.unsqueeze(0).expand(B, -1, -1),
                                P_view], dim=1)                  # [B, V+num_vrt, hidden]
       logits = torch.einsum("bld,bvd->blv", hidden, extended_lm)
       return logits
   ```
   注意 LM head 的 `cat` 是 per-batch 的,因为 prototype 跟图像绑定。

6. **`forward_dynamic`** — 调用 `compute_dynamic_logits` 时不再传 `static_logits`(因为 Qwen 内部 lm_head 那一步可以**完全跳过**,节省一次大 matmul!这是 Phase 2 方案 B 的天然延伸)。具体做法:
   - Qwen forward 用 `output_hidden_states=True` 但不要 `outputs.logits`(可以设 `labels=None` 让内部不算 CE,但 logits 还是会算)
   - 更彻底:patch 一下 Qwen forward 让它**跳过 lm_head matmul**(`Qwen2_5_VLForConditionalGeneration` 内部有一个 `logits = self.lm_head(hidden)`,可以通过 monkey-patch 或子类绕开)
   - 然后**直接用 hidden + 自定义 lm_head cat 路径**

7. **`custom_vrt_decode`** — eval 端的 AR 解码,每步要算 next-token logits。原本流程:
   ```python
   next_token_logits = compute_dynamic_logits(hidden=..., static_logits=current_outputs.logits, ...)
   patch_logits = next_token_logits[:, vrt_start_id : vrt_end_id]
   ```
   改后:
   ```python
   # 直接算 VRT 段 logits,跳过全词表
   patch_logits = torch.einsum("bd,bnd->bn", hidden[:, -1, :], P_view)
   patch_ids = torch.argmax(patch_logits, dim=-1)
   ```
   **AR 解码每步省一次全 vocab matmul**(forward_dynamic 也是),这对 eval 速度是真实增益。

### 对齐原始 PaDT

✅ **完全对齐**(`padt.py:185-300` 的核心机制 1:1 复现)。

### Phase 2 与 Phase 7 的整合

Phase 7 完成后,**Phase 2 的 `compute_dynamic_logits` clone 问题自动消失**(根本不再有 static_logits 入参)。所以:

- **如果决定做 Phase 7**:Phase 2 改用方案 A 简版(代码改动最小,反正之后还要重写)
- **如果暂缓 Phase 7**:Phase 2 改用方案 B(为 Phase 7 铺路)

**建议:走方案 A 简版,Phase 7 时一次性重写到 cat 路径。**

### 验证

- **Tokenization 等价性**:用同一句 instruction + teacher solution,在新旧两版 `build_padt_inputs` 下,对比 `input_ids` 序列。non-VRT 部分必须完全相同;VRT 部分新版应为 `[vrt_start_id, ...)` 而旧版是 tokenizer 分配的真实 ID,**ID 值不同但语义对齐**(都是"在这个位置吃 prototype")。
- **前向数值等价性**:同 P_ref 输入,对比 `inputs_embeds`(在 VRT 位置)和 `next_token_logits`(在 VRT 段)的数值。max-abs-diff < bf16 噪声。**这一步在新代码可以与旧代码并行跑做差**,不需要重训。
- **重训后 loss 曲线**:在 Phase 8 重训里观察,跟"如果走 Phase 6 单独重训"做对比,确认 Phase 7 没引入回归。

---

## Phase 8 — 重训 + 最终验证

### 目标

在 Phase 1-7 全部 merge 的代码上,从同一个 Qwen2.5-VL pretrained ckpt 从零训练一次,得到新 PaDTPI ckpt。

### 配置

使用 `starVLA/config/training/starvla_padtpi_libero_v3.yaml`(Phase 6 新增的)。

关键超参完全照搬原始 PaDT(`padt_sft_trainer.py:151-160`):

| 项 | v2(旧) | v3(新) |
|---|---|---|
| `decoder_hidden_size` | 2048 隐式 | **1280** |
| `decoder_num_heads` | 8 隐式 | **16** |
| `decoder_intermediate` | 8192 隐式(`4×hidden`) | **3420** |
| VRT 在词表 | 内 | **外** |
| `learning_rate.padt_decoder` | 5e-5 | **5e-5**(暂不调) |
| `learning_rate.qwen_vl_interface` | 1e-5 | **1e-5** |
| `loss_weights.act` | 1.0 | **1.0**(不变) |
| `loss_weights.vrt` | 0.1 | **0.5**(向原始 PaDT 对齐,LLM grounding 信号加强 5×) |
| `loss_weights.bbox` | 0.25 | **0.5**(配套,跟 vrt 同量级) |
| `loss_weights.mask` | 0.25 | **0.5**(配套,跟 vrt 同量级) |
| `loss_weights.score` | 0.1 | **0.1**(本身量级小,不变) |
| `enable_gradient_checkpointing` | true | **true**(Phase 4 实测后决定 decoder 自己是否还要 GC) |

### Loss weight 调整说明

**原始 PaDT 用 `sft_loss + bbox_loss + score_loss + mask_loss`(全 1.0,见 `padt_sft_trainer.py:539`)**,sft_loss 是整个 completion 区域的 CE 平均(VRT 占 ~74%),所以 VRT 的等效权重 ~0.74。starVLA 当前 vrt=0.1 比原始低 ~7×,理论上会导致:

- LLM hidden state 在 VRT 位置缺乏强 grounding 信号
- decoder query 信号弱,只能靠 decoder 内部参数硬学
- bbox/mask 监督信号无法有效回流到 LLM(被 decoder 3 层 transformer "吸收")

**v3 调到 vrt=0.5**(原始等效 0.74 的 ~68%)+ bbox/mask=0.5(配套):

- VRT 重新作为 LLM grounding 的核心信号,但不到原始的 1.0,给 action loss 留空间
- bbox/mask 跟 vrt 同量级,decoder 监督和 LLM 监督力度匹配
- score 量级本身就小(~0.05-0.1 收敛),0.1 已够

**收敛时各项预期贡献**(act 0.005-0.05、vrt 0.15-0.5、bbox 0.25-0.75、mask 0.15-0.35、score 0.005-0.01):

- VRT + bbox + mask 合计占 ~70% — 提供足够的 grounding 监督
- action 占 ~5-10% — 保持作为主任务,不被淹没
- 比 v2 配置下"bbox 单项就 25-50%"的失衡情况更健康

⚠️ **风险点**:Phase 8 同时改了 3 类变量(decoder 结构、VRT 词表位置、loss weights)。如果重训出来 success rate 不如预期,**无法直接定位是哪类变量的锅**。**缓解方案**见下方"分步验证策略"。

### 分步验证策略(可选,降低 Phase 8 风险)

如果担心一次性改太多变量带来回归风险,可以把 Phase 8 拆成两步:

**Step 8a(可选)— 仅调 loss weights,不动结构**:
- 在现有 v2 ckpt 基础上继续训 5k-10k step,只把 loss weights 调到 v3 值
- 观察 success rate 变化,确认 loss weight 调整本身的方向是对的
- 如果 success rate ↑ → loss weight 调整方向 OK,继续 Step 8b
- 如果 success rate ↓ → 退回 v2 weights,Step 8b 只做结构改造

**Step 8b — 重训(结构 + loss weights 一起)**:
- 从同一个 Qwen2.5-VL pretrained 起步
- 用 Step 8a 确认过的 loss weights
- 完成 Phase 6+7 结构改造

总训练成本 = ~1.2× 单次重训(Step 8a 短,Step 8b 完整)。

**默认走法**:**直接走 Step 8b**(单次完整重训),只在结果不符合预期时再回头跑 Step 8a 做 ablation。原因:loss weight 调整本身风险不大(只是相对权重变了 5×,不是数量级跳跃),没必要为它单独花 5k step。

### 跑前准备 checklist

```
□ Phase 0 audit 全部完成
□ Phase 1-5 全部 merge 到主分支
□ Phase 5 验证矩阵全部通过
□ Phase 6 代码 merge
□ Phase 7 代码 merge
□ 新 yaml 文件 v3 创建并 review
□ 小规模 sanity check:跑 200 step,确认 loss 正常下降无 NaN
□ 决定 effective batch size(目标:由于 1280 decoder + 所有优化,单卡 BS 应能到 12-16)
□ 估算总训练时间(参考:QwenPI 在同环境跑 50k step 需要 X 小时,新 PaDTPI 应该在 1.2-1.5×X 之间)
```

### 训练 + 验证流程

1. **Sanity train**:200 step,确认无 NaN、loss 下降、辅助 loss 都激活。
2. **Short train**:5k step,检查曲线形状是否健康(action loss 平稳下降,辅助 loss 在 2-3k step 之后逐步收敛)。
3. **Eval mid-point**:5k step 时跑 1-2 个 LIBERO task,成功率应已显著非零(若为零,排查问题再继续)。
4. **Full train**:跑到原计划的总 step 数(默认 100k,可视 5k 中检收敛速度调整)。
5. **Final eval**:跑完整 LIBERO benchmark(4 个 suite × 多任务),对比 QwenPI baseline。

### 验收标准

- ✅ 训练单步耗时 < QwenPI baseline 的 1.5×(Phase 1-5 + 1280 decoder 完成后预期目标)
- ✅ 单卡 BS 上限 ≥ 12
- ✅ LIBERO benchmark 总成功率 ≥ 当前 v2 PaDTPI ckpt(不能因为重训倒退)
- ✅ Eval 单帧延迟 < QwenPI baseline 的 1.5×
- ✅ 训练曲线无异常 spike,辅助 loss 健康收敛

如果 ✅ 全过 → plan 完成,新 ckpt 作为 PaDTPI 的 v3 baseline。

如果某项不过:
- 单步耗时未达标:profile 找剩余瓶颈
- BS 不够:检查 GC 是否真接上、是否还有未释放的 activation
- 成功率倒退 > 5%:回到 Phase 6 检查 decoder 重构是否丢了关键信号(比如 `input_projection` 初始化、`object_memory_proj` 出口投影是否合理)

---

## 时间线(已根据重训决策调整)

| 周次 | 任务 |
|---|---|
| Week 1 | Phase 0 audit + Phase 1 实现 + PR |
| Week 2 | Phase 2 + Phase 3 + Phase 4 实现 + PR(并行) |
| Week 3 | Phase 5 集成验证 + 阶段性报告;Phase 6 / 7 开始 implement |
| Week 4 | Phase 6 + Phase 7 完成 + 单元数值等价测试 + 小规模 sanity train |
| Week 5-6 | Phase 8 正式重训(预计 ~3-5 天 wall clock) |
| Week 7 | LIBERO benchmark + 写最终报告 |

**总周期 ~7 周**。前 3 周拿到工程优化收益(可在 v2 ckpt 上立刻使用),后 4 周完成结构对齐 + 重训。

---

## 最终对齐总结

| 维度 | 原始 PaDT | Phase 8 完成后的 PaDTPI |
|---|---|---|
| ViT 前向次数 / step | 1 | 1 ✅ |
| Logits 构造 | `cat([lm_head, proto])` 一次 matmul | 同款 ✅(Phase 7) |
| LLM/ViT gradient checkpointing | 默认开 | 默认开 ✅(Phase 3) |
| Decoder hidden_size | 1280(对齐 ViT) | **1280** ✅(Phase 6) |
| Decoder FFN 扩张比 | 2.67× | **2.67×** ✅(Phase 6) |
| Decoder num_heads | 16(80 head_dim) | **16** ✅(Phase 6) |
| 高分支特征来源 | ViT 原生 1280-d | ViT 原生 1280-d ✅(Phase 6 删 `high_res_proj`) |
| VRT token 位置 | 词表外 | **词表外** ✅(Phase 7) |
| Loss 权重 | sft+bbox+score+mask 全 1.0(VRT 等效 ~0.74) | act:1.0, vrt:0.5, bbox:0.5, mask:0.5, score:0.1 ✅(Phase 8) |
| AR decode | generate 路径 | `custom_vrt_decode` 54 步 KV-cache ➖(实现合理,不动) |

**仅剩的非对齐项**:
1. `custom_vrt_decode` 用的是自定义 AR 路径而非 HF `model.generate`。starVLA 为了精确控制 force-token 序列做的合理工程选择,**不算偏离原始算法**。
2. Loss 权重相对原始 PaDT 略低(vrt 0.5 vs 原始 ~0.74,bbox/mask 0.5 vs 原始 1.0)。这是为给 action loss 留出 contribution 空间的**合理调整**,原始 PaDT 无 action task。

---

## Open Questions / 决策点

1. **Phase 2 走方案 A 还是直接 B?**
   - **决策**:既然要做 Phase 7,Phase 2 走**方案 A 简版**(去 clone,minimal 改动),Phase 7 时重写到 cat 路径。✅

2. **Phase 4 在 Phase 6 之后是否保留?**
   - **建议**:Phase 6 完成后实测 decoder activation 占比,若 < 10% 总显存则取消 GC(因为 GC 多 20% 算力,得不偿失)。Phase 4 实现时设计成可关。✅

3. **Phase 6 中 1280 配置的 LR 是否需要调?**
   - **决策**:**先不调**,完全照搬原始 PaDT food recipe(decoder=5e-5, qwen=1e-5)。Phase 8 重训出曲线后若发现欠拟合再调。✅

4. **Phase 8 训多少 step?**
   - **决策**:默认 100k(跟当前 yaml 一致),5k step 时根据曲线决定是否提前停或加长。

5. **Phase 8 中 BS 设多少?**
   - **决策**:在 Phase 5 拿到实测 BS 上限后定,目标 effective BS = 64-128(跟 QwenPI baseline 对齐)。

6. **是否同时升级 dataset(比如加更多 LIBERO suite)?**
   - **决策**:**不**。Phase 8 重训的 dataset 与现 v2 ckpt 训练完全一致,确保对比公平。dataset 升级是另一个独立实验。

---

## 不在本 plan 范围内(留作下一轮)

- **PaDT decoder 的 `× O × V` 复制改成 attention mask 表达**:能再砍 8× decoder activation。等 Phase 8 出结果后,看是否还需要进一步优化。
- **`high_res_tokens_per_view` 调小**:会改效果,独立实验。
- **`max_task_objects` / `num_core_vrt_tokens` 调小**:会改效果,独立实验。
- **`custom_vrt_decode` 改用 HF `model.generate`**:工程清理,不影响性能/效果。
- **跨 VLM backbone 移植(InternVL / SmolVLM 等)**:Phase 7 完成后这件事变可行,但属于另一个项目。

---

## 变更历史

- **2026-05-26 v1**:初版 plan,Phase 6 为 1280 vs 2048 A/B 实验。
- **2026-05-26 v2**:用户确认接受重训,Phase 6 直接 commit 1280,新增 Phase 7(VRT 词表外)+ Phase 8(重训),时间线从 5 周延至 7 周。
- **2026-05-26 v3**:Phase 8 loss weights 调整向原始 PaDT 对齐 — vrt: 0.1→0.5, bbox: 0.25→0.5, mask: 0.25→0.5, score: 0.1 不变, act: 1.0 不变。新增"分步验证策略"小节,允许将 Phase 8 拆成 8a(loss weight ablation)+ 8b(完整重训),默认走单次重训。
