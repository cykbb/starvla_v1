确认一下 `QWen2_5_PaDT.py` 是否只服务于 padtpi（避免 lang_summary 修改影响 baseline）。

# 计划：padtpi train/eval 对齐 + 参数对齐 baseline

## 隔离边界（首要约束）

**不会被触碰的文件 / 链路**（保证 qwenpi baseline 0 影响）：
- `starVLA/model/framework/QwenPI.py`
- `starVLA/model/modules/vlm/QWen2_5.py`
- `examples/LIBERO/eval_files/eval_libero.py`
- `examples/LIBERO/eval_files/model2libero_interface.py`
- `examples/LIBERO/eval_files/auto_eval_scripts/*.sh`
- `qwenpi_libero_baseline` config 和 ckpt

**会被修改的范围**：
- `starVLA/model/framework/QwenPaDTPI.py`（仅 `predict_action`）
- `starVLA/model/modules/vlm/QWen2_5_PaDT.py`（仅 `custom_vrt_decode` 的 `lang_summary` 计算；已验证此文件只在 `framework.name == "QwenPaDTPI"` 时加载，不会影响 baseline）
- 新增 `starVLA/config/training/starvla_padtpi_libero_aligned.yaml`（不覆盖原 yaml）

---

## Phase A — 推理对齐训练（无需重训，可直接在现有 v2_4 ckpt 上验证）

### A1. 修 `lang_summary`（`QWen2_5_PaDT.py:828`）

**当前 bug**：`custom_vrt_decode` 里用 `prompt_outputs.hidden_states[-1][:, -1]`（单个最后 token）。

**对齐改法**：复用训练侧的 `_compute_lang_summary` 同款 mean-pool。在 prompt-only 前向里，所有 prompt 位置都还没有"被生成"，所以 `labels == IGNORE_INDEX` 等价于"prompt mask = attention_mask"。

```python
# 替换 line 828
prompt_input_ids = prompt_inputs["input_ids"]
prompt_attention_mask = prompt_inputs["attention_mask"]
prompt_final_hidden = prompt_outputs.hidden_states[-1]

prompt_mask = prompt_attention_mask.bool()
prompt_mask = prompt_mask & (prompt_input_ids != self.model.config.image_token_id)
prompt_mask = prompt_mask & ~((prompt_input_ids >= self.vrt_start_id) & (prompt_input_ids < self.vrt_end_id))
prompt_mask = prompt_mask.unsqueeze(-1)
pooled = (prompt_final_hidden * prompt_mask).sum(dim=1) / prompt_mask.sum(dim=1).clamp_min(1)
lang_summary = self.lang_summary_proj(pooled)
```

因果 mask 下 prompt 位置在 (prompt-only) 与 (prompt+solution) 两次 forward 中 hidden state 数值相同，所以这一步严格等同训练。

### A2. 修 `predict_action` 不传 visibility 的 bug（`QwenPaDTPI.py:344-349`）

**当前 bug**：`predict_action` 调用 `padt_decoder` 时不传 `target_visible_by_view`，导致 absent 槽 `object_memory` 走到 `query.mean(dim=(2,3))` 分支，norm ≈ 250；训练时 absent 槽 norm = 0。

**对齐改法（一遍 forward 版）**：把 `object_presence_mask` 广播到 V 维当 visibility 传给 decoder。
```python
B, O = batch.object_presence_mask.shape
V = self.num_vrt_views
visibility_proxy = (
    batch.object_presence_mask
    .to(decoded.final_hidden.device, dtype=patch_features["all"].dtype)
    .unsqueeze(-1)
    .expand(B, O, V)
)
decoder_outputs = self.padt_decoder(
    grouped.vrt_token_sequences,
    patch_features["all"],
    patch_features.get("high_res_all", None),
    target_visible_by_view=visibility_proxy,   # ← 新增
    object_presence_mask=batch.object_presence_mask.to(decoded.final_hidden.device),
)
```

效果：absent 槽 `object_memory = 0`（与 train 一致）；present 槽跨两个视角等权平均（与 train 一致，因为多视角监督数据里大部分 present 物体确实在两个视角都可见）。**唯一遗留偏差**：训练中"present 但某个视角不可见"的样本会在该视角不参与 pooling；推理 proxy 把它当成可见。这是个小偏差，可在 Phase C 处理。

### A3. 在 v2_4 ckpt 上 sanity test（不重训）

跑一遍 libero_goal 的 task_id=0（已有 task_meta_aligned eval 流程），重点核对：
- `lang_summary_norm`：应接近训练日志中 `eval_vrt_teacher_hidden_norm`（量级 ~16）
- `object_memory_norm`：slot 1/2/3（absent）应 ≈ 0
- 至少 task_id=0 出现 1 个 Success=True 才算可上量
- 单 task 50 trials 跑通后再扩展到全 suite

预计 Phase A 全部工时：代码 < 20 行 diff + 1 次 eval 验证。如果 Phase A 后成功率仍 < 50%，再走 Phase B。

---

## Phase B — 参数对齐 baseline（需要重训新 ckpt）

### B1. 创建新 config

复制 `starVLA/config/training/starvla_padtpi_libero.yaml` 为 `starVLA/config/training/starvla_padtpi_libero_aligned.yaml`，改动：

| key | 旧 | 新 | 说明 |
|---|---|---|---|
| `framework.qwenvl.num_vl_layers` | 16 | **36** | 同时控制 DiT 深度 |
| `framework.action_model.diffusion_model_cfg.num_layers` | 16 | **36** | 与上面联动（实际通过 `DiTConfig["num_layers"] = num_vl_layers` 自动同步，yaml 可以删掉这条避免冲突） |
| `framework.action_model.num_inference_timesteps` | 4 | **8** | |
| `trainer.repeated_diffusion_steps` | (无) | **2** | baseline 有 |
| `trainer.max_train_steps` | 60000 | **100000** | |
| `datasets.vla_data.per_device_batch_size` | 6 | **3-4**（待测） | 36 层 DiT + 448 图 + dual-view VRAM 压力大，可能需要降一半；如配合 grad-accum 保 effective batch |
| `trainer.gradient_accumulation_steps` | 1 | **2**（如需） | 维持 effective batch |
| `output_dir` / `run_id` / `wandb` | v2_4 | **`qwen_padtpi_libero_aligned_v3`** | |

`framework.padt.*` / `loss_weights` / `noisy_teacher_probability` 等保持不变（先不动 train 侧逻辑）。

### B2. 训练脚本

新建 `examples/LIBERO/train_files/run_libero_train_padtpi_aligned.sh`（不覆盖现有 `run_libero_train.sh`），仅改 `config_yaml` 指向新 config + 调整 `--num_processes` / GPU 数。不动 deepspeed 配置（zero2 足够）。

### B3. 训练验证

每 10k step 跑一次 sanity eval（task_id=0, 5 trials），重点关注：
- `loss_action_fm` 是否能降到 baseline 的 0.02-0.03 量级（baseline 100k step 时大概这个数）
- 36 层 DiT 收敛是否明显比 16 层快
- 5 trial 成功率 > 0 才说明方向对

### B4. 完整 eval

ckpt 训完后用现有 `auto_eval_libero.sh`（`CKPT_PATH` 指向新 ckpt 即可，`INJECT_TASK_META_VALUE=auto` 会自动识别 framework.name 然后注入 meta），跑全部 libero_goal/libero_10 50 trials/task。

---

## Phase C（可选 / 仅在 A+B 后成功率仍未到 80%+ 时启用）

不纳入主线，但备好以便快速试：
- **exposure bias**：v3 config 里把 `noisy_teacher_probability` 设 0.3，或 `use_sampled_branch: true` + `sampled_branch_weight: 0.2`（代码已就位）
- **VRT collapse**：`custom_vrt_decode` 内 sampled_token 改 top-k 采样（k=5）+ 去重，避免 [167,167,167,167,167] 退化
- **per-view visibility 偏差**：在 `predict_action` 改用 2-pass（pass1 得 score → derive visibility → pass2 重算 object_memory）

---

## 风险与决策点

1. **VRAM**：36 层 DiT + 448 图 + dual-view 可能 OOM。Phase B1 时 batch 从 6 降到 3-4。如果 zero2 不够，再切 zero3。
2. **Phase A 成功率上限未知**：v2_4 ckpt 训练时见的 condition 分布本来就是带 bug 的；fix 后 condition 分布更"干净"但模型未必学过对应映射。可能要 1-2 个 fine-tune step 才能用，**最坏情况 Phase A 只能从 0% 提到 20-40%**，仍需 Phase B。
3. **Train 侧改动顺序**：Phase B 只改 config，不改 train 代码，所以风险低；Phase C 若启用 sampled_branch 会增加每 step ~1.5x 时间。

---

## 估时 / 估算 GPU 占用

| Phase | 代码量 | GPU | 钟表时间 |
|---|---|---|---|
| A1+A2 | ~25 行 diff | 0（写代码） | < 1 小时 |
| A3 sanity eval | 0 | 1 卡 | ~10 分钟（task_id=0, 50 trials） |
| B1+B2 | ~1 个新 yaml + 1 个 sh | 0 | 30 分钟 |
| B3 训练 | 0 | 8 卡（与 baseline 同） | ~3-5 天（参考 baseline 100k 步用时） |
| B4 完整 eval | 0 | 1 卡 | ~3 小时（10 task × 50 trials × 2 suite） |

建议立刻动 Phase A（成本低、可证伪 / 部分修复），同时准备 B1 的 config，等 Phase A 结果决定是否启动 B3 重训。

要我现在就开 Phase A 的代码改动吗？