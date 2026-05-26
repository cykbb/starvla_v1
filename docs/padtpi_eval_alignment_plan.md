# PaDTPI Eval 输入对齐计划

## Summary

这份文档记录 PaDTPI checkpoint 在 LIBERO eval 时需要与训练输入对齐的改动方案。本文档只描述计划，不修改 runtime 行为。

当前 `standard eval` 没有默认注入训练时使用的 `task_objects / object_role`。因此 `QwenPaDTPI` 在 inference 时会 fallback 到 `slot_1..slot_4`，并将 4 个 object slot 全部设为 active。这可以让 eval 跑通，但和训练时基于 ordered task objects 的输入条件不完全一致。

对齐目标是：PaDTPI eval 使用 task-level metadata 提供 ordered object slot 和 role 信息，但不使用 `bbox / mask / segmentation` ground truth。也就是说，eval 只补齐任务级 object 语义，不给模型任何在线视觉标注答案。

## 当前 Eval 与训练的差异

训练时，PaDTPI sample 中包含：

- `image`
- `lang`
- `state`
- `action`
- `objects`
- `task_objects`
- `object_role`
- `bbox_by_view`
- `patch_mask_by_view`
- `valid_patch_ids`
- `core_patch_ids`
- `visible_by_view`

其中 `bbox_by_view / patch_mask_by_view / valid_patch_ids / core_patch_ids / visible_by_view` 是训练监督，用于计算 `VRT / bbox / mask / score` loss。它们不应该在 eval 时作为输入。

当前 eval 每一步主要传入：

- `image=[agentview, wrist]`
- `lang=task_description`
- `state=16-step history`

如果没有 `task_objects / object_role`，`preprocess_raw_dict(...)` 会进入 inference fallback：生成 `slot_1..slot_4`，并把所有 object slot 设为 present。这会弱化 PaDTPI 训练时学到的 ordered object slot 和 role embedding 约束。

## 需要改动的文件

后续实现时需要修改以下文件：

- `examples/LIBERO/eval_files/auto_eval_scripts/auto_eval_libero.sh`
- `examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh`
- `examples/LIBERO/eval_files/eval_libero.py`

建议新增两个 eval launcher 参数：

- `INJECT_TASK_META_VALUE`
- `TASK_META_PATH_VALUE`

其中 `INJECT_TASK_META_VALUE` 默认使用 `auto`：

- checkpoint `framework.name == QwenPaDTPI` 时自动开启 task metadata 注入。
- checkpoint 是 `QwenPI` 或其他 baseline 时默认关闭。
- 用户可以手动设置 `true / false` 做 ablation。

`TASK_META_PATH_VALUE` 用于手动覆盖 `padt_task_specs.jsonl` 路径。如果不指定，则从 checkpoint `config.yaml` 的 `datasets.vla_data.data_root_dir` 和当前 `task_suite_name` 自动解析。

### `auto` 解析放在哪一层

`eval_libero.py:76` 的 `inject_task_meta: bool` 是布尔，不直接接受 `"auto"`。最终落地方式：

- **由 shell 层解析 `auto`**：在 `eval_libero_parall.sh` 启动 policy server 之前，用一段小 Python（或 `yq`）读取 `${your_ckpt}` 同级目录的 `config.yaml`，提取 `framework.name`：
  - `QwenPaDTPI` → 实际传 `--args.inject-task-meta`；
  - 其他 → 不传。
- `eval_libero.py` 保持现有 `bool` 接口不动，避免 Python 侧新增一份 framework 名字映射。
- 这样 `INJECT_TASK_META_VALUE` 在 shell 内只有三种取值：`auto / true / false`，对应 `auto-resolve / force-on / force-off`。

### `padt_task_specs.jsonl` 缺失时的行为

`_resolve_task_meta_path(...)` 当前在 jsonl 不存在时只 `logging.warning` 然后 fallback 到无 metadata 路径。这对 PaDTPI eval 是危险的 silent fallback —— 表面跑的是"PaDTPI eval"，实际全部走 `slot_1..slot_4`。规则改为：

- 若 `INJECT_TASK_META_VALUE` 解析后为 `true`（不管是 `auto` 推出来还是手动指定）且最终 `resolved_task_meta_path is None`，必须 **fail-fast**：抛 `RuntimeError` 退出 eval，而不是 warn 后继续。
- 仅当用户显式 `INJECT_TASK_META_VALUE=false` 时才允许无 jsonl 路径运行（baseline / ablation）。
- 错误信息要列出尝试过的路径，方便定位"需要补 jsonl"还是"路径解析逻辑错"。

### `padt_task_specs.jsonl` 的前置生成

`add_to_each_dataset/meta/padt_task_specs.jsonl.example` 只是 schema，每个 LIBERO suite 的真实 jsonl 还需要单独生成，放到：

```
<data_root_dir>/<dataset_dir>/meta/padt_task_specs.jsonl
```

其中 `<dataset_dir>` 对应 `TASK_SUITE_TO_DATASET_DIR[task_suite_name]`：

- `libero_goal_no_noops_1.0.0_lerobot`
- `libero_object_no_noops_1.0.0_lerobot`
- `libero_spatial_no_noops_1.0.0_lerobot`
- `libero_10_no_noops_1.0.0_lerobot`

每行字段必须包含：`task_index / task / task_objects / object_role`。`objects` 可选，用于记录当前 task suite 的完整场景 object list、debug 和未来 prompt 扩展；当前 PaDTPI eval/action path 不直接依赖它。`task_index` 必须和 LIBERO benchmark 内置 task 顺序一致，`task` 字符串和 `task.language` 完全匹配（`_resolve_task_meta_record` 优先 by_index、再 fallback by_task 文本匹配）。

**本计划不负责生成 jsonl**，但实施 eval 改动之前，这 4 份 jsonl 必须先准备好；否则 `auto` 模式直接 fail-fast 阻止 eval 启动。

## 预期 PaDTPI Eval 输入

对齐后，PaDTPI `standard eval` 每一步应传给 policy server：

- `image=[agentview, wrist]`
- `lang=task_description`
- `state=16-step history`
- `task_objects`
- `object_role`

其中：

- `task_objects` 是 ordered task-relevant object subset。
- `object_role` 描述每个 task object 的 role，例如 `primary / secondary / receptacle / tool`。
- `objects` 可选传入；当前 action inference 不直接使用它，只建议保留在 metadata 文件里用于 schema consistency、debug 和未来显式 object-name prompt 扩展。

这样 `QwenPaDTPI.predict_action(...)` 中的 prompt、object slot、role embedding、object presence mask 会更接近训练时的输入分布。

## `objects` 是否必须注入

当前 PaDTPI eval/action path 中，`objects` 不是必需输入。

模型当前实际使用的是：

- `task_objects`：决定 active object slot 的数量和顺序。
- `object_role`：决定 prompt 中的 role 描述和 role embedding。
- `lang`：提供自然语言 task description，其中通常已经包含 object name。

训练时 `objects` 有实际作用，因为每个 object record 里会挂 `bbox_by_view / patch_mask_by_view / valid_patch_ids / core_patch_ids / visible_by_view` 等监督字段，训练代码通过 object id 去取这些 label。Eval 时这些监督字段不注入，因此 `objects` 对当前 action 输出没有直接影响。

结论：

- PaDTPI eval 必须注入 `task_objects / object_role`。
- PaDTPI eval 可以不注入 `objects`。
- 如果 metadata 文件已经有 `objects`，可以保留并写入 debug manifest，但不要把它作为 PaDTPI standard eval 的 hard requirement。

## Eval 时不注入的训练监督

正式 eval 不应注入以下字段：

- `bbox_by_view`
- `patch_mask_by_view`
- `valid_patch_ids`
- `core_patch_ids`
- `visible_by_view`

原因是这些字段来自 segmentation / bbox annotation，是训练阶段用于监督 `VRT / bbox / mask / score` 的 label。真实 closed-loop eval 不应该提供这些信息。Eval 时应由模型自己完成：

- autoregressive dual-view `VRT` generation
- `PaDTObjectDecoder` 的 `bbox / mask / score` prediction
- `object_memory / object_view_memory` construction
- final action chunk prediction

## 与 QwenPI Baseline 的关系

`QwenPI` baseline 不使用 PaDT object slot、`task_objects` 或 `object_role`。因此 baseline 的 `standard eval` 应保持默认不注入 metadata，避免改变已有评测语义。

建议行为：

- `QwenPaDTPI`: `INJECT_TASK_META_VALUE=auto` 时开启。
- `QwenPI`: `INJECT_TASK_META_VALUE=auto` 时关闭。
- 手动 `INJECT_TASK_META_VALUE=false` 可用于 PaDTPI no-meta ablation。
- 手动 `INJECT_TASK_META_VALUE=true` 可用于强制 metadata eval。

## 与 `diagnostic_meta` 模式的边界

`eval_libero_parall.sh:192-195` 已经有 `diagnostic_meta` 模式，它本质上也是 `inject_task_meta=true`。两者职责分开：

- `standard` + `INJECT_TASK_META_VALUE=true/auto-on`：**closed-loop performance eval**。只跑成功率，不写 debug dump、不跑 `analyze_padt_libero_diagnostics`，输出体积小，适合刷分数。
- `diagnostic_meta`：**带 metadata 的诊断 eval**。会落 `debug_dump_dir`、`steps.jsonl`、`actions.npy`、frame 截图，并自动 `run_analysis` 出 report。用于排查模型行为，不用于报成绩。

实施时不要把这两个模式合并：`standard` 里仍然 **不** 触发 `--args.debug-dump-dir`，避免在大规模 eval 时把磁盘写爆。

## 已经对齐、本计划不再覆盖的部分

为避免后人误以为本计划是 PaDTPI eval 全部对齐工作，显式列出已经在其它代码路径里完成的对齐项：

- **Image 顺序**：`example_dict["image"] = [agentview, wrist]`（`eval_libero.py:243`）和训练 `view_names: [agentview, wrist]` 一致。
- **State history length**：`ModelClient` 在 `model2libero_interface.py:122-137` 按 `state_history_len` 拼历史并左 pad，`state_history_includes_current` 默认 `False` 与训练 loader 行为一致。
- **Action chunk size**：从 `model_config["framework"]["action_model"]["future_action_window_size"] + 1` 派生（`model2libero_interface.py:251`），不在本计划范围。
- **Normalization stats**：由 `read_mode_config` 在 server 启动时加载，eval 端不重复处理。

本计划只解决 PaDT object slot / role 这一块训练-eval 输入分布的差异。

## Test Plan

文档检查：

- 确认新文件存在于 `starVLA_origin/docs/padtpi_eval_alignment_plan.md`。
- 确认 Markdown 标题、路径、变量名、technical terms 拼写正确。
- 确认内容没有描述 eval 使用 `bbox / mask / segmentation` ground truth。

后续实现 runtime 改动时再运行：

```bash
bash -n examples/LIBERO/eval_files/auto_eval_scripts/auto_eval_libero.sh
bash -n examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh
python -m py_compile examples/LIBERO/eval_files/eval_libero.py
```

PaDTPI smoke eval 建议（注意：`INJECT_TASK_META_VALUE / TASK_META_PATH_VALUE` 为本计划新增变量，需先在 `auto_eval_libero.sh` 和 `eval_libero_parall.sh` 中实现并 `export` 后才能生效；当前脚本直接传入会被忽略）：

```bash
CKPT_PATH=/home/users/astar/i2r/chengzy/starVLA_origin/results/Checkpoints/qwen_padtpi_libero_dualvrt_v2_4/checkpoints/steps_50000_pytorch_model.pt \
TASK_SUITE_NAME=libero_goal \
EVAL_MODE=standard \
TASK_IDS=0 \
MAX_EPISODES_PER_TASK=2 \
NUM_TRIALS_PER_TASK=2 \
INJECT_TASK_META_VALUE=auto \
SAVE_VIDEO_VALUE=false \
TRACE_ENV_STEPS_VALUE=false \
sbatch --export=ALL examples/LIBERO/eval_files/auto_eval_scripts/auto_eval_libero.sh
```

期望 log 中能看到（实施 eval 改动时也需新增/加强对应日志）：

- 启动时一行 `inject_task_meta=true source=auto framework=QwenPaDTPI`，说明 auto 解析命中。
- 启动时一行 `task_meta_path=<绝对路径>`，证明 jsonl 已被解析。
- 每个 task 的第一步 inference 前，打印一次该 task 解析出的 `task_objects=[...] object_role={...}`，确认与 LIBERO benchmark `task_id` 对得上，而不是空 list 静默通过。
- 每个 episode summary 中 `task_meta_applied=true`。
- 若 `auto` 推出 PaDTPI 但 `padt_task_specs.jsonl` 缺失，应 `RuntimeError` 退出，错误信息列出尝试过的路径，而不是 warn + 继续。

## Assumptions

- 文档语言以中文为主，保留 `PaDTPI / QwenPI / eval / task_objects / object_role / VRT / bbox / mask` 等英文专业名词。
- 本计划只记录 eval 输入对齐方案，不实施 eval 代码改动。
- PaDTPI 正式 eval 应该使用 task-level metadata，因为训练 prompt 和 role embedding 都依赖 ordered object slots。
- Eval 不使用实时 `bbox / mask / segmentation` 标注，避免和真实部署不一致。
- 4 个 LIBERO suite 的 `meta/padt_task_specs.jsonl` 已经按 schema 准备好，并放在训练时使用的 `data_root_dir` 下。本计划默认这一前置已经满足；若未满足，`auto` 模式按上文规则 fail-fast。
- `auto` 解析在 shell 层完成（读 checkpoint 同级 `config.yaml` 的 `framework.name`），`eval_libero.py` 接口保持 `inject_task_meta: bool` 不变。
