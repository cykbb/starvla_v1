# QwenPaDTPI 当前架构说明

这份文档描述当前 `starVLA_origin` 中 `QwenPaDTPI` 的完整路径：从
`LeRobot` dataloader 传入 raw sample 开始，到训练时计算各项 loss，以及
eval 时生成最终 LIBERO action。文档中 class/function/path/tensor shape/loss
名称尽量保持英文，解释性文字使用中文。

## 0. Main Files

主要代码位置如下：

- Data loading:
  - `starVLA/dataloader/__init__.py`
  - `starVLA/dataloader/lerobot_datasets.py`
  - `starVLA/dataloader/gr00t_lerobot/datasets.py`
  - `starVLA/dataloader/padt_segmentation_adapter.py`
- PaDT batch / VRT utilities:
  - `starVLA/model/modules/vlm/padt_data_utils.py`
- Framework orchestration:
  - `starVLA/model/framework/QwenPaDTPI.py`
- Qwen2.5-VL PaDT interface:
  - `starVLA/model/modules/vlm/QWen2_5_PaDT.py`
- PaDT-style decoder:
  - `starVLA/model/modules/vlm/padt_object_decoder.py`
- Action condition bridge:
  - `starVLA/model/modules/action_model/PaDTConditionBridge.py`
- Flow-matching action head:
  - `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`

## 1. Shape Convention

下文使用这些符号：

- `B`: batch size
- `O`: object slot 数量上限，目前 `max_task_objects=4`
- `V`: view 数量，目前 `decoder_num_views=2`
- `K`: 每个 object 每个 view 的 core VRT 数量，目前 `num_core_vrt_tokens=5`
- `N`: 每个 view 的 low-resolution VRT patch token 数量，目前 `num_vrt_tokens=256`
- `HN`: 每个 view 的 high-resolution visual token 数量，目前 `high_res_tokens_per_view=1024`
- `D`: Qwen hidden size，通常是 `2048`
- `A`: action dimension，目前 `7`
- `H`: action horizon，目前 `future_action_window_size + 1 = 8`

两个 view name 固定为：

```text
agentview, wrist
```

## 2. Dataloader Output

训练入口在 `train_starvla.py`。`prepare_data(...)` 会调用：

```text
build_dataloader(cfg, dataset_py="lerobot_datasets")
```

当前 dataloader 使用：

```python
collate_fn(batch): return batch
```

所以 framework 收到的是一个 Python `List[dict]`，不是提前 stack 好的 tensor batch。

每个 raw sample 至少包含：

```text
image:  list[PIL.Image]       # primary image 在前，wrist image 在后
lang:   str                   # task instruction
action: np.ndarray            # action chunk，shape 近似 [T_action, A]
state:  np.ndarray optional   # include_state=true 时提供 state / proprio
```

还可能包含：

```text
robot_tag
language
__padt_meta__
```

对于 `QwenPaDTPI` 训练，`PaDTSegmentationSourceAdapter` 会继续给 sample
补充 object-centric supervision：

```text
objects
task_objects
object_role
task_index
task_name
__padt_source__
```

当前 LIBERO 配置中相关字段是：

```yaml
padt_use_segmentation_source: true
padt_task_meta_required: true
padt_segmentation_fields:
  agentview: segmentation.agentview_bbox_mask
  wrist: segmentation.wrist_bbox_mask
padt_patch_grid_size: 16
padt_valid_patch_threshold: 0.30
padt_num_core_patches: 5
```

## 3. Segmentation Adapter

`PaDTSegmentationSourceAdapter` 从 task-level metadata 读取 object 信息：

```text
meta/padt_task_specs.jsonl
```

每条 task spec 提供：

```text
task_index
task
objects
task_objects
object_role
```

每个 step sample 会读取两路 segmentation payload：

```text
segmentation.agentview_bbox_mask
segmentation.wrist_bbox_mask
```

对每个 object label 和每个 view，adapter 会生成：

```text
bbox_by_view[view]              # normalized xyxy bbox
patch_mask_by_view[view]        # flattened 16x16 patch coverage，长度 256
visible_by_view[view]           # bool
valid_patch_ids_by_view[view]   # coverage >= threshold 的 patch ids
core_patch_ids_by_view[view]    # 选出的 5 个 core patch ids
```

`core_patch_ids` 的生成逻辑：

1. 将 dense mask 转成 `16 x 16` coverage grid。
2. 用 `coverage >= padt_valid_patch_threshold` 得到 valid patches。
3. 如果没有 valid patch，但 coverage max 大于 0，则使用 max-coverage patch。
4. 对 valid mask 做一次 erosion，优先选择 object 内部 patch。
5. 对 core region 做 farthest-point sampling，选出 `padt_num_core_patches=5` 个 patch。
6. 如果不足 5 个，从 valid patches 补齐；仍然不足时重复最后一个 patch。

因此，正常 visible 的 object-view 会得到固定 `K=5` 个 core VRT label。

## 4. Canonical PaDT Batch

`QwenPaDTPI.forward(...)` 首先调用：

```python
batch = preprocess_raw_dict(examples, config)
```

它把 `List[dict]` 转成 `PaDTRawBatch`：

```text
images:                     List[List[Any]]
instructions:               List[str]
actions:                    Tensor or None, [B, T_action, A]
state:                      Tensor or None
task_object_ids:            List[List[str]]
task_object_roles:          List[List[str]]
objects:                    List[List[dict]]
object_lookup:              List[dict]
target_core_patch_ids:      LongTensor [B, O, V, K]
target_valid_patch_mask:    FloatTensor [B, O, V, N]
target_patch_masks_by_view: FloatTensor [B, O, V, N]
target_boxes_by_view:       FloatTensor [B, O, V, 4]
target_visible_by_view:     BoolTensor [B, O, V]
object_presence_mask:       BoolTensor [B, O]
```

这里有几个重要行为：

- `image` 会被规范成固定 view 顺序 `[agentview, wrist]`。
- `task_objects` 决定 fixed object slots。
- 如果没有 `task_objects`，inference 时会 fallback 到 `slot_1 ... slot_4`，并把这些 slot 标记为 present。
- `target_core_patch_ids` 中缺失或 inactive slot 用 `-1`。
- `target_valid_patch_mask` 表示每个 object-view 的 valid patch set。
- `target_patch_masks_by_view` 是 decoder mask loss 的 target。

## 5. Visual Feature Extraction

framework 调用：

```python
patch_features = qwen_vl_interface.extract_patch_features(
    images=batch.images,
    instructions=batch.instructions,
    object_roles=batch.task_object_roles,
)
```

这里会构造 Qwen chat prompt：

```text
Task: {instruction}
Ordered object slots: obj1={role}; obj2={role}; ...
Return fixed slots only.
```

然后通过 custom dual-resolution path 跑 `Qwen2.5-VL` visual tower：

- `low_res_features`: Qwen visual merger 之后的 features
- `high_res_features`: Qwen visual merger 之前的 features，再投影到 Qwen hidden dim

返回的 tensor：

```text
agentview:          [B, N, D]
wrist:              [B, N, D]
all:                [B, V*N, D]
high_res_agentview: [B, HN, D]
high_res_wrist:     [B, HN, D]
high_res_all:       [B, V*HN, D]
vrt_bank:           [B, V, N, D]
```

随后构造 dynamic VRT prototype bank：

```python
P_ref = build_prototypes(patch_features["vrt_bank"])
```

`P_ref` 的 shape 是：

```text
[B, V, N, D]
```

`P_ref` 用在两个地方：

1. 替换 teacher/generated VRT token 的 input embedding。
2. 用 hidden state 和 prototype 做 dot product，生成 dynamic logits over VRT patch ids。

## 6. Teacher VRT Sequence During Training

训练时使用 structured teacher forcing：

```python
teacher = build_structured_teacher_seq(batch, token_table, noisy_teacher_probability)
```

assistant solution 的 schema 固定为：

```text
<|padt_begin|>
<|obj1|> <|view_agentview|> <|padt_vrt_xxx|> ... 5 VRTs ...
        <|view_wrist|>     <|padt_vrt_xxx|> ... 5 VRTs ...
<|obj2|> ...
<|padt_end|>
```

如果 `O=4, V=2, K=5`，每个 sample 最多监督：

```text
4 objects * 2 views * 5 VRTs = 40 VRT tokens
```

如果 `use_noisy_teacher_branch=true`，`noisy_teacher_probability` 可以把
teacher core ids 从同一个 object 的 valid patch set 中重新采样。当前训练脚本里
`noisy_teacher_probability=0`，所以 teacher VRT label 就是 dataset 里的
`core_patch_ids_by_view`。

teacher builder 返回：

```text
teacher.core_patch_ids              [B, O, V, K]
teacher.valid_patch_mask_per_token  [B, O*V*K, N]
```

## 7. Dynamic Qwen Forward And VRT Loss

训练时调用：

```python
dynamic_outputs = qwen_vl_interface.forward_dynamic(
    images=batch.images,
    instructions=batch.instructions,
    object_roles=batch.task_object_roles,
    solutions=teacher.solutions,
    P_ref=P_ref,
    teacher_core_patch_ids=teacher.core_patch_ids,
    valid_patch_mask_per_token=teacher.valid_patch_mask_per_token,
)
```

`forward_dynamic` 内部流程：

1. 用 `prompt + teacher solution` 构造 Qwen inputs。
2. 对 prompt token 和 padding token 的 labels 置 `IGNORE_INDEX`。
3. 用 visual embeddings 替换 image placeholder embeddings。
4. 用 `P_ref` 中对应 prototype 替换每个 VRT token embedding。
5. 运行 Qwen，并设置 `output_hidden_states=True`。
6. 将 VRT token range 内的 normal vocabulary logits 替换成 dynamic logits：

```text
dynamic_vrt_logits = hidden @ P_ref[view].T
```

输出包括：

```text
final_hidden:  [B, S, D]
logits:        [B, S, vocab]
input_ids:     [B, S]
labels:        [B, S]
lang_summary:  [B, D]
vrt_loss:      scalar
```

`lang_summary` 的计算方式是：对 prompt hidden states 做 average pooling，
但排除 image tokens 和 VRT tokens，然后经过 `lang_summary_proj`。

### VRT Loss

对每个 supervised VRT token：

1. 找到 teacher VRT token 的 `label_pos`。
2. 使用 `logit_pos = label_pos - 1`，因为 Qwen 是 causal LM。
3. 根据当前 token 所属 view 选择 `P_ref[:, view]`。
4. 计算 `N=256` 个 patch ids 的 logits。
5. 当前代码会把 `valid-but-not-core` patch logits 置为 `-inf`。
6. 对 logits 做 `log_softmax`。
7. 对 target core patch id 计算 NLL。

所以当前实现里，target core patch 不会和同一个 object 的其他 valid patch 竞争。
它主要和 non-valid patches 竞争。这一点对理解 `vrt_teacher_top1_acc` 很重要。

可选 diagnostics：

```text
vrt_supervised_tokens
vrt_teacher_top1_acc
vrt_teacher_hidden_norm
vrt_teacher_proto_norm
vrt_teacher_target_logit
vrt_teacher_top1_margin
vrt_teacher_logit_abs_max
vrt_proto_global_norm
vrt_proto_global_norm_max
```

## 8. Group VRT Hidden States Into Object-View Queries

Qwen forward 结束后，framework 调用：

```python
grouped = group_vrt_hidden_by_slots(
    final_hidden=dynamic_outputs.final_hidden,
    input_ids=dynamic_outputs.input_ids,
    ...
)
```

它按照 fixed schema 扫描 sequence：

```text
<|obj_i|> <|view_agentview|> VRT... <|view_wrist|> VRT...
```

然后构造：

```text
grouped.vrt_token_sequences: [B, O, V, K, D]
grouped.predicted_patch_ids: [B, O, V, K]
```

随后 `QwenPaDTPI` 给每个 object slot 加上 learned role embedding：

```text
vrt_token_sequences += role_embedding(object_role)
```

这样 decoder 输入保留 token-level VRT evidence，同时保留 `object_role`
信息。这里没有对 VRT tokens 做 mean pooling 后再送入 decoder。

## 9. PaDT-Style Object Decoder

训练时 decoder 调用如下：

```python
decoder_outputs = padt_decoder(
    grouped.vrt_token_sequences,
    patch_features["all"],
    patch_features["high_res_all"],
    target_boxes_by_view=batch.target_boxes_by_view,
    target_patch_masks_by_view=batch.target_patch_masks_by_view,
    target_visible_by_view=batch.target_visible_by_view,
    object_presence_mask=batch.object_presence_mask,
)
```

输入 shape：

```text
vrt_token_sequences: [B, O, V, K, D]
low_res_features:    [B, V*N, D]
high_res_features:   [B, V*HN, D]
```

### Query Construction

对每个 object-view item，decoder 会构造：

```text
[bbox_token, score_token, mask_token, VRT_1, ..., VRT_K]
```

shape：

```text
queries: [B, O, V, 3+K, D]
```

其中：

- VRT tokens 会加 learned `vrt_embedding`。
- image memory 会加 2D xy position embedding。
- image memory 也会加 view embedding。

### Attention Scope

decoder 会把 object-view item 展平：

```text
[B, O, V, L, D] -> [B*O*V, L, D]
```

每个 object-view item 只看对应 view 的 image memory：

1. 选择 matching view 的 low-res memory。
2. 跑一个 PaDT-style low-res block：
   - query self-attention
   - query-to-image cross-attention
   - query MLP
   - image-to-query memory update
3. 将 low-res memory repeat 后加到对应 high-res memory 上。
4. 跑两个 PaDT-style high-res blocks，attention pattern 与 low-res block 一致。

decoder block 内部没有 cross-view attention。`agentview` 和 `wrist` 分别 decode，
之后才在 `object_memory` 中聚合。

### Decoder Predictions

decoder heads 输出：

```text
bbox_by_view:       [B, O, V, 4]     # xyxy normalized
patch_mask_by_view: [B, O, V, N]     # high-res mask downsample 到 low-res patch logits
score_logits:       [B, O]           # aggregated score
visibility_logits:  [B, O, V]        # 当前等于 score_logits_by_view
object_memory:      [B, O, D]
object_view_memory: [B, O, V, D]
```

`visibility_logits` 目前是为了 eval/analyzer compatibility 保留的字段。
当前代码没有独立 visibility head。

### Object Memory Aggregation

训练时如果有 `target_visible_by_view`：

```text
object_memory = visible-weighted mean over views of mean(query tokens)
```

inference 时没有 target visibility，所以：

```text
object_memory = mean over views and query tokens
```

`object_view_memory` 保留 view-specific object memory：

```text
object_view_memory = mean over query tokens, per object-view
```

action bridge 会同时消费 `object_memory` 和 `object_view_memory`。

## 10. Decoder Losses

decoder 计算三个 auxiliary losses。

### Bbox Loss

输入：

```text
pred bbox_cxcywh:       [B, O, V, 4]
target_boxes_by_view:   [B, O, V, 4]  # xyxy
target_visible_by_view: [B, O, V]
```

loss：

```text
L_bbox = L1(cxcywh) + (1 - GIoU)
```

只有 visible object-view pair 参与计算。

### Mask Loss

mask head 使用：

```text
mask_token + high_res_memory
```

预测 high-res mask。target 是 dataset 里生成的 `16 x 16` patch coverage。
当 shape 可以对齐时，会把 target upsample 到 high-res prediction 对应分辨率。

loss：

```text
L_mask = DiceLoss + SigmoidFocalLoss
```

只有 visible object-view pair 参与计算。

### Score Loss

score head 对每个 object-view 预测一个 scalar：

```text
score_logits_by_view: [B, O, V]
```

预测值映射到 `[-1, 1]`：

```text
pred_score = sigmoid(score_logits) * 2 - 1
```

target：

```text
visible view:   target_score = detached GIoU
invisible view: target_score = -1
```

loss：

```text
L_score = MSE(pred_score, target_score)
```

这个 loss 的 mask 来自 `object_presence_mask`，并 expand 到 view 维度。

## 11. Action Condition Bridge

framework 构造 action condition：

```python
action_condition = condition_bridge(
    lang_summary=dynamic_outputs.lang_summary,
    object_memory=decoder_outputs.object_memory,
    object_view_memory=decoder_outputs.object_view_memory,
    state=state,
)
```

`PaDTConditionBridge` 不做复杂语义推理，它的作用是把 decoder 输出包装成
PI action head 需要的 condition token 格式。

它会创建：

```text
lang token:         [B, 1, D]
object tokens:      [B, up to bridge_max_object_tokens, D]
object-view tokens: [B, up to bridge_max_view_tokens, D]
```

然后 pad/truncate 到：

```text
cond_tokens: [B, bridge_max_condition_tokens, D]
```

当前配置：

```text
bridge_max_object_tokens = 4
bridge_max_view_tokens = 8
bridge_max_condition_tokens = 16
```

再加 learned slot embedding，并将同一个 `cond_tokens` 复制给每个 Qwen layer：

```text
action_condition: List[Tensor]
len = num_vl_layers
each tensor = [B, 16, D]
```

注意：`state` 不在 bridge 内融合，它仍然作为 explicit input 传给 action head。

## 12. Flow-Matching Action Head

训练 target：

```python
action_target = actions[:, -(future_action_window_size + 1):, :]
```

当前 shape：

```text
action_target: [B, H, A] = [B, 8, 7]
```

`LayerwiseFlowmatchingActionHead` 使用 flow matching：

1. sample Gaussian noise：

```text
noise: [B, H, A]
```

2. 从 Beta distribution sample time `t`。
3. 构造 noisy trajectory：

```text
noisy_trajectory = (1 - t) * noise + t * action_target
```

4. 定义 target velocity：

```text
velocity = action_target - noise
```

5. 用 timestep embedding 编码 noisy actions。
6. 如果存在 `state`，用 `state_encoder` 编码。
7. 拼接：

```text
state_features + future_tokens + action_features
```

8. 跑 DiT transformer blocks。每个 block 会 cross-attend 到对应 layer 的
   PaDT condition tokens。
9. action decoder 输出 predicted velocity。

action loss：

```text
L_action = mean((pred_velocity - velocity)^2)
```

## 13. Total Training Loss

`QwenPaDTPI` 聚合总 loss：

```text
L_total =
    lambda_act   * (L_action + L_sampled_action)
  + lambda_vrt   * L_vrt
  + lambda_bbox  * L_bbox
  + lambda_mask  * L_mask
  + lambda_score * L_score
```

当前配置：

```yaml
loss_weights:
  act: 1.0
  vrt: 0.1
  bbox: 0.25
  mask: 0.25
  score: 0.1
```

返回给 trainer 的主 loss key 是：

```text
action_loss: L_total
```

原因是当前 trainer 会读取 `output_dict["action_loss"]` 作为 backward 的 scalar loss。

额外 logging metrics：

```text
loss_action_fm
loss_action_sampled
loss_vrt
loss_bbox
loss_patch_mask
loss_score
vrt diagnostics
```

当前 sampled branch 是关闭的：

```yaml
use_sampled_branch: false
sampled_branch_weight: 0.0
```

如果打开 sampled branch，它会在训练时额外跑 eval-style autoregressive VRT decode，
再经过 decoder 和 bridge，最后加一个 weighted action loss。

## 14. Inference / Eval Path

eval 时，`eval_libero.py` 通过 `ModelClient` 把当前 environment step 发给
model server。

每一步 example 包含：

```text
image: [agentview_image, wrist_image]
lang: task_description
state: robot state / state history
```

如果启用 task metadata injection，还可能包含：

```text
objects
task_objects
object_role
task_index
task_name
```

model server 调用：

```python
QwenPaDTPI.predict_action(examples=[example])
```

inference 流程：

1. `preprocess_raw_dict` 规范化 example。
2. 如果没有 object metadata，则 fallback 到 default object slots。
3. 提取 low/high visual features。
4. 构造 `P_ref`。
5. 运行 `custom_vrt_decode`。

`custom_vrt_decode` 是 fixed-schema cached decode：

```text
force <|padt_begin|>
for obj_idx in 1..O:
  force <|obj_i|> or <|padt_null|>
  for each view:
    force <|view_agentview|> / <|view_wrist|>
    repeat K times:
      compute dynamic VRT logits
      argmax patch id
      feed selected VRT token back
force <|padt_end|>
```

生成：

```text
predicted_patch_ids: [B, O, V, K]
final_hidden:        [B, S_prompt + S_generated, D]
lang_summary:        [B, D]
```

之后复用训练时的 downstream path：

```text
group VRT hidden -> PaDT decoder -> condition bridge -> action head
```

inference 不计算 `VRT/bbox/mask/score loss`。

action head 从 Gaussian action noise 开始，跑 `num_inference_timesteps` 次
Euler integration。当前 `num_inference_timesteps=4`。

输出：

```text
normalized_actions: [B, H, A]
```

`ModelClient` 后处理：

1. cache 当前 action chunk。
2. 将 normalized actions clip 到 `[-1, 1]`。
3. 对 normalized gripper channel 做 threshold，变成二值 gripper。
4. 使用 `dataset_statistics.json` 对 action unnormalize。
5. 返回当前 chunk offset 对应的 LIBERO action fields：

```text
world_vector:    raw_actions[:3]
rotation_delta:  raw_actions[3:6]
open_gripper:    raw_actions[6:7]
```

LIBERO 最后拼成：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

并执行：

```python
env.step(delta_action)
```

## 15. End-To-End Training Summary

训练链路可以概括为：

```text
LeRobot sample
  -> segmentation adapter adds object/view supervision
  -> preprocess_raw_dict
  -> Qwen visual tower extracts low/high visual memories
  -> prototype bank P_ref
  -> structured teacher VRT sequence
  -> Qwen dynamic teacher-forced forward
  -> VRT CE/NLL loss
  -> group VRT hidden states by object/view
  -> PaDT decoder
  -> bbox/mask/score losses
  -> condition bridge
  -> flow-matching action head
  -> action FM loss
  -> weighted total loss returned as action_loss
```

## 16. End-To-End Eval Summary

eval 链路可以概括为：

```text
LIBERO obs
  -> [agentview, wrist], language, state
  -> preprocess_raw_dict
  -> Qwen visual tower extracts low/high visual memories
  -> prototype bank P_ref
  -> custom fixed-schema VRT autoregressive decode
  -> group VRT hidden states by object/view
  -> PaDT decoder
  -> condition bridge
  -> flow-matching action sampling
  -> normalized action chunk
  -> unnormalize with dataset statistics
  -> LIBERO env.step action
```
