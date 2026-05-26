# PaDTPI 时间一致性问题与改造方案

## 问题描述

将 PaDT 的训练方式从 VLM（COCO 数据集，单帧独立同分布）迁移到 VLA（LIBERO，episode 内数百时间步、agentview 几乎不变、wristview 物体进出 FOV）后，eval 出现以下现象：

> 同一个 episode、同一个任务、不同时间步，对**同一个物体**预测出的 `bbox / patch_mask / VRT token` 在帧间反复跳变，**有时正确、有时完全错误**。

该现象直接影响 closed-loop 成功率，因为：

- `bbox / patch_mask / VRT` 决定 `object_memory / object_view_memory`，进而决定 action chunk 的 conditioning。
- conditioning 在帧间抖动 → action chunk 在 chunk 边界跳变 → 机械臂轨迹卡顿、抓取失败。
- 即使某些帧的 grounding 正确，被错误帧污染后的 cached action chunk 仍会驱动 N 步错误动作。

本文档的目标是 **从根因上彻底解决该抖动，提升 closed-loop 成功率**。重训成本不是约束。

## 根因分析

PaDT 在 VLM/COCO 上的训练假设是**每张图独立同分布**，模型只在单帧上把 bbox/mask 解对即可。该假设在 VLA 上彻底破裂，原因叠加如下：

1. **训练损失全部是 per-frame、per-sample 独立计算**。`loss_bbox / loss_patch_mask / loss_score / loss_vrt` 全都只看当前一帧。一个 LIBERO episode 的 200 帧被 dataloader 当成 200 个独立样本喂进去，模型完全没有"同一物体在不同帧应该给出相同预测"的 incentive。

2. **VRT 是 autoregressive sampling**。即使 `do_sample=False`，softmax 顶端两个 patch_id 的 logit 差 1e-3 即可翻盘；连锁到后续 token，误差被放大。不同时间步的图像有微小渲染差异（gripper 阴影、机械臂遮挡、轻微相机抖动），同一物体在两步可能落到完全不同的 patch token 序列上。

3. **每步从零开始 ground**。eval 时 `object_memory / object_view_memory / bbox / mask` 在每次 VLM forward 里都重新解一遍，没有跨步状态、没有任何先验。VLM 训练分布里根本不存在"上一帧的 bbox 是 X，这一帧应该也是 X 附近"这种监督。

4. **action_chunk_size 把抖动可视化成阶梯**。`model2libero_interface.py` 中 `if step % action_chunk_size == 0` 才调一次 VLM，于是抖动表现为"每隔 chunk_size 步突变一次"，对动作的破坏更显眼。

5. **`visible_by_view` 的硬 0/1 label 是震荡源**。训练时基于 segmentation 给的硬标签，物体进出 FOV 时在 0/1 间快速切换；eval 时只要 score logit 在阈值附近，就会在两个相邻 chunk 上输出完全不同的可见性。

6. **PaDT 预训练（COCO 风格）的 patch 表示分散度跟 LIBERO 不匹配**。COCO 上 patch 之间天然差异大，softmax 比较稳定；LIBERO agentview 大片背景 patch 表示太相近，logit 差距小，sample 一抖就翻。

**核心矛盾：PaDT 缺的就是"时间一致性"——既缺监督，又缺架构。**

## 改造方案（按对成功率的实际贡献排序）

下列方案均假设可以重训。投入顺序按"对最终成功率的边际贡献"排，而不是按"实现成本"排。

### 方案 1：sequence-level dataloader + 跨帧一致性 loss（最大杠杆）

**思路**：把"时间一致性"作为训练目标，让模型自己学会在 episode 内对同一物体给出稳定预测。

**Dataloader 改造**：
- 每个 batch 元素从"单帧 sample"改成"同 episode 内的一段 clip"，长度 K=4~8 帧。
- clip 内的 K 帧共享同一份 `task_objects / object_role`，但每帧各自有自己的 `image / state / action / bbox_by_view / patch_mask_by_view / visible_by_view`。
- 采样策略：从同一 episode 随机选起始帧 t，按固定步长（或随机步长）取 K 帧。优先覆盖"phase 转换"区间（接触前 / 接触中 / 接触后），避免全是同景帧。
- batch 拼接：把 K 帧 flatten 进 batch 维，并记录 clip_id 用于跨帧 loss 聚合。

**新增 loss 项**（在原 `loss_action_fm / loss_vrt / loss_bbox / loss_patch_mask / loss_score` 之外）：

| Loss 名 | 形式 | 触发条件 | 建议权重 |
|---|---|---|---|
| `loss_bbox_consistency` | `L1(pred_bbox_t, pred_bbox_{t+1})` 或 GIoU | 两帧 `visible_by_view` 都为 True，且 GT bbox 中心位移 < 阈值（避免约束运动物体） | 0.1 |
| `loss_mask_consistency` | `BCE(pred_mask_t, pred_mask_{t+1})` 或 dice | 同上 | 0.1 |
| `loss_score_consistency` | `MSE(sigmoid(score_t), sigmoid(score_{t+1}))` | 两帧 visibility label 相同 | 0.05 |
| `loss_vrt_consistency` | cosine sim 上的 contrastive，pull 同 slot 跨帧、push 不同 slot | clip 内所有可见 slot 的 vrt token 表示 | 0.05 |

**实现位置**：
- Dataloader：`starVLA/dataloader/gr00t_lerobot/datasets.py` 加 clip sampler，`lerobot_datasets.py` 加 clip-level collate。
- Loss：`starVLA/model/framework/QwenPaDTPI.py` 的 `forward` 末尾，从 `decoder_outputs` 取出 per-frame 预测后，按 clip_id 聚合算一致性 loss。
- Config：`starVLA/config/training/starvla_padtpi_libero.yaml` 加：
  ```yaml
  framework:
    padt:
      clip_length: 4
      clip_stride: 8
      consistency_loss:
        bbox: 0.1
        mask: 0.1
        score: 0.05
        vrt: 0.05
        movement_threshold_px: 16
  ```

**预期收益**：直接对症。eval 时帧间抖动会显著下降。这是单项收益最高的改动。

### 方案 2：跨步 object memory 模块（架构层）

**思路**：在 `PaDTObjectDecoder` 上加跨时间 cross-attention，让当前步的 grounding 显式吸收前 K 步的 object 表示，不再从零解。

**架构改动**：
- 在 `PaDTObjectDecoder` 的 forward 前面加一个 `TemporalObjectMemory` 模块：
  - 输入：当前帧的 slot embedding `[B, max_task_objects, D]` + 前 K 步缓存的 slot embedding `[B, K, max_task_objects, D]`。
  - 输出：经过时间 cross-attention 后的 slot embedding `[B, max_task_objects, D]`，作为 decoder 的 query。
  - 实现：一层 Transformer，slot 维度做 self-attention、时间维度做 causal cross-attention。参数量 < 10M。
- 训练时：clip_length=K 帧顺序 forward，每帧的 memory = 前 (k-1) 帧的 slot embedding（teacher-forcing 用 GT bbox/mask 监督，但 memory 用 model 自己的 slot embedding，避免 train/eval gap）。
- eval 时：每个 episode 维护一个 ring buffer 存最近 K 步的 slot embedding，调 VLM 时一并传进去。

**改动位置**：
- 新文件：`starVLA/model/modules/vlm/padt_temporal_memory.py`，实现 `TemporalObjectMemory`。
- `starVLA/model/modules/vlm/padt_object_decoder.py` forward 接受 `prev_slot_embeddings` 参数，传给 memory 模块。
- `starVLA/model/framework/QwenPaDTPI.py` 的 `predict_action` 增加 `episode_state: dict` 参数（由 ModelClient 维护），里面存 slot embedding ring buffer。
- `examples/LIBERO/eval_files/model2libero_interface.py` 的 `ModelClient.reset` 重置 buffer，`ModelClient.step` 把 buffer 传给 policy server。

**预期收益**：天然时间稳定，且能跟踪缓慢移动的物体。比方案 1 更彻底，但实现成本更高，需要重训。建议方案 1 走通后立即上方案 2。

### 方案 3：anchor + refine 两段式 grounding

**思路**：episode 开头第一帧做一次完整 detection（anchor），后续步只对 anchor 做 delta refine（类似 SOT/tracker）。

**改动**：
- 训练数据增加一对帧：`(t=0_frame, t=k_frame)`，t=0 用 full PaDT decode，t=k 用一个轻量 "RefineHead" 输入 anchor bbox + 当前帧 patch feature，预测 delta bbox + delta mask。
- 损失加 `loss_refine_bbox / loss_refine_mask`，比 full detection loss 权重更高（因为这是 eval 主路径）。
- eval：episode 第一帧 full decode → 之后每步 refine。检测到大幅运动或 score 显著下降时触发 re-anchor。

**预期收益**：计算量降一半（不用每步跑 full decoder），稳定性最好。代价是物体被推走/被遮挡时无法重新 ground，必须有可靠的 re-anchor 触发器。

**结合方案 2 的形态**：方案 2 是"软记忆"（学出来的），方案 3 是"硬锚点"（结构上的）。两者可以并存：用方案 2 提供短期稳定性，用方案 3 提供长期 anchor。

### 方案 4：task-phase conditioning

**思路**：把"episode 进度 / 接触状态 / gripper 历史"做成额外 condition token 喂进 decoder。让"抖动"被重写成"phase 切换"——可解释、可控。

**改动**：
- 新增 phase encoder：输入 [t / episode_length, gripper_open_history, ee_velocity]，输出一个 phase token。
- decoder 的 query 拼上 phase token，role embedding + phase embedding 共同决定 slot 的语义角色。
- 训练时 phase 信号从 dataset 计算（如 `action.gripper` 的变化点定义 phase 边界）。

**预期收益**：让模型主动区分"接近 / 接触 / 操作 / 放置 / 退出"阶段，对应不同的 attention 模式。对 libero_10 这类长时序 multi-stage 任务尤其有用。

**改动位置**：
- `starVLA/dataloader/gr00t_lerobot/datasets.py` 计算 phase 信号字段。
- `starVLA/model/modules/vlm/padt_object_decoder.py` 加 phase token cross-attention。

### 方案 5：visibility label 软化 + 边界平滑

**思路**：消除 `visible_by_view` 的 0/1 硬切换带来的 score 震荡。

**改动**：
- 在 dataloader 阶段，把 `visible_by_view` 从硬 0/1 改成时间窗滑动平均（窗口 3~5 帧），label 变成 [0,1] 软概率。
- `loss_score` 改成对 soft label 的 BCE 或 KL。
- eval 时 score threshold 用滞回阈值（hysteresis）：进入 0.6 才算"出现"，退出 0.3 才算"消失"，避免在 0.5 附近震荡。

**实现成本极低，可与任何方案叠加。**

### 方案 6：VRT autoregressive 改成 episode-anchored sticky decode

**思路**：当前 `custom_vrt_decode` 每步从零 sample；改成跨步缓存 VRT prefix，本质让 VRT 序列在时间上做 soft anchor。

**改动**：
- `QWen2_5_PaDT.custom_vrt_decode` 增加 `prev_vrt_tokens` 参数；如果传入，作为 KV prefix 让本次 decode 在它的基础上 continue / refine，而不是 cold start。
- 训练时同时支持两种 mode（with prefix / without prefix），随机切换以避免过拟合到 always-with-prefix。
- eval 时 ModelClient 维护 `last_vrt_tokens` per episode，每步喂给 server。

**预期收益**：在不动 decoder 结构的前提下显著降 VRT 跳变。配合方案 2 使用效果最好。

### 方案 7：训练时的 episode-aware 数据增强

**思路**：让 VLM 在训练时就见过"同一物体在轻微扰动下"的样本，强迫输出对小扰动鲁棒。

**改动**：
- 对方案 1 的 clip 内 K 帧，额外做一份"clone + 局部增强"的副本（颜色抖动、轻微遮挡、小幅 crop）。
- 增加 `loss_aug_consistency`：pred(原帧) 与 pred(增强帧) 之间的 bbox/mask 必须一致。
- 这等于用数据增强模拟 LIBERO 渲染的真实噪声（gripper 阴影、机械臂遮挡），逼模型学到 invariant 表示。

**和方案 1 完全互补**：方案 1 约束"不同时间但同物体"，方案 7 约束"同时间但加噪"。两者合起来才能覆盖 eval 的实际噪声分布。

## 实施 Roadmap

按对成功率的实际贡献 + 解耦程度排执行顺序：

**Phase A（基础时间一致性，预计 +10~20% 成功率）**
1. 方案 5（visibility 软化）—— 1 天，对所有方案有 enabling 作用，先做。
2. 方案 1（sequence-level dataloader + 一致性 loss）—— 1~2 周，单项收益最高。
3. 方案 7（augmentation consistency）—— 与方案 1 同时做，复用 clip dataloader。

**Phase B（架构稳定性，预计再 +5~15% 成功率）**
4. 方案 2（temporal object memory）—— 2~3 周，方案 1 验证有效后跟进。
5. 方案 6（VRT sticky decode）—— 1 周，可与方案 2 并行做。

**Phase C（长时序任务专项）**
6. 方案 4（task-phase conditioning）—— 1~2 周，对 libero_10 / libero_90 这类长 horizon 任务尤其有效。
7. 方案 3（anchor + refine）—— 3~4 周，最重的架构改动；如果方案 2 已经做到饱和、剩余抖动主要在长 episode，再上。

## 评测协议

每个 phase 完成后跑同一套对照实验，避免"感觉变好"：

- **指标**：
  - 主指标：libero_goal / libero_object / libero_spatial / libero_10 四套各 50 episodes 的平均成功率。
  - 辅助指标 1：同 episode 内连续 chunk 边界上的 bbox IoU 均值（应 > 0.85）。
  - 辅助指标 2：score 在阈值附近震荡次数 / episode（应 → 0）。
  - 辅助指标 3：VRT token 序列在连续 chunk 之间的 Hamming 距离（应 → 0）。

- **诊断脚本**：从 `diagnostic_meta` 模式产出的 `steps.jsonl` 抽 bbox / score / patch_ids，按 episode × slot 画 trajectory 和 IoU 曲线。可以放到 `examples/LIBERO/eval_files/analyze_padt_libero_diagnostics.py` 里扩展。

- **回归对照**：每个 phase 上线时保留上一个 phase 的 checkpoint，跑同样 task 子集做 A/B；如果主指标不升或辅助指标恶化，回退该方案。

## 备注

- 本方案默认 LIBERO 4 套 task suite 的 `padt_task_specs.jsonl` 已经按 `docs/padtpi_eval_alignment_plan.md` 准备好。
- 方案 1 / 2 / 7 都依赖 clip-level dataloader，应该统一在一份 dataloader 改造里实现，避免做三份。
- 重训成本不是约束，但 wall-clock 时间是；建议方案 1 + 5 + 7 一起 retrain 一次（共享 clip dataloader），不要拆三次。
- 所有方案完成后，回头评估"PaDT 是不是 VLA 最优形态"。如果时间一致性问题需要这么多专项补丁，说明 PaDT 在 VLA 上结构性不匹配；届时可考虑替换为天然 video-grounding 的 backbone（如 video transformer + slot attention），但那是另一个项目的范畴。
