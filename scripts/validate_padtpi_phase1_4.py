"""Phase 1-4 integration validation for PaDTPI perf optimization.

Runs on a GPU node. Loads a small batch from the LIBERO dataset, runs:
  1. one full forward (action_loss) with the optimized code path
  2. measures max memory, wall-clock, and reports loss values

Usage (on a GPU node, with the starVLA conda env activated):
  cd /home/users/astar/i2r/chengzy/starVLA_origin
  python scripts/validate_padtpi_phase1_4.py \
      --config starVLA/config/training/starvla_padtpi_libero.yaml

What to verify by reading the output:
  - action_loss / loss_vrt / loss_bbox / loss_mask / loss_score finite & reasonable
  - max_memory_allocated lower than baseline (you have to compare to a previous run)
  - "ViT call count" reports 1 (single forward thanks to Phase 1)
  - "GC enabled" prints True (Phase 3 wired correctly)

This is a smoke + diagnostic script, NOT a full Phase 5 validation matrix. The
full matrix (50-step loss curves, eval rollouts) requires running the regular
training and eval pipelines.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf  # noqa: E402

# Counter for ViT forward calls to verify Phase 1.
_VISUAL_CALL_COUNT = 0


@contextmanager
def count_visual_calls(model):
    """Patch model.visual.__call__ to count invocations during a forward."""
    global _VISUAL_CALL_COUNT
    _VISUAL_CALL_COUNT = 0
    original_forward = model.visual.forward

    def counting_forward(*args, **kwargs):
        global _VISUAL_CALL_COUNT
        _VISUAL_CALL_COUNT += 1
        return original_forward(*args, **kwargs)

    model.visual.forward = counting_forward
    try:
        yield
    finally:
        model.visual.forward = original_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to yaml config")
    parser.add_argument("--batch_size", type=int, default=2, help="micro batch size for smoke")
    parser.add_argument("--n_steps", type=int, default=3, help="number of forward+backward to measure")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available — run this on a GPU node.")
        sys.exit(1)

    print(f"[info] cuda device: {torch.cuda.get_device_name(0)}")
    print(f"[info] config: {args.config}")

    cfg = OmegaConf.load(args.config)
    cfg.framework.qwenvl.base_vlm = os.path.abspath(cfg.framework.qwenvl.base_vlm)

    print("[info] building model …")
    from starVLA.model.framework.QwenPaDTPI import QwenPaDTPI
    model = QwenPaDTPI(config=cfg).to("cuda").train()

    # Report Phase 3 wiring
    gc_on = (
        not model.qwen_vl_interface.model.config.use_cache
        and getattr(model.qwen_vl_interface.model.visual, "gradient_checkpointing", False)
    )
    print(f"[Phase 3] LLM gradient checkpointing enabled: {gc_on}")
    print(f"[Phase 4] padt_decoder._use_grad_ckpt: {model.padt_decoder._use_grad_ckpt}")

    # Build a tiny synthetic-but-realistic batch by sampling from the dataloader.
    print("[info] building a tiny LIBERO batch …")
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    examples = [dataset[i] for i in range(args.batch_size)]

    # Run a single forward and count ViT calls (Phase 1 check).
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("\n[Phase 1] running 1 forward, counting ViT calls …")
    with count_visual_calls(model.qwen_vl_interface.model):
        out = model(examples=examples)
        loss = out["action_loss"]
    print(f"[Phase 1] ViT call count for 1 training forward: {_VISUAL_CALL_COUNT}")
    print(f"          (expected 1; 2 = old buggy path)")

    # Print loss values
    print("\n[loss snapshot]")
    for k, v in out.items():
        if torch.is_tensor(v) and v.numel() == 1:
            print(f"  {k:24s} = {float(v.detach().cpu()):.6f}")

    # Backward to verify the whole graph still works under GC
    print("\n[Phase 3/4] running .backward() to exercise gradient checkpoints …")
    loss.backward()
    print("           backward OK")

    # Per-step timing for the next n_steps
    print(f"\n[timing] running {args.n_steps} forward+backward iterations …")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(args.n_steps):
        model.zero_grad()
        out = model(examples=examples)
        out["action_loss"].backward()
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_step = elapsed / args.n_steps
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"           per-step wall-clock: {per_step:.3f} s")
    print(f"           peak memory:        {peak_mem:.2f} GiB")
    print(f"           batch size:         {args.batch_size}")
    print("\n[done] compare these against your v2 baseline run with the same BS.")
    print("       expected: per-step ↓ 30-50%, peak mem ↓ 40-50% vs unoptimized v2.")


if __name__ == "__main__":
    main()
