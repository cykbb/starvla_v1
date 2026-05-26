# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._termination_requested = False
        self._termination_signal = None
        self._last_saved_checkpoint_step = None

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._install_signal_handlers()
        self._init_wandb()

    def _install_signal_handlers(self):
        """Request a checkpoint when SLURM sends a preemption/timeout signal."""

        def _request_checkpoint(signum, _frame):
            self._termination_requested = True
            self._termination_signal = signal.Signals(signum).name

        for signum in (signal.SIGUSR1, signal.SIGTERM):
            try:
                signal.signal(signum, _request_checkpoint)
            except (OSError, ValueError) as exc:
                logger.warning(f"Could not install handler for signal {signum}: {exc}")

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self, reason: str = "periodic"):
        """Save current training state."""
        if self.accelerator.is_main_process:
            if self._last_saved_checkpoint_step == self.completed_steps:
                self.accelerator.print(
                    f"Checkpoint for step {self.completed_steps} already saved; skipping duplicate {reason} save."
                )
                self.accelerator.wait_for_everyone()
                return

            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self._last_saved_checkpoint_step = self.completed_steps
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path} ({reason})")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _termination_requested_across_ranks(self) -> bool:
        """Synchronize a local signal request so every rank exits together."""
        requested = 1 if self._termination_requested else 0
        if dist.is_initialized():
            flag = torch.tensor([requested], device=self.accelerator.device, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            requested = int(flag.item())
            if requested:
                self._termination_requested = True
        return bool(requested)

    def _maybe_save_and_stop_for_signal(self) -> bool:
        """Save a last-step checkpoint after SIGUSR1/SIGTERM and stop training."""
        if not self._termination_requested_across_ranks():
            return False

        signal_name = self._termination_signal or "remote-rank-signal"
        logger.warning(
            f"Received {signal_name}; saving preemption checkpoint at step {self.completed_steps} before exit."
        )
        self._save_checkpoint(reason=f"signal:{signal_name}")
        return True

    @staticmethod
    def _to_scalar(value):
        """Best-effort conversion for logging-friendly scalar metrics."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().float().item())
            return float(value.detach().float().mean().item())
        if isinstance(value, np.ndarray):
            return float(value.mean())
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return None

    def _extract_loss_metrics(self, output_dict, prefix: str = ""):
        """Collect scalar loss terms returned by the framework forward pass."""
        metric_aliases = {
            "action_loss": "action_dit_loss",
            "loss_action_fm": "loss_action_fm",
            "loss_action_sampled": "loss_action_sampled",
            "loss_vrt": "loss_vrt",
            "loss_bbox": "loss_bbox",
            "loss_patch_mask": "loss_patch_mask",
            "loss_score": "loss_score",
            "vrt_supervised_tokens": "vrt_supervised_tokens",
            "vrt_teacher_top1_acc": "vrt_teacher_top1_acc",
            "vrt_teacher_hidden_norm": "vrt_teacher_hidden_norm",
            "vrt_teacher_proto_norm": "vrt_teacher_proto_norm",
            "vrt_teacher_target_logit": "vrt_teacher_target_logit",
            "vrt_teacher_top1_margin": "vrt_teacher_top1_margin",
            "vrt_teacher_logit_abs_max": "vrt_teacher_logit_abs_max",
            "vrt_proto_global_norm": "vrt_proto_global_norm",
            "vrt_proto_global_norm_max": "vrt_proto_global_norm_max",
        }

        metrics = {}
        for source_key, metric_name in metric_aliases.items():
            if source_key not in output_dict:
                continue
            scalar_value = self._to_scalar(output_dict[source_key])
            if scalar_value is not None:
                metrics[f"{prefix}{metric_name}"] = scalar_value

        return metrics

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            metrics = dict(metrics)
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def _visualize_padt_batch(self, batch: list, num_samples: int) -> None:
        """Dump bbox + patch-mask overlays for PaDT alignment debugging. Rank-0 only."""
        from PIL import Image, ImageDraw, ImageFont

        PALETTE = [
            (255, 99,  71), (30,  144, 255), (60,  179, 113),
            (255, 165,   0), (186,  85, 211), (255, 215,   0),
        ]
        PATCH_GRID = 16
        VIEW_ORDER = [("agentview", 0), ("wrist", 1)]

        save_dir = Path(self.config.output_dir) / "viz_padt" / f"step_{self.completed_steps:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for s_idx, sample in enumerate(batch[:num_samples]):
            images  = sample.get("image", [])
            objects = sample.get("objects", [])
            task    = sample.get("task_name", sample.get("lang", ""))

            panels = []
            for view_name, img_idx in VIEW_ORDER:
                if img_idx >= len(images):
                    continue
                img     = images[img_idx].copy().convert("RGBA")
                W, H    = img.size
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw    = ImageDraw.Draw(overlay, "RGBA")
                font    = ImageFont.load_default()
                cell_h  = H / PATCH_GRID
                cell_w  = W / PATCH_GRID

                for o_idx, obj in enumerate(objects):
                    visible = obj.get("visible_by_view", {}).get(view_name, False)
                    color   = PALETTE[o_idx % len(PALETTE)]
                    label   = obj.get("label", f"obj{o_idx}")

                    # bbox rectangle
                    bbox = obj.get("bbox_by_view", {}).get(view_name, [0, 0, 0, 0])
                    if visible and any(v > 0 for v in bbox):
                        x0 = int(bbox[0] * W)
                        y0 = int(bbox[1] * H)
                        x1 = int(bbox[2] * W)
                        y1 = int(bbox[3] * H)
                        draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=3)
                        draw.text((x0 + 4, max(2, y0 + 4)), label, fill=(*color, 255), font=font)

                    # patch mask coverage (semi-transparent fill)
                    patch_mask = obj.get("patch_mask_by_view", {}).get(view_name, [])
                    if patch_mask and visible:
                        grid = np.array(patch_mask, dtype=np.float32).reshape(PATCH_GRID, PATCH_GRID)
                        for r in range(PATCH_GRID):
                            for c in range(PATCH_GRID):
                                cov = grid[r, c]
                                if cov > 0.05:
                                    alpha = int(min(120, 80 * cov))
                                    px0, py0 = int(c * cell_w), int(r * cell_h)
                                    px1, py1 = int((c + 1) * cell_w), int((r + 1) * cell_h)
                                    draw.rectangle([px0, py0, px1, py1], fill=(*color, alpha))

                    # valid patch ids — outlined in same color as the bbox (bright, high alpha)
                    for pid in obj.get("valid_patch_ids", []):
                        r, c = divmod(int(pid), PATCH_GRID)
                        px0, py0 = int(c * cell_w), int(r * cell_h)
                        px1, py1 = int((c + 1) * cell_w), int((r + 1) * cell_h)
                        draw.rectangle([px0, py0, px1, py1], outline=(*color, 220), width=1)

                    # Core patch ids — cross in same color as the bbox, only on agentview.
                    # Each cross should sit inside its object's colored bbox if mask alignment is correct.
                    if view_name == "agentview":
                        for pid in obj.get("core_patch_ids", []):
                            r, c = divmod(int(pid), PATCH_GRID)
                            cx = int((c + 0.5) * cell_w)
                            cy = int((r + 0.5) * cell_h)
                            draw.line([(cx - 5, cy), (cx + 5, cy)], fill=(*color, 255), width=3)
                            draw.line([(cx, cy - 5), (cx, cy + 5)], fill=(*color, 255), width=3)

                composed = Image.alpha_composite(img, overlay).convert("RGB")
                ImageDraw.Draw(composed).text((4, 4), view_name, fill=(255, 255, 255), font=font)
                panels.append(composed)

            if not panels:
                continue

            gap    = 4
            tw     = sum(p.width for p in panels) + gap * (len(panels) - 1)
            th     = max(p.height for p in panels) + 20
            canvas = Image.new("RGB", (tw, th), (24, 24, 24))
            x = 0
            for p in panels:
                canvas.paste(p, (x, 20))
                x += p.width + gap
            ImageDraw.Draw(canvas).text(
                (4, 4), f"[{s_idx}] {task}", fill=(255, 255, 255), font=ImageFont.load_default()
            )
            canvas.save(save_dir / f"sample_{s_idx:03d}.png")

        self.accelerator.print(f"[VizPaDT] Saved {min(num_samples, len(batch))} samples to {save_dir}")

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        interrupted = False
        while self.completed_steps < self.config.trainer.max_train_steps:
            if self._maybe_save_and_stop_for_signal():
                interrupted = True
                break

            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # PaDT alignment visualizer — debug only, rank-0, step 0 only
            viz_n = int(self.config.trainer.get("visualize_padt_samples", 0))
            if viz_n > 0 and self.accelerator.is_local_main_process and self.completed_steps == 0:
                self._visualize_padt_batch(batch_vla, viz_n)

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self._maybe_save_and_stop_for_signal():
                interrupted = True
                break

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self._maybe_save_and_stop_for_signal():
                interrupted = True
                break

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        if interrupted:
            self._finalize_interrupted_training()
        else:
            self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Run action eval on a fresh batch and attach eval losses plus prediction error."""
        step_metrics = {} if step_metrics is None else step_metrics
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    eval_forward_dict = self.model.forward(examples)
                output_dict = self.model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)
        finally:
            self.model.train(was_training)

        step_metrics.update(self._extract_loss_metrics(eval_forward_dict, prefix="eval_"))

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                total_loss = output_dict["action_loss"]

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

        return self._extract_loss_metrics(output_dict)

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()

    def _finalize_interrupted_training(self):
        """Clean shutdown after a preemption checkpoint without writing final_model."""
        if self.accelerator.is_main_process:
            logger.warning(
                "Training interrupted after saving a checkpoint. "
                "Skipping final_model because max_train_steps was not reached."
            )
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
