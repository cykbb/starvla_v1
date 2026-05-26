#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (255, 99, 71),
    (30, 144, 255),
    (60, 179, 113),
    (255, 165, 0),
    (186, 85, 211),
    (255, 215, 0),
]

LOSS_KEYS = [
    "action_dit_loss",
    "loss_action_fm",
    "loss_action_sampled",
    "loss_vrt",
    "loss_bbox",
    "loss_patch_mask",
    "loss_score",
]


@dataclass
class ModeDiagnostics:
    name: str
    manifest: dict[str, Any]
    summaries: pd.DataFrame
    step_records: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PaDT LIBERO short diagnostic outputs.")
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-debug-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=30000)
    parser.add_argument(
        "--save-overlay-frames",
        action="store_true",
        help="Save per-step overlay images on top of the original agentview / wrist frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_metrics = parse_train_metrics(args.train_log)
    run_config = load_run_config(args.run_dir / "config.yaml")
    mode_dirs = discover_mode_dirs(args.eval_debug_dir)
    if not mode_dirs:
        raise FileNotFoundError(
            f"No diagnostic run directories found under {args.eval_debug_dir}. "
            "Expected either a run directory with run_manifest.json or a parent directory containing "
            "subdirectories such as current_eval/ and meta_aligned_eval/."
        )

    modes = [load_mode_diagnostics(mode_dir) for mode_dir in mode_dirs]
    rollout_summary_df = build_rollout_summary(modes)
    step_debug_df = build_step_debug_dataframe(modes)
    mode_stats = compute_mode_stats(modes)
    train_snapshot = extract_train_snapshot(train_metrics, target_step=args.target_step)

    if not rollout_summary_df.empty:
        rollout_summary_df.to_csv(args.output_dir / "eval_rollout_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "eval_rollout_summary.csv", index=False)

    if not step_debug_df.empty:
        step_debug_df.to_csv(args.output_dir / "step_debug.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "step_debug.csv", index=False)

    plot_loss_curves(train_metrics, target_step=args.target_step, output_path=args.output_dir / "loss_curves.png")
    plot_action_trajectories(modes, output_path=args.output_dir / "action_trajectories.png")
    plot_slot_diagnostics(modes, output_path=args.output_dir / "slot_diagnostics.png")
    plot_overlay_examples(modes, output_path=args.output_dir / "overlay_examples.png")
    if args.save_overlay_frames:
        dump_overlay_frames(modes, output_root=args.output_dir / "overlay_frames")
    write_report(
        output_path=args.output_dir / "report.md",
        run_config=run_config,
        train_snapshot=train_snapshot,
        mode_stats=mode_stats,
        modes=modes,
        target_step=args.target_step,
        save_overlay_frames=bool(args.save_overlay_frames),
    )


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def parse_train_metrics(train_log: Path) -> pd.DataFrame:
    text = strip_ansi(train_log.read_text(encoding="utf-8", errors="ignore"))
    lines = text.splitlines()
    records: list[dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = re.search(r"Step\s+(\d+), Loss:", line)
        if not match:
            idx += 1
            continue

        step = int(match.group(1))
        brace_parts: list[str] = []
        brace_balance = 0
        found_brace = False
        line_cursor = idx
        while line_cursor < len(lines):
            current = lines[line_cursor]
            if not found_brace:
                brace_pos = current.find("{")
                if brace_pos < 0:
                    line_cursor += 1
                    continue
                current = current[brace_pos:]
                found_brace = True

            brace_parts.append(current.strip())
            brace_balance += current.count("{") - current.count("}")
            line_cursor += 1
            if found_brace and brace_balance <= 0:
                break

        idx = line_cursor
        if not brace_parts:
            continue

        payload = " ".join(brace_parts)
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end >= start:
            payload = payload[start : end + 1]
        try:
            metrics = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(metrics, dict):
            continue
        metrics["step"] = step
        records.append(metrics)

    if not records:
        return pd.DataFrame(columns=["step", *LOSS_KEYS, *[f"eval_{key}" for key in LOSS_KEYS]])

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["step"], keep="last").sort_values("step").reset_index(drop=True)
    return df


def load_run_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def discover_mode_dirs(root: Path) -> list[Path]:
    if (root / "run_manifest.json").exists():
        return [root]
    return sorted([path for path in root.iterdir() if path.is_dir() and (path / "run_manifest.json").exists()])


def load_mode_diagnostics(mode_dir: Path) -> ModeDiagnostics:
    manifest = load_json(mode_dir / "run_manifest.json")
    mode_name = str(manifest.get("mode_name") or mode_dir.name)

    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(mode_dir.rglob("episode_summary.json")):
        summary = load_json(summary_path)
        summary["mode"] = mode_name
        summaries.append(summary)

    step_records: list[dict[str, Any]] = []
    for steps_path in sorted(mode_dir.rglob("steps.jsonl")):
        with open(steps_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["mode"] = mode_name
                step_records.append(record)

    summaries_df = pd.DataFrame(summaries)
    return ModeDiagnostics(name=mode_name, manifest=manifest, summaries=summaries_df, step_records=step_records)


def build_rollout_summary(modes: list[ModeDiagnostics]) -> pd.DataFrame:
    frames = [mode.summaries for mode in modes if not mode.summaries.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["success_rate"] = combined.groupby("mode")["success"].transform("mean")
    return combined


def build_step_debug_dataframe(modes: list[ModeDiagnostics]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        summary_lookup = {
            (int(row["task_id"]), int(row["episode_idx"])): row
            for _, row in mode.summaries.iterrows()
        }
        for record in mode.step_records:
            task_id = int(record.get("task_id", -1))
            episode_idx = int(record.get("episode_idx", -1))
            summary = summary_lookup.get((task_id, episode_idx), {})
            delta_action = np.asarray(record.get("delta_action", []), dtype=np.float32).reshape(-1)
            normalized_action = np.asarray(record.get("current_normalized_action", []), dtype=np.float32).reshape(-1)
            rows.append(
                {
                    "mode": mode.name,
                    "task_id": task_id,
                    "task_description": str(record.get("task_description", "")),
                    "episode_idx": episode_idx,
                    "step": int(record.get("step", -1)),
                    "task_meta_applied": bool(record.get("task_meta_applied", False)),
                    "success": bool(summary.get("success", False)),
                    "policy_latency_sec": float(record.get("policy_latency_sec", 0.0)),
                    "chunk_step_offset": int(record.get("chunk_step_offset", 0)),
                    "chunk_reused": bool(record.get("chunk_reused", False)),
                    "delta_action_norm": float(np.linalg.norm(delta_action)) if delta_action.size else 0.0,
                    "normalized_action_norm": float(np.linalg.norm(normalized_action)) if normalized_action.size else 0.0,
                    "max_abs_action": float(np.abs(delta_action).max()) if delta_action.size else 0.0,
                    "gripper_value": float(delta_action[-1]) if delta_action.size else 0.0,
                    "predicted_patch_ids_json": json.dumps(record.get("predicted_patch_ids", []), ensure_ascii=False),
                    "task_object_roles_json": json.dumps(record.get("task_object_roles", []), ensure_ascii=False),
                    "score_logits_json": json.dumps(record.get("score_logits", []), ensure_ascii=False),
                    "visibility_logits_json": json.dumps(record.get("visibility_logits", []), ensure_ascii=False),
                    "agentview_path": record.get("agentview_path", None),
                    "wrist_path": record.get("wrist_path", None),
                }
            )
    return pd.DataFrame(rows)


def compute_mode_stats(modes: list[ModeDiagnostics]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for mode in modes:
        summaries = mode.summaries
        step_records = mode.step_records
        action_norm_mean = float(summaries["action_norm_mean"].mean()) if not summaries.empty else 0.0
        max_abs_action = float(summaries["max_abs_action"].max()) if not summaries.empty else 0.0
        gripper_flips = float(summaries["gripper_flips"].mean()) if not summaries.empty else 0.0
        patch_change_rate = compute_patch_change_rate(step_records)
        score_prob_mean = compute_score_probability_mean(step_records)
        visibility_prob_mean = compute_visibility_probability_mean(step_records)
        stats[mode.name] = {
            "success_rate": float(summaries["success"].mean()) if not summaries.empty else 0.0,
            "episodes": int(len(summaries)),
            "action_norm_mean": action_norm_mean,
            "max_abs_action": max_abs_action,
            "gripper_flips_mean": gripper_flips,
            "patch_change_rate": patch_change_rate,
            "score_prob_mean": score_prob_mean,
            "visibility_prob_mean": visibility_prob_mean,
            "task_meta_applied_any": bool(summaries["task_meta_applied"].any()) if not summaries.empty else False,
        }
    return stats


def extract_train_snapshot(train_metrics: pd.DataFrame, *, target_step: int) -> dict[str, Any]:
    if train_metrics.empty:
        return {}
    if target_step in set(train_metrics["step"].tolist()):
        row = train_metrics.loc[train_metrics["step"] == target_step].iloc[-1]
    else:
        row = train_metrics.iloc[(train_metrics["step"] - target_step).abs().argmin()]
    return row.to_dict()


def plot_loss_curves(train_metrics: pd.DataFrame, *, target_step: int, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    plot_keys = [
        ("action_dit_loss", "eval_action_dit_loss", "Action DiT"),
        ("loss_action_fm", "eval_loss_action_fm", "Action FM"),
        ("loss_action_sampled", "eval_loss_action_sampled", "Action Sampled"),
        ("loss_vrt", "eval_loss_vrt", "VRT"),
        ("loss_bbox", "eval_loss_bbox", "BBox"),
        ("loss_patch_mask", "eval_loss_patch_mask", "Patch Mask"),
        ("loss_score", "eval_loss_score", "Score"),
    ]

    if train_metrics.empty:
        for ax in axes:
            ax.axis("off")
        axes[0].text(0.5, 0.5, "No train metrics parsed.", ha="center", va="center")
        plt.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    for ax, (train_key, eval_key, title) in zip(axes, plot_keys):
        if train_key in train_metrics:
            ax.plot(train_metrics["step"], train_metrics[train_key], label=f"train:{train_key}")
        if eval_key in train_metrics:
            ax.plot(train_metrics["step"], train_metrics[eval_key], label=f"eval:{eval_key}")
        ax.axvline(target_step, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(plot_keys):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_action_trajectories(modes: list[ModeDiagnostics], *, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    axes = axes.flatten()
    action_labels = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    dim_colors = {
        modes[idx].name: color
        for idx, color in enumerate(["tab:blue", "tab:orange", "tab:green", "tab:red"])
        if idx < len(modes)
    }

    for dim, label in enumerate(action_labels):
        ax = axes[dim]
        for mode in modes:
            trajectory = first_episode_actions(mode)
            if trajectory.size == 0:
                continue
            ax.plot(trajectory[:, dim], label=mode.name, color=dim_colors.get(mode.name, None))
        ax.set_title(label)
        ax.set_xlabel("policy step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes[-1]
    width = 0.35
    mode_names = [mode.name for mode in modes]
    means = [compute_mode_stats([mode])[mode.name]["action_norm_mean"] for mode in modes]
    max_vals = [compute_mode_stats([mode])[mode.name]["max_abs_action"] for mode in modes]
    x = np.arange(len(mode_names))
    ax.bar(x - width / 2, means, width=width, label="mean action norm")
    ax.bar(x + width / 2, max_vals, width=width, label="max abs action")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, rotation=20, ha="right")
    ax.set_title("Action magnitude summary")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_slot_diagnostics(modes: list[ModeDiagnostics], *, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    slot_indices = np.arange(4)
    width = 0.8 / max(len(modes), 1)

    for mode_idx, mode in enumerate(modes):
        stats = compute_slot_stats(mode.step_records)
        offset = (mode_idx - (len(modes) - 1) / 2.0) * width
        axes[0].bar(slot_indices + offset, stats["patch_change_rate"], width=width, label=mode.name)
        axes[1].bar(slot_indices + offset, stats["score_prob_mean"], width=width, label=mode.name)
        axes[2].bar(slot_indices + offset, stats["visibility_agentview_mean"], width=width, label=mode.name)
        axes[3].bar(slot_indices + offset, stats["visibility_wrist_mean"], width=width, label=mode.name)

    titles = [
        "Patch id change rate by slot",
        "Mean score probability by slot",
        "Mean visibility probability (agentview)",
        "Mean visibility probability (wrist)",
    ]
    for ax, title in zip(axes, titles):
        ax.set_xticks(slot_indices)
        ax.set_xticklabels([f"slot_{idx + 1}" for idx in slot_indices], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overlay_examples(modes: list[ModeDiagnostics], *, output_path: Path) -> None:
    panels: list[Image.Image] = []
    for mode in modes:
        selected_records = select_overlay_records(mode.step_records)
        if not selected_records:
            continue
        row_images = []
        for record in selected_records:
            row_images.append(
                build_overlay_pair(
                    record=record,
                    title=f"{mode.name} | ep{record.get('episode_idx')} step{record.get('step')}",
                )
            )
        panels.append(stitch_row(row_images, title=mode.name))

    if not panels:
        image = Image.new("RGB", (640, 120), (24, 24, 24))
        draw = ImageDraw.Draw(image)
        draw.text((20, 50), "No overlay examples available.", fill=(255, 255, 255), font=ImageFont.load_default())
        image.save(output_path)
        return

    canvas = stitch_column(panels)
    canvas.save(output_path)


def write_report(
    *,
    output_path: Path,
    run_config: dict[str, Any],
    train_snapshot: dict[str, Any],
    mode_stats: dict[str, dict[str, Any]],
    modes: list[ModeDiagnostics],
    target_step: int,
    save_overlay_frames: bool,
) -> None:
    lines: list[str] = []
    lines.append("# PaDT LIBERO Diagnostic Report")
    lines.append("")

    summary_line = infer_primary_summary(run_config=run_config, train_snapshot=train_snapshot, mode_stats=mode_stats)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- {summary_line}")
    lines.append("")

    lines.append("## Training Snapshot")
    lines.append("")
    if train_snapshot:
        lines.append(f"- Target step: `{int(train_snapshot.get('step', target_step))}`")
        for key in [
            "action_dit_loss",
            "loss_action_fm",
            "loss_action_sampled",
            "loss_vrt",
            "loss_bbox",
            "loss_patch_mask",
            "loss_score",
            "eval_action_dit_loss",
            "eval_loss_action_fm",
            "eval_loss_action_sampled",
            "eval_loss_vrt",
            "eval_loss_bbox",
            "eval_loss_patch_mask",
            "eval_loss_score",
        ]:
            if key in train_snapshot:
                lines.append(f"- `{key}`: `{float(train_snapshot[key]):.6f}`")
    else:
        lines.append("- No step metrics were parsed from the training log.")

    padt_cfg = run_config.get("framework", {}).get("padt", {})
    loss_weights = padt_cfg.get("loss_weights", {})
    lines.append("")
    lines.append("## Run Config")
    lines.append("")
    lines.append(f"- `noisy_teacher_probability`: `{padt_cfg.get('noisy_teacher_probability', 'n/a')}`")
    lines.append(f"- `use_sampled_branch`: `{padt_cfg.get('use_sampled_branch', 'n/a')}`")
    lines.append(f"- `sampled_branch_weight`: `{padt_cfg.get('sampled_branch_weight', 'n/a')}`")
    lines.append(f"- `loss_weights.vrt`: `{loss_weights.get('vrt', 'n/a')}`")
    lines.append("")

    lines.append("## Eval Comparison")
    lines.append("")
    if mode_stats:
        lines.append("| mode | success_rate | mean_action_norm | max_abs_action | mean_gripper_flips | patch_change_rate | mean_score_prob | mean_visibility_prob |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for mode_name, stats in mode_stats.items():
            lines.append(
                f"| `{mode_name}` | `{stats['success_rate']:.3f}` | `{stats['action_norm_mean']:.3f}` | "
                f"`{stats['max_abs_action']:.3f}` | `{stats['gripper_flips_mean']:.3f}` | "
                f"`{stats['patch_change_rate']:.3f}` | `{stats['score_prob_mean']:.3f}` | "
                f"`{stats['visibility_prob_mean']:.3f}` |"
            )
    else:
        lines.append("- No eval diagnostic outputs were found.")
    lines.append("")

    lines.append("## Likely Causes")
    lines.append("")
    for finding in infer_findings(run_config=run_config, train_snapshot=train_snapshot, mode_stats=mode_stats):
        lines.append(f"- {finding}")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append("- `loss_curves.png`")
    lines.append("- `action_trajectories.png`")
    lines.append("- `slot_diagnostics.png`")
    lines.append("- `overlay_examples.png`")
    if save_overlay_frames:
        lines.append("- `overlay_frames/`")
    lines.append("- `eval_rollout_summary.csv`")
    lines.append("- `step_debug.csv`")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def infer_primary_summary(
    *,
    run_config: dict[str, Any],
    train_snapshot: dict[str, Any],
    mode_stats: dict[str, dict[str, Any]],
) -> str:
    findings = infer_findings(run_config=run_config, train_snapshot=train_snapshot, mode_stats=mode_stats)
    if findings:
        return findings[0]
    return "The available diagnostics do not isolate a single dominant failure mode yet."


def infer_findings(
    *,
    run_config: dict[str, Any],
    train_snapshot: dict[str, Any],
    mode_stats: dict[str, dict[str, Any]],
) -> list[str]:
    findings: list[str] = []
    current = mode_stats.get("current_eval", None)
    meta = mode_stats.get("meta_aligned_eval", None)

    if current and meta:
        meta_helped = (
            meta["success_rate"] > current["success_rate"] + 0.05
            or meta["action_norm_mean"] > max(current["action_norm_mean"] * 1.2, current["action_norm_mean"] + 0.05)
        )
        if meta_helped:
            findings.append(
                "`meta_aligned_eval` 的动作幅值/成功率明显高于 `current_eval`，更像是 eval 缺少 "
                "`task_objects/object_role` 导致 object-centric 条件在推理时退化。"
            )

    current_like = current or next(iter(mode_stats.values()), None)
    if current_like:
        if current_like["patch_change_rate"] > 0.35:
            findings.append(
                "`predicted_patch_ids` 在相邻 step 间变化频繁，说明 VRT / object decoder 端的 slot 定位不稳定。"
            )
        if current_like["action_norm_mean"] < 0.15 or current_like["max_abs_action"] < 0.12:
            findings.append(
                "动作整体幅值偏小，轨迹更像接近静止的保守输出，action head / condition bridge 可能没有得到足够有效的条件。"
            )
        if current_like["score_prob_mean"] < 0.35 and current_like["visibility_prob_mean"] < 0.35:
            findings.append(
                "object decoder 的 `score/visibility` 概率整体偏低，说明模型没有稳定地激活与任务相关的 object slots。"
            )

    padt_cfg = run_config.get("framework", {}).get("padt", {})
    if (
        train_snapshot
        and float(train_snapshot.get("eval_loss_vrt", 999.0)) < 5.0
        and float(train_snapshot.get("eval_loss_action_fm", 999.0)) < 0.4
        and not any("task_objects/object_role" in finding for finding in findings)
        and current_like
        and current_like["patch_change_rate"] > 0.25
    ):
        findings.append(
            "训练指标在目标 checkpoint 处并不算发散，但当前 checkpoint 仍处于 Stage A warmup "
            f"(`noisy_teacher_probability={padt_cfg.get('noisy_teacher_probability', 'n/a')}`, "
            f"`use_sampled_branch={padt_cfg.get('use_sampled_branch', 'n/a')}`)，"
            "更像是 teacher-forced 训练与 sampled decode 评测之间的 train/infer gap。"
        )

    if not findings:
        findings.append("现有指标没有指向单一故障点，需要结合新的短重跑产物继续判断。")
    return findings


def compute_patch_change_rate(step_records: list[dict[str, Any]]) -> float:
    slot_stats = compute_slot_stats(step_records)
    return float(np.nanmean(slot_stats["patch_change_rate"])) if slot_stats["patch_change_rate"].size else 0.0


def compute_score_probability_mean(step_records: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for record in step_records:
        score_logits = np.asarray(record.get("score_logits", []), dtype=np.float32)
        if score_logits.size:
            values.extend(sigmoid(score_logits).reshape(-1).tolist())
    return float(np.mean(values)) if values else 0.0


def compute_visibility_probability_mean(step_records: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for record in step_records:
        visibility_logits = np.asarray(record.get("visibility_logits", []), dtype=np.float32)
        if visibility_logits.size:
            values.extend(sigmoid(visibility_logits).reshape(-1).tolist())
    return float(np.mean(values)) if values else 0.0


def compute_slot_stats(step_records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    patch_change_numer = np.zeros(4, dtype=np.float32)
    patch_change_denom = np.zeros(4, dtype=np.float32)
    score_values = [[] for _ in range(4)]
    visibility_agentview = [[] for _ in range(4)]
    visibility_wrist = [[] for _ in range(4)]

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in step_records:
        key = (int(record.get("task_id", -1)), int(record.get("episode_idx", -1)))
        grouped.setdefault(key, []).append(record)

    for records in grouped.values():
        records = sorted(records, key=lambda item: int(item.get("step", 0)))
        prev_slots = None
        for record in records:
            predicted_patch_ids = np.asarray(record.get("predicted_patch_ids", []), dtype=np.int64)
            if predicted_patch_ids.ndim == 3:
                predicted_patch_ids = predicted_patch_ids[0]
            if predicted_patch_ids.ndim != 2:
                predicted_patch_ids = np.zeros((4, 3), dtype=np.int64)

            if prev_slots is not None:
                slot_changed = np.any(predicted_patch_ids != prev_slots, axis=-1)
                patch_change_numer[: slot_changed.shape[0]] += slot_changed.astype(np.float32)
                patch_change_denom[: slot_changed.shape[0]] += 1.0
            prev_slots = predicted_patch_ids

            score_logits = np.asarray(record.get("score_logits", []), dtype=np.float32).reshape(-1)
            if score_logits.size:
                for slot_idx, value in enumerate(sigmoid(score_logits[:4])):
                    score_values[slot_idx].append(float(value))

            visibility_logits = np.asarray(record.get("visibility_logits", []), dtype=np.float32)
            if visibility_logits.ndim == 3:
                visibility_logits = visibility_logits[0]
            if visibility_logits.ndim == 2:
                vis_probs = sigmoid(visibility_logits[:4, :2])
                for slot_idx in range(vis_probs.shape[0]):
                    visibility_agentview[slot_idx].append(float(vis_probs[slot_idx, 0]))
                    if vis_probs.shape[1] > 1:
                        visibility_wrist[slot_idx].append(float(vis_probs[slot_idx, 1]))

    patch_change_rate = np.divide(
        patch_change_numer,
        np.maximum(patch_change_denom, 1.0),
        out=np.zeros_like(patch_change_numer),
        where=patch_change_denom > 0,
    )
    return {
        "patch_change_rate": patch_change_rate,
        "score_prob_mean": np.asarray([np.mean(values) if values else 0.0 for values in score_values], dtype=np.float32),
        "visibility_agentview_mean": np.asarray(
            [np.mean(values) if values else 0.0 for values in visibility_agentview],
            dtype=np.float32,
        ),
        "visibility_wrist_mean": np.asarray(
            [np.mean(values) if values else 0.0 for values in visibility_wrist],
            dtype=np.float32,
        ),
    }


def first_episode_actions(mode: ModeDiagnostics) -> np.ndarray:
    if mode.summaries.empty:
        return np.zeros((0, 7), dtype=np.float32)
    first_row = mode.summaries.sort_values(["task_id", "episode_idx"]).iloc[0]
    actions_path = first_row.get("actions_npy_path", None)
    if not actions_path:
        return np.zeros((0, 7), dtype=np.float32)
    path = Path(actions_path)
    if not path.exists():
        return np.zeros((0, 7), dtype=np.float32)
    return np.load(path).astype(np.float32)


def select_overlay_records(step_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not step_records:
        return []

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in step_records:
        key = (int(record.get("task_id", -1)), int(record.get("episode_idx", -1)))
        grouped.setdefault(key, []).append(record)
    first_key = sorted(grouped.keys())[0]
    records = sorted(grouped[first_key], key=lambda item: int(item.get("step", 0)))
    if not records:
        return []
    indices = sorted({0, len(records) // 2, len(records) - 1})
    return [records[idx] for idx in indices]


def dump_overlay_frames(modes: list[ModeDiagnostics], *, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for record in mode.step_records:
            key = (int(record.get("task_id", -1)), int(record.get("episode_idx", -1)))
            grouped.setdefault(key, []).append(record)

        for (task_id, episode_idx), records in grouped.items():
            episode_dir = output_root / mode.name / f"task_{task_id:03d}" / f"episode_{episode_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            for record in sorted(records, key=lambda item: int(item.get("step", 0))):
                step = int(record.get("step", 0))
                source_episode_dir = _source_episode_dir_from_record(record)
                source_overlay_dir = source_episode_dir / "overlays" if source_episode_dir is not None else None
                if source_overlay_dir is not None:
                    source_overlay_dir.mkdir(parents=True, exist_ok=True)
                saved_images: list[Image.Image] = []
                saved_labels: list[str] = []
                step_label = f"{mode.name} | task {task_id} ep {episode_idx} step {step}"

                agentview_path = record.get("agentview_path", None)
                if agentview_path:
                    agent_overlay = overlay_prediction(
                        Path(agentview_path),
                        record,
                        view_idx=0,
                        header_text=f"step {step} | agentview",
                    )
                    agent_overlay.save(episode_dir / f"step_{step:04d}_agentview_overlay.jpg", quality=95)
                    if source_overlay_dir is not None:
                        agent_overlay.save(source_overlay_dir / f"step_{step:04d}_agentview_overlay.jpg", quality=95)
                    saved_images.append(agent_overlay)
                    saved_labels.append("agentview")

                wrist_path = record.get("wrist_path", None)
                if wrist_path:
                    wrist_overlay = overlay_prediction(
                        Path(wrist_path),
                        record,
                        view_idx=1,
                        header_text=f"step {step} | wrist",
                    )
                    wrist_overlay.save(episode_dir / f"step_{step:04d}_wrist_overlay.jpg", quality=95)
                    if source_overlay_dir is not None:
                        wrist_overlay.save(source_overlay_dir / f"step_{step:04d}_wrist_overlay.jpg", quality=95)
                    saved_images.append(wrist_overlay)
                    saved_labels.append("wrist")

                if saved_images:
                    pair = stitch_row(
                        saved_images,
                        title=step_label,
                        labels=saved_labels,
                    )
                    pair.save(episode_dir / f"step_{step:04d}_pair.jpg", quality=95)
                    if source_overlay_dir is not None:
                        pair.save(source_overlay_dir / f"step_{step:04d}_pair.jpg", quality=95)


def _source_episode_dir_from_record(record: dict[str, Any]) -> Path | None:
    for key in ("agentview_path", "wrist_path"):
        path_value = record.get(key, None)
        if not path_value:
            continue
        path = Path(path_value)
        if path.parent.name == "frames":
            return path.parent.parent
    return None


def build_overlay_pair(record: dict[str, Any], *, title: str) -> Image.Image:
    agentview_path = record.get("agentview_path", None)
    wrist_path = record.get("wrist_path", None)
    images = []
    step = int(record.get("step", 0))
    if agentview_path:
        images.append(
            (
                "agentview",
                overlay_prediction(Path(agentview_path), record, view_idx=0, header_text=f"step {step} | agentview"),
            )
        )
    if wrist_path:
        images.append(
            (
                "wrist",
                overlay_prediction(Path(wrist_path), record, view_idx=1, header_text=f"step {step} | wrist"),
            )
        )
    return stitch_row([image for _, image in images], title=title, labels=[label for label, _ in images])


def overlay_prediction(
    image_path: Path,
    record: dict[str, Any],
    *,
    view_idx: int,
    header_text: str | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()

    width, height = image.size
    bbox_by_view = np.asarray(record.get("bbox_by_view", []), dtype=np.float32)
    patch_mask_by_view = np.asarray(record.get("patch_mask_by_view", []), dtype=np.float32)
    predicted_patch_ids = np.asarray(record.get("predicted_patch_ids", []), dtype=np.int64)
    object_presence_mask = np.asarray(record.get("object_presence_mask", []), dtype=bool)
    score_logits = np.asarray(record.get("score_logits", []), dtype=np.float32).reshape(-1)
    visibility_logits = np.asarray(record.get("visibility_logits", []), dtype=np.float32)
    task_object_roles = list(record.get("task_object_roles", []))

    if bbox_by_view.ndim == 4:
        bbox_by_view = bbox_by_view[0]
    if patch_mask_by_view.ndim == 4:
        patch_mask_by_view = patch_mask_by_view[0]
    if predicted_patch_ids.ndim == 3:
        predicted_patch_ids = predicted_patch_ids[0]
    if object_presence_mask.ndim == 2:
        object_presence_mask = object_presence_mask[0]
    if visibility_logits.ndim == 3:
        visibility_logits = visibility_logits[0]
    if task_object_roles and isinstance(task_object_roles[0], list):
        task_object_roles = list(task_object_roles[0])

    score_probs = sigmoid(score_logits) if score_logits.size else np.zeros((0,), dtype=np.float32)
    visibility_probs = sigmoid(visibility_logits) if visibility_logits.size else np.zeros((0, 2), dtype=np.float32)

    num_slots = min(4, bbox_by_view.shape[0] if bbox_by_view.ndim >= 3 else 0)
    for slot_idx in range(num_slots):
        is_present = bool(object_presence_mask[slot_idx]) if slot_idx < object_presence_mask.shape[0] else False
        if not is_present:
            continue

        color = PALETTE[slot_idx % len(PALETTE)]
        score_prob = float(score_probs[slot_idx]) if slot_idx < len(score_probs) else 0.0
        view_prob = float(visibility_probs[slot_idx, view_idx]) if visibility_probs.ndim == 2 and slot_idx < visibility_probs.shape[0] and view_idx < visibility_probs.shape[1] else 0.0
        bbox = bbox_by_view[slot_idx, view_idx]
        patch_mask = patch_mask_by_view[slot_idx, view_idx] if patch_mask_by_view.ndim == 3 else np.zeros((0,), dtype=np.float32)
        role = task_object_roles[slot_idx] if slot_idx < len(task_object_roles) else f"slot_{slot_idx + 1}"
        if predicted_patch_ids.ndim == 3 and slot_idx < predicted_patch_ids.shape[0] and view_idx < predicted_patch_ids.shape[1]:
            slot_patch_ids = predicted_patch_ids[slot_idx, view_idx]
        elif predicted_patch_ids.ndim == 2 and slot_idx < predicted_patch_ids.shape[0]:
            slot_patch_ids = predicted_patch_ids[slot_idx]
        else:
            slot_patch_ids = np.asarray([], dtype=np.int64)
        slot_patch_ids = [int(idx) for idx in np.asarray(slot_patch_ids).reshape(-1).tolist() if int(idx) >= 0]
        view_patch_ids = slot_patch_ids

        if patch_mask.size:
            grid_size = int(round(math.sqrt(patch_mask.size)))
            if grid_size * grid_size == patch_mask.size:
                patch_grid = patch_mask.reshape(grid_size, grid_size)
                flat_probs = patch_grid.reshape(-1)
                selected_patch_ids = set(view_patch_ids)
                top_k = min(6, flat_probs.size)
                selected_patch_ids.update(int(idx) for idx in np.argsort(flat_probs)[-top_k:] if flat_probs[int(idx)] >= 0.45)

                for patch_id in sorted(selected_patch_ids):
                    if not (0 <= patch_id < flat_probs.size):
                        continue
                    row_idx = patch_id // grid_size
                    col_idx = patch_id % grid_size
                    prob = float(flat_probs[patch_id])
                    x0 = int(col_idx * width / grid_size)
                    y0 = int(row_idx * height / grid_size)
                    x1 = int((col_idx + 1) * width / grid_size)
                    y1 = int((row_idx + 1) * height / grid_size)

                    alpha = int(min(110, 30 + 110 * prob))
                    draw.rectangle([x0, y0, x1, y1], fill=(*color, alpha))

                    outline_width = 4 if patch_id in view_patch_ids else 2
                    outline_color = (*color, 255) if patch_id in view_patch_ids else (*color, 180)
                    draw.rectangle([x0, y0, x1, y1], outline=outline_color, width=outline_width)

                    if patch_id in view_patch_ids:
                        patch_label = f"p{patch_id}"
                        patch_text_bbox = draw.textbbox((x0 + 2, y0 + 2), patch_label, font=font)
                        draw.rectangle(patch_text_bbox, fill=(0, 0, 0, 180))
                        draw.text((x0 + 2, y0 + 2), patch_label, fill=(255, 255, 255, 255), font=font)

        if bbox.shape[-1] == 4:
            x0, y0, x1, y1 = bbox.tolist()
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
                x0 *= width
                x1 *= width
                y0 *= height
                y1 *= height
            if not np.all(np.isfinite([x0, y0, x1, y1])):
                continue
            x0, x1 = sorted((float(x0), float(x1)))
            y0, y1 = sorted((float(y0), float(y1)))
            x0 = max(0.0, min(float(width - 1), x0))
            x1 = max(0.0, min(float(width - 1), x1))
            y0 = max(0.0, min(float(height - 1), y0))
            y1 = max(0.0, min(float(height - 1), y1))
            if x1 > x0 and y1 > y0:
                draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=3)
                patch_text = ",".join(str(idx) for idx in view_patch_ids[:3]) if view_patch_ids else "-"
                label_suffix = "core" if view_idx == 0 else "mask"
                label = f"{role} s={score_prob:.2f} v={view_prob:.2f} {label_suffix}={patch_text}"
                text_bbox = draw.textbbox((int(x0) + 4, int(y0) + 4), label, font=font)
                draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
                draw.text((int(x0) + 4, int(y0) + 4), label, fill=(*color, 255), font=font)

    composed = Image.alpha_composite(image, overlay).convert("RGB")
    if header_text:
        composed = _draw_image_header(composed, header_text)
    return composed


def _draw_image_header(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((8, 8), text, font=font)
    padded_bbox = (
        text_bbox[0] - 4,
        text_bbox[1] - 2,
        text_bbox[2] + 4,
        text_bbox[3] + 2,
    )
    draw.rectangle(padded_bbox, fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return image


def stitch_row(images: list[Image.Image], *, title: str | None = None, labels: list[str] | None = None) -> Image.Image:
    if not images:
        return Image.new("RGB", (320, 80), (24, 24, 24))
    labels = labels or [""] * len(images)
    gap = 8
    title_height = 24 if title else 0
    label_height = 18
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images) + title_height + label_height
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if title:
        draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    x = 0
    for label, image in zip(labels, images):
        canvas.paste(image, (x, title_height))
        draw.text((x + 6, title_height + 4), label, fill=(255, 255, 255), font=font)
        x += image.width + gap
    return canvas


def stitch_column(images: list[Image.Image]) -> Image.Image:
    if not images:
        return Image.new("RGB", (320, 80), (24, 24, 24))
    gap = 12
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    return canvas


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-value))


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
