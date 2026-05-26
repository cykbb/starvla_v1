#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from starVLA.dataloader.padt_segmentation_adapter import build_task_meta_scaffold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact PaDT task-level scaffold from LeRobot meta/tasks.parquet"
    )
    parser.add_argument("dataset_root", type=Path, help="Path to one dataset root, e.g. libero_10_no_loops_lerobot")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to <dataset_root>/meta/padt_task_specs.jsonl",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    tasks_parquet = dataset_root / "meta" / "tasks.parquet"
    if args.output is None:
        output_jsonl = dataset_root / "meta" / "padt_task_specs.jsonl"
    else:
        output_jsonl = args.output

    build_task_meta_scaffold(tasks_parquet=tasks_parquet, output_jsonl=output_jsonl)
    print(f"[OK] wrote scaffold to {output_jsonl}")
    print("Fill task_objects / object_role / objects manually so labels match segmentation JSON labels exactly.")


if __name__ == "__main__":
    main()
