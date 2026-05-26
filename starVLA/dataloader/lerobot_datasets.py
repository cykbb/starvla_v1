# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].
# Modified by [OpenAI] in [2026].
# Modification: [optional PaDT annotation wrapper and metadata passthrough for QwenPaDTPI].

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag


def collate_fn(batch):
    return batch


class PaDTAnnotationWrapper(Dataset):
    """Attach PaDT supervision sidecar annotations without rewriting the core loader.

    Expected key format in JSON / JSONL sidecar:
        "{dataset_name}/{trajectory_id}/{step_index}"

    Minimal sidecar payload per sample should include:
        objects, task_objects, object_role, bbox_by_view, patch_mask_by_view,
        valid_patch_ids, core_patch_ids, visible_by_view
    """

    def __init__(self, base_dataset: Dataset, annotation_path: str | None = None, required: bool = False):
        self.base_dataset = base_dataset
        self.annotation_path = annotation_path
        self.required = required
        self.annotation_db = self._load_annotation_db(annotation_path) if annotation_path else {}

    def _load_annotation_db(self, annotation_path: str) -> Dict[str, Dict[str, Any]]:
        path = Path(annotation_path)
        if not path.exists():
            if self.required:
                raise FileNotFoundError(f"PaDT annotation file not found: {path}")
            return {}
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            raise ValueError(f"Expected dict JSON annotation file at {path}")
        if path.suffix == ".jsonl":
            db: Dict[str, Dict[str, Any]] = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    key = record.get("key")
                    if key is None:
                        dataset_name = record.get("dataset_name")
                        trajectory_id = record.get("trajectory_id")
                        step_index = record.get("step_index")
                        key = f"{dataset_name}/{trajectory_id}/{step_index}"
                    db[str(key)] = record
            return db
        raise ValueError(f"Unsupported PaDT annotation sidecar format: {path}")

    def _make_key(self, meta: Dict[str, Any]) -> str:
        return f"{meta['dataset_name']}/{meta['trajectory_id']}/{meta['step_index']}"

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.base_dataset[index]
        meta = sample.get("__padt_meta__", None)
        if meta is None:
            if self.required:
                raise KeyError("PaDTAnnotationWrapper expected `__padt_meta__` in base sample")
            return sample
        annotation = self.annotation_db.get(self._make_key(meta), None)
        if annotation is None:
            if self.required:
                raise KeyError(
                    f"Missing PaDT sidecar annotation for {self._make_key(meta)} in {self.annotation_path}"
                )
            return sample
        merged = dict(sample)
        merged.update(annotation)
        return merged

    def __getattr__(self, name: str):
        return getattr(self.base_dataset, name)


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    lerobot_version: str | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param lerobot_version: Explicit lerobot version override ("v2.0" or "v3.0"). If None, auto-detected from dataset file structure.
    :return: A LeRobotSingleDataset object.
    """

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        lerobot_version=lerobot_version,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> Dataset:
    """Get a LeRobot mixture dataset.

    Two PaDT data-source modes are supported:
    1) legacy sidecar-per-step annotations via `padt_annotation_path`
    2) compact task-level metadata + step-level segmentation parquet fields via
       `padt_use_segmentation_source=true` (preferred for YK-Bai/new_data).
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for entry in mixture_spec:
        # Support both 3-tuple (name, weight, robot_type) and 4-tuple (name, weight, robot_type, lerobot_version)
        d_name, d_weight, robot_type = entry[0], entry[1], entry[2]
        d_version = entry[3] if len(entry) > 3 else None
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type, d_version))

    dataset_mixture = []
    for d_name, d_weight, robot_type, d_version in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    Path(data_root_dir),
                    d_name,
                    robot_type,
                    delete_pause_frame=delete_pause_frame,
                    data_cfg=data_cfg,
                    lerobot_version=d_version,
                ),
                d_weight,
            )
        )

    dataset: Dataset = LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )

    use_segmentation_source = bool(data_cfg.get("padt_use_segmentation_source", False))
    annotation_path = data_cfg.get("padt_annotation_path", None)
    if annotation_path not in [None, "", "null"] and not use_segmentation_source:
        dataset = PaDTAnnotationWrapper(
            dataset,
            annotation_path=str(annotation_path),
            required=bool(data_cfg.get("padt_annotation_required", False)),
        )
    return dataset


if __name__ == "__main__":

    # import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_behavior.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()
    args.config_yaml = "./examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # cfg.datasets.vla_data.data_mix = "robotwin"
    vla_dataset_cfg = cfg.datasets.vla_data
    # cfg.datasets.vla_data.include_state = True
    vla_dataset_cfg.task_id = 1
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass
