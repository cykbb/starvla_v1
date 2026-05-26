from collections import deque
from typing import Any, Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import numpy as np
from pathlib import Path
from PIL import Image

from starVLA.model.tools import read_mode_config


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = True,
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: Optional[Sequence[int]] = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
        state_history_len: int = 16,
        state_history_includes_current: bool = False,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.model_config, norm_stats = read_mode_config(Path(policy_ckpt_path))
        self.image_size = self.get_image_size(self.model_config, requested_image_size=image_size)

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, image_size: {self.image_size} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.state_history_len = state_history_len
        self.state_history: deque = deque(maxlen=state_history_len)
        self.state_history_includes_current = state_history_includes_current
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats_from_loaded_config(
            self.unnorm_key, norm_stats=norm_stats
        )
        self.action_chunk_size = self.get_action_chunk_size_from_loaded_config(self.model_config)
        self.raw_actions: Optional[np.ndarray] = None
        self._cached_response_debug: Optional[dict[str, Any]] = None
        self._cached_normalized_actions: Optional[np.ndarray] = None
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.state_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None
        self._cached_response_debug = None
        self._cached_normalized_actions = None


    def step(
        self, 
        example: dict,
        step: int = 0,
        return_debug: bool = False,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        task_description = example.get("lang", None)
        images = example["image"]  # list of images for history

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        example["image"] = images

        # Match the training loader by default: state history excludes the current state.
        current_state = example.pop("state", None)
        if current_state is not None:
            current_state = np.asarray(current_state, dtype=np.float32)
            history = list(self.state_history)
            if self.state_history_includes_current:
                history.append(current_state)
            elif not history:
                history = [current_state]
            while len(history) < self.state_history_len:
                history.insert(0, history[0])
            example["state"] = np.stack(history[-self.state_history_len :], axis=0)
            self.state_history.append(current_state)

        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "return_debug": bool(return_debug),
        }
        

        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            response = self.client.predict_action(vla_input)
            response_data = response.get("data")
            if not isinstance(response_data, dict) or "normalized_actions" not in response_data:
                print(f"Response data: {response}")
                error_message = None
                if isinstance(response.get("error"), dict):
                    error_message = response["error"].get("message")
                if error_message:
                    raise RuntimeError(f"Policy server returned an error: {error_message}")
                raise KeyError(
                    f"Key 'normalized_actions' not found in response data. Top-level response keys: {list(response.keys())}"
                )

            normalized_actions = np.asarray(response_data["normalized_actions"]) # B, chunk, D
            self._cached_normalized_actions = normalized_actions
            response_debug = response_data.get("debug", None)
            self._cached_response_debug = response_debug if isinstance(response_debug, dict) else None
            normalized_actions = normalized_actions[0]    
            self.raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        
        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        result = {"raw_action": raw_action}
        if return_debug:
            result["debug"] = self._build_step_debug(
                step=step,
                chunk_offset=step % action_chunk_size,
                raw_action=raw_action,
            )
        return result

    def _build_step_debug(
        self,
        *,
        step: int,
        chunk_offset: int,
        raw_action: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        debug: dict[str, Any] = {
            "step": int(step),
            "chunk_step_offset": int(chunk_offset),
            "action_chunk_size": int(self.action_chunk_size),
            "chunk_reused": bool(chunk_offset != 0),
            "current_raw_action": {
                key: np.asarray(value, dtype=np.float32)
                for key, value in raw_action.items()
            },
        }

        if self._cached_normalized_actions is not None and self._cached_normalized_actions.size > 0:
            normalized_actions = np.asarray(self._cached_normalized_actions)
            debug["normalized_actions"] = normalized_actions[0]
            debug["current_normalized_action"] = normalized_actions[0, chunk_offset]

        if self._cached_response_debug:
            for key, value in self._cached_response_debug.items():
                debug[key] = value

        return debug

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        return ModelClient.get_action_stats_from_loaded_config(unnorm_key, norm_stats)

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        return ModelClient.get_action_chunk_size_from_loaded_config(model_config)

    @staticmethod
    def get_action_stats_from_loaded_config(unnorm_key: str, norm_stats: dict) -> dict:
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size_from_loaded_config(model_config: dict) -> int:
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    @staticmethod
    def _parse_image_size(value: Optional[Sequence[int]]) -> Optional[list[int]]:
        if value is None:
            return None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) >= 2:
                return [int(value[-2]), int(value[-1])]
            if len(value) == 1:
                edge = int(value[0])
                return [edge, edge]
            return None
        edge = int(value)
        return [edge, edge]

    @staticmethod
    def get_image_size(model_config: dict, requested_image_size: Optional[Sequence[int]] = None) -> list[int]:
        explicit_image_size = ModelClient._parse_image_size(requested_image_size)
        if explicit_image_size is not None:
            return explicit_image_size

        cfg_image_size = ModelClient._parse_image_size(
            model_config.get("datasets", {}).get("vla_data", {}).get("image_size", None)
        )
        if cfg_image_size is not None:
            return cfg_image_size

        # Old checkpoints may only contain `default_image_resolution`, which historically did
        # not guarantee the actual train-time resize used by the loader. Fall back to 224 here
        # instead of trusting that field and silently reintroducing a train/eval mismatch.
        return [224, 224]


    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
