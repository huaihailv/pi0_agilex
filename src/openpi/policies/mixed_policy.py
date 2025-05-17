import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

def make_mixed_example() -> dict:
    """Creates a random input example for the Mixed policy."""
    return {
        "state": np.ones((26,)),  # 14 joint states + 12 EEF states
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

@dataclasses.dataclass(frozen=True)
class MixedInputs(transforms.DataTransformFn):
    """Inputs for the Mixed policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [26]
    - actions: [action_horizon, 26]
    """

    action_dim: int
    adapt_to_pi: bool = True
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_mixed(data, adapt_to_pi=self.adapt_to_pi)

        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class MixedOutputs(transforms.DataTransformFn):
    """Outputs for the Mixed policy."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :26])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}

def _joint_flip_mask() -> np.ndarray:
    """Used to convert between mixed and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1] + [1]*12)

def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def _gripper_to_angular(value):
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return _normalize(value, min_val=0.4, max_val=1.5)

def _gripper_from_angular(value):
    value = _unnormalize(value, min_val=0.4, max_val=1.5)
    return _normalize(value, min_val=-0.6213, max_val=1.4910)

def _gripper_from_angular_inv(value):
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)

def _decode_mixed(data: dict, *, adapt_to_pi: bool = False) -> dict:
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data

def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        state = _joint_flip_mask() * state
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state

def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions

def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
