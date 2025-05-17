"""
support dir format:

raw_dir/
├── task_1/
│   ├── inst.txt
│   │── episode_0.hdf5
│   │── episode_1.hdf5
│   └── ···
├── task_2/
│   ├── inst.txt
│   │── episode_0.hdf5
│   │── episode_1.hdf5
│   └── ···
...
"""

import os
import random
from pathlib import Path
import h5py
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LEROBOT_HOME

def qpos_2_joint_positions(qpos: np.ndarray):
    l_joint_pos = qpos[:, 50:56]
    r_joint_pos = qpos[:, 0:6]
    l_gripper_pos = np.array([qpos[:, 60]]).reshape(-1, 1)
    r_gripper_pos = np.array([qpos[:, 10]]).reshape(-1, 1)
    l_pos = np.concatenate((l_joint_pos, l_gripper_pos), axis=1)
    r_pos = np.concatenate((r_joint_pos, r_gripper_pos), axis=1)
    return np.concatenate((l_pos, r_pos), axis=1)

def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4
        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam

def load_raw_episode_data(ep_path: Path):
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(qpos_2_joint_positions(ep["/observations/qpos"][:]))
        action = torch.from_numpy(qpos_2_joint_positions(ep["/action"][:]))
        velocity = torch.from_numpy(ep["/observations/qvel"][:]) if "/observations/qvel" in ep else None
        effort = torch.from_numpy(ep["/observations/effort"][:]) if "/observations/effort" in ep else None
        imgs_per_cam = load_raw_images_per_camera(ep, ["cam_high", "cam_left_wrist", "cam_right_wrist"])
    return imgs_per_cam, state, action, velocity, effort

def create_empty_dataset(repo_id: str, robot_type: str, mode: str = "video") -> LeRobotDataset:
    motors = [
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
        "right_wrist_angle", "right_wrist_rotate", "right_gripper",
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    ]
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": {"motors": motors}},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": {"motors": motors}},
        "task_index": {"dtypes": "int32", "shape": (1), "names": {"task_index": "task_index"}},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    dataset_path = LEROBOT_HOME / repo_id
    if dataset_path.exists():
        import shutil
        shutil.rmtree(dataset_path)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=(mode == "video"),
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
        video_backend=None,
    )

def process_multiple_tasks(raw_dirs, repo_id: str, robot_type: str = "aloha", mode: str = "video"):
    dataset = create_empty_dataset(repo_id=repo_id, robot_type=robot_type, mode=mode)
    for task_index, task_dir_str in enumerate(raw_dirs):
        task_dir = Path(task_dir_str)
        if not task_dir.is_dir():
            continue
        inst_file = task_dir / "inst.txt"
        if not inst_file.exists():
            continue
        with open(inst_file, "r") as f:
            instructions = [line.strip() for line in f if line.strip()]
        if not instructions:
            continue
        selected_instruction = random.choice(instructions)
        for ep_file in task_dir.glob("episode_*.hdf5"):
            imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_file)
            num_frames = state.shape[0]
            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task_index": task_index,
                }
                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]
                if velocity is not None:
                    frame["observation.velocity"] = velocity[i]
                if effort is not None:
                    frame["observation.effort"] = effort[i]
                dataset.add_frame(frame)
            dataset.save_episode(task=selected_instruction)
    dataset.consolidate()
    dataset.push_to_hub(repo_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process multiple tasks and convert to LeRobot dataset.")
    parser.add_argument("--raw_dirs", type=str, required=True, help="List of task directories to be processed.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID for the LeRobot dataset.")
    parser.add_argument("--robot_type", type=str, default="aloha", help="Type of robot (default: aloha).")
    parser.add_argument("--mode", type=str, choices=["video", "image"], default="video", help="Data mode (video or image).")
    args = parser.parse_args()
    raw_dirs = [d for d in Path(args.raw_dirs).iterdir() if d.is_dir()]
    
    process_multiple_tasks(raw_dirs, args.repo_id, args.robot_type, args.mode)
    
# uv run /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/convert_aloha_multisubtask_to_lerobot.py --raw_dirs /mnt/hpfs/baaiei/robot_data/agilex/robohetero/groceries_bag --repo_id HuaihaiLyu/groceries_bag --robot_type aloha --mode video
