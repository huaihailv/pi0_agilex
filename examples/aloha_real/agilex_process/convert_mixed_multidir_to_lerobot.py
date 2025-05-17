"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: 
uv run examples/aloha_real/convert_mixed_data_to_lerobot.py --raw-dir /mnt/hpfs/baaiei/robot_data/agilex/stack_basket/task_put_brown_black_basket_4.1 --repo-id HuaihaiLyu/stack_basket  --mode="video"  --task="stack the brown basket on the black basket"
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /mnt/hpfs/baaiei/robot_data/agilex/groceries_dual/task_take_brown_long_bread_Egg_yolk_pasry_4.3 --repo-id HuaihaiLyu/groceries  --mode="video"  --task="Pick the brown long bread and Egg yolk pasry into package"
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /share/project/lvhuaihai/robot_data/agilex/robohetero/task_Basket_banana_500_5.13/task_black_left_Basket_banana_125_5.12  --repo_id=HuaihaiLyu/Basket_banana --task="realman test"  --mode="video" 
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import pdb
import pytransform3d.rotations as rotations

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_joint_position1",
        "left_joint_position2",
        "left_joint_position3",
        "left_joint_position4",
        "left_joint_position5",
        "left_joint_position6",
        "left_gripper",
        "right_joint_position1",
        "right_joint_position2",
        "right_joint_position3",
        "right_joint_position4",
        "right_joint_position5",
        "right_joint_position6",
        "right_gripper",
        "agilex_left_x",
        "agilex_left_y",
        "agilex_left_z",
        "agilex_left_delta_x",
        "agilex_left_delta_y",
        "agilex_left_delta_z",
        "agilex_right_x",
        "agilex_right_y",
        "agilex_right_z",
        "agilex_right_delta_x",
        "agilex_right_delta_y",
        "agilex_right_delta_z",
        "vp_left_x",
        "vp_left_y",
        "vp_left_z",
        "vp_left_delta_x",
        "vp_left_delta_y",
        "vp_left_delta_z",
        "vp_right_x",
        "vp_right_y",
        "vp_right_z",
        "vp_right_delta_x",
        "vp_right_delta_y",
        "vp_right_delta_z"
    ]
    cameras = [
        "cam_high",
        # "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": {"motors":motors},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": {"motors":motors},
        },
        "task_index": {"dtypes": "int32", "shape": (1), "names": {"task_index": "task_index"}},
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4
        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2
            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam

def qpos_2_joint_positions(qpos:np.ndarray):

        l_joint_pos = qpos[:, 50:56]
        r_joint_pos = qpos[:, 0:6]
        l_gripper_pos = np.array([qpos[:,60]]).reshape(-1,1)
        r_gripper_pos = np.array([qpos[:,10]]).reshape(-1,1)

        l_pos = np.concatenate((l_joint_pos,l_gripper_pos), axis=1)
        r_pos = np.concatenate((r_joint_pos,r_gripper_pos), axis=1)

        return np.concatenate((l_pos,r_pos), axis=1)
    
def pose6D2quat(pose:np.ndarray):
        
        column_1 = pose[:,:3]
        column_2 = pose[:,3:]

        R = np.column_stack((column_1, column_2, np.cross(column_1, column_2)))

        # quat = rotations.quaternion_from_matrix(R)
        # 初始化欧拉角数组
        euler_angles = np.zeros((pose.shape[0], 3))

        # 循环处理每个旋转矩阵
        for i in range(pose.shape[0]):
            euler_angles[i] = rotations.euler_from_matrix(R[i].reshape(3,3), 0, 1, 2, extrinsic=True)
        
        return euler_angles
    
def qpos_2_ee_pose(qpos:np.ndarray):

        # r_joint_pos = qpos[0:10]
        # l_joint_pos = qpos[50:60]

        # l_gripper_joint_pos = qpos[60:65]
        # r_gripper_joint_pos = qpos[25:30]

        # l_pose6d = qpos[83:89]
        # r_pose6d = qpos[33:39]
        # l_quat = pose6D2quat(l_pose6d)
        # r_quat = pose6D2quat(r_pose6d)
        
        l_pose6d = qpos[:,83:89]
        r_pose6d = qpos[:,33:39]
        l_quat = pose6D2quat(l_pose6d)
        r_quat = pose6D2quat(r_pose6d)
        l_ee_trans = qpos[:,80:83]
        r_ee_trans = qpos[:,30:33]
        l_gripper_pos = np.array([qpos[:,60]]).reshape(-1,1)
        r_gripper_pos = np.array([qpos[:,10]]).reshape(-1,1)

        return np.concatenate((l_ee_trans, l_quat, r_ee_trans, r_quat), axis=1)
    

def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    
    with h5py.File(ep_path, "r") as ep:

        print(ep_path)
        state = torch.from_numpy(np.concatenate((qpos_2_joint_positions(ep["/observations/qpos"][:]), qpos_2_ee_pose(ep["/observations/qpos"][:]), np.zeros_like(qpos_2_ee_pose(ep["/observations/qpos"][:]))), axis=1))
        action = torch.from_numpy(np.concatenate((qpos_2_joint_positions(ep["/observations/qpos"][:]), qpos_2_ee_pose(ep["/observations/qpos"][:]), np.zeros_like(qpos_2_ee_pose(ep["/observations/qpos"][:]))), axis=1))
        
        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    return dataset


def port_aloha(
    raw_dir: Path, # hdf5 path
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)
        # 删除已存在目录

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub(repo_id)

import random
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
            print(instructions)
            cleaned_list = []
            for s in instructions:
                s = s.replace('\u200b', '')
                if s and s[0].isdigit():  # 判断第一个字符是不是数字
                    dot_pos = s.find('.')  # 找到第一个点的位置
                    if 0 <= dot_pos <= 3:  # 点的位置靠前（最多3位数字，比如1., 10., 100.）
                        cleaned_list.append(s[dot_pos + 1:].lstrip())  # 去掉前缀，顺便清空多余空格
                    else:
                        cleaned_list.append(s)
                else:
                    cleaned_list.append(s)
            instructions = cleaned_list
            print(instructions)
            
        if not instructions:
            continue
        # selected_instruction = random.choice(instructions)
        selected_instruction = instructions[0]
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
    
# uv run /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/agilex_process/convert_mixed_multidir_to_lerobot.py --raw_dirs /mnt/hpfs/baaiei/robot_data/agilex/robohetero/build_blocks --repo_id=agilex/build_blocks --mode="video" 

# now ready tasks build_blocks groceries_bag organize_pants pour_bowl pour_tea stack_basket

# save_data_root = /mnt/hpfs/baaiei/qianpusun/data/lerobot_data/agilex