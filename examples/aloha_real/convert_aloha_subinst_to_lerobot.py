"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: 

uv run examples/aloha_real/convert_aloha_subinst_to_lerobot.py --raw-dir /share/project/lvhuaihai/robot_data/agilex/hirobot/humber_5.7 --repo-id HuaihaiLyu/hamburger0513_4cam

made for hirobot demo.

uv runpport dir format:

raw_dir/
├── task_1/
│   ├── inst.txt
│   ├── episode_0/
│   │   ├── episode_0.hdf5
│   │   └── ...
│   └── episode_1/
│       └── episode_1.hdf5
├── task_2/
│   ├── inst.txt
│   └── episode_0/
│       └── episode_0.hdf5
...


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

cameras = [
        "cam_high",
        "cam_high_realsense",
        "cam_left_wrist",
        "cam_right_wrist",
        ]

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
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
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
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam

def qpos_2_joint_positions(qpos:np.ndarray):

        l_joint_pos = qpos[:, 50:56]
        r_joint_pos = qpos[:, 0:6]
        l_gripper_pos = np.array([qpos[:,60]]).reshape(-1,1)
        r_gripper_pos = np.array([qpos[:,10]]).reshape(-1,1)

        # import pdb
        # pdb.set_trace()
        l_pos = np.concatenate((l_joint_pos,l_gripper_pos), axis=1)
        r_pos = np.concatenate((r_joint_pos,r_gripper_pos), axis=1)

        return np.concatenate((r_pos,l_pos), axis=1)
    
def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(qpos_2_joint_positions(ep["/observations/qpos"][:]))
        action = torch.from_numpy(qpos_2_joint_positions(ep["/action"][:]))

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            cameras,
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
    dataset: LeRobotDataset | None = None,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    # import pdb
    # pdb.set_trace()

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    # 读取所有语言指令
    inst_file = raw_dir / "inst.txt"
    with open(inst_file, "r") as f:
        all_tasks = [line.strip() for line in f.readlines()]
    
    
    for ep_folder in sorted(raw_dir.glob("*")):
        if ep_folder.is_dir():
            hdf5_files = []
            hdf5_files.extend(sorted(ep_folder.glob("episode_*.hdf5")))

            if len(hdf5_files) != len(all_tasks):
                raise ValueError(f"Mismatch: {len(hdf5_files)} hdf5 files vs {len(all_tasks)} tasks in {raw_dir}")
            
            if not hdf5_files:
                print(f"No hdf5 files found in {ep_folder}")
                continue  # 如果这个子目录没有hdf5文件就跳过
            
        # 每个 hdf5 单独处理并配对任务
        for i, ep_path in enumerate(tqdm.tqdm(hdf5_files)):
            imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
            num_frames = state.shape[0]

            for j in range(num_frames):
                frame = {
                    "observation.state": state[j],
                    "action": action[j],
                }
                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[j]
                if velocity is not None:
                    frame["observation.velocity"] = velocity[j]
                if effort is not None:
                    frame["observation.effort"] = effort[j]

                dataset.add_frame(frame)
            print(f"the {i}-th file: {ep_path}\n")
            dataset.save_episode(task=all_tasks[i])
        
    # dataset.consolidate()

    # if push_to_hub:
        # dataset.push_to_hub(repo_id)


def convert_all_tasks(
    raw_dir: Path,
    repo_id: str,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    all_task_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and (d / "inst.txt").exists()])

    print(f"Found {len(all_task_dirs)} tasks.")
    
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    
    for task_dir in tqdm.tqdm(all_task_dirs, desc="Processing tasks"):
        task_name = task_dir.name  # Use directory name as task name
        print(f"Processing task: {task_name}")
        port_aloha(
            raw_dir=task_dir,
            repo_id=repo_id,
            # task_prefix=task_name,
            push_to_hub=False,  # consolidate after all tasks
            is_mobile=is_mobile,
            mode=mode,
            dataset_config=dataset_config,
            dataset = dataset,
        )
    
    # Final consolidation & optional push
    dataset.consolidate()
    if push_to_hub:
        dataset.push_to_hub(repo_id)

if __name__ == "__main__":
    tyro.cli(convert_all_tasks)
    
# python /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/convert_aloha_data_to_lerobot.py \
#     --raw_dir="/mnt/hpfs/baaiei/lvhuaihai/agilex_data/test/task_put_black_brown_basket_4.1" \
#     --repo_id="/mnt/hpfs/baaiei/lvhuaihai/agilex_data/test/save" \
#     --task="DEBUG" \
#     --mode="video"

# python /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/convert_aloha_data_to_lerobot.py --raw_dir="/mnt/hpfs/baaiei/lvhuaihai/agilex_data/test/task_put_black_brown_basket_4.1"  --repo_id="/mnt/hpfs/baaiei/lvhuaihai/agilex_data/test/save" --task="DEBUG"  --mode="video" 