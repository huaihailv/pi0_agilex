"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: 
uv run examples/aloha_real/vp_process/convert_mixed_multidir_to_lerobot.py --raw_dirs '/mnt/hpfs/baaiei/robot_data/vp/pour_tea/' --repo_id=HuaihaiLyu/vp_pour_tea --mode="video"
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
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
        "vp_right_delta_z",
    ]
    cameras = [
        "cam_high",
        # "cam_low",
        # "cam_left_wrist",
        # "cam_right_wrist",
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
    
import cv2

def load_video_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.transpose(2, 0, 1))  # CHW
    cap.release()
    return np.stack(frames)

def load_raw_images_from_videos(ep_folder: Path) -> dict[str, np.ndarray]:
    return {
        "cam_high": load_video_frames(ep_folder / "camera_c.mp4"),
    }
    
from scipy.spatial.transform import Rotation as R    
def load_raw_episode_data(
    ep_folder: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, None, None]:
    hdf5_path = ep_folder
    with h5py.File(hdf5_path, "r") as ep:
        print(ep_folder)
        head_pose_array = ep['head_pose']
        head_pose_array = np.array(head_pose_array)
        non_zero_indices = np.nonzero(head_pose_array.any(axis=1))[0]
        actual_length = non_zero_indices[-1] + 1
        
        # 裁剪有效长度
        head_pose_array = head_pose_array[:actual_length]
        left_pose_array = np.array(ep['left_pose'])[:actual_length]
        right_pose_array = np.array(ep['right_pose'])[:actual_length]
        img_front = np.array(ep['img_front'])[:actual_length]

        def convert_pose_to_xyz_euler(pose_array):
            xyz = pose_array[:, :3]
            quat = pose_array[:, 3:]  # qw, qx, qy, qz 或其他顺序？
            # 确保顺序是 [qx, qy, qz, qw]，如果原始是 [x, y, z, qw, qx, qy, qz] 需调整
            quat_xyzw = quat[:, [1, 2, 3, 0]]  # -> qx, qy, qz, qw
            euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=False)  # (T, 3)
            return xyz, euler
        
        l_xyz, l_euler = convert_pose_to_xyz_euler(left_pose_array)
        r_xyz, r_euler = convert_pose_to_xyz_euler(right_pose_array)

        # 拼接：每一帧为 26 零 + 12 pose（xyz + euler 左右手）
        zeros_26 = np.zeros((actual_length, 26), dtype=np.float32)
        pose_12 = np.concatenate([l_xyz, l_euler, r_xyz, r_euler], axis=1)  # (T, 12)
        full_vec = np.concatenate([zeros_26, pose_12], axis=1)  # (T, 38)

        state = torch.from_numpy(full_vec)
        action = torch.from_numpy(full_vec)

        velocity = None
        effort = None

        # img_front = img_front[..., ::-1]  # BGR to RGB
        img_front = img_front.transpose(0, 3, 1, 2)  # TCHW
        imgs_per_cam = {"cam_high": img_front,}
        
        # 这里没有 velocity 和 effort
        velocity = None
        effort = None

    return imgs_per_cam, state, action, velocity, effort

def process_multiple_tasks(raw_dirs, repo_id: str, robot_type: str = "aloha", mode: str = "video"):
    dataset = create_empty_dataset(repo_id=repo_id, robot_type=robot_type, mode=mode)
    for task_index, task_dir_str in enumerate(raw_dirs):
        task_dir = Path(task_dir_str)
        if not task_dir.is_dir():
            print(f"{task_dir} is not a directory")
            continue
        print(task_dir)
        inst_file = task_dir / "inst.txt"
        if not inst_file.exists():
            print(f"inst.txt not found in {task_dir}")
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
            print(f"no instructions found in {task_dir}")
            continue
        # selected_instruction = random.choice(instructions)
        selected_instruction = instructions[0]
        for ep_file in task_dir.glob("data*/*.h5"):
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
import os
def get_deepest_subdirectories(path):
    deepest_dirs = []
    max_depth = -1

    def find_deepest_dirs(current_path, current_depth):
        nonlocal deepest_dirs, max_depth
        subdirs = [os.path.join(current_path, d) for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]
        
        if not subdirs:
            if current_depth > max_depth:
                deepest_dirs = [current_path]
                max_depth = current_depth
            elif current_depth == max_depth:
                deepest_dirs.append(current_path)
        else:
            for subdir in subdirs:
                find_deepest_dirs(subdir, current_depth + 1)

    find_deepest_dirs(path, 0)
    return deepest_dirs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process multiple tasks and convert to LeRobot dataset.")
    parser.add_argument("--raw_dirs", type=str, required=True, help="List of task directories to be processed.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID for the LeRobot dataset.")
    parser.add_argument("--robot_type", type=str, default="aloha", help="Type of robot (default: aloha).")
    parser.add_argument("--mode", type=str, choices=["video", "image"], default="video", help="Data mode (video or image).")
    args = parser.parse_args()
    deepest = get_deepest_subdirectories(args.raw_dirs)
    parent_of_deepest = list(set([os.path.dirname(d) for d in deepest]))
    print(parent_of_deepest)
    process_multiple_tasks(parent_of_deepest, args.repo_id, args.robot_type, args.mode)
    
# python /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/convert_aloha_data_to_lerobot.py \
#     --raw_dir="/mnt/hpfs/baaiei/lvhuaihai/agilex_data/test/task_put_black_brown_basket_4.1" \
#     --repo_id="/mnt/hpfs/baaiei/lvhuaihai/agilex_data/test/save" \
#     --task="DEBUG" \
#     --mode="video"

# python /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/convert_aloha_data_to_lerobot.py --raw_dir="/mnt/hpfs/baaiei/lvhuaihai/agilex_data/test/task_put_black_brown_basket_4.1"  --repo_id="/mnt/hpfs/baaiei/lvhuaihai/agilex_data/test/save" --task="DEBUG"  --mode="video" 
# python /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/convert_aloha_data_to_lerobot.py --raw_dir="/mnt/hpfs/baaiei/robot_data/agilex/robohetero/test/task_Egg_yolk_pastry__bowl_blue_40_4.23"  --repo_id=HuaihaiLyu/test --task="realman test"  --mode="video" 