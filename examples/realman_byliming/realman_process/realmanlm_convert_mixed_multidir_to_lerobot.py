"""
将realman机器人的hdf5数据转换为LeRobot数据集v2.0格式的脚本。

使用示例: 
uv run examples/aloha_real/convert_mixed_data_to_lerobot.py --raw-dir /mnt/hpfs/baaiei/robot_data/agilex/stack_basket/task_put_brown_black_basket_4.1 --repo-id HuaihaiLyu/stack_basket  --mode="video"  --task="stack the brown basket on the black basket"
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /mnt/hpfs/baaiei/robot_data/agilex/groceries_dual/task_take_brown_long_bread_Egg_yolk_pasry_4.3 --repo-id HuaihaiLyu/groceries  --mode="video"  --task="Pick the brown long bread and Egg yolk pasry into package"
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /mnt/hpfs/baaiei/robot_data/realman/18.37realman  --repo_id=HuaihaiLyu/test --task="realman test"  --mode="video" 
"""

# 导入必要的库
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
#pdb是用来断点调试的
import pdb
import pytransform3d.rotations as rotations

# 数据集配置类，用于设置数据集参数
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True  # 是否使用视频格式存储
    tolerance_s: float = 0.0001  # 时间容差(秒)
    image_writer_processes: int = 10  # 图像写入进程数
    image_writer_threads: int = 5  # 图像写入线程数
    video_backend: str | None = None  # 视频后端类型


DEFAULT_DATASET_CONFIG = DatasetConfig()  # 默认配置


# 创建空的LeRobot数据集
def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # 定义所有电机和传感器名称 - 修改为RealMan的7自由度机械臂
    motors = [
        # 左臂关节位置 - 从6个修改为7个关节
        "left_joint_position1",
        "left_joint_position2",
        "left_joint_position3",
        "left_joint_position4",
        "left_joint_position5",
        "left_joint_position6",
        "left_joint_position7",  # 增加第7关节
        "left_gripper",
        # 右臂关节位置 - 从6个修改为7个关节
        "right_joint_position1",
        "right_joint_position2",
        "right_joint_position3",
        "right_joint_position4",
        "right_joint_position5",
        "right_joint_position6",
        "right_joint_position7",  # 增加第7关节
        "right_gripper",
        # Realman左手位置和增量（注意这个以后会和pika数据混训）
        "realman_left_x",
        "realman_left_y",
        "realman_left_z",
        "realman_left_delta_x",
        "realman_left_delta_y",
        "realman_left_delta_z",
        # Realman右手位置和增量（注意这个以后会和pika数据混训）
        "realman_right_x",
        "realman_right_y",
        "realman_right_z",
        "realman_right_delta_x",
        "realman_right_delta_y",
        "realman_right_delta_z",
        # Pika左手位置和增量（现在要改成vp数据了5月7日）
        "vp_left_x",
        "vp_left_y",
        "vp_left_z",
        "vp_left_delta_x",
        "vp_left_delta_y",
        "vp_left_delta_z",
        # Pika右手位置和增量（现在要改成vp数据了5月7日）
        "vp_right_x",
        "vp_right_y",
        "vp_right_z",
        "vp_right_delta_x",
        "vp_right_delta_y",
        "vp_right_delta_z",
        
    ]
    # 相机列表
    cameras = [
        "cam_high",  # 高位俯视相机
        # "cam_low",
        "cam_left_wrist",  # 左手腕相机
        "cam_right_wrist",  # 右手腕相机
    ]

    # 定义数据集特征
    features = {
        "observation.state": {  # 机器人状态
            "dtype": "float32",
            "shape": (len(motors),),
            "names": {"motors":motors},
        },
        "action": {  # 机器人动作
            "dtype": "float32",
            "shape": (len(motors),),
            "names": {"motors":motors},
        },
        "task_index": {"dtypes": "int32", "shape": (1), "names": {"task_index": "task_index"}},
    }

    # 添加速度特征（如果有）
    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    # 添加力矩特征（如果有）
    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    # 添加相机图像特征
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,  # 使用指定的模式（视频或图像）
            "shape": (3, 480, 640),  # 图像大小：RGB三通道，480x640像素
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    # 如果数据集已存在，先删除
    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # 创建并返回新数据集
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,  # 帧率30fps
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


# 从HDF5文件中获取相机列表
def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # 忽略深度通道
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]


# 检查HDF5文件是否包含速度数据
def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


# 检查HDF5文件是否包含力矩数据
def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


# 从HDF5文件加载相机图像
def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4  # 检查是否为未压缩图像
        if uncompressed:
            # 直接加载未压缩图像
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2
            # 加载并解压缩图像
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam

# 将原始位置数据转换为关节位置数组
def qpos_2_joint_positions(qpos:np.ndarray):
    # 从qpos中提取左右手臂的关节位置和夹爪位置
    l_joint_pos = qpos[:, 50:57]  # 左臂7个关节位置 (原来是50:56)
    r_joint_pos = qpos[:, 0:7]    # 右臂7个关节位置 (原来是0:6)
    l_gripper_pos = np.array([qpos[:,60]]).reshape(-1,1)  # 左夹爪位置
    r_gripper_pos = np.array([qpos[:,10]]).reshape(-1,1)  # 右夹爪位置

    # 合并左右手臂数据
    l_pos = np.concatenate((l_joint_pos, l_gripper_pos), axis=1)
    r_pos = np.concatenate((r_joint_pos, r_gripper_pos), axis=1)

    return np.concatenate((l_pos, r_pos), axis=1)
    
# 将6D姿态转换为欧拉角
def pose6D2quat(pose:np.ndarray):
        # 从6D位姿中提取两个列向量
        column_1 = pose[:,:3]
        column_2 = pose[:,3:]

        # 构建旋转矩阵（两个列向量外积得到第三个列向量）
        R = np.column_stack((column_1, column_2, np.cross(column_1, column_2)))

        # 初始化欧拉角数组
        euler_angles = np.zeros((pose.shape[0], 3))

        # 将旋转矩阵转换为欧拉角
        for i in range(pose.shape[0]):
            euler_angles[i] = rotations.euler_from_matrix(R[i].reshape(3,3), 0, 1, 2, extrinsic=True)
        
        return euler_angles
    
# 从位置数据提取末端执行器位姿
def qpos_2_ee_pose(qpos:np.ndarray):
        # 提取左右手臂的6D位姿
        l_pose6d = qpos[:,83:89]
        r_pose6d = qpos[:,33:39]
        # 转换为欧拉角
        l_quat = pose6D2quat(l_pose6d)
        r_quat = pose6D2quat(r_pose6d)
        # 提取末端执行器位置
        l_ee_trans = qpos[:,80:83]  # 左末端执行器位置
        r_ee_trans = qpos[:,30:33]  # 右末端执行器位置
        # 提取夹爪位置
        l_gripper_pos = np.array([qpos[:,60]]).reshape(-1,1)
        r_gripper_pos = np.array([qpos[:,10]]).reshape(-1,1)

        # 合并所有数据
        return np.concatenate((l_ee_trans, l_quat, r_ee_trans, r_quat), axis=1)
    
# 加载单个剧集的原始数据
def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    
    with h5py.File(ep_path, "r") as ep:
        print(ep_path)
        # 处理状态和动作数据
        # 连接关节位置、末端执行器位姿和预留的零填充数据（可能用于其他特征）
        state = torch.from_numpy(np.concatenate((
            qpos_2_joint_positions(ep["/observations/qpos"][:]), 
            qpos_2_ee_pose(ep["/observations/qpos"][:]), 
            np.zeros_like(qpos_2_ee_pose(ep["/observations/qpos"][:]))
        ), axis=1))
        # 动作与状态相同
        action = torch.from_numpy(np.concatenate((
            qpos_2_joint_positions(ep["/observations/qpos"][:]), 
            qpos_2_ee_pose(ep["/observations/qpos"][:]), 
            np.zeros_like(qpos_2_ee_pose(ep["/observations/qpos"][:]))
        ), axis=1))
        
        # 加载速度数据（如果存在）
        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        # 加载力矩数据（如果存在）
        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        # 加载相机图像
        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


# 填充数据集
def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    # 如果未指定剧集，则处理所有剧集
    if episodes is None:
        episodes = range(len(hdf5_files))

    # 遍历所有指定的剧集
    for ep_idx in tqdm.tqdm(episodes):
        
        ep_path = hdf5_files[ep_idx]

        # 加载剧集数据
        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        # 遍历剧集中的每一帧
        for i in range(num_frames):
            # 构建帧数据
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }
            # 添加相机图像
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            # 添加速度和力矩数据（如果有）
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            # 将帧添加到数据集
            dataset.add_frame(frame)

        # 保存完整剧集并关联任务描述
        dataset.save_episode(task=task)

    return dataset


# 将realman数据转换为LeRobot数据集格式
def port_realman(
    raw_dir: Path, # 原始hdf5文件目录
    repo_id: str,  # 数据集仓库ID
    raw_repo_id: str | None = None,  # 原始数据仓库ID
    task: str = "DEBUG",  # 任务描述
    *,
    episodes: list[int] | None = None,  # 要处理的剧集列表
    push_to_hub: bool = False,  # 是否推送到HuggingFace Hub
    is_mobile: bool = False,  # 是否为移动机器人
    mode: Literal["video", "image"] = "image",  # 存储模式
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,  # 数据集配置
):
    # 如果数据集目录已存在，先删除
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # 如果原始数据目录不存在，尝试下载
    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    # 查找所有HDF5文件
    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))

    # 创建空数据集
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_realman" if is_mobile else "realman",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    # 填充数据集
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # 整合数据集
    dataset.consolidate()

    # 推送到HuggingFace Hub
    if push_to_hub:
        dataset.push_to_hub(repo_id)

import random
# 处理多个任务目录的函数
def process_multiple_tasks(raw_dirs, repo_id: str, robot_type: str = "realman", mode: str = "video"):
    # 创建空数据集
    dataset = create_empty_dataset(repo_id=repo_id, robot_type=robot_type, mode=mode)
    # 遍历每个任务目录
    for task_index, task_dir_str in enumerate(raw_dirs):

        task_dir = Path(task_dir_str)
        if not task_dir.is_dir():
            continue
        
        # 读取任务指令文件
        inst_file = task_dir / "inst.txt"
        if not inst_file.exists():
            continue
            
        with open(inst_file, "r") as f:
            # 读取所有非空行
            instructions = [line.strip() for line in f if line.strip()]
            print(instructions)
            
            # 清理指令文本（移除前缀编号等）
            cleaned_list = []
            for s in instructions:
                s = s.replace('\u200b', '')  # 移除零宽空格
                if s and s[0].isdigit():  # 检查是否以数字开头（如"1.任务"）
                    dot_pos = s.find('.')  # 找到数字后的点
                    if 0 <= dot_pos <= 3:  # 确保是编号（最多3位数）
                        cleaned_list.append(s[dot_pos + 1:].lstrip())  # 移除编号部分
                    else:
                        cleaned_list.append(s)
                else:
                    cleaned_list.append(s)
            instructions = cleaned_list
            print(instructions)
            
        if not instructions:
            continue
            
        # 选择第一条指令作为任务描述
        # selected_instruction = random.choice(instructions)  # 随机选择一条指令
        selected_instruction = instructions[0]  # 使用第一条指令
        
        # 处理该任务目录下的所有剧集
        for ep_file in task_dir.glob("episode_*.hdf5"):
            # 加载剧集数据
            imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_file)
            num_frames = state.shape[0]
            
            # 添加每一帧到数据集
            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task_index": task_index,  # 记录任务索引
                }
                # 添加相机图像
                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]
                    
                # 添加速度和力矩数据（如果有）
                if velocity is not None:
                    frame["observation.velocity"] = velocity[i]
                if effort is not None:
                    frame["observation.effort"] = effort[i]
                    
                # 将帧添加到数据集
                dataset.add_frame(frame)
                
            # 保存剧集，关联任务描述
            dataset.save_episode(task=selected_instruction)
            
    # 整合数据集
    dataset.consolidate()
    # 推送到HuggingFace Hub
    dataset.push_to_hub(repo_id)


# 主程序入口
if __name__ == "__main__":
    import argparse
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="处理多个任务并转换为LeRobot数据集。")
    parser.add_argument("--raw_dirs", type=str, required=True, help="要处理的任务目录列表。")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot数据集的仓库ID。")
    parser.add_argument("--robot_type", type=str, default="realman", help="机器人类型（默认：realman）。")
    parser.add_argument("--mode", type=str, choices=["video", "image"], default="video", help="数据模式（视频或图像）。")
    args = parser.parse_args()
    
    # 获取所有子目录
    raw_dirs = [d for d in Path(args.raw_dirs).iterdir() if d.is_dir()]
    # 处理多个任务目录
    process_multiple_tasks(raw_dirs, args.repo_id, args.robot_type, args.mode)
    
# 使用示例
# uv run /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/agilex_process/convert_mixed_multidir_to_lerobot.py --raw_dirs /mnt/hpfs/baaiei/robot_data/agilex/robohetero/build_blocks --repo_id=agilex/build_blocks --mode="video" 
# uv run /mnt/hpfs/baaiei/chenliming/openpi/examples/realman_byliming/realman_process/realmanlm_convert_mixed_multidir_to_lerobot.py --raw_dirs /mnt/hpfs/baaiei/robot_data/realman/groceries_bag --repo_id=realman/groceries_bag --mode="video" 
# 已准备好的任务: build_blocks groceries_bag organize_pants pour_bowl pour_tea stack_basket

# 保存数据根目录: /mnt/hpfs/baaiei/chenliming/data/lerobot_data/realman