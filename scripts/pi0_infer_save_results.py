from openpi.training import config
from openpi.policies import policy_config
import numpy as np
import os
import json
from typing import List, Dict, Tuple
import time
import h5py
import cv2
import re
from datetime import datetime

def parse_joint_states(lines: List[str], start_idx: int) -> Tuple[str, np.ndarray]:
    """
    解析日志行中的关节状态数据，处理跨行的情况
    
    Args:
        lines (List[str]): 日志行列表
        start_idx (int): 起始行索引
    
    Returns:
        Tuple[str, np.ndarray]: (arm_type, joint_states)
    """
    # 使用正则表达式提取时间戳和关节状态
    first_line = lines[start_idx].strip()
    match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - jointstate_(\w+): \[(.*)', first_line)
    if not match:
        raise ValueError(f"无法解析日志行: {first_line}")
    
    timestamp, arm_type, joint_values_start = match.groups()
    
    # 合并两行的数据
    joint_values = joint_values_start.strip()
    if start_idx + 1 < len(lines):  # 确保有下一行
        next_line = lines[start_idx + 1].strip()
        if next_line.endswith(']'):  # 确认是续行
            joint_values += ' ' + next_line.strip('[] ')
    
    # 将字符串转换为numpy数组
    try:
        values = [float(x) for x in joint_values.split()]
        joint_array = np.array(values)
        
        if len(joint_array) != 7:
            print(f"Warning: {arm_type}臂关节维度不是7维: {len(joint_array)}维")
            print(f"原始数据: {joint_values}")
            print(f"解析值: {values}")
            # 如果维度不够，填充0；如果维度过多，截断
            if len(joint_array) < 7:
                joint_array = np.pad(joint_array, (0, 7 - len(joint_array)), 'constant')
            else:
                joint_array = joint_array[:7]
            print(f"已调整为7维: {joint_array}")
    except Exception as e:
        print(f"解析关节值时出错: {str(e)}")
        print(f"原始数据: {joint_values}")
        # 返回7维的零向量
        joint_array = np.zeros(7)
    
    return arm_type, joint_array

def load_joint_states_from_log(log_file: str) -> np.ndarray:
    """
    从日志文件加载关节状态
    
    Args:
        log_file (str): 日志文件路径
    
    Returns:
        np.ndarray: 合并的左右臂关节状态 (14维)
    """
    with open(log_file, 'r') as f:
        # 读取所有行
        lines = [line.strip() for line in f.readlines()]
        
        try:
            # 解析左右臂状态（每个状态可能占用两行）
            left_type, left_joints = parse_joint_states(lines, 0)  # 左臂在第0,1行
            right_type, right_joints = parse_joint_states(lines, 2)  # 右臂在第2,3行
            
            # 确保两个数组都是7维
            assert len(left_joints) == 7, f"左臂关节维度错误: {len(left_joints)}"
            assert len(right_joints) == 7, f"右臂关节维度错误: {len(right_joints)}"
            
            # 合并左右臂状态为14维向量
            joint_states = np.concatenate([left_joints, right_joints])
            assert len(joint_states) == 14, f"合并后维度错误: {len(joint_states)}"
            
            return joint_states
        except Exception as e:
            print(f"解析关节状态时出错: {str(e)}")
            print("日志文件内容:")
            for i, line in enumerate(lines):
                print(f"Line {i}: {line}")
            raise

def load_data_from_states(states_dir: str = "/share/project/lvhuaihai/lvhuaihai/openpi/scripts/states") -> Dict:
    """
    从states目录加载数据
    
    Args:
        states_dir (str): states数据目录
    
    Returns:
        Dict: 包含图像和状态数据的字典
    """
    # 构建图像路径
    image_paths = {
        "cam_high": os.path.join(states_dir, "cam_high.png"),
        "cam_left_wrist": os.path.join(states_dir, "cam_left_wrist.png"),
        "cam_right_wrist": os.path.join(states_dir, "cam_right_wrist.png")
    }
    
    # 读取图像
    frames = {}
    for cam_name, img_path in image_paths.items():
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在：{img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"无法读取图像：{img_path}")
        # 转换为RGB并调整维度
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[cam_name] = frame.transpose(2, 0, 1)
    
    # 读取joint states从日志文件
    joints_log_path = os.path.join(states_dir, "joints.txt")
    if not os.path.exists(joints_log_path):
        raise FileNotFoundError(f"关节状态日志文件不存在：{joints_log_path}")
    state = load_joint_states_from_log(joints_log_path)
    
    return {
        "frames": frames,
        "state": state,
    }

def save_inference_results(
    save_dir: str,
    states_dir: str = "/share/project/lvhuaihai/lvhuaihai/openpi/scripts/states",
    prompt: str = "Use the left arm to place the bottom bun upright.",
    chunk_size: int = 50
) -> None:
    """
    执行推理并保存结果
    
    Args:
        save_dir (str): 结果保存目录
        states_dir (str): states数据目录
        prompt (str): 提示语句
        chunk_size (int): 每次推理的动作序列长度
    """

    try:
        # 记录数据加载开始时间
        data_loading_start = time.time()
        # 加载数据
        print("Loading data...")
        data_dict = load_data_from_states(states_dir)
        
        # 构建模型输入
        example = {
            "state": data_dict["state"],
            "images": data_dict["frames"],
            "prompt": prompt
        }
        data_loading_time = time.time() - data_loading_start
        print(f"Data loading completed in {data_loading_time:.2f} seconds")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化策略模型
        print("Initializing policy...")
        config_obj = config.get_config("pi0_aloha_hamburger")
        checkpoint_dir = "/share/project/lvhuaihai/lvhuaihai/openpi/checkpoints/pi0_aloha_hamburger/aloha_hamburger/19999"
        policy = policy_config.create_trained_policy(config_obj, checkpoint_dir)
        print("Policy created successfully")
    
        # 记录推理开始时间
        
        
        while True:
            # 执行推理
            print("Running inference...")
            start_time = time.time()
            action_chunk = policy.infer(example)["actions"]
            
            # 记录推理结束时间
            inference_time = time.time() - start_time
            print(f"Inference completed in {inference_time:.2f} seconds")
            import pdb; pdb.set_trace()
        
        # 保存结果到HDF5文件
        h5_path = os.path.join(save_dir, "predictions.h5")
        with h5py.File(h5_path, 'w') as f:
            # 创建数据集
            f.create_dataset("predicted_actions", data=action_chunk)
            # 保存元数据
            f.attrs["prompt"] = prompt
            f.attrs["inference_time"] = inference_time
        
        print(f"Results saved to: {h5_path}")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
    
    print("\nProcessing completed.")

if __name__ == "__main__":
    # 设置参数
    SAVE_DIR = "/share/project/lvhuaihai/robot_data/lerobot_data/HuaihaiLyu/0509hirobot/inference_results"
    STATES_DIR = "/share/project/lvhuaihai/lvhuaihai/openpi/scripts/states"
    PROMPT = "Use the left arm to place the bottom bun upright."
    
    # 运行推理和保存
    save_inference_results(
        save_dir=SAVE_DIR,
        states_dir=STATES_DIR,
        prompt=PROMPT
    ) 