from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

import cv2
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
# from agilex_env import get_obs

def extract_frame(video_path, save_path=None, frame_number=0):
    """
    读取视频的指定帧图像并保存（可选）

    Args:
        video_path (str): 视频文件的路径
        save_path (str or None): 保存指定帧图像的路径，如果为 None 则不保存
        frame_number (int): 要读取的帧号（从0开始计数）
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频文件不存在: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {video_path}")
        return

    # 跳过前 frame_number - 1 帧
    for _ in range(frame_number - 1):
        success = cap.grab()
        if not success:
            print(f"[ERROR] 无法读取第 {_+1} 帧")
            cap.release()
            return

    # 读取第 frame_number 帧
    success, frame = cap.read()
    cap.release()

    if not success:
        print(f"[ERROR] 无法读取第 {frame_number} 帧")
        return

    print(f"[INFO] 成功读取第 {frame_number} 帧")

    if save_path:
        cv2.imwrite(save_path, frame)
        print(f"[INFO] 第 {frame_number} 帧已保存至: {save_path}")

    return frame

def load_data_for_frame_with_prompt(frame_number, episode_index=0,
                        data_dir="/share/project/lvhuaihai/robot_data/lerobot_data/HuaihaiLyu/0509hirobot",
                        meta_path="meta/episodes.jsonl"):
    """
    加载指定 episode 中的帧数据，并附带任务描述（prompt）

    Args:
        frame_number (int): 要读取的帧号
        episode_index (int): episode 序号
        data_dir (str): 数据根目录
        meta_path (str): episodes.jsonl 的相对路径

    Returns:
        dict: 包含视频帧、状态、完整 dataframe 及任务 prompt 的字典
    """
    episode_str = f"episode_{episode_index:06d}"
    
    # 加载视频帧
    video_paths = {
        "cam_high": os.path.join(data_dir, f"videos/chunk-000/observation.images.cam_high/{episode_str}.mp4"),
        "cam_left_wrist": os.path.join(data_dir, f"videos/chunk-000/observation.images.cam_left_wrist/{episode_str}.mp4"),
        "cam_right_wrist": os.path.join(data_dir, f"videos/chunk-000/observation.images.cam_right_wrist/{episode_str}.mp4")
    }

    frames = {}
    for cam_name, video_path in video_paths.items():
        frame = extract_frame(video_path, frame_number=frame_number)
        if frame is None:
            raise ValueError(f"无法读取 {cam_name} 的第 {frame_number} 帧，文件路径：{video_path}")
        frames[cam_name] = frame.transpose(2, 0, 1)

    # 加载状态数据
    parquet_path = os.path.join(data_dir, f"data/chunk-000/{episode_str}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"状态数据文件不存在：{parquet_path}")
    data = pd.read_parquet(parquet_path)
    if frame_number >= len(data):
        raise IndexError(f"帧号 {frame_number} 超过数据帧数 {len(data)}")
    state = data['action'][frame_number]

    # 加载 episode 对应的 prompt
    full_meta_path = os.path.join(data_dir, meta_path)
    prompt = None
    with open(full_meta_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record["episode_index"] == episode_index:
                prompt = record["tasks"][0]  # 默认取第一个 task
                break
    if prompt is None:
        raise ValueError(f"未在 {meta_path} 中找到 episode_index={episode_index} 的任务描述")

    return {
        "frames": frames,
        "state": state,
        "data": data,
        "prompt": prompt
    }


def process_multiple_episodes_with_prompt_swap(
    frame_numbers: List[int],
    episode_indices: List[int],
    prompt_indices: List[int],
    data_dir="/share/project/lvhuaihai/robot_data/lerobot_data/HuaihaiLyu/0509hirobot"
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    对多个 episode 的帧进行处理，支持使用其他 episode 的 prompt，用于评估语言泛化能力。
    
    Args:
        frame_numbers (List[int]): 起始帧号列表（如 [0, 50, 100, 150]）
        episode_indices (List[int]): 对应的 episode index 列表（与 frame_numbers 对应）
        prompt_indices (List[int]): 提供 prompt 的 episode index 列表（与 episode_indices 对应）
        data_dir (str): 根数据路径

    Returns:
        Dict[(state_episode, prompt_episode)]: 包含两类 loss 的字典
    """
    assert len(frame_numbers) == len(episode_indices) == len(prompt_indices), "输入列表长度不一致"

    results = {}

    for frame_number, state_epi, prompt_epi in zip(frame_numbers, episode_indices, prompt_indices):
        try:
            # 加载状态 episode_i 的数据
            data_i = load_data_for_frame_with_prompt(
                frame_number=frame_number,
                episode_index=state_epi,
                data_dir=data_dir
            )

            # 加载 prompt episode_j 的 prompt
            prompt_j_data = load_data_for_frame_with_prompt(
                frame_number=0,
                episode_index=prompt_epi,
                data_dir=data_dir
            )
            prompt_j = prompt_j_data["prompt"]

            # 构建输入
            example = {
                "state": data_i["state"],
                "images": {
                    "cam_high": data_i["frames"]["cam_high"],
                    "cam_left_wrist": data_i["frames"]["cam_left_wrist"],
                    "cam_right_wrist": data_i["frames"]["cam_right_wrist"]
                },
                "prompt": prompt_j
            }

            print("inference start:")
            import time
            start_time = time.time()
            # 推理
            action_chunk = policy.infer(example)["actions"]
            end_time = time.time()
            print(f"inference time: {end_time - start_time:.2f} seconds")  

            import pdb; pdb.set_trace()

            # 取 state_epi 的 ground truth
            gt_i = np.stack(data_i["data"]['action'][frame_number:frame_number+50].tolist())

            # 如果 prompt_epi ≠ state_epi，也取 prompt_epi 的 ground truth
            if prompt_epi != state_epi:
                data_j = load_data_for_frame_with_prompt(
                    frame_number=frame_number,
                    episode_index=prompt_epi,
                    data_dir=data_dir
                )
                gt_j = np.stack(data_j["data"]['action'][frame_number:frame_number+50].tolist())
            else:
                gt_j = gt_i

            # 三种对比
            mse_i = np.mean((action_chunk - gt_i) ** 2)  # 当前状态 vs 自身 prompt
            mse_j = np.mean((action_chunk - gt_j) ** 2)  # 当前状态 vs 外部 prompt 的 GT
            mse_swapped = mse_i  # 本质上等于 mse_i（因为比较的一直是 state_i 的 GT）

            # 计算每一步的 MSE（假设 action_chunk 和 gt_i shape: [N, action_dim]）
            mse_each_step = np.mean((action_chunk - gt_i) ** 2, axis=1)  # shape: [N]
            mse_each_step_prompt = np.mean((action_chunk - gt_j) ** 2, axis=1)

            results[(state_epi, prompt_epi)] = {
                "MSE(state_vs_gt)": mse_i,
                "MSE(prompt_vs_gt)": mse_j,
                "MSE(state_gt_vs_swapped_prompt)": mse_i if prompt_epi != state_epi else None,
                "MSE_each_step_state_gt": mse_each_step.tolist(),
                "MSE_each_step_prompt_gt": mse_each_step_prompt.tolist()
            }

            print(f"[State {state_epi} | Prompt {prompt_epi}] -> "
                  f"MSE(state_gt): {mse_i:.6f} | MSE(prompt_gt): {mse_j:.6f} | "
                  f"MSE(state_vs_swapped_prompt): {mse_swapped:.6f}" if prompt_epi != state_epi else ""
                  f"MSE_each_step_state_gt: {mse_each_step.tolist()} | MSE_each_step_prompt_gt: {mse_each_step_prompt.tolist()}")

        except Exception as e:
            print(f"处理 Episode {state_epi} / Prompt {prompt_epi} 出错: {str(e)}")
            results[(state_epi, prompt_epi)] = {
                "MSE(state_vs_gt)": None,
                "MSE(prompt_vs_gt)": None,
                "MSE(state_gt_vs_swapped_prompt)": None,
                "MSE_each_step_state_gt": None,
                "MSE_each_step_prompt_gt": None
            }

    return results

# 初始化训练好的 policy
config = config.get_config("pi0_aloha_hamburger")
checkpoint_dir = "/share/project/lvhuaihai/lvhuaihai/openpi/checkpoints/pi0_aloha_hamburger/aloha_hamburger/19999"
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print("policy created")

# 要测试的帧号列表（可根据 chunk size 设置）
frame_numbers = [0, 50, 100, 150]

# 每个帧号对应的状态 episode（即从哪个 episode 读取 state 和图像）
episode_indices = [0, 0, 1, 1]

# 每个帧号对应的 prompt episode（即使用哪个任务描述 prompt）
prompt_indices = [0, 1, 1, 0]  # 其中 1 和 2 为跨 episode prompt，对应语言泛化测试

# 运行评估
results = process_multiple_episodes_with_prompt_swap(
    frame_numbers,
    episode_indices,
    prompt_indices,
    data_dir="/share/project/lvhuaihai/robot_data/lerobot_data/HuaihaiLyu/0509hirobot"
)

# 打印结果
print("\n=== 每对 Episode 的 MSE 结果 ===")
for (state_ep, prompt_ep), losses in results.items():
    s_loss = losses["MSE(state_vs_gt)"]
    p_loss = losses["MSE(prompt_vs_gt)"]
    s_loss_each_step = losses["MSE_each_step_state_gt"]
    p_loss_each_step = losses["MSE_each_step_prompt_gt"]
    print(f"[State Ep {state_ep}, Prompt Ep {prompt_ep}] -> MSE(state_gt): {s_loss:.6f} | MSE(prompt_gt): {p_loss:.6f}")
    print(f"MSE_each_step_state_gt: {s_loss_each_step} | MSE_each_step_prompt_gt: {p_loss_each_step}")

# 计算平均值（排除失败项）
valid_state_losses = [v["MSE(state_vs_gt)"] for v in results.values() if v["MSE(state_vs_gt)"] is not None]
valid_prompt_losses = [v["MSE(prompt_vs_gt)"] for v in results.values() if v["MSE(prompt_vs_gt)"] is not None]

if valid_state_losses:
    print(f"\n平均 MSE（正确 prompt）: {np.mean(valid_state_losses):.6f}")
if valid_prompt_losses:
    print(f"平均 MSE（语言泛化 prompt）: {np.mean(valid_prompt_losses):.6f}")

import pdb; pdb.set_trace()