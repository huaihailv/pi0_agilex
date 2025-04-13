from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

import cv2
import os
# from agilex_env import get_obs

def extract_frame(video_path, save_path=None, frame_number=150):
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


config = config.get_config("pi0_aloha_stack_basket")
checkpoint_dir = "/mnt/hpfs/baaiei/lvhuaihai/openpi/checkpoints/pi0_aloha_stack_basket/aloha_stack_basket_more_steps/40000"

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

cam_front_path = "/mnt/hpfs/baaiei/robot_data/pi0_data/agilex/HuaihaiLyu/stack_basket/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
cam_left_path = "/mnt/hpfs/baaiei/robot_data/pi0_data/agilex/HuaihaiLyu/stack_basket/videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4"
cam_right_path = "/mnt/hpfs/baaiei/robot_data/pi0_data/agilex/HuaihaiLyu/stack_basket/videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4"

cam_front = extract_frame(cam_front_path)
cam_left = extract_frame(cam_left_path)
cam_right = extract_frame(cam_right_path)


state = [-0.4628709 ,  1.9265295 , -0.8959641 , -0.15232489,  0.39396533,
        1.3374132 ,  0.01022   , -0.26855722,  1.0227504 , -0.90336055,
        0.6278081 ,  0.81776065, -0.466447  ,  0.05628  ]
# Run inference on a dummy example.
example = {
    "state": state,
    "images": {
        "cam_high": cam_front.transpose(2, 0, 1), # [720, 1280, 3]
        "cam_left_wrist": cam_left.transpose(2, 0, 1),
        "cam_right_wrist": cam_right.transpose(2, 0, 1)
    },
    "prompt": "stack the brown basket on the black basket"
}

action_chunk = policy.infer(example)["actions"]
print(action_chunk)
import pdb
pdb.set_trace()
