import sys
sys.path.append('/home/agilex/zbl/openpi/openpi-main/src')
import logging
logging.basicConfig(
        filename='/home/agilex/zbl/openpi/openpi-main/scripts/output.txt',  # 日志文件路径
        filemode='w',  # 'a' 为追加模式，'w' 为覆盖模式
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        level=logging.INFO  # 设置日志等级（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    )
# logging.info(sys.path)
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import openpi.transforms as _transforms
import torch
import cv2
import os
import numpy as np
from dora import Node
from PIL import Image
import pyarrow as pa

def qpos_2_joint_positions(qpos:np.ndarray):

        l_joint_pos = qpos[50:56]
        r_joint_pos = qpos[0:6]
        l_gripper_pos = np.array([qpos[60]])
        r_gripper_pos = np.array([qpos[10]])

        l_pos = np.concatenate((l_joint_pos,l_gripper_pos))
        r_pos = np.concatenate((r_joint_pos,r_gripper_pos))

        return np.concatenate((l_pos,r_pos))
def main():
    
    cfg = config.get_config("pi0_aloha_stack_basket")
    checkpoint_dir = "/home/agilex/zbl/openpi/openpi-main/checkpoints/pi0_aloha_stack_basket/aloha_stack_basket/45000"

    # Create a trained policy.
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir)

    node = Node()
    frames = {}
    joints = {}
    pose = {}
    
    with torch.no_grad():
        for event in node:
            event_type = event["type"]
            if event_type == "INPUT":
                event_id = event["id"]

                if "image" in event_id:
                    storage = event["value"]
                    metadata = event["metadata"]
                    encoding = metadata["encoding"]

                    if encoding == "bgr8" or encoding == "rgb8" or encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                        channels = 3
                        storage_type = np.uint8
                    else:
                        raise RuntimeError(f"Unsupported image encoding: {encoding}")

                    if encoding == "bgr8":
                        width = metadata["width"]
                        height = metadata["height"]
                        frame = (
                            storage.to_numpy()
                            .astype(storage_type)
                            .reshape((height, width, channels))
                        )
                        frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
                    elif encoding == "rgb8":
                        width = metadata["width"]
                        height = metadata["height"]
                        frame = (
                            storage.to_numpy()
                            .astype(storage_type)
                            .reshape((height, width, channels))
                        )
                    elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                        storage = storage.to_numpy()
                        frame = cv2.imdecode(storage, cv2.IMREAD_COLOR)
                        frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
                    else:
                        raise RuntimeError(f"Unsupported image encoding: {encoding}")
                    frames[f"last_{event_id}"] = frames.get(
                        event_id, Image.fromarray(frame),
                    )
                    frames[event_id] = Image.fromarray(frame)
                elif "qpos" in event_id:
                    joints[event_id] = event["value"].to_numpy()
                elif "pose" in event_id:
                    pose[event_id] = event["value"].to_numpy()
                
                elif "tick" in event_id:
                    ## Wait for all images
                    # logging.info(f"frames: {frames.keys()}")
                    # logging.info(f"joints: {joints.keys()}")
                    # logging.info(f"pose: {pose.keys()}")
                    if len(frames.keys()) < 3:
                        continue
                    if len(joints.keys()) < 2:
                        continue
                    if len(pose.keys()) < 2:
                        continue
                    right_arm_joint = joints["/observations/qpos_right"]
                    left_arm_joint = joints["/observations/qpos_left"]
                    ## Embed images
                    obs = {
                        "state": np.concatenate(
                        [
                            right_arm_joint,
                            left_arm_joint,
                        ],
                    ).squeeze(),
                        "images": {
                            "cam_high": np.array(frames['/observations/images/cam_high']).transpose(2, 0, 1), # [720, 1280, 3]
                            "cam_left_wrist": np.array(frames['/observations/images/cam_left_wrist']).transpose(2, 0, 1),
                            "cam_right_wrist": np.array(frames['/observations/images/cam_right_wrist']).transpose(2, 0, 1)
                        },
                        "prompt": "stack the brown basket on the black basket"
                    }
                    
                    delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
                    data_transforms = data_transforms.push(
                        inputs=[_transforms.DeltaActions(delta_action_mask)],
                        outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
                    
                    action_chunk = policy.infer(obs)["actions"]
                    logging.info(f"[ROS] Executing action 1: {action_chunk[0]}")
                    for i in range (len(action_chunk)):
                        left_action = action_chunk[i,:7]
                        right_action = action_chunk[i,7:]
                        # action = action.detach().float().to("cpu").numpy()
                        node.send_output("jointstate_left", pa.array(left_action.ravel()))
                        node.send_output("jointstate_right", pa.array(right_action.ravel()))
                        
if __name__ == "__main__":
    main()