import numpy as np
import requests
import cv2
import logging
import time
import base64
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def encode_image(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def call_pi0_server(state, images, prompt):
    """调用PI0服务器进行推理
    
    Args:
        state: 机器人状态数组
        images: 包含相机图像的字典
        prompt: 提示文本
    
    Returns:
        预测的动作序列
    """
    url = "http://localhost:5000/infer"
    
    # 准备请求数据
    data = {
        "state": state,
        "images": {
            "cam_high": encode_image(images["cam_high"]),
            "cam_left_wrist": encode_image(images["cam_left_wrist"]),
            "cam_right_wrist": encode_image(images["cam_right_wrist"])
        },
        "prompt": prompt
    }
    
    if "cam_high_realsense" in images:
        data["images"]["cam_high_realsense"] = encode_image(images["cam_high_realsense"])
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # 检查请求是否成功
        result = response.json()
        
        if result["status"] == "success":
            return np.array(result["actions"])
        else:
            logging.error(f"Inference failed: {result.get('message', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None

def main():
    # 示例：准备输入数据
    # 1. 机器人状态
    state = [0.006716111209243536, 0, -0.4529101252555847, -0.07096400111913681,
             1.0769851207733154, 0.10430033504962921, -0.00007000000186963007,
             -0.26855722069740295, 0, -0.4851300120353699, 0.19741877913475037,
             1.0484634637832642, -0.09125188738107681, 0.0002800000074785203]
    
    # 2. 相机图像 (这里使用空白图像作为示例)
    images = {
        # 创建示例图像
        "cam_high": np.zeros((720, 1280, 3), dtype=np.uint8),
        "cam_left_wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
        "cam_right_wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
        "cam_high_realsense": np.zeros((720, 1280, 3), dtype=np.uint8)
    }
    
    # 3. 提示文本
    prompt = "Use the right arm to pick up a tomato slice and place it on the lettuce."
    
    # 调用服务进行推理
    logging.info("Sending inference request...")
    start_time = time.time()
    
    for i in range(3):
        actions = call_pi0_server(state, images, prompt)
        
        if actions is not None:
            inference_time = time.time() - start_time
            logging.info(f"第{i+1}次推理完成,用时 {inference_time:.2f} 秒")
            logging.info(f"收到 {len(actions)} 个动作")
            start_time = time.time()  # 重置开始时间用于下一次推理
        
        # 处理返回的动作序列
        # for i, action in enumerate(actions[:20]):  # 只处理前20个动作
            # left_action = action[0:7]
            # right_action = action[7:14]
            # logging.info(f"Action {i}:")
            # logging.info(f"  Left arm: {left_action}")
            # logging.info(f"  Right arm: {right_action}")
            
            # 这里可以添加动作执行代码
            # time.sleep(0.02)  # 模拟动作执行时间

if __name__ == "__main__":
    main() 