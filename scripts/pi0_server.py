import sys
sys.path.append("/home/agilex/zbl/openpi/openpi-main/src")
import os
from openpi.training import config
from openpi.policies import policy_config
import numpy as np
import torch
import logging
import base64
import cv2
from flask import Flask, request, jsonify


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/pi0_server.log"),
        logging.StreamHandler()
    ],
    force=True  # 强制覆盖之前配置
)

app = Flask(__name__)

def decode_image(b64str):
    """将 base64 字符串解码为图像数组"""
    image_bytes = base64.b64decode(b64str)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

class PI0Model:
    def __init__(self):
        self.load_model()
        
    def load_model(self):
        """加载PI0模型"""
        try:
            config_data = config.get_config("pi0_aloha_hamburger_4cam")
            checkpoint_dir = "/share/project/lvhuaihai/lvhuaihai/openpi/checkpoints/pi0_aloha_hamburger_4cam/aloha_hamburger_4cam_50000steps/40000"
            
            with torch.no_grad():
                self.policy = policy_config.create_trained_policy(config_data, checkpoint_dir)
            logging.info("PI0 model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load PI0 model: {e}")
            raise

    def infer(self, state, images, prompt):
        """执行推理"""
        try:
            example = {
                "state": state,
                "images": {
                    "cam_high": images["cam_high"].transpose(2, 0, 1),
                    "cam_left_wrist": images["cam_left_wrist"].transpose(2, 0, 1),
                    "cam_right_wrist": images["cam_right_wrist"].transpose(2, 0, 1),
                    "cam_high_realsense": images.get("cam_high_realsense", np.zeros((720, 1280, 3), dtype=np.uint8)).transpose(2, 0, 1)
                },
                "prompt": prompt
            }
            
            
            import time
            start_time = time.time()
            output = self.policy.infer(example)
            inference_time = time.time() - start_time
            logging.info(f"Inference time: {inference_time:.2f} seconds")
                
            return output["actions"]
        except Exception as e:
            logging.error(f"Inference error: {e}")
            raise

# 创建全局模型实例
model = PI0Model()

@app.route('/infer', methods=['POST'])
def infer():
    try:
        logging.info("receive input data")
        import time
        start_time_outer = time.time()
        data = request.get_json()
        state = data['state']
        prompt = data['prompt']
        # 用decode_image解码
        images = {
            'cam_high': decode_image(data['images']['cam_high']),
            'cam_left_wrist': decode_image(data['images']['cam_left_wrist']),
            'cam_right_wrist': decode_image(data['images']['cam_right_wrist'])
        }
        if 'cam_high_realsense' in data['images']:
            images['cam_high_realsense'] = decode_image(data['images']['cam_high_realsense'])

        data_loading_time = time.time() - start_time_outer
        logging.info(f"Data loading time: {data_loading_time:.2f} seconds")
        actions = model.infer(state, images, prompt)
        logging.info("results output")
        inference_time_outer = time.time() - start_time_outer
        logging.info(f"Inference time outer: {inference_time_outer:.2f} seconds")

        return jsonify({
            'status': 'success',
            'actions': actions.tolist()
        })
    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 