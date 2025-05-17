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

@dataclasses.dataclass
class Args:
    repo_id: str  # Replace with your repo_id

if __name__ == "__main__":
    args = tyro.cli(Args)

    dataset = LeRobotDataset(args.repo_id, local_files_only=True)

    # 上传到 Hugging Face Hub
    dataset.push_to_hub(args.repo_id)