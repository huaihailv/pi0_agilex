"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize_gpu as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import torch
import jax.numpy as jnp


class KeepOnlyKeys(transforms.DataTransformFn):
    def __init__(self, keep_keys):
        self.keep_keys = keep_keys

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if k in self.keep_keys}

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
            KeepOnlyKeys(["state", "actions"]), # 必须加上，只留小数据
        ],
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=8192,
        num_workers=32,            # ⚡ 更多worker
        shuffle=shuffle,
        num_batches=num_frames,
    )
    
    device = "cuda"
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    accumulate_batches = {key: [] for key in keys}  # 用来攒 batch
    accumulate_steps = 8     # 每8个batch一起 update

    with torch.no_grad():  # 🚀 全局no_grad加速
        for idx, batch in enumerate(tqdm.tqdm(data_loader, total=(num_frames // 8192 + 1), desc="Computing stats")):
            for key in keys:
                values = batch[key][0]
                if isinstance(values, np.ndarray):
                    values = torch.from_numpy(values)
                # 统一转成 numpy
                if isinstance(values, np.ndarray):
                    pass
                elif isinstance(values, jnp.ndarray) or hasattr(values, "device_buffer"):  
                    # 说明是JAX array
                    values = np.asarray(values)  # 转成 numpy
                else:
                    raise TypeError(f"Unsupported type: {type(values)} for batch[{key}]")
                
                values = torch.from_numpy(values).to(torch.float32).to(device)
                accumulate_batches[key].append(values)

            # 每 accumulate_steps 个batch做一次 update
            if (idx + 1) % accumulate_steps == 0 or (idx + 1) == (num_frames // 8192 + 1):
                for key in keys:
                    merged = torch.cat(accumulate_batches[key], dim=0)
                    stats[key].update(merged.reshape(-1, merged.shape[-1]))
                for key in keys:
                    accumulate_batches[key].clear()

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)



if __name__ == "__main__":
    tyro.cli(main)
