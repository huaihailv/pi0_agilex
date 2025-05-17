import json
import pathlib

import torch
import pydantic


@pydantic.dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class NormStats:
    mean: torch.Tensor
    std: torch.Tensor
    q01: torch.Tensor | None = None  # 1st quantile
    q99: torch.Tensor | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors on GPU."""

    def __init__(self, device="cuda", num_quantile_bins=5000):
        self.device = device
        self.num_quantile_bins = num_quantile_bins

        self.count = 0
        self.sum = None
        self.sqsum = None
        self.min = None
        self.max = None
        self.histograms = None
        self.bin_edges = None

    def update(self, batch: torch.Tensor) -> None:
        """Update the running statistics with a batch of vectors."""
        if not isinstance(batch, torch.Tensor):
            raise TypeError(f"Batch must be a torch.Tensor, got {type(batch)}")
        batch = batch.to(self.device)

        if batch.ndim == 1:
            batch = batch.unsqueeze(0)

        num_elements, vector_length = batch.shape

        batch_min = batch.min(dim=0).values
        batch_max = batch.max(dim=0).values

        if self.count == 0:
            self.sum = batch.sum(dim=0)
            self.sqsum = (batch ** 2).sum(dim=0)
            self.min = batch_min
            self.max = batch_max

            self.histograms = torch.zeros((vector_length, self.num_quantile_bins), device=self.device)
            self.bin_edges = []
            for i in range(vector_length):
                edges = torch.linspace(
                    self.min[i] - 1e-10,
                    self.max[i] + 1e-10,
                    self.num_quantile_bins + 1,
                    device=self.device,
                )
                self.bin_edges.append(edges)
        else:
            self.sum += batch.sum(dim=0)
            self.sqsum += (batch ** 2).sum(dim=0)

            new_max = torch.max(self.max, batch_max)
            new_min = torch.min(self.min, batch_min)

            max_changed = (new_max > self.max).any()
            min_changed = (new_min < self.min).any()

            self.max = new_max
            self.min = new_min

            if max_changed or min_changed:
                self._adjust_histograms()

        self.count += num_elements
        self._update_histograms(batch)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self.histograms)):
            old_edges = self.bin_edges[i]
            new_edges = torch.linspace(
                self.min[i],
                self.max[i],
                self.num_quantile_bins + 1,
                device=self.device,
            )

            centers = (old_edges[:-1] + old_edges[1:]) / 2
            hist_values = self.histograms[i]

            new_hist = torch.zeros_like(hist_values)
            bin_idx = torch.bucketize(centers, new_edges) - 1
            bin_idx = torch.clamp(bin_idx, 0, self.num_quantile_bins - 1)

            for j in range(len(hist_values)):
                new_hist[bin_idx[j]] += hist_values[j]

            self.histograms[i] = new_hist
            self.bin_edges[i] = new_edges

    def _update_histograms(self, batch: torch.Tensor) -> None:
        """Update histograms with new batch."""
        for i in range(batch.shape[1]):
            hist = torch.histc(
                batch[:, i],
                bins=self.num_quantile_bins,
                min=self.bin_edges[i][0].item(),
                max=self.bin_edges[i][-1].item(),
            )
            self.histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles from histograms."""
        results = []
        for q in quantiles:
            target_count = q * self.count
            q_values = []
            for i in range(len(self.histograms)):
                cumsum = torch.cumsum(self.histograms[i], dim=0)
                idx = torch.searchsorted(cumsum, target_count)
                idx = torch.clamp(idx, 0, self.num_quantile_bins - 1)
                q_value = self.bin_edges[i][idx]
                q_values.append(q_value)
            results.append(torch.stack(q_values))
        return results

    def get_statistics(self) -> NormStats:
        """Return the computed statistics."""
        if self.count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        mean = self.sum / self.count
        variance = self.sqsum / self.count - mean ** 2
        std = torch.sqrt(variance.clamp(min=1e-6))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=mean, std=std, q01=q01, q99=q99)


# 保存 / 加载接口
def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        key: {
            "mean": value.mean.cpu().tolist(),
            "std": value.std.cpu().tolist(),
            "q01": value.q01.cpu().tolist() if value.q01 is not None else None,
            "q99": value.q99.cpu().tolist() if value.q99 is not None else None,
        }
        for key, value in norm_stats.items()
    }
    path.write_text(json.dumps(data, indent=2))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")

    data = json.loads(path.read_text())
    return {
        key: NormStats(
            mean=torch.tensor(value["mean"]),
            std=torch.tensor(value["std"]),
            q01=torch.tensor(value["q01"]) if value["q01"] is not None else None,
            q99=torch.tensor(value["q99"]) if value["q99"] is not None else None,
        )
        for key, value in data.items()
    }
