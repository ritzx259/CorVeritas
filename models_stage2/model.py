"""PyTorch architecture shared by training and loading; imports perform no training."""
import random

import numpy as np
import torch
from torch import nn


class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim, hidden=(128, 64), dropout=(.3, .2)):
        super().__init__()
        # LayerNorm avoids BatchNorm's batch-size-one failure and unstable small-batch statistics.
        self.network = nn.Sequential(nn.Linear(input_dim, hidden[0]), nn.ReLU(), nn.LayerNorm(hidden[0]),
            nn.Dropout(dropout[0]), nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout[1]), nn.Linear(hidden[1], 3))

    def forward(self, x):
        return self.network(x)


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS was requested but is unavailable. Use --device cpu.")
    return torch.device(requested)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def neural_scores(model, matrix, device="cpu", batch_size=1024):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            tensor = torch.as_tensor(np.asarray(matrix[start:start + batch_size], dtype=np.float32), device=device)
            chunks.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, 3))
