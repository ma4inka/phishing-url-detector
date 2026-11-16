"""PyTorch-датасет для URL."""
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .preprocessing import URLTokenizer


class URLDataset(Dataset):
    """Зберігає закодовані URL та мітки у вигляді тензорів."""

    def __init__(self, urls: Sequence[str], labels: Sequence[int], tokenizer: URLTokenizer):
        self.x = torch.from_numpy(tokenizer.encode_batch(urls)).long()
        self.y = torch.tensor(list(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
