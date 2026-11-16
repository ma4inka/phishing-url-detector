"""Конфігурація проєкту: шляхи та гіперпараметри."""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODELS_DIR / "phishing_model.pt"
DEFAULT_DATASET_PATH = DATA_DIR / "urls.csv"


@dataclass
class ModelConfig:
    """Гіперпараметри архітектури Char-CNN + BiLSTM + Attention."""

    embed_dim: int = 48
    conv_filters: int = 64
    kernel_sizes: tuple = (3, 5, 7)  # лише непарні значення
    lstm_hidden: int = 64
    dense_dim: int = 64
    dropout: float = 0.35


@dataclass
class TrainConfig:
    """Параметри навчання."""

    max_len: int = 150
    epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    grad_clip: float = 1.0
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    use_class_weights: bool = True
