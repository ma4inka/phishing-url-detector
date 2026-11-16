"""Клас TrainingPipeline: повний цикл «дані → навчання → оцінка → збереження»."""
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import MODEL_PATH, REPORTS_DIR, ModelConfig, TrainConfig
from .dataset import URLDataset
from .evaluator import Evaluator
from .model import PhishingNet
from .preprocessing import URLTokenizer
from .trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingPipeline:
    def __init__(self, model_cfg: Optional[ModelConfig] = None, train_cfg: Optional[TrainConfig] = None,
                 model_path: Path = MODEL_PATH, reports_dir: Path = REPORTS_DIR):
        self.model_cfg = model_cfg or ModelConfig()
        self.train_cfg = train_cfg or TrainConfig()
        self.model_path = Path(model_path)
        self.reports_dir = Path(reports_dir)

    def _split(self, df: pd.DataFrame):
        c = self.train_cfg
        train_df, test_df = train_test_split(df, test_size=c.test_size, stratify=df["label"], random_state=c.seed)
        train_df, val_df = train_test_split(
            train_df, test_size=c.val_size / (1 - c.test_size), stratify=train_df["label"], random_state=c.seed
        )
        return train_df, val_df, test_df

    def run(self, df: pd.DataFrame, progress_callback: Optional[Callable[[float, str], None]] = None) -> dict:
        cb = progress_callback or (lambda frac, msg: None)
        c = self.train_cfg
        set_seed(c.seed)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        cb(0.02, "Підготовка даних…")
        df = df[["url", "label"]].drop_duplicates("url").reset_index(drop=True)
        train_df, val_df, test_df = self._split(df)

        tokenizer = URLTokenizer(max_len=c.max_len)
        make = lambda d: URLDataset(d["url"].tolist(), d["label"].tolist(), tokenizer)  # noqa: E731
        train_loader = DataLoader(make(train_df), batch_size=c.batch_size, shuffle=True)
        val_loader = DataLoader(make(val_df), batch_size=c.batch_size)
        test_loader = DataLoader(make(test_df), batch_size=c.batch_size)

        pos_weight = None
        if c.use_class_weights:
            n_pos = int((train_df["label"] == 1).sum())
            n_neg = int((train_df["label"] == 0).sum())
            if n_pos > 0:
                pos_weight = float(np.clip(n_neg / n_pos, 0.1, 10.0))

        model = PhishingNet(vocab_size=tokenizer.vocab_size, **asdict(self.model_cfg))
        trainer = Trainer(model, c, pos_weight=pos_weight)
        print(f"Пристрій: {trainer.device} | параметрів: {model.count_parameters():,} | "
              f"train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

        def on_epoch(epoch, total, logs):
            cb(0.05 + 0.85 * epoch / total,
               f"Епоха {epoch}/{total}: val_acc={logs['val_acc']:.4f}, val_F1={logs['val_f1']:.4f}")

        history = trainer.fit(train_loader, val_loader, progress_callback=on_epoch)

        cb(0.92, "Оцінювання на тестовій вибірці…")
        evaluator = Evaluator(trainer.model, trainer.device)
        y_true, y_prob = evaluator.predict_proba(test_loader)
        metrics = evaluator.compute_metrics(y_true, y_prob, threshold=0.5)
        print("Тестові метрики:", json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"},
                                              ensure_ascii=False))

        evaluator.plot_training_history(history, self.reports_dir / "training_history.png")
        evaluator.plot_confusion_matrix(metrics["confusion_matrix"], self.reports_dir / "confusion_matrix.png")
        if metrics["roc_auc"] is not None:
            evaluator.plot_roc(y_true, y_prob, self.reports_dir / "roc_curve.png")

        cb(0.97, "Збереження моделі…")
        dataset_info = {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "phishing_share": float(df["label"].mean()),
        }
        ckpt = {
            "model_state": {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
            "model_config": {**asdict(self.model_cfg), "kernel_sizes": list(self.model_cfg.kernel_sizes)},
            "max_len": int(c.max_len),
            "threshold": 0.5,
            "metrics": metrics,
            "history": {k: [float(x) for x in v] for k, v in history.items()},
            "dataset_info": dataset_info,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        torch.save(ckpt, self.model_path)
        with open(self.reports_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "dataset_info": dataset_info}, f, ensure_ascii=False, indent=2)

        cb(1.0, "Готово!")
        return {"metrics": metrics, "history": history, "dataset_info": dataset_info, "model_path": str(self.model_path)}
