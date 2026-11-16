"""Клас Trainer: цикл навчання, валідація, early stopping."""
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import TrainConfig


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, pos_weight: Optional[float] = None,
                 device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        pw = torch.tensor([pos_weight], device=self.device) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=1)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
        self.best_state = None

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, n = 0.0, 0
        tp = fp = fn = correct = 0
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                pred = (logits > 0).float()
                total_loss += loss.item() * len(yb)
                n += len(yb)
                correct += (pred == yb).sum().item()
                tp += ((pred == 1) & (yb == 1)).sum().item()
                fp += ((pred == 1) & (yb == 0)).sum().item()
                fn += ((pred == 0) & (yb == 1)).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return total_loss / max(n, 1), correct / max(n, 1), f1

    def fit(self, train_loader, val_loader,
            progress_callback: Optional[Callable[[int, int, dict], None]] = None) -> dict:
        best_val, bad_epochs = float("inf"), 0
        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss, tr_acc, _ = self._run_epoch(train_loader, train=True)
            va_loss, va_acc, va_f1 = self._run_epoch(val_loader, train=False)
            self.scheduler.step(va_loss)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)
            self.history["val_f1"].append(va_f1)

            logs = {"train_loss": tr_loss, "val_loss": va_loss, "train_acc": tr_acc,
                    "val_acc": va_acc, "val_f1": va_f1}
            print(f"Епоха {epoch:02d}/{self.cfg.epochs} | loss {tr_loss:.4f} / {va_loss:.4f} | "
                  f"acc {tr_acc:.4f} / {va_acc:.4f} | val F1 {va_f1:.4f}")
            if progress_callback:
                progress_callback(epoch, self.cfg.epochs, logs)

            if va_loss < best_val - 1e-4:
                best_val, bad_epochs = va_loss, 0
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.patience:
                    print(f"Early stopping на епосі {epoch}")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history
