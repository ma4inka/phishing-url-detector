"""Клас Evaluator: метрики якості та графіки."""
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict_proba(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        ys, ps = [], []
        for xb, yb in loader:
            probs = torch.sigmoid(self.model(xb.to(self.device))).cpu().numpy()
            ps.append(probs)
            ys.append(yb.numpy())
        return np.concatenate(ys).astype(int), np.concatenate(ps)

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = None
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "threshold": float(threshold),
            "n_samples": int(len(y_true)),
        }

    # ---------- графіки ----------
    @staticmethod
    def plot_training_history(history: dict, path: Path) -> None:
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epochs, history["train_loss"], label="train")
        axes[0].plot(epochs, history["val_loss"], label="val")
        axes[0].set_title("Функція втрат (BCE)")
        axes[0].set_xlabel("Епоха")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(epochs, history["train_acc"], label="train")
        axes[1].plot(epochs, history["val_acc"], label="val")
        axes[1].set_title("Точність (accuracy)")
        axes[1].set_xlabel("Епоха")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    @staticmethod
    def plot_confusion_matrix(cm, path: Path) -> None:
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        names = ["Легітимний", "Фішинг"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.set_xlabel("Передбачено")
        ax.set_ylabel("Фактично")
        ax.set_title("Матриця помилок")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    @staticmethod
    def plot_roc(y_true, y_prob, path: Path) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-крива")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
