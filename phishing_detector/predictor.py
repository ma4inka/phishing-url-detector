"""Клас PhishingPredictor: завантаження моделі та інференс."""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import MODEL_PATH
from .model import PhishingNet
from .preprocessing import URLTokenizer


class PhishingPredictor:
    def __init__(self, model: PhishingNet, tokenizer: URLTokenizer, threshold: float = 0.5,
                 metadata: Optional[dict] = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "PhishingPredictor":
        ckpt = torch.load(str(path), map_location="cpu")
        tokenizer = URLTokenizer(max_len=ckpt["max_len"])
        cfg = dict(ckpt["model_config"])
        cfg["kernel_sizes"] = tuple(cfg["kernel_sizes"])
        model = PhishingNet(vocab_size=tokenizer.vocab_size, **cfg)
        model.load_state_dict(ckpt["model_state"])
        meta = {k: ckpt.get(k) for k in ("metrics", "history", "dataset_info", "trained_at", "model_config", "max_len")}
        return cls(model, tokenizer, threshold=ckpt.get("threshold", 0.5), metadata=meta)

    @torch.no_grad()
    def _proba_ids(self, ids: np.ndarray, batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(ids), batch_size):
            xb = torch.from_numpy(ids[i:i + batch_size]).long().to(self.device)
            out.append(torch.sigmoid(self.model(xb)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict_proba(self, urls: Sequence[str]) -> np.ndarray:
        """Ймовірності фішингу для списку URL."""
        return self._proba_ids(self.tokenizer.encode_batch(urls))

    def predict(self, url: str, threshold: Optional[float] = None) -> Dict:
        thr = self.threshold if threshold is None else threshold
        prob = float(self.predict_proba([url])[0])
        return {"url": url, "probability": prob, "is_phishing": prob >= thr, "threshold": thr}

    def explain(self, url: str, window: int = 3) -> List[Tuple[str, float]]:
        """Важливість кожного символу методом оклюзії.

        Для кожної позиції i символи в вікні навколо i замінюються на UNK, а
        важливість = p(фішинг) до заміни − p після. Додатне значення означає,
        що символ підвищує підозру моделі, від'ємне — знижує.
        """
        chars = self.tokenizer.tokens(url)
        n = len(chars)
        if n == 0:
            return []
        ids = self.tokenizer.encode(url)
        base = float(self._proba_ids(ids[None, :])[0])
        half = window // 2
        variants = np.tile(ids, (n, 1))
        for i in range(n):
            variants[i, max(0, i - half): min(n, i + half + 1)] = URLTokenizer.UNK
        probs = self._proba_ids(variants)
        return [(chars[i], float(base - probs[i])) for i in range(n)]
