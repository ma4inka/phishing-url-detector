"""Попередня обробка URL: нормалізація та символьна токенізація."""
import re
import string
from typing import Iterable, List

import numpy as np


class URLTokenizer:
    """Символьний токенізатор URL.

    Кожен символ нормалізованого URL перетворюється на ціле число.
    0 — PAD (доповнення), 1 — UNK (невідомий символ, напр. юнікод).
    Схема (http/https) та префікс www. відкидаються, а довгі URL
    обрізаються справа (початок URL з доменом найінформативніший).
    """

    PAD = 0
    UNK = 1
    CHARSET = string.ascii_lowercase + string.digits + "-._~:/?#[]@!$&'()*+,;=%"

    def __init__(self, max_len: int = 150):
        self.max_len = int(max_len)
        self.char2idx = {c: i + 2 for i, c in enumerate(self.CHARSET)}

    @property
    def vocab_size(self) -> int:
        return len(self.CHARSET) + 2

    @staticmethod
    def normalize(url: str) -> str:
        url = str(url).strip().lower()
        url = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", url)
        url = re.sub(r"^www\d*\.", "", url)
        return url

    def tokens(self, url: str) -> List[str]:
        """Символи нормалізованого URL після обрізання до max_len."""
        return list(self.normalize(url)[: self.max_len])

    def encode(self, url: str) -> np.ndarray:
        ids = [self.char2idx.get(c, self.UNK) for c in self.tokens(url)]
        out = np.full(self.max_len, self.PAD, dtype=np.int64)
        out[: len(ids)] = ids
        return out

    def encode_batch(self, urls: Iterable[str]) -> np.ndarray:
        rows = [self.encode(u) for u in urls]
        if not rows:
            return np.zeros((0, self.max_len), dtype=np.int64)
        return np.stack(rows)

    def to_config(self) -> dict:
        return {"max_len": self.max_len}

    @classmethod
    def from_config(cls, cfg: dict) -> "URLTokenizer":
        return cls(max_len=cfg["max_len"])
