"""Завантаження та підготовка датасетів URL."""
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DEFAULT_DATASET_PATH
from .data_generator import SyntheticURLGenerator

POSITIVE_LABELS = {"1", "phishing", "phish", "bad", "malicious", "spam"}
NEGATIVE_LABELS = {"0", "legitimate", "legit", "good", "benign", "safe"}

URL_COLUMN_CANDIDATES = ("url", "urls", "link", "website")
LABEL_COLUMN_CANDIDATES = ("label", "labels", "class", "target", "status", "result", "type")


def _map_label(value) -> Optional[int]:
    s = str(value).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    if s in POSITIVE_LABELS:
        return 1
    if s in NEGATIVE_LABELS:
        return 0
    return None


class DatasetLoader:
    """Читає CSV (url + label) або створює синтетичний датасет."""

    @staticmethod
    def load_csv(path, url_col: Optional[str] = None, label_col: Optional[str] = None,
                 invert_labels: bool = False) -> pd.DataFrame:
        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}
        url_col = url_col or next((cols[c] for c in URL_COLUMN_CANDIDATES if c in cols), None)
        label_col = label_col or next((cols[c] for c in LABEL_COLUMN_CANDIDATES if c in cols), None)
        if url_col is None or label_col is None:
            raise ValueError(
                "Не вдалося визначити стовпці з URL та міткою. "
                f"Наявні стовпці: {list(df.columns)}. Вкажіть --url-col і --label-col."
            )
        out = pd.DataFrame({
            "url": df[url_col].astype(str).str.strip(),
            "label": df[label_col].map(_map_label),
        })
        out = out.dropna(subset=["label"])
        out["label"] = out["label"].astype(int)
        if invert_labels:
            out["label"] = 1 - out["label"]
        out = out[out["url"].str.len() > 0].drop_duplicates("url").reset_index(drop=True)
        if out["label"].nunique() < 2:
            raise ValueError("У датасеті має бути обидва класи (0 і 1).")
        return out

    @staticmethod
    def synthetic(n_samples: int = 20000, seed: int = 42, save_path: Optional[Path] = DEFAULT_DATASET_PATH) -> pd.DataFrame:
        df = SyntheticURLGenerator(seed=seed).generate(n_samples)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
        return df
