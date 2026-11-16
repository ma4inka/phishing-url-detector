"""Тести: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishing_detector.data_generator import SyntheticURLGenerator  # noqa: E402
from phishing_detector.data_loader import DatasetLoader  # noqa: E402
from phishing_detector.features import URLFeatureAnalyzer  # noqa: E402
from phishing_detector.preprocessing import URLTokenizer  # noqa: E402


def test_tokenizer_normalization_and_shape():
    tok = URLTokenizer(max_len=20)
    assert tok.normalize("HTTPS://www.Example.com/Login") == "example.com/login"
    ids = tok.encode("https://example.com/login")
    assert ids.shape == (20,) and ids.dtype == np.int64
    assert ids[0] != URLTokenizer.PAD and ids[-1] == URLTokenizer.PAD


def test_tokenizer_truncates_and_handles_unicode():
    tok = URLTokenizer(max_len=10)
    assert (tok.encode("a" * 100) != URLTokenizer.PAD).all()
    assert URLTokenizer.UNK in tok.encode("http://пейпал.com")


def test_generator_is_balanced_and_unique():
    df = SyntheticURLGenerator(seed=1).generate(600)
    assert df["url"].is_unique
    assert set(df["label"].unique()) == {0, 1}
    assert abs(df["label"].mean() - 0.5) < 0.05


def test_generator_reproducible():
    a = SyntheticURLGenerator(seed=7).generate(200)
    b = SyntheticURLGenerator(seed=7).generate(200)
    assert a["url"].tolist() == b["url"].tolist()


def test_dataset_loader_label_mapping(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"URL": ["a.com", "b.com", "c.com"], "Label": ["good", "bad", "weird"]}).to_csv(p, index=False)
    df = DatasetLoader.load_csv(p)
    assert dict(zip(df["url"], df["label"])) == {"a.com": 0, "b.com": 1}
    inv = DatasetLoader.load_csv(p, invert_labels=True)
    assert dict(zip(inv["url"], inv["label"])) == {"a.com": 1, "b.com": 0}


def test_feature_analyzer_flags_ip_and_at():
    rows = {r["Ознака"]: r for r in URLFeatureAnalyzer().analyze("http://1.2.3.4/login@x")}
    assert rows["IP-адреса замість домену"]["Підозріло"] != "—"
    assert rows["Символ @ в URL"]["Підозріло"] != "—"


# ---------- тести, що потребують PyTorch ----------
torch = pytest.importorskip("torch")


def test_model_forward_shape_and_padding_invariance():
    from phishing_detector.model import PhishingNet

    tok = URLTokenizer(max_len=50)
    model = PhishingNet(tok.vocab_size).eval()
    x = torch.from_numpy(tok.encode_batch(["paypal.com.login.xyz/verify", "google.com", ""])).long()
    out = model(x)
    assert out.shape == (3,) and torch.isfinite(out).all()


def test_full_pipeline_trains_and_predicts(tmp_path):
    from phishing_detector.config import ModelConfig, TrainConfig
    from phishing_detector.pipeline import TrainingPipeline
    from phishing_detector.predictor import PhishingPredictor

    df = SyntheticURLGenerator(seed=3).generate(800)
    model_path = tmp_path / "m.pt"
    pipe = TrainingPipeline(ModelConfig(), TrainConfig(epochs=2, batch_size=64, max_len=80),
                            model_path=model_path, reports_dir=tmp_path / "reports")
    res = pipe.run(df)
    assert model_path.exists() and 0.0 <= res["metrics"]["accuracy"] <= 1.0

    predictor = PhishingPredictor.load(model_path)
    p = predictor.predict("http://paypal.com.secure-login.xyz/verify.php")
    assert 0.0 <= p["probability"] <= 1.0
    assert len(predictor.explain("http://example.com/a")) == len("example.com/a")
