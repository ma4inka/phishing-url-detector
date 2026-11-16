"""Streamlit-застосунок: виявлення фішингових URL нейромережею."""
import html

import pandas as pd
import streamlit as st

from phishing_detector.config import MODEL_PATH, REPORTS_DIR, ModelConfig, TrainConfig
from phishing_detector.data_loader import DatasetLoader
from phishing_detector.features import URLFeatureAnalyzer
from phishing_detector.pipeline import TrainingPipeline
from phishing_detector.predictor import PhishingPredictor

st.set_page_config(page_title="Детектор фішингових URL", page_icon="🎣", layout="wide")

PLACEHOLDER = "— оберіть приклад —"
EXAMPLES = {
    "Легітимний: Wikipedia": "https://www.wikipedia.org/wiki/Neural_network",
    "Легітимний: GitHub": "https://github.com/pytorch/pytorch",
    "Легітимний: Google Accounts": "https://accounts.google.com/signin/v2/identifier",
    "Підозрілий: бренд у піддомені": "http://paypal.com.secure-login-verify.xyz/update-account.php?id=83af92c1",
    "Підозрілий: typosquatting": "http://paypa1-account-verify.top/login.php",
    "Підозрілий: IP-адреса": "http://192.168.44.12:8080/wp-includes/secure/login.php",
    "Підозрілий: символ @": "https://google.com@free-gift-claim.tk/auth",
    "Підозрілий: безкоштовний хостинг": "https://monobank-security-x7k2q.weebly.com/signin",
}


@st.cache_resource(show_spinner=False)
def load_predictor(mtime: float) -> PhishingPredictor:
    return PhishingPredictor.load(MODEL_PATH)


def get_predictor():
    if not MODEL_PATH.exists():
        return None
    return load_predictor(MODEL_PATH.stat().st_mtime)


def training_form(key: str):
    """Форма навчання/перенавчання моделі прямо в застосунку."""
    with st.form(key):
        source = st.radio("Джерело даних", ["Синтетичний датасет", "Власний CSV"], horizontal=True)
        n_samples = st.number_input("Розмір синтетичного датасету", 2000, 200000, 20000, step=2000)
        epochs = st.slider("Кількість епох", 2, 30, 8)
        csv_file = st.file_uploader("CSV зі стовпцями url та label", type=["csv"])
        invert = st.checkbox("Інвертувати мітки (у CSV 1 = легітимний)")
        submitted = st.form_submit_button("🚀 Навчити модель")

    if not submitted:
        return
    try:
        if source == "Власний CSV":
            if csv_file is None:
                st.error("Завантажте CSV-файл.")
                return
            df = DatasetLoader.load_csv(csv_file, invert_labels=invert)
        else:
            df = DatasetLoader.synthetic(int(n_samples))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Помилка підготовки даних: {exc}")
        return

    bar = st.progress(0.0, text="Старт…")

    def cb(frac: float, msg: str):
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    pipeline = TrainingPipeline(ModelConfig(), TrainConfig(epochs=int(epochs)))
    pipeline.run(df, progress_callback=cb)
    st.session_state["just_trained"] = True
    st.rerun()


def importance_html(pairs) -> str:
    """Теплова карта символів: червоний — підвищує підозру, зелений — знижує."""
    if not pairs:
        return ""
    max_abs = max(abs(v) for _, v in pairs) or 1.0
    spans = []
    for ch, v in pairs:
        alpha = min(1.0, abs(v) / max_abs) * 0.85
        color = f"rgba(220,53,69,{alpha:.2f})" if v > 0 else f"rgba(40,167,69,{alpha:.2f})"
        spans.append(f'<span style="background-color:{color};padding:2px 0;">{html.escape(ch)}</span>')
    return ('<div style="font-family:monospace;font-size:1.15rem;word-break:break-all;line-height:2;">'
            + "".join(spans) + "</div>")


# ------------------------------------------------------------------ UI
st.title("🎣 Виявлення фішингових вебсайтів за URL")
st.caption("Нейромережа Char-CNN + BiLSTM + Attention аналізує URL посимвольно, без ручних ознак.")

predictor = get_predictor()

if st.session_state.pop("just_trained", False):
    st.success("Модель успішно навчено та збережено!")

if predictor is None:
    st.warning("Навчену модель не знайдено. Навчіть її (займе кілька хвилин на CPU) або виконайте `python train.py`.")
    training_form("train_first")
    st.stop()

with st.sidebar:
    st.header("⚙️ Налаштування")
    threshold = st.slider("Поріг класифікації", 0.05, 0.95, float(predictor.threshold), 0.05,
                          help="Якщо ймовірність фішингу ≥ порогу, URL вважається фішинговим.")
    st.caption("Менший поріг → вища повнота (менше пропущених фішингових сайтів), але більше хибних тривог.")
    n_par = sum(p.numel() for p in predictor.model.parameters())
    st.metric("Параметрів моделі", f"{n_par:,}")
    with st.expander("🔁 Перенавчити модель"):
        training_form("retrain")

tab_check, tab_batch, tab_metrics, tab_about = st.tabs(
    ["🔍 Перевірка URL", "📋 Пакетна перевірка", "📊 Якість моделі", "📘 Про метод"]
)

# ---------------------------------------------------------- перевірка одного URL
with tab_check:
    def _fill_example():
        choice = st.session_state.get("example_select")
        if choice and choice != PLACEHOLDER:
            st.session_state["url_input"] = EXAMPLES[choice]

    st.selectbox("Приклади", [PLACEHOLDER] + list(EXAMPLES), key="example_select", on_change=_fill_example)
    url = st.text_input("Введіть URL для перевірки", key="url_input", placeholder="https://example.com/login")

    if st.button("Перевірити", type="primary") and url.strip():
        res = predictor.predict(url, threshold=threshold)
        prob = res["probability"]
        c1, c2 = st.columns([1, 2])
        with c1:
            if res["is_phishing"]:
                st.error("🚨 Ймовірно ФІШИНГ")
            else:
                st.success("✅ Схоже на легітимний URL")
            st.metric("Ймовірність фішингу", f"{prob:.1%}")
            st.progress(prob)
        with c2:
            st.markdown("**Внесок символів у рішення моделі** (метод оклюзії; "
                        "🟥 підвищує підозру, 🟩 знижує)")
            st.markdown(importance_html(predictor.explain(url)), unsafe_allow_html=True)

        st.markdown("**Інтерпретовані ознаки URL** (довідково; модель їх не використовує)")
        st.dataframe(pd.DataFrame(URLFeatureAnalyzer().analyze(url)), hide_index=True)

# ---------------------------------------------------------- пакетна перевірка
with tab_batch:
    st.write("Завантажте CSV зі стовпцем `url` або вставте список URL (по одному в рядок).")
    up = st.file_uploader("CSV-файл", type=["csv"], key="batch_csv")
    text = st.text_area("або список URL", height=150, key="batch_text")

    urls = []
    if up is not None:
        try:
            bdf = pd.read_csv(up)
            col = next((c for c in bdf.columns if c.lower().strip() in ("url", "urls", "link")), bdf.columns[0])
            urls = bdf[col].astype(str).tolist()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не вдалося прочитати CSV: {exc}")
    elif text.strip():
        urls = [u.strip() for u in text.splitlines() if u.strip()]

    if urls and st.button("Перевірити всі", type="primary"):
        urls = urls[:20000]
        probs = predictor.predict_proba(urls)
        out = pd.DataFrame({
            "URL": urls,
            "Ймовірність фішингу": probs.round(4),
            "Вердикт": ["Фішинг" if p >= threshold else "Легітимний" for p in probs],
        })
        st.metric("Знайдено фішингових", f"{int((probs >= threshold).sum())} із {len(urls)}")
        st.dataframe(out, hide_index=True,
                     column_config={"Ймовірність фішингу": st.column_config.ProgressColumn(
                         "Ймовірність фішингу", min_value=0.0, max_value=1.0, format="%.3f")})
        st.download_button("⬇️ Завантажити результати (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")

# ---------------------------------------------------------- метрики
with tab_metrics:
    meta = predictor.metadata
    m = meta.get("metrics") or {}
    if m:
        st.subheader("Метрики на тестовій вибірці")
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{m['accuracy']:.4f}")
        cols[1].metric("Precision", f"{m['precision']:.4f}")
        cols[2].metric("Recall", f"{m['recall']:.4f}")
        cols[3].metric("F1", f"{m['f1']:.4f}")
        cols[4].metric("ROC AUC", f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "—")
        info = meta.get("dataset_info") or {}
        if info:
            st.caption(f"Датасет: {info['total']} URL (train {info['train']} / val {info['val']} / "
                       f"test {info['test']}), частка фішингу {info['phishing_share']:.1%}. "
                       f"Навчено: {meta.get('trained_at', '—')}")
    img_cols = st.columns(3)
    for col, (fname, cap) in zip(img_cols, [("training_history.png", "Криві навчання"),
                                            ("confusion_matrix.png", "Матриця помилок"),
                                            ("roc_curve.png", "ROC-крива")]):
        path = REPORTS_DIR / fname
        if path.exists():
            col.image(str(path), caption=cap)

# ---------------------------------------------------------- про метод
with tab_about:
    st.markdown("""
### Архітектура
```
URL → нормалізація → символьні індекси → Embedding
    → 3 паралельні Conv1D (ядра 3, 5, 7) + BatchNorm
    → BiLSTM
    → Attention-pooling ⊕ Max-pooling
    → Dense → Sigmoid (ймовірність фішингу)
```
* **Char-CNN** виділяє локальні символьні n-грами (`paypa`, `-login`, `.com.`, `xn--`).
* **BiLSTM** враховує порядок та контекст частин URL (домен, шлях, параметри).
* **Attention + Max-pooling** агрегують інформацію з усієї послідовності.
* **Пояснення** будується методом оклюзії: символи по черзі маскуються, і вимірюється зміна ймовірності.

### Обмеження
Модель аналізує лише текст URL. Вона не бачить вміст сторінки, вік домену, TLS-сертифікат тощо.
Результат — ймовірнісна оцінка, а не гарантія безпеки; для production її слід комбінувати з іншими джерелами
(чорні списки, аналіз сторінки). Якість залежить від навчального датасету — для достовірних результатів
використовуйте реальні дані (PhishTank, OpenPhish, Tranco тощо).
""")
