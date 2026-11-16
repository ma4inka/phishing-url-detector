# 🎣 Виявлення фішингових вебсайтів за URL-адресами нейромережевими засобами

Курсовий проєкт. Нейромережа **Char-CNN + BiLSTM + Attention** (PyTorch) класифікує URL як фішинговий або легітимний,
аналізуючи його **посимвольно** (без ручного конструювання ознак). Веб-інтерфейс — **Streamlit**.

## Що вміє застосунок

* перевірка одного URL з ймовірністю фішингу та регульованим порогом;
* **пояснення рішення**: теплова карта символів (метод оклюзії) + таблиця інтерпретованих ознак;
* пакетна перевірка (CSV або список URL) з експортом результатів;
* вкладка з метриками (Accuracy, Precision, Recall, F1, ROC AUC), кривими навчання, матрицею помилок;
* навчання / перенавчання моделі прямо в інтерфейсі (синтетичні дані або власний CSV).

## Структура проєкту

```
phishing-url-detector/
├── app.py                         # Streamlit-застосунок
├── train.py                       # CLI для навчання
├── requirements.txt
├── README.md
├── phishing_detector/
│   ├── config.py                  # ModelConfig, TrainConfig, шляхи
│   ├── data_generator.py          # SyntheticURLGenerator — синтетичний датасет
│   ├── data_loader.py             # DatasetLoader — читання CSV / генерація
│   ├── preprocessing.py           # URLTokenizer — нормалізація та кодування символів
│   ├── dataset.py                 # URLDataset (torch.utils.data.Dataset)
│   ├── model.py                   # PhishingNet — нейромережа
│   ├── trainer.py                 # Trainer — цикл навчання, early stopping
│   ├── evaluator.py               # Evaluator — метрики та графіки
│   ├── features.py                # URLFeatureAnalyzer — ознаки для пояснення
│   ├── predictor.py               # PhishingPredictor — інференс і пояснення
│   └── pipeline.py                # TrainingPipeline — повний цикл
├── tests/test_pipeline.py
├── data/      models/      reports/   # створюються/наповнюються автоматично
```

## Архітектура моделі

```
URL → нормалізація (lowercase, без http(s):// та www.) → символьні індекси (довжина 150)
    → Embedding (48)
    → 3 паралельні Conv1D (ядра 3, 5, 7; по 64 фільтри) + BatchNorm
    → BiLSTM (64 × 2)
    → Attention-pooling ⊕ Max-pooling
    → Dense (64) → Dropout → 1 логіт → Sigmoid
```

Функція втрат — `BCEWithLogitsLoss` (з ваговим коефіцієнтом класу при дисбалансі), оптимізатор — `AdamW`,
`ReduceLROnPlateau`, gradient clipping, early stopping по val loss. Розбиття: 70 / 15 / 15 (train / val / test, стратифіковано).

---

## 🚀 Запуск (послідовні команди)

Потрібен **Python 3.10–3.12** та `pip`.

### Windows (PowerShell)

```powershell
cd phishing-url-detector
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

> Якщо PowerShell забороняє виконання скриптів:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` і повторіть активацію.

### Linux / macOS

```bash
cd phishing-url-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Після `streamlit run app.py` відкрийте http://localhost:8501.

> Крок `python train.py` можна пропустити: якщо моделі ще немає, застосунок запропонує навчити її
> кнопкою в інтерфейсі.

### Корисні команди

```bash
# Навчання з іншими параметрами
python train.py --samples 40000 --epochs 12 --batch-size 128 --max-len 150

# Навчання на власному датасеті (CSV зі стовпцями url, label)
python train.py --data my_dataset.csv

# Kaggle "Phishing Site URLs" (стовпці URL, Label зі значеннями good/bad)
python train.py --data phishing_site_urls.csv --url-col URL --label-col Label

# Датасет, де 1 = легітимний (наприклад PhiUSIIL) — інвертувати мітки
python train.py --data PhiUSIIL_Phishing_URL_Dataset.csv --url-col URL --label-col label --invert-labels

# Тести
pytest -q

# Запуск на іншому порту / доступ з мережі
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

Мітки: `1` / `phishing` / `bad` — фішинг; `0` / `legitimate` / `good` — легітимний.

---

## ☁️ Розгортання на Streamlit Community Cloud

1. Навчіть модель локально: `python train.py` (файл `models/phishing_model.pt` невеликий, його можна закомітити).
2. Створіть репозиторій на GitHub і завантажте проєкт:
   ```bash
   git init
   git add .
   git commit -m "Phishing URL detector"
   git branch -M main
   git remote add origin https://github.com/<ваш-логін>/phishing-url-detector.git
   git push -u origin main
   ```
3. На https://share.streamlit.io натисніть **New app**, оберіть репозиторій, гілку `main`, файл `app.py` → **Deploy**.
4. Якщо `models/phishing_model.pt` не закомічено, застосунок запропонує навчити модель при першому запуску.

## 📚 Дані: важливе зауваження

За замовчуванням використовується **синтетичний датасет** (`SyntheticURLGenerator`), щоб проєкт працював одразу,
без завантаження файлів. Він містить типові фішингові прийоми (бренд у піддомені, typosquatting і гомогліфи,
IP замість домену, `@` в URL, безкоштовні хостинги, підозрілі TLD, «ключові слова») та «важкі» приклади в обох класах.
Метрики на синтетичних даних **завищені** порівняно з реальними — для розділу «Експерименти» курсової
навчіть модель на реальних джерелах:

* фішингові URL: PhishTank, OpenPhish;
* легітимні URL: Tranco list, Alexa/Majestic Million;
* готові набори: Kaggle «Phishing Site URLs», UCI/Kaggle PhiUSIIL, ISCX-URL-2016.

Рекомендується прибрати дублікати та розбивати вибірки так, щоб один домен не потрапляв і в train, і в test.

## 🧪 Ідеї для розділів курсової роботи

1. **Актуальність і постановка задачі:** фішинг, бінарна класифікація URL.
2. **Огляд методів:** чорні списки; ручні ознаки + класичний ML; глибоке навчання (Char-CNN, RNN, URLNet, трансформери).
3. **Обґрунтування архітектури:** символьні моделі не залежать від словника та стійкі до нових доменів;
   CNN ловить локальні n-грами, BiLSTM — послідовні залежності, attention — важливі фрагменти.
4. **Експерименти:** порівняння з абляціями (лише CNN, лише BiLSTM, без attention), вплив `max_len`, розміру ембеддингу,
   dropout; аналіз помилок (FP/FN).
5. **Метрики:** Accuracy, Precision, Recall, F1, ROC AUC; для фішингу особливо важливий Recall.
6. **Інтерпретація:** карта важливості символів (оклюзія) у застосунку.
7. **Обмеження:** аналіз лише тексту URL, дрейф даних, adversarial-приклади, скорочувачі посилань.

Орієнтири для огляду літератури (перевірте вихідні дані самостійно): Le et al., *URLNet: Learning a URL Representation
with Deep Learning for Malicious URL Detection* (2018); Bahnsen et al., *Classifying Phishing URLs Using Recurrent Neural
Networks* (2017); Saxe & Berlin, *eXpose* (2017).

## ⚠️ Застереження

Інструмент навчальний. Ймовірнісна оцінка моделі не гарантує безпеку сайту; для реального захисту поєднуйте її з
чорними списками, аналізом вмісту сторінки та перевіркою сертифікатів.
