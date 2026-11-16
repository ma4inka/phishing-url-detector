"""CLI для навчання моделі.

Приклади:
    python train.py                              # синтетичний датасет, налаштування за замовчуванням
    python train.py --samples 40000 --epochs 12  # більший синтетичний датасет
    python train.py --data my_urls.csv           # власний CSV (стовпці url, label)
    python train.py --data kaggle.csv --url-col URL --label-col Label
"""
import argparse

from phishing_detector.config import DEFAULT_DATASET_PATH, MODEL_PATH, ModelConfig, TrainConfig
from phishing_detector.data_loader import DatasetLoader
from phishing_detector.pipeline import TrainingPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Навчання Char-CNN+BiLSTM+Attention для виявлення фішингових URL")
    p.add_argument("--data", type=str, default=None, help="шлях до CSV; якщо не вказано — генерується синтетичний датасет")
    p.add_argument("--url-col", type=str, default=None, help="назва стовпця з URL")
    p.add_argument("--label-col", type=str, default=None, help="назва стовпця з міткою")
    p.add_argument("--invert-labels", action="store_true", help="інвертувати мітки (якщо в датасеті 1 = легітимний)")
    p.add_argument("--samples", type=int, default=20000, help="розмір синтетичного датасету")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--max-len", type=int, default=TrainConfig.max_len)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--out", type=str, default=str(MODEL_PATH), help="куди зберегти модель")
    return p.parse_args()


def main():
    args = parse_args()

    if args.data:
        print(f"Завантаження датасету: {args.data}")
        df = DatasetLoader.load_csv(args.data, args.url_col, args.label_col, args.invert_labels)
    else:
        print(f"Генерація синтетичного датасету ({args.samples} URL) → {DEFAULT_DATASET_PATH}")
        df = DatasetLoader.synthetic(args.samples, seed=args.seed)
    print(f"Розмір датасету: {len(df)} | частка фішингу: {df['label'].mean():.2%}")

    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len,
                            lr=args.lr, seed=args.seed)
    pipeline = TrainingPipeline(ModelConfig(), train_cfg, model_path=args.out)
    result = pipeline.run(df)
    print(f"\nМодель збережено: {result['model_path']}")
    print("Графіки та metrics.json збережено в папці reports/")
    print("Запуск застосунку:  streamlit run app.py")


if __name__ == "__main__":
    main()
