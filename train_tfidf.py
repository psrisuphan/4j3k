"""Train a lightweight TF-IDF + linear classifier for Thai hate-speech detection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABEL2ID: Dict[str, int] = {"nonhatespeech": 0, "hatespeech": 1}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/HateThaiSent.csv"),
        help="Path to the labelled CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tfidf-linear"),
        help="Directory to store the trained pipeline.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Holdout size fraction for evaluation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=200000,
        help="Limit on TF-IDF vocabulary size (None for unlimited).",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n-gram length; char n-grams capture Thai tokens well.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=5,
        help="Maximum n-gram length used in TF-IDF features.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-5,
        help="Regularisation strength for the SGD linear classifier.",
    )
    parser.add_argument(
        "--loss",
        default="log_loss",
        choices=("log_loss", "hinge"),
        help="Classification loss; log_loss -> logistic regression, hinge -> linear SVM.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallelism for the SGD classifier (-1 = use all cores).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max SGD epochs over the training data.",
    )
    return parser.parse_args()


def _load_dataframe(csv_path: Path) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Message", "Hatespeech"])
    df = df[df["Hatespeech"].isin(LABEL2ID)]
    X = df["Message"].astype(str)
    y = df["Hatespeech"].map(LABEL2ID)
    return X, y


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(args.ngram_min, args.ngram_max),
        lowercase=False,
        max_features=None if args.max_features <= 0 else args.max_features,
        sublinear_tf=True,
    )
    classifier = SGDClassifier(
        loss=args.loss,
        alpha=args.alpha,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
        class_weight="balanced",
        random_state=args.seed,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = _load_dataframe(args.data)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    pipeline = _build_pipeline(args)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_eval)
    accuracy = accuracy_score(y_eval, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_eval, y_pred, average="binary", zero_division=0
    )

    joblib.dump(pipeline, args.output_dir / "model.joblib")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classes": ID2LABEL,
    }
    with (args.output_dir / "eval_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
