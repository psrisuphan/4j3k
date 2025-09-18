"""Fine-tune WangchanBERTa on the Thai hate-speech dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABEL2ID: Dict[str, int] = {"nonhatespeech": 0, "hatespeech": 1}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Message", "Hatespeech"])
    df = df[df["Hatespeech"].isin(LABEL2ID)]
    df = df.assign(label=df["Hatespeech"].map(LABEL2ID))
    return df[["Message", "label"]]


def _split_dataset(
    df: pd.DataFrame, test_size: float, seed: int
) -> Tuple[Dataset, Dataset]:
    train_df, eval_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    return Dataset.from_pandas(train_df, preserve_index=False), Dataset.from_pandas(
        eval_df, preserve_index=False
    )


def _tokenize(tokenizer: AutoTokenizer, dataset: Dataset, max_length: int) -> Dataset:
    text_columns = [col for col in dataset.column_names if col not in {"label"}]
    return dataset.map(
        lambda examples: tokenizer(
            examples["Message"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
        remove_columns=text_columns,
    )


def _compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/HateThaiSent.csv"),
        help="Path to the labelled CSV dataset.",
    )
    parser.add_argument(
        "--model-name",
        default="airesearch/wangchanberta-base-att-spm-uncased",
        help="Backbone model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/wangchanberta-hatespeech"),
        help="Directory to store checkpoints.",
    )
    parser.add_argument(
        "--epochs", type=float, default=2.0, help="Number of fine-tuning epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size used during training.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Number of update accumulation steps to smooth GPU load.",
    )
    parser.add_argument(
        "--max-gpu-memory-fraction",
        type=float,
        default=1.0,
        help=(
            "Upper bound on fraction of a visible GPU's memory the trainer may allocate. "
            "Values <1.0 attempt to leave headroom to avoid pegging the device."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length; reducing this lowers CPU compute cost.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Only train the classification head (useful when CPU-bound).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Optional cap on PyTorch CPU threads for reproducibility/perf tuning.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=None,
        help="Override number of workers for data loading (defaults to min(4, cores)).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay applied during training.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(args.data)
    train_ds, eval_ds = _split_dataset(df, args.test_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_train = _tokenize(tokenizer, train_ds, args.max_length)
    tokenized_eval = _tokenize(tokenizer, eval_ds, args.max_length)

    dataset_dict = DatasetDict({"train": tokenized_train, "eval": tokenized_eval})

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if args.freeze_encoder:
        encoder = getattr(model, "base_model", None)
        if encoder is None and hasattr(model, "base_model_prefix"):
            encoder = getattr(model, model.base_model_prefix, None)
        if encoder is None:
            raise ValueError("Could not identify base encoder module to freeze.")
        for param in encoder.parameters():
            param.requires_grad = False

    dataloader_workers = (
        args.dataloader_workers
        if args.dataloader_workers is not None
        else max(1, min(4, os.cpu_count() or 1))
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, args.gradient_accumulation),
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        do_eval=True,
        dataloader_num_workers=dataloader_workers,
        torch_compile=False,
    )

    if torch.cuda.is_available():
        if 0.0 < args.max_gpu_memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_fraction)
    else:
        if args.cpu_threads:
            threads = max(1, min(args.cpu_threads, os.cpu_count() or 1))
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(threads)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["eval"],
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate(dataset_dict["eval"])
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    metrics_path = args.output_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(serializable_metrics, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
