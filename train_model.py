"""Fine-tune WangchanBERTa on the Thai hate-speech dataset."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import requests
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import zipfile
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

LABEL2ID: Dict[str, int] = {"nonhatespeech": 0, "hatespeech": 1}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _is_rocm_runtime() -> bool:
    """Return True when PyTorch is built with ROCm/HIP support."""

    torch_version = getattr(torch, "version", None)
    hip_version = getattr(torch_version, "hip", None) if torch_version is not None else None
    return bool(hip_version)


def _resolve_target_device(requested: str) -> str:
    """Resolve the execution device, preferring accelerators when available."""

    choice = requested.lower()
    if choice == "cpu":
        return "cpu"
    if choice in {"cuda", "gpu", "rocm", "hip"}:
        if torch.cuda.is_available():
            return "cuda"
        if _is_rocm_runtime():
            raise RuntimeError(
                "ROCm runtime detected but no GPU is visible. Check your AMD drivers and permissions."
            )
        raise RuntimeError("CUDA/ROCm requested but no compatible GPU is available.")
    if choice == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS requested but no Apple GPU backend is available.")
    if choice != "auto":
        raise ValueError(f"Unsupported device selection '{requested}'.")

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if _is_rocm_runtime():
        # hip runtime present but no visible device – fall back to CPU to stay safe
        return "cpu"
    return "cpu"


def _configure_trainable_layers(
    model: AutoModelForSequenceClassification, layer_count: int
) -> None:
    """Freeze or partially unfreeze encoder layers based on *layer_count*."""

    if layer_count < 0:
        return

    encoder = getattr(model, "base_model", None)
    if encoder is None:
        prefix = getattr(model, "base_model_prefix", None)
        if prefix:
            encoder = getattr(model, prefix, None)
    if encoder is None:
        raise ValueError("Could not identify the base encoder module to adjust trainable layers.")

    for param in encoder.parameters():
        param.requires_grad = False

    if layer_count == 0:
        return

    encoder_module = getattr(encoder, "encoder", None)
    layers = getattr(encoder_module, "layer", None) if encoder_module is not None else None
    if layers is None:
        raise ValueError("Encoder module does not expose an iterable 'layer' attribute.")

    layers = list(layers)
    if layer_count > len(layers):
        layer_count = len(layers)

    for layer in layers[-layer_count:]:
        for param in layer.parameters():
            param.requires_grad = True

def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Message", "Hatespeech"])
    df = df[df["Hatespeech"].isin(LABEL2ID)]
    df = df.assign(label=df["Hatespeech"].map(LABEL2ID))
    return df[["Message", "label"]]


def _load_thai_toxicity_dataset(split: str) -> Dataset:
    """Download and load the Thai Toxicity Tweet dataset without relying on HF scripts."""

    if split != "train":
        raise ValueError(
            "SEACrowd/thai_toxicity_tweet only provides a 'train' split; "
            f"received '{split}'."
        )

    zip_url = "https://archive.org/download/ThaiToxicityTweetCorpus/data.zip"
    try:
        response = requests.get(zip_url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Failed to download Thai Toxicity Tweet corpus from archive.org. "
            "Check your network connection or specify --no-hf-dataset to skip it."
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            with zf.open("data/train.jsonl") as fh:
                df = pd.read_json(fh, lines=True)
    except (KeyError, zipfile.BadZipFile, ValueError) as exc:
        raise RuntimeError("Downloaded Thai toxicity archive is corrupted or has unexpected format.") from exc

    return Dataset.from_pandas(df, preserve_index=False)


def _load_hf_dataframe(
    dataset_id: str,
    text_column: str,
    label_column: str,
    dataset_config: str | None,
    dataset_split: str,
) -> pd.DataFrame:
    """Load and normalize a Hugging Face dataset to match the CSV schema."""

    dataset_kwargs: Dict[str, Any] = {}
    if dataset_config:
        dataset_kwargs["name"] = dataset_config

    dataset_normalized = dataset_id.lower()
    if dataset_normalized == "seacrowd/thai_toxicity_tweet":
        raw_dataset = _load_thai_toxicity_dataset(dataset_split)
    elif dataset_normalized == "tmu-nlp/thai_toxicity_tweet":
        raw_dataset = _load_thai_toxicity_dataset(dataset_split)
    else:
        raw_dataset = cast(
            Dataset,
            load_dataset(
                dataset_id,
                split=dataset_split,
                **dataset_kwargs,
            ),
        )

    if not isinstance(text_column, str):  # defensive guard for static type checkers
        raise TypeError("hf_text_column must resolve to a string value")
    if text_column not in raw_dataset.column_names:
        raise ValueError(
            f"Text column '{text_column}' not found in dataset '{dataset_id}'. Available columns: {raw_dataset.column_names}"
        )
    if not isinstance(label_column, str):
        raise TypeError("hf_label_column must resolve to a string value")
    if label_column not in raw_dataset.column_names:
        raise ValueError(
            f"Label column '{label_column}' not found in dataset '{dataset_id}'. Available columns: {raw_dataset.column_names}"
        )

    df = cast(pd.DataFrame, raw_dataset.to_pandas())
    df = df[[text_column, label_column]].copy()
    df = df.rename(columns={text_column: "Message", label_column: "label"})
    df = df.dropna(subset=["Message", "label"])

    df["Message"] = df["Message"].astype(str).str.strip()
    df = df[df["Message"].str.len() > 0]
    df = df[df["Message"].str.upper() != "TWEET_NOT_FOUND"]

    label_feature = raw_dataset.features[label_column]

    alias_map = {
        "toxic": LABEL2ID["hatespeech"],
        "1": LABEL2ID["hatespeech"],
        "non-toxic": LABEL2ID["nonhatespeech"],
        "nontoxic": LABEL2ID["nonhatespeech"],
        "0": LABEL2ID["nonhatespeech"],
    }

    def _coerce_label(value: Any) -> int | None:
        if isinstance(label_feature, ClassLabel):
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            if isinstance(value, (int, np.integer)):
                names = label_feature.names
                if 0 <= int(value) < len(names):
                    lookup = names[int(value)]
                    mapped = LABEL2ID.get(str(lookup))
                    if mapped is not None:
                        return mapped
                if int(value) in LABEL2ID.values():
                    return int(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in LABEL2ID:
                    return LABEL2ID[lowered]
                if lowered in alias_map:
                    return alias_map[lowered]
        else:
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in LABEL2ID:
                    return LABEL2ID[lowered]
                if lowered in alias_map:
                    return alias_map[lowered]
                if lowered.isdigit():
                    value = int(lowered)
            if isinstance(value, (int, np.integer)):
                if int(value) in LABEL2ID.values():
                    return int(value)
                if int(value) in (0, 1):
                    return int(value)
        return None

    df["label"] = df["label"].map(_coerce_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df[["Message", "label"]]


def _split_dataset(
    df: pd.DataFrame, test_size: float, seed: int
) -> Tuple[Dataset, Dataset]:
    label_names = [ID2LABEL[idx] for idx in range(len(ID2LABEL))]
    features = Features(
        {
            "Message": Value("string"),
            "label": ClassLabel(num_classes=len(label_names), names=label_names),
        }
    )
    base_dataset = Dataset.from_pandas(df, preserve_index=False, features=features)
    split_dataset = base_dataset.train_test_split(
        test_size=test_size,
        seed=seed,
        stratify_by_column="label",
    )
    return split_dataset["train"], split_dataset["test"]


def _tokenize(tokenizer: PreTrainedTokenizerBase, dataset: Dataset, max_length: int) -> Dataset:
    text_columns = [col for col in dataset.column_names if col not in {"label"}]
    def _batch_tokenize(examples: Dict[str, List[str]]):
        return tokenizer(
            examples["Message"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        _batch_tokenize,
        batched=True,
        remove_columns=text_columns,
        load_from_cache_file=True,
        desc="Tokenizing dataset",
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
        "--hf-dataset-id",
        default="SEACrowd/thai_toxicity_tweet",
        help=(
            "Hugging Face dataset identifier to append to the CSV data (defaults to "
            "'SEACrowd/thai_toxicity_tweet')."
        ),
    )
    parser.add_argument(
        "--hf-dataset-config",
        default=None,
        help="Optional dataset configuration name used with --hf-dataset-id.",
    )
    parser.add_argument(
        "--hf-dataset-split",
        default="train",
        help="Hugging Face dataset split to load when --hf-dataset-id is set.",
    )
    parser.add_argument(
        "--hf-text-column",
        default=None,
        help=(
            "Text column in the Hugging Face dataset (defaults to 'tweet_text' when "
            "using SEACrowd/thai_toxicity_tweet)."
        ),
    )
    parser.add_argument(
        "--hf-label-column",
        default=None,
        help=(
            "Label column in the Hugging Face dataset (defaults to 'is_toxic' when "
            "using SEACrowd/thai_toxicity_tweet)."
        ),
    )
    parser.add_argument(
        "--no-hf-dataset",
        action="store_true",
        help="Disable loading the Hugging Face dataset (use CSV-only training).",
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
        "--trainable-layer-count",
        type=int,
        default=-1,
        help=(
            "Number of encoder transformer blocks to fine-tune. "
            "-1 trains the entire encoder, 0 freezes it, positive values unfreeze only the top-N layers."
        ),
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
        "--device",
        choices=("auto", "cpu", "cuda", "mps", "rocm", "hip", "gpu"),
        default="auto",
        help="Preferred execution device; defaults to automatic accelerator detection.",
    )
    parser.add_argument(
        "--hip-gfx-override",
        default=None,
        help="Optional HSA_OVERRIDE_GFX_VERSION value for older AMD GPUs when using ROCm.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=(
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ),
        default="linear",
        help="Scheduler applied to the optimizer learning rate.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of warmup steps for the scheduler (overrides warmup ratio when >0).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Warmup proportion of total training steps when warmup steps is 0.",
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
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision (fp16) training when the hardware supports it.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable mixed precision (bf16) training on supported hardware.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Use gradient checkpointing to trade compute for lower memory usage.",
    )
    parser.add_argument(
        "--group-by-length",
        action="store_true",
        help="Group batches by sequence length to reduce padding overhead.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for potential speedups (requires PyTorch 2.0+).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("fp16 and bf16 modes are mutually exclusive; choose only one.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataframes: List[pd.DataFrame] = []

    hf_dataset_id = None if args.no_hf_dataset else (args.hf_dataset_id or None)

    if args.data is not None:
        if args.data.exists():
            dataframes.append(_load_dataframe(args.data))
        elif not hf_dataset_id:
            raise FileNotFoundError(f"CSV dataset '{args.data}' not found.")
        else:
            print(f"Warning: CSV dataset '{args.data}' not found; continuing without it.")

    if hf_dataset_id:
        text_column = args.hf_text_column
        label_column = args.hf_label_column
        if hf_dataset_id.lower() == "seacrowd/thai_toxicity_tweet":
            text_column = text_column or "tweet_text"
            label_column = label_column or "is_toxic"
        missing_fields = []
        if text_column is None:
            missing_fields.append("--hf-text-column")
        if label_column is None:
            missing_fields.append("--hf-label-column")
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(
                f"Missing required argument(s) {missing} for dataset '{hf_dataset_id}'."
            )

        hf_df = _load_hf_dataframe(
            dataset_id=hf_dataset_id,
            text_column=text_column,
            label_column=label_column,
            dataset_config=args.hf_dataset_config,
            dataset_split=args.hf_dataset_split,
        )
        dataframes.append(hf_df)

    if not dataframes:
        raise ValueError(
            "No training data sources were provided. Supply --data and/or --hf-dataset-id."
        )

    df = pd.concat(dataframes, ignore_index=True)
    df = df.drop_duplicates(subset=["Message", "label"]).reset_index(drop=True)

    train_ds, eval_ds = _split_dataset(df, args.test_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_train = _tokenize(tokenizer, train_ds, args.max_length)
    tokenized_eval = _tokenize(tokenizer, eval_ds, args.max_length)

    dataset_dict = DatasetDict({"train": tokenized_train, "eval": tokenized_eval})
    train_dataset = cast(Any, dataset_dict["train"])  # appease static type checkers
    eval_dataset = cast(Any, dataset_dict["eval"])  # appease static type checkers

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    effective_layer_count = args.trainable_layer_count
    if args.freeze_encoder:
        effective_layer_count = 0
    _configure_trainable_layers(model, effective_layer_count)

    dataloader_workers = (
        args.dataloader_workers
        if args.dataloader_workers is not None
        else max(1, min(4, os.cpu_count() or 1))
    )

    target_device = _resolve_target_device(args.device)
    use_cuda = target_device == "cuda"
    use_mps = target_device == "mps"

    pad_to_multiple = 8 if use_cuda else None
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple,
    )

    if use_mps:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if use_cuda and _is_rocm_runtime():
        if args.hip_gfx_override:
            os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", args.hip_gfx_override)
        else:
            try:
                name = torch.cuda.get_device_name(0).lower()
            except torch.cuda.CudaError:
                name = ""
            if any(token in name for token in {"5700", "5600", "navi 10"}):
                os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        os.environ.setdefault("HIP_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES", "0"))

    if target_device == "cpu":
        threads = args.cpu_threads or (os.cpu_count() or 1)
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 4)))
        if torch.backends.mkldnn.is_available():
            torch.backends.mkldnn.enabled = True
        torch.set_float32_matmul_precision("medium")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, args.gradient_accumulation),
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        do_eval=True,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=use_cuda or use_mps,
        no_cuda=not (use_cuda or use_mps),
        use_mps_device=use_mps,
        fp16=args.fp16,
        bf16=args.bf16,
        warmup_steps=max(0, args.warmup_steps),
        warmup_ratio=max(0.0, args.warmup_ratio) if args.warmup_steps <= 0 else 0.0,
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length,
        torch_compile=args.torch_compile,
    )

    if use_cuda and 0.0 < args.max_gpu_memory_fraction < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_fraction)
        except (AttributeError, RuntimeError):
            pass

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
    )
    trainer.tokenizer = tokenizer

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate(cast(Any, eval_dataset))
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    metrics_path = args.output_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(serializable_metrics, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
