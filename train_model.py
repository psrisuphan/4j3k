"""Fine-tune WangchanBERTa on the Thai hate-speech dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

def _enforce_memory_safe_defaults(args: argparse.Namespace, target_device: str) -> None:
    """Best-effort adjustments to curb OOM risk across devices."""

    if target_device == "mps":
        if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.80"
        if args.batch_size > 2:
            print(
                f"[memory-safe] Reducing MPS batch size from {args.batch_size} to 2 to ease memory pressure.",
                flush=True,
            )
            args.batch_size = 2
    elif target_device == "cuda":
        if not (0.0 < args.max_gpu_memory_fraction <= 1.0):
            args.max_gpu_memory_fraction = 0.9
        elif args.max_gpu_memory_fraction > 0.9:
            print(
                f"[memory-safe] Clamping CUDA memory fraction from {args.max_gpu_memory_fraction:.2f} to 0.90.",
                flush=True,
            )
            args.max_gpu_memory_fraction = 0.9
        if args.batch_size > 4:
            print(
                f"[memory-safe] Reducing CUDA batch size from {args.batch_size} to 4 to conserve VRAM.",
                flush=True,
            )
            args.batch_size = 4

    if not args.gradient_checkpointing and args.batch_size <= 4:
        print(
            "[memory-safe] Enabling gradient checkpointing to shrink activation footprint.",
            flush=True,
        )
        args.gradient_checkpointing = True


def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Message", "Hatespeech"])
    df = df[df["Hatespeech"].isin(LABEL2ID)]
    df = df.assign(label=df["Hatespeech"].map(LABEL2ID))
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
        "--extra-data",
        type=Path,
        nargs="*",
        default=[Path("data/ThaiToxicityTweet_converted.csv")],
        help=(
            "Optional additional CSV dataset(s) with the same schema. "
            "Specify --extra-data without values to disable the defaults."
        ),
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

    if args.data.exists():
        dataframes.append(_load_dataframe(args.data))
    else:
        raise FileNotFoundError(f"Primary dataset '{args.data}' not found.")

    extra_paths = args.extra_data or []
    for extra_path in extra_paths:
        if extra_path.exists():
            dataframes.append(_load_dataframe(extra_path))
        else:
            print(
                f"Warning: extra dataset '{extra_path}' not found; continuing without it.",
                flush=True,
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
    _enforce_memory_safe_defaults(args, target_device)
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
