"""Inference helper that applies age-aware thresholds to model outputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from contextlib import nullcontext

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from age_policy import AgePolicy, resolve_policy

try:  # Optional dependency for Windows DirectML acceleration
    import torch_directml  # type: ignore
except Exception:  # pragma: no cover - optional runtime import
    torch_directml = None


def _has_directml() -> bool:
    if torch_directml is None:
        return False
    try:
        torch_directml.device()
    except Exception:
        return False
    return True


def _select_device(explicit: Optional[str] = None) -> str:
    """Choose an execution device, falling back gracefully when accelerators fail."""

    if explicit:
        choice = explicit.lower()
        if choice in {"cuda", "gpu"}:
            if torch.cuda.is_available():
                try:
                    torch.zeros(1).to("cuda")
                    return "cuda"
                except RuntimeError:
                    print("[device] CUDA reported available but failed; using CPU instead.", flush=True)
                    return "cpu"
            print("[device] CUDA requested but not available; using CPU instead.", flush=True)
            return "cpu"
        if choice == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend and torch.backends.mps.is_available():
                try:
                    torch.zeros(1).to("mps")
                    return "mps"
                except RuntimeError:
                    print("[device] MPS backend failed; falling back to CPU.", flush=True)
                    return "cpu"
            raise RuntimeError("MPS requested but not available on this system.")
        if choice in {"dml", "directml"}:
            if _has_directml():
                return "dml"
            raise RuntimeError("DirectML requested but torch-directml is not installed.")
        if choice == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported device override '{explicit}'.")
    if torch.cuda.is_available():
        try:
            torch.zeros(1).to("cuda")
            return "cuda"
        except RuntimeError:
            print("[device] CUDA reported available but failed; using CPU instead.", flush=True)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and torch.backends.mps.is_available():
        try:
            torch.zeros(1).to("mps")
            return "mps"
        except RuntimeError:
            print("[device] MPS backend failed to initialise; falling back to CPU.", flush=True)
    if _has_directml():
        return "dml"
    return "cpu"


class TransformerAgeAwareClassifier:
    """Wrap a Transformers sequence classifier with age-based policies."""

    def __init__(self, model_path: Path, device: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory {self.model_path} not found")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

        max_len = getattr(self.tokenizer, "model_max_length", 512)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 4096:
            max_len = 512
        self.max_length = max_len

        self.device = _select_device(device)
        self._dml_device = None
        try:
            if self.device == "dml":
                if torch_directml is None:
                    raise RuntimeError(
                        "DirectML device requested but torch-directml is not installed."
                    )
                self._dml_device = torch_directml.device()
                self.model.to(self._dml_device)
            else:
                self.model.to(self.device)
                if self.device == "cuda" and hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.benchmark = True
        except RuntimeError as exc:
            message = str(exc).lower()
            if "hip error" in message or "invalid device function" in message:
                print("[device] Accelerator failed with HIP error; retrying on CPU.", flush=True)
                self.device = "cpu"
                self.model.to(self.device)
            elif self.device == "cuda" and any(
                token in message for token in {"cuda", "cublas", "cudnn"}
            ):
                print("[device] CUDA initialisation failed; retrying on CPU.", flush=True)
                self.device = "cpu"
                self.model.to(self.device)
            elif self.device == "dml" and "directml" in message:
                print("[device] DirectML initialisation failed; retrying on CPU.", flush=True)
                self.device = "cpu"
                self._dml_device = None
                self.model.to(self.device)
            else:
                raise

    def _score_text(self, text: str) -> float:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        target_device = self._dml_device if self.device == "dml" else self.device
        encoded = {k: v.to(target_device) for k, v in encoded.items()}
        def _autocast_context(device: str):
            if device == "cuda":
                return torch.autocast(device_type="cuda", dtype=torch.float16)
            return nullcontext()

        with torch.no_grad(), _autocast_context(self.device):
            try:
                logits = self.model(**encoded).logits
            except RuntimeError as exc:
                message = str(exc).lower()
                if "hip error" in message or "invalid device function" in message:
                    print("[device] Runtime HIP failure during inference; switching to CPU.", flush=True)
                    self.device = "cpu"
                    self.model.to(self.device)
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    with _autocast_context(self.device):
                        logits = self.model(**encoded).logits
                elif self.device == "cuda" and any(
                    token in message for token in {"cuda", "cublas", "cudnn"}
                ):
                    print("[device] CUDA runtime failure; switching to CPU.", flush=True)
                    self.device = "cpu"
                    self.model.to(self.device)
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    with _autocast_context(self.device):
                        logits = self.model(**encoded).logits
                elif "directml" in message and self.device == "dml":
                    print("[device] DirectML execution failed; switching to CPU.", flush=True)
                    self.device = "cpu"
                    self._dml_device = None
                    self.model.to(self.device)
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    with _autocast_context(self.device):
                        logits = self.model(**encoded).logits
                else:
                    raise
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return float(probabilities[:, 1].item())

    def classify(self, text: str, age: Optional[int]) -> Dict[str, Any]:
        score = self._score_text(text)
        policy: AgePolicy = resolve_policy(age)
        should_block = score >= policy.threshold
        return {
            "score": score,
            "should_block": should_block,
            "age_policy": asdict(policy),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Input text to score.")
    parser.add_argument(
        "--age",
        type=int,
        default=None,
        help="Age provided by the user; influences the post-processing threshold.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/wangchanberta-hatespeech"),
        help="Directory holding the fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override (e.g. 'cpu', 'cuda', or 'dml').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = TransformerAgeAwareClassifier(args.model, device=args.device)
    result = classifier.classify(args.text, args.age)
    print(result)


if __name__ == "__main__":
    main()
