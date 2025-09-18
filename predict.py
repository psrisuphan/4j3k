"""Inference helper that applies age-aware thresholds to model outputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from age_policy import AgePolicy, resolve_policy


class TransformerAgeAwareClassifier:
    """Wrap a Transformers sequence classifier with age-based policies."""

    def __init__(self, model_path: Path, device: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory {self.model_path} not found")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _score_text(self, text: str) -> float:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
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
        help="Optional device override (e.g. 'cpu' or 'cuda').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = TransformerAgeAwareClassifier(args.model, device=args.device)
    result = classifier.classify(args.text, args.age)
    print(result)


if __name__ == "__main__":
    main()
