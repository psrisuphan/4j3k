"""Run the hate-speech classifier on a bundle of sample sentences."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List

from age_policy import resolve_policy
from predict import TransformerAgeAwareClassifier


DEFAULT_SENTENCES_PATH = Path(__file__).with_name("sample_sentences.jsonl")


def _load_jsonl_sentences(path: Path) -> List[dict]:
    sentences: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - user input errors
                raise SystemExit(
                    f"Failed to parse JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if "text" not in record:
                raise SystemExit(
                    f"Missing 'text' field on line {line_number} of {path}."
                )
            sentences.append({"text": record["text"], "expected": record.get("expected")})
    return sentences


def _load_default_sentences() -> List[dict]:
    if DEFAULT_SENTENCES_PATH.exists():
        return _load_jsonl_sentences(DEFAULT_SENTENCES_PATH)
    raise SystemExit(
        f"Default sentences file not found. Expected {DEFAULT_SENTENCES_PATH} to exist."
    )


def _read_sentences(input_path: Path | None) -> List[dict]:
    if input_path is None:
        return _load_default_sentences()
    if input_path.suffix.lower() in {".jsonl", ".json"}:
        return _load_jsonl_sentences(input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        return [{"text": line.strip(), "expected": None} for line in handle if line.strip()]


def _write_csv(rows: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/wangchanberta"),
        help="Directory containing the fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--age",
        type=int,
        default=15,
        help="User age for policy evaluation (default: 15).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override such as 'cpu', 'cuda', or 'dml'.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional UTF-8 file to score. "
            "Use .jsonl to include expected labels or plain text (one sentence per line)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path to store the scoring results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentences = _read_sentences(args.input)

    if not sentences:
        raise SystemExit("No sentences provided for scoring.")

    classifier = TransformerAgeAwareClassifier(args.model, device=args.device)
    policy = resolve_policy(args.age)

    print(f"Using age policy: {policy}")

    rows = []
    correct = 0
    total_with_labels = 0

    for idx, sample in enumerate(sentences, start=1):
        text = sample["text"]
        expected = sample.get("expected")

        result = classifier.classify(text, args.age)
        score = result["score"]
        should_block = result["should_block"]
        is_correct = expected is not None and expected == should_block
        if expected is not None:
            total_with_labels += 1
            if is_correct:
                correct += 1

        expected_str = "?" if expected is None else str(expected)
        print(
            f"{idx:02d}. score={score:.4f} block={should_block} expected={expected_str} text={text}"
        )
        rows.append(
            {
                "index": idx,
                "text": text,
                "score": score,
                "should_block": should_block,
                "threshold": policy.threshold,
                "age": args.age,
                "expected": expected,
                "correct": is_correct if expected is not None else None,
            }
        )

    if total_with_labels:
        print(f"Accuracy: {correct}/{total_with_labels}")

    if args.output is not None:
        _write_csv(rows, args.output)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
