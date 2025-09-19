"""Run the hate-speech classifier on a bundle of sample sentences."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

from age_policy import resolve_policy
from predict import TransformerAgeAwareClassifier


DEFAULT_SENTENCES: Sequence[dict] = (
    {"text": "วันนี้อากาศดีมาก ออกไปวิ่งกับพ่อแม่สดชื่นสุดๆ", "expected": False},
    {"text": "เลิกโง่สักทีเถอะ คิดอะไรไม่เป็นก็อย่ามาโชว์", "expected": True},
    {"text": "ขอบคุณน้องๆ ที่ช่วยกันเก็บขยะริมทะเลเมื่อเช้า", "expected": False},
    {"text": "พวกมันโกงเลือกตั้งอีกแล้ว ประชาชนอย่ายอม", "expected": False},
    {"text": "คอนเสิร์ตเมื่อคืนสนุกมาก นักร้องทักทายแฟนคลับน่ารักสุด", "expected": False},
    {"text": "ไปต่อคิวซื้อน้ำชาไข่มุกสองชั่วโมง เหนื่อยแต่คุ้ม", "expected": False},
    {"text": "เธอเอาแต่บ่น ไม่ทำงานอะไรเลย เบื่อจะตาย", "expected": False},
    {"text": "ดีใจที่ทีมชาติชนะในบ้าน คนไทยเฮทั้งประเทศ", "expected": False},
    {"text": "อ่านคอมเมนต์เฟซแล้วสงสาร พวกคนดูถูกคนจนมันน่ารังเกียจ", "expected": False},
    {"text": "พนักงานรถไฟติดแอร์ไม่เปิด หนาวจะตาย ยังจะหัวเราะใส่ผู้โดยสารอีก", "expected": True},
    {"text": "#saveบางกลอย ชาวบ้านแค่ขออยู่บ้านเกิด ช่วยกันแชร์ให้คนเห็น", "expected": False},
    {"text": "ทวิตเตอร์เมื่อคืนดราม่าแรงมาก ด่าไปด่ามาไม่มีใครรับผิดชอบ", "expected": True},
)


def _read_sentences(input_path: Path | None) -> List[dict]:
    if input_path is None:
        return list(DEFAULT_SENTENCES)
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
        help="Optional device override such as 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional UTF-8 text file with one sentence per line to score.",
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
