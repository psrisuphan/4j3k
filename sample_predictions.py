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
    {"text": "ไอ้พวกขายชาติ ขอให้ชีวิตมันพังพินาศทุกคน", "expected": True},
    {"text": "วันนี้ได้โบนัสจากที่ทำงาน จะพาแม่ไปกินอาหารญี่ปุ่น", "expected": False},
    {"text": "เดินตลาดนัดแล้วโดนชนยังไม่ขอโทษ มารยาทไม่มีเลย", "expected": False},
    {"text": "แกงเขียวหวานร้านนี้เผ็ดจนปากพอง ใครไปก็เตรียมน้ำไว้ด้วย", "expected": False},
    {"text": "อย่ามาทำตัวเป็นนักบุญ ทั้งที่เบื้องหลังสกปรกกว่าขยะ", "expected": True},
    {"text": "เพื่อนๆ ในดิสคอร์ดช่วยติวเลขให้จนทำข้อสอบได้ สุดยอดมาก", "expected": False},
    {"text": "เห็นข่าวนักเรียนโดนบูลลี่แล้วโมโหมาก อยากให้โรงเรียนทำอะไรสักอย่าง", "expected": False},
    {"text": "เธอจะทำอะไรก็พังไปหมด นี่มันเรื่องง่ายๆ ยังทำไม่ได้", "expected": True},
    {"text": "เมื่อคืนไปเดินงานกาชาด คนเยอะมากแต่สนุกดี", "expected": False},
    {"text": "รถเมล์วันนี้ตรงเวลาจนแทบไม่เชื่อ ขอบคุณคนขับมาก", "expected": False},
    {"text": "ร้านกาแฟแถวบ้านพูดจากับลูกค้าหยาบคายมาก เห็นแล้วของขึ้น", "expected": True},
    {"text": "เด็กๆ โรงเรียนวัดแข่งบาสได้ที่หนึ่ง ภูมิใจแทนชุมชนจริงๆ", "expected": False},
    {"text": "เพื่อนร่วมงานเอาแต่โยนงาน แล้วมายืนสั่งอย่างกับหัวหน้า", "expected": False},
    {"text": "ไอ้คนขับแท็กซี่โกงค่าโดยสาร ขอให้โดนจับให้หมด", "expected": True},
    {"text": "แม่ค้าตลาดช่วยกันบริจาคอาหารให้หมาแมวจรน่ารักสุดๆ", "expected": False},
    {"text": "เด็กแว้นพวกนี้ส่งเสียงดังรบกวนทั้งคืน มันช่างไร้สมอง", "expected": True},
    {"text": "อาสาสมัครกู้ภัยช่วยผู้ประสบอุบัติเหตุทันเวลา ชื่นชมมาก", "expected": False},
    {"text": "ข้าวแกงร้านป้าจืดขึ้นราคาอีกแล้ว แต่รสชาติยังดีเหมือนเดิม", "expected": False},
    {"text": "คนแถวนี้ยังชอบดูกถูกคนต่างจังหวัดอยู่เลย น่าหดหู่", "expected": False},
    {"text": "ครูประจำชั้นประกาศผลสอบด้วยรอยยิ้ม ทุกคนลุ้นกันสุดๆ", "expected": False},
    {"text": "แก๊งบ้านี่น่าเกลียดยิ่งกว่าอะไรทั้งนั้น อยู่ใกล้แล้วปวดหัว", "expected": True},
    {"text": "วันนี้ฝนตกหนักแต่ทีมทำความสะอาดก็ยังมาทำงานครบ", "expected": False},
    {"text": "สนามบินใหม่บริการดีจนเซอร์ไพรส์ อยากให้รักษามาตรฐานไว้", "expected": False},
    {"text": "ผู้บริหารพูดไม่รู้เรื่องแถมโยนความผิดให้น้องในทีม", "expected": False},
    {"text": "เจอคอมเมนต์เหยียดคนอีสานเต็มฟีด หัวร้อนสุดๆ", "expected": False},
    {"text": "เมื่อคืนฝันดีมาก ได้เจอศิลปินที่ชอบมาตั้งแต่เด็ก", "expected": False},
    {"text": "อย่ามาทำตัวต่ำๆ แบบพวกขี้แพ้ จะอ้วก", "expected": True},
    {"text": "ไฟดับทั้งหมู่บ้าน แต่ช่างไฟมากู้ทัน ก่อนคืนนั้นจะมืดสนิท", "expected": False},
    {"text": "วัยรุ่นแถวนี้ช่วยกันเก็บขยะหลังคอนเสิร์ตน่ารักมาก", "expected": False},
    {"text": "หัวหน้าฝ่ายบัญชีไม่ฟังใครสักคน เอาแต่สั่ง", "expected": False},
    {"text": "คนที่ชอบข่มเหงผู้อื่นนี่น่าขยะแขยงที่สุด", "expected": True},
    {"text": "วันนี้รถติดน้อยเกินคาด ถึงที่ทำงานก่อนเวลาอีก", "expected": False},
    {"text": "ไปบริจาคเลือดครั้งแรก ตื่นเต้นแต่รู้สึกดีมาก", "expected": False},
    {"text": "พวกขี้โกงในวงการศึกษา สมควรโดนไล่ออกให้หมด", "expected": True},
    {"text": "ลูกค้าประจำซื้อขนมครบทุกชนิด ดีใจสุดๆ", "expected": False},
    {"text": "เพื่อนพาไปเดินป่าครั้งแรก เหนื่อยแต่คุ้มกับวิว", "expected": False},
    {"text": "คนขับรถเมล์ตะคอกใส่ผู้โดยสารผู้สูงอายุ โกรธมาก", "expected": True},
    {"text": "สมาชิกชมรมดนตรีซ้อมหนักเพื่อการแสดงใหญ่สัปดาห์หน้า", "expected": False},
    {"text": "แฟนคลับช่วยสะสมเงินซื้อโฆษณาอวยพรวันเกิดศิลปิน", "expected": False},
    {"text": "คนบางกลุ่มชอบใช้คำเหยียดคนลาว เห็นแล้วทนไม่ไหว", "expected": False},
    {"text": "ทีมอีสปอร์ตของโรงเรียนชนะรายการระดับประเทศ", "expected": False},
    {"text": "นายกสมาคมพูดจาไม่ให้เกียรติคนฟังเลย สุดจะทน", "expected": False},
    {"text": "พนักงานร้านหนังสือช่วยหาของขวัญให้แม่รวดเร็วมาก", "expected": False},
    {"text": "เด็กที่บอกว่าเพื่อนต่างจังหวัดโง่นี่มันเลวจริง", "expected": True},
    {"text": "ช่างซ่อมแอร์มาทันเวลาทำให้ไม่ต้องดับร้อนทั้งคืน", "expected": False},
    {"text": "กลุ่มวิ่งตอนเช้าชวนกันบริจาคค่าน้ำดื่มให้ รพ.", "expected": False},
    {"text": "เพื่อนในแชทด่าคนอ้วนด้วยคำพูดรุนแรง ไม่โอเคเลย", "expected": True},
    {"text": "พนักงานต้อนรับสนามกีฬายิ้มแย้มตลอดบริการดีมาก", "expected": False},
    {"text": "ประชุมวันนี้ยาวไปหน่อยแต่ก็ได้ข้อสรุปดีๆ", "expected": False},
    {"text": "คนที่ด่าแรงๆ ว่าคนจนขี้เกียจนี่น่าตบ", "expected": True},
    {"text": "เด็กๆ ศิลปะชนะประกวดวาดภาพระดับเขต ครูดีใจมาก", "expected": False},
    {"text": "นักเรียนหมู่บ้านจัดงานดนตรีช่วยระดมทุนซ่อมสนาม", "expected": False},
    {"text": "เพื่อนร่วมงานใหม่สุภาพมาก ช่วยงานละเอียด", "expected": False},
    {"text": "แก๊งมิจฉาชีพโทรมาหลอกด่าเราโง่ น่าถีบให้หาย", "expected": True},
    {"text": "รายการทีวีเมื่อคืนเชิญนักวิทยาศาสตร์ไทยมาเล่าแรงบันดาลใจ", "expected": False},
    {"text": "อธิบดีตอบคำถามสื่อแบบไม่จริงใจเลย เสียเวลาฟัง", "expected": False},
    {"text": "คนขับรถปาดหน้าแล้วลงมาด่าแม่เรา น่าเจ็บใจ", "expected": True},
    {"text": "ชมรมอ่านหนังสือเปิดรับสมาชิกใหม่ ใครสนใจเชิญนะ", "expected": False},
    {"text": "ข้างบ้านเปิดเพลงเสียงดังแต่พอคุยดีๆ ก็ยอมลด", "expected": False},
    {"text": "ดาราคนนี้เอาแต่สร้างข่าวฉาว สันดานไม่เคยเปลี่ยน", "expected": True},
    {"text": "หมออาสาไปตรวจคนในพื้นที่ห่างไกลอีกแล้ว นับถือ", "expected": False},
    {"text": "พนักงานส่งของทำแพ็คเกจเสียหายแต่รีบรับผิดชอบ", "expected": False},
    {"text": "ไอ้พวกฟาสสิสต์นี่มันชั่วร้ายไม่มีวันสำนึก", "expected": True},
    {"text": "เช้านี้ฟังพอดแคสต์แรงบันดาลใจ ทำให้มีพลัง", "expected": False},
    {"text": "ร้านข้าวต้มเปิดเพลงลูกทุ่งเพราะๆ ให้ลูกค้าฟัง", "expected": False},
    {"text": "ตำรวจจับคนปล่อยข่าวลือทำลายชื่อเสียงโรงพยาบาลได้", "expected": False},
    {"text": "นักร้องพูดว่าคนจนไม่มีค่า ฟังแล้วโกรธสุดๆ", "expected": True},
    {"text": "ทีมกู้ภัยดำน้ำช่วยนักท่องเที่ยวติดถ้ำได้สำเร็จ", "expected": False},
    {"text": "ชมรมหมากรุกโรงเรียนคว้าแชมป์ภาคเป็นครั้งแรก", "expected": False},
    {"text": "ร้านอาหารมังสวิรัติเปิดใหม่ บรรยากาศดีมาก", "expected": False},
    {"text": "คอมเมนต์ด่าคนไทยเชื้อสายเขมรนี่มันเหยียดจริงๆ", "expected": True},
    {"text": "สตาร์ทอัพไทยพัฒนาแอปช่วยเหลือผู้พิการทางสายตา", "expected": False},
    {"text": "เพื่อนในทวิตชมงานศิลป์เราแล้วมีกำลังใจขึ้นเยอะ", "expected": False},
    {"text": "คนที่ชอบเหยียดเพศอื่นคือพวกไร้ค่า น่าเกลียดมาก", "expected": True},
    {"text": "ทริปเที่ยวเชียงใหม่กับครอบครัวเต็มไปด้วยความสุข", "expected": False},
    {"text": "นักเรียนช่วยกันทำความสะอาดห้องเรียนก่อนกลับบ้าน", "expected": False},
    {"text": "เพื่อนบ้านที่ขี้นินทาไม่เคยหยุดปั่นเรื่องคนอื่น", "expected": False},
    {"text": "พวกคอมเมนต์เหยียดคนอีสานนี่มันต้องโดนสั่งสอนแรงๆ", "expected": True},
    {"text": "ศิลปินฝึกซ้อมคอนเสิร์ตใหญ่เพื่อแฟนคลับทั่วประเทศ", "expected": False},
    {"text": "อาจารย์ชวนเด็กๆ ปลูกต้นไม้ริมคลองช่วยลดน้ำท่วม", "expected": False},
    {"text": "เพื่อนร่วมทีมคอยช่วยเก็บรายละเอียดงานจนเสร็จทัน", "expected": False},
    {"text": "พวกสลิ่มหัวแข็งจะทำประเทศพังทั้งก๊ก", "expected": True},
    {"text": "แม่บ้านออฟฟิศทำความสะอาดเรียบร้อย ขอบคุณจริง", "expected": False},
    {"text": "คุณหมอแนะนำให้ปรับอาหารและออกกำลัง นำไปทำตามเลย", "expected": False},
    {"text": "นักศึกษาระดมทุนช่วยเพื่อนที่ประสบอุบัติเหตุ", "expected": False},
    {"text": "คนบางพวกชอบดูถูกเพศทางเลือก จิตสำนึกอยู่ไหน", "expected": True},
    {"text": "สนามฟุตบอลชุมชนได้ทาสีใหม่ เด็กๆ แห่ไปลองเล่น", "expected": False},
    {"text": "เพื่อนรุ่นพี่ให้คำแนะนำเรื่องการสมัครงานอย่างละเอียด", "expected": False},
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
        help="Optional device override such as 'cpu', 'cuda', or 'dml'.",
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
