"""Convert the Thai Toxicity Tweet corpus to the HateThaiSent CSV schema."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Final

import pandas as pd
import requests
import zipfile

ARCHIVE_URL: Final[str] = "https://archive.org/download/ThaiToxicityTweetCorpus/data.zip"

LABEL_MAP: Final[dict[int, str]] = {
    0: "nonhatespeech",
    1: "hatespeech",
}


def _download_archive(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _load_dataframe(archive_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        with zf.open("data/train.jsonl") as fh:
            df = pd.read_json(fh, lines=True)
    df = df.rename(columns={"tweet_text": "Message", "is_toxic": "Hatespeech"})
    df = df.dropna(subset=["Message", "Hatespeech"])
    df = df[df["Message"].astype(str).str.upper() != "TWEET_NOT_FOUND"]
    df["Message"] = df["Message"].astype(str).str.strip()
    df = df[df["Message"].str.len() > 0]
    df["Hatespeech"] = df["Hatespeech"].map(LABEL_MAP)
    df = df[df["Hatespeech"].notna()]
    df = df.assign(Sentiment="Unknown")
    return df[["Message", "Hatespeech", "Sentiment"]]


def convert(output_path: Path) -> None:
    archive_bytes = _download_archive(ARCHIVE_URL)
    df = _load_dataframe(archive_bytes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ThaiToxicityTweet_converted.csv"),
        help="Destination CSV path matching the HateThaiSent schema.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        convert(args.output)
    except requests.HTTPError as err:
        raise SystemExit(f"Failed to download archive: {err}") from err
    except zipfile.BadZipFile as err:
        raise SystemExit(f"Corrupt archive received from {ARCHIVE_URL}") from err


if __name__ == "__main__":
    main()
