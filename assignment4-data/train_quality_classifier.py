"""
Train a fastText binary quality classifier: __label__wiki vs __label__cc.

Positive examples (wiki): the high-quality Wikipedia reference fixture text,
  slid into overlapping chunks at 50-, 100-, and 200-word granularities.
Negative examples (cc): plain text extracted from Common Crawl WARC responses,
  split into 200-word chunks.

Usage:
    uv run python train_quality_classifier.py [--warc PATH] [--out PATH]

Outputs:
    cs336_text_cleaning/quality_classifier.bin  (fastText model)
    cs336_text_cleaning/quality_train.txt       (training data, kept for inspection)
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

DEFAULT_WARC = pathlib.Path(
    "cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
)
DEFAULT_MODEL_OUT = pathlib.Path("cs336_text_cleaning/quality_classifier.bin")
DEFAULT_TRAIN_OUT = pathlib.Path("cs336_text_cleaning/quality_train.txt")
WIKI_FIXTURE = pathlib.Path("tests/fixtures/high_quality_wiki_reference.txt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    """Collapse all whitespace to single spaces (fastText expects one doc per line)."""
    return " ".join(text.split())


def sliding_chunks(words: list[str], size: int, min_words: int = 20) -> list[str]:
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i : i + size])
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Build positive (wiki) training examples
# ---------------------------------------------------------------------------

def build_wiki_examples(fixture_path: pathlib.Path) -> list[str]:
    text = fixture_path.read_text()
    words = text.split()
    examples: list[str] = []
    for size in (50, 100, 200):
        examples.extend(sliding_chunks(words, size))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for ex in examples:
        key = ex[:120]  # cheap key – full dedup would be too slow
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


# ---------------------------------------------------------------------------
# Build negative (cc) training examples
# ---------------------------------------------------------------------------

def build_cc_examples(warc_path: pathlib.Path, max_docs: int = 400) -> list[str]:
    from fastwarc.warc import ArchiveIterator, WarcRecordType
    from cs336_text_cleaning.process_text import extract_text_from_html_bytes

    examples: list[str] = []
    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if len(examples) >= max_docs * 10:  # rough upper bound
                break
            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)
            if not text or len(text.split()) < 20:
                continue
            words = text.split()
            chunks = sliding_chunks(words, 200)
            examples.extend(clean(c) for c in chunks)
            if sum(1 for _ in []) >= max_docs:  # count docs processed
                break

    # Re-count by collecting docs first then chunking
    return examples


def build_cc_examples_v2(warc_path: pathlib.Path, max_docs: int = 400) -> list[str]:
    from fastwarc.warc import ArchiveIterator, WarcRecordType
    from cs336_text_cleaning.process_text import extract_text_from_html_bytes

    examples: list[str] = []
    n_docs = 0
    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if n_docs >= max_docs:
                break
            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)
            if not text or len(text.split()) < 20:
                continue
            n_docs += 1
            for chunk in sliding_chunks(text.split(), 200):
                examples.append(clean(chunk))
    return examples


# ---------------------------------------------------------------------------
# Write fastText training file
# ---------------------------------------------------------------------------

def write_training_file(
    wiki_examples: list[str],
    cc_examples: list[str],
    out_path: pathlib.Path,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for ex in wiki_examples:
            f.write(f"__label__wiki {ex}\n")
        for ex in cc_examples:
            f.write(f"__label__cc {ex}\n")


# ---------------------------------------------------------------------------
# Train fastText classifier
# ---------------------------------------------------------------------------

def train(train_path: pathlib.Path, model_out: pathlib.Path) -> None:
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None

    print("Training fastText classifier...")
    model = fasttext.train_supervised(
        input=str(train_path),
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        dim=100,
        loss="ova",
        thread=4,
    )
    model.save_model(str(model_out))
    print(f"Model saved to {model_out}")
    return model


# ---------------------------------------------------------------------------
# Sanity-check on the two fixture files
# ---------------------------------------------------------------------------

def sanity_check(model) -> bool:
    import fasttext

    checks = [
        (WIKI_FIXTURE, "wiki"),
        (pathlib.Path("tests/fixtures/low_quality_cc.txt"), "cc"),
    ]
    all_ok = True
    print("\nSanity check:")
    for path, expected in checks:
        text = clean(path.read_text())
        labels, scores = model.predict(text, k=1)
        label = labels[0].replace("__label__", "")
        score = float(min(scores[0], 1.0))
        ok = label == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {path.name}: predicted={label}, expected={expected}, score={score:.4f}")
        all_ok = all_ok and ok
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train quality classifier")
    parser.add_argument("--warc", type=pathlib.Path, default=DEFAULT_WARC,
                        help="Path to Common Crawl WARC .gz file")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_MODEL_OUT,
                        help="Output path for trained model (.bin)")
    parser.add_argument("--train-file", type=pathlib.Path, default=DEFAULT_TRAIN_OUT,
                        help="Output path for training data text file")
    parser.add_argument("--max-cc-docs", type=int, default=400,
                        help="Maximum number of CC documents to use for negatives")
    args = parser.parse_args()

    if not WIKI_FIXTURE.exists():
        sys.exit(f"ERROR: wiki fixture not found at {WIKI_FIXTURE}")
    if not args.warc.exists():
        sys.exit(f"ERROR: WARC file not found at {args.warc}")

    # 1. Build training data
    print(f"Building wiki examples from {WIKI_FIXTURE}...")
    wiki_ex = build_wiki_examples(WIKI_FIXTURE)
    print(f"  {len(wiki_ex)} wiki training examples")

    print(f"Building CC examples from {args.warc} (max {args.max_cc_docs} docs)...")
    cc_ex = build_cc_examples_v2(args.warc, max_docs=args.max_cc_docs)
    print(f"  {len(cc_ex)} cc training examples")

    # 2. Write training file
    write_training_file(wiki_ex, cc_ex, args.train_file)
    print(f"Training file written to {args.train_file}")

    # 3. Train
    model = train(args.train_file, args.out)

    # 4. Sanity check
    ok = sanity_check(model)
    if not ok:
        sys.exit("ERROR: sanity check failed — model may need more training data or hyperparameter tuning")
    print("\nDone.")


if __name__ == "__main__":
    main()
