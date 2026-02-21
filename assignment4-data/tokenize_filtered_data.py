"""
tokenize_filtered_data.py — Tokenize filtered CC data with the GPT-2 tokenizer.

Reads one or more JSONL files produced by filter_wet_files.py (each line is a
JSON object {"url": ..., "text": ...}), extracts the text field, appends the
GPT-2 <|endoftext|> EOS token, and serialises the concatenated token IDs as a
numpy uint16 binary file compatible with the CS336 training script.

Usage:
    uv run python tokenize_filtered_data.py                        # default paths
    uv run python tokenize_filtered_data.py \\
        --input-glob "filtered_output/*.jsonl" \\
        --output tokenized/train.bin
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import multiprocessing
import pathlib

import numpy as np
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer

transformers.logging.set_verbosity_error()  # suppress "sequence length > 1024" warnings

# ---------------------------------------------------------------------------
# Worker (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

_tokenizer: AutoTokenizer | None = None


def _init_worker():
    """Initialise the tokenizer once per worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained("gpt2")


def _tokenize_text(text: str) -> list[int]:
    """Encode `text` and append the GPT-2 EOS token."""
    assert _tokenizer is not None
    return _tokenizer.encode(text) + [_tokenizer.eos_token_id]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenize filtered CC JSONL → uint16 binary")
    p.add_argument(
        "--input-glob",
        default="filtered_output/*.jsonl",
        help="Glob pattern for input JSONL files (default: filtered_output/*.jsonl)",
    )
    p.add_argument(
        "--output",
        default="tokenized/train.bin",
        help="Output binary file path (default: tokenized/train.bin)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of tokenizer worker processes (default: cpu_count)",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=256,
        help="imap chunksize (default: 256)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Discover input files
    input_files = sorted(pathlib.Path(p) for p in _glob.glob(args.input_glob))
    if not input_files:
        raise FileNotFoundError(
            f"No files matched: {args.input_glob!r}. "
            "Run filter_wet_files.py first, or provide --input-glob."
        )
    print(f"Input files  : {len(input_files)} JSONL file(s)")

    # Read and extract text fields
    texts: list[str] = []
    for path in input_files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    pass  # skip malformed lines

    print(f"Documents    : {len(texts):,}")

    # Tokenize in parallel
    n_workers = args.workers or multiprocessing.cpu_count()
    print(f"Workers      : {n_workers}  |  chunksize={args.chunksize}")

    results: list[list[int]] = []
    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_worker,
    ) as pool:
        for token_ids in tqdm(
            pool.imap(_tokenize_text, texts, chunksize=args.chunksize),
            total=len(texts),
            desc="Tokenizing",
        ):
            results.append(token_ids)

    # Flatten and serialise
    all_ids = [tid for sublist in results for tid in sublist]
    n_tokens = len(all_ids)
    print(f"Total tokens : {n_tokens:,}")

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)
    print(f"Saved to     : {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
