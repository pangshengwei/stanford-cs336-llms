"""
filter_wet_files.py – CLI entry point for the CC WET filtering pipeline.

Usage (cluster):
    uv run python filter_wet_files.py --wet-dir /data/CC --output-dir filtered_output

Usage (local, single sample file):
    uv run python filter_wet_files.py --output-dir filtered_output

Optional dedup post-processing (run after all WET files are filtered):
    uv run python filter_wet_files.py --dedup-only --output-dir filtered_output

Pipeline per document:
  1. Language identification  → keep English  (lang="en", score≥lang_threshold)
  2. Gopher quality filter    → keep if passes all structural checks
  3. NSFW classifier          → drop NSFW pages
  4. Toxic-speech classifier  → drop toxic/hate-speech pages
  5. Quality classifier       → drop pages labelled "cc" with score≥quality_cc_threshold
  6. PII masking              → replace emails / phone numbers / IPs with placeholders

Post-processing (optional, requires all JSONL outputs to be complete):
  7. Exact-line deduplication across all output JSONL files

The script reports the number of documents removed by each filter step.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter CC WET files → JSONL training data")
    p.add_argument(
        "--wet-dir",
        default="/data/CC",
        help="Directory containing CC*.warc.wet.gz files (default: /data/CC)",
    )
    p.add_argument(
        "--local-wet-glob",
        default="cs336_data/*.warc.wet.gz",
        help="Glob for local WET files used when --wet-dir is unavailable",
    )
    p.add_argument(
        "--output-dir",
        default="filtered_output",
        help="Output directory for .jsonl files (default: filtered_output)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: all CPUs)",
    )
    p.add_argument(
        "--lang-threshold",
        type=float,
        default=0.65,
        help="Minimum language-ID confidence to accept English (default: 0.65)",
    )
    p.add_argument(
        "--min-alpha-words",
        type=int,
        default=200,
        help="Minimum alpha-word count for Gopher filter (default: 200; original Gopher uses 50)",
    )
    p.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable the wiki/CC quality classifier step (disabled by default; our "
             "model was trained on limited data and may be over-aggressive)",
    )
    p.add_argument(
        "--quality-cc-threshold",
        type=float,
        default=0.5,
        help="Reject docs labelled 'cc' with at least this confidence (default: 0.5)",
    )
    p.add_argument(
        "--dedup",
        action="store_true",
        help="Run exact-line deduplication on output JSONL files after filtering",
    )
    p.add_argument(
        "--dedup-only",
        action="store_true",
        help="Skip filtering; run dedup only on existing files in --output-dir",
    )
    return p.parse_args()


def run_dedup(output_dir: pathlib.Path) -> None:
    """Run exact-line deduplication across all JSONL files in output_dir."""
    from cs336_text_cleaning.deduplication import exact_line_deduplication

    jsonl_files = sorted(output_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No .jsonl files found for deduplication.")
        return

    dedup_dir = output_dir / "deduped"
    print(f"\nRunning exact-line deduplication on {len(jsonl_files)} files → {dedup_dir}")
    t0 = time.time()
    exact_line_deduplication(jsonl_files, dedup_dir)
    elapsed = time.time() - t0

    # Count lines before / after
    before = sum(sum(1 for _ in f.open()) for f in jsonl_files)
    after = sum(sum(1 for _ in f.open()) for f in sorted(dedup_dir.glob("*.jsonl")))
    print(f"Dedup complete in {elapsed:.1f}s: {before:,} → {after:,} lines "
          f"({100*(before-after)/max(before,1):.1f}% removed)")


def main() -> None:
    args = _parse_args()

    from cs336_text_cleaning.process_text import process_wet_files_parallel

    output_dir = pathlib.Path(args.output_dir)

    if not args.dedup_only:
        t0 = time.time()
        stats = process_wet_files_parallel(
            output_dir=str(output_dir),
            wet_dir=args.wet_dir,
            local_wet_glob=args.local_wet_glob,
            max_workers=args.max_workers,
            lang_threshold=args.lang_threshold,
            min_alpha_words=args.min_alpha_words,
            quality_filter=args.quality_filter,
            quality_cc_threshold=args.quality_cc_threshold,
        )
        elapsed = time.time() - t0
        print(f"\nFiltering done in {elapsed:.1f}s  ({elapsed/60:.1f} min)")

        # Estimate time for full 100k WET dump
        n_jsonl = len(list(output_dir.glob("*.jsonl")))
        if n_jsonl > 0:
            per_file = elapsed / n_jsonl          # wall-clock time per file (already parallelised)
            workers = args.max_workers or max(1, os.cpu_count() or 4)
            # On the cluster we'd have more CPUs; estimate for a few scenarios
            print(f"\nTiming: {per_file:.1f}s/file (wall clock with {n_jsonl} file(s) × {workers} workers)")
            for n_total, label in [(5_000, "5k WET files"), (100_000, "100k WET files")]:
                for w, w_label in [(workers, f"{workers} workers (this machine)"), (64, "64 workers (cluster)")]:
                    est_min = per_file * n_total / w / 60
                    print(f"  {label} with {w_label}: ~{est_min:.0f} min ({est_min/60:.1f} h)")

    if args.dedup or args.dedup_only:
        run_dedup(output_dir)


if __name__ == "__main__":
    main()
