#!/usr/bin/env python3
"""
Sweep over different model sizes to benchmark performance.

This script runs the benchmarking script for each model size preset
and collects the results.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

def run_benchmark(
    model_size: str,
    batch_size: int,
    sequence_length: int,
    warmup_steps: int,
    measure_steps: int,
    pass_type: str,
    device: str,
    dtype: str,
    use_amp: bool,
    use_compile: bool,
    memory_snapshot_prefix: str | None = None,
) -> dict:
    """Run a single benchmark configuration."""
    cmd = [
        sys.executable,
        "benchmark.py",
        "--model-size", model_size,
        "--batch-size", str(batch_size),
        "--sequence-length", str(sequence_length),
        "--warmup-steps", str(warmup_steps),
        "--measure-steps", str(measure_steps),
        "--pass-type", pass_type,
        "--device", device,
        "--dtype", dtype,
    ]

    if use_amp:
        cmd.append("--use-amp")

    if use_compile:
        cmd.append("--use-compile")

    if memory_snapshot_prefix:
        snapshot_file = f"{memory_snapshot_prefix}_{model_size}.pickle"
        cmd.extend(["--memory-snapshot", snapshot_file])

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # Set environment to disable output buffering for real-time progress bars
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    # Run without capturing output to see progress bars in real-time
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"\nError running benchmark for {model_size}")
        return None

    return {"model_size": model_size, "completed": True}


def main():
    parser = argparse.ArgumentParser(
        description="Sweep over model sizes for benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-sizes", nargs="+",
                       default=["small", "medium", "large"],
                       choices=["small", "medium", "large", "xl", "2.7B"],
                       help="Model sizes to benchmark")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=512,
                       help="Sequence length")
    parser.add_argument("--warmup-steps", type=int, default=10,
                       help="Number of warmup steps")
    parser.add_argument("--measure-steps", type=int, default=100,
                       help="Number of steps to measure")
    parser.add_argument("--pass-type", type=str, default="forward_backward",
                       choices=["forward", "forward_backward"],
                       help="Whether to measure forward only or forward+backward")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="Device to run on")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float16", "bfloat16"],
                       help="Data type to use")
    parser.add_argument("--use-amp", action="store_true",
                       help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--use-compile", action="store_true",
                       help="Use torch.compile")
    parser.add_argument("--memory-snapshot-prefix", type=str, default=None,
                       help="Prefix for memory snapshot files (e.g., 'snapshots/mem')")
    parser.add_argument("--output", type=str, default="model_size_sweep_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    results = []

    for model_size in tqdm(args.model_sizes, desc="Sweeping over model sizes"):
        result = run_benchmark(
            model_size=model_size,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            pass_type=args.pass_type,
            device=args.device,
            dtype=args.dtype,
            use_amp=args.use_amp,
            use_compile=args.use_compile,
            memory_snapshot_prefix=args.memory_snapshot_prefix,
        )

        if result:
            results.append(result)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()