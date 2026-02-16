#!/usr/bin/env python3
"""
Sweep over different context lengths to benchmark performance.

This script runs the benchmarking script for each sequence length
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
    sequence_length: int,
    model_size: str,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    pass_type: str,
    device: str,
    dtype: str,
    use_amp: bool,
    use_compile: bool,
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

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # Set environment to disable output buffering for real-time progress bars
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    # Run without capturing output to see progress bars in real-time
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"\nError running benchmark for sequence_length={sequence_length}")
        return None

    return {"sequence_length": sequence_length, "completed": True}


def main():
    parser = argparse.ArgumentParser(
        description="Sweep over context lengths for benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--sequence-lengths", nargs="+", type=int,
                       default=[128, 256, 512, 1024],
                       help="Sequence lengths to benchmark")
    parser.add_argument("--model-size", type=str, default="small",
                       choices=["small", "medium", "large", "xl", "2.7B"],
                       help="Model size to use")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--warmup-steps", type=int, default=10,
                       help="Number of warmup steps")
    parser.add_argument("--measure-steps", type=int, default=100,
                       help="Number of steps to measure")
    parser.add_argument("--pass-type", type=str, default="forward_backward",
                       choices=["forward", "forward_backward"],
                       help="Whether to measure forward only or forward+backward")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float16", "bfloat16"],
                       help="Data type to use")
    parser.add_argument("--use-amp", action="store_true",
                       help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--use-compile", action="store_true",
                       help="Use torch.compile")
    parser.add_argument("--output", type=str, default="context_length_sweep_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    results = []

    for sequence_length in tqdm(args.sequence_lengths, desc="Sweeping over context lengths"):
        result = run_benchmark(
            sequence_length=sequence_length,
            model_size=args.model_size,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            pass_type=args.pass_type,
            device=args.device,
            dtype=args.dtype,
            use_amp=args.use_amp,
            use_compile=args.use_compile,
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
