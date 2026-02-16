#!/usr/bin/env python3
"""
Benchmarking script for Transformer model forward and backward passes.

This script performs end-to-end benchmarking of the Transformer model,
measuring throughput and memory usage for different configurations.
"""

import argparse
import sys
import timeit
from typing import Literal
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn

# Import the model from cs336-basics
sys.path.insert(0, str(Path(__file__).parent / "cs336-basics"))

from cs336_basics.model import BasicsTransformerLM


def benchmark_model(
    # Model hyperparameters
    vocab_size: int = 50257,
    context_length: int = 1024,
    d_model: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    d_ff: int = 3072,
    rope_theta: float = 10000.0,
    # Batch configuration
    batch_size: int = 8,
    sequence_length: int = 512,
    # Benchmarking parameters
    warmup_steps: int = 10,
    measure_steps: int = 100,
    pass_type: Literal["forward", "forward_backward"] = "forward_backward",
    # Device and precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "float32",
    use_amp: bool = False,
    # Additional options
    use_compile: bool = False,
    profile_memory: bool = True,
    memory_snapshot: str | None = None,
):
    """
    Benchmark the Transformer model.

    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum context length for the model
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        rope_theta: RoPE theta value
        batch_size: Batch size for benchmarking
        sequence_length: Sequence length for benchmarking
        warmup_steps: Number of warmup steps before timing
        measure_steps: Number of steps to measure
        pass_type: Whether to measure "forward" only or "forward_backward"
        device: Device to run on ("cuda" or "cpu")
        dtype: Data type to use ("float32", "float16", "bfloat16")
        use_amp: Whether to use Automatic Mixed Precision (AMP) with autocast
        use_compile: Whether to use torch.compile
        profile_memory: Whether to profile memory usage
        memory_snapshot: If provided, save detailed memory snapshot to this file path

    Returns:
        Dictionary with benchmarking results
    """
    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    print("=" * 80)
    print("BENCHMARK CONFIGURATION")
    print("=" * 80)
    print(f"Model: {num_layers} layers, {d_model} dim, {num_heads} heads, {d_ff} ff_dim")
    print(f"Batch: {batch_size} x {sequence_length}")
    print(f"Device: {device}, Dtype: {dtype}")
    if use_amp:
        amp_dtype = "bfloat16" if dtype == "bfloat16" or torch.cuda.is_bf16_supported() else "float16"
        print(f"Mixed Precision: Enabled (AMP with {amp_dtype})")
    print(f"Pass type: {pass_type}")
    print(f"Warmup: {warmup_steps}, Measure: {measure_steps}")
    if use_compile:
        print("Using torch.compile")
    if memory_snapshot:
        print(f"Memory Snapshot: {memory_snapshot}")
    print("=" * 80)

    # Initialize model
    print("\nInitializing model...")
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )

    # Move model to device
    # If using AMP, keep model in FP32 and let autocast handle precision
    # Otherwise, cast model to specified dtype
    if use_amp:
        model = model.to(device=device)
    else:
        model = model.to(device=device, dtype=torch_dtype)

    # Optionally compile the model
    if use_compile:
        print("Compiling model...")
        model = torch.compile(model)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")

    # Generate random batch of data
    print(f"\nGenerating random data (batch_size={batch_size}, seq_len={sequence_length})...")
    input_ids = torch.randint(
        0, vocab_size, (batch_size, sequence_length), device=device
    )

    # Create dummy targets for loss computation (if doing backward pass)
    if pass_type == "forward_backward":
        targets = torch.randint(
            0, vocab_size, (batch_size, sequence_length), device=device
        )
        loss_fn = nn.CrossEntropyLoss()

    # Setup AMP context
    # Determine autocast dtype: prefer bfloat16 if supported, otherwise float16
    if use_amp and device == "cuda":
        amp_dtype = torch.bfloat16 if (dtype == "bfloat16" or torch.cuda.is_bf16_supported()) else torch.float16
    else:
        amp_dtype = None

    # Reset peak memory stats
    if device == "cuda" and profile_memory:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Warmup
    print(f"\nRunning {warmup_steps} warmup steps...")
    model.train() if pass_type == "forward_backward" else model.eval()

    for _ in tqdm(range(warmup_steps), desc="Warmup", file=sys.stdout, ncols=80, miniters=1):
        if pass_type == "forward":
            with torch.no_grad():
                if use_amp and device == "cuda":
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        outputs = model(input_ids)
                else:
                    outputs = model(input_ids)
        else:  # forward_backward
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    outputs = model(input_ids)
                    # Compute loss
                    logits = outputs.view(-1, vocab_size)
                    targets_flat = targets.view(-1)
                    loss = loss_fn(logits, targets_flat)
                loss.backward()
            else:
                outputs = model(input_ids)
                # Compute loss
                logits = outputs.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                loss = loss_fn(logits, targets_flat)
                loss.backward()
            # Zero gradients for next iteration
            model.zero_grad()

        # Synchronize to ensure all operations complete
        if device == "cuda":
            torch.cuda.synchronize()

    # Memory before measurement
    if device == "cuda" and profile_memory:
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3  # GB

    # Start detailed memory profiling if requested
    if device == "cuda" and memory_snapshot:
        print(f"\nStarting detailed memory profiling...")
        print(f"Snapshot will be saved to: {memory_snapshot}")
        try:
            torch.cuda.memory._record_memory_history(
                enabled=True,
                max_entries=100000,
            )
        except Exception as e:
            print(f"Warning: Could not start memory profiling: {e}")
            memory_snapshot = None

    # Measure performance
    print(f"\nMeasuring {measure_steps} steps...")

    def run_step():
        if pass_type == "forward":
            with torch.no_grad():
                if use_amp and device == "cuda":
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        outputs = model(input_ids)
                else:
                    outputs = model(input_ids)
        else:  # forward_backward
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    outputs = model(input_ids)
                    logits = outputs.view(-1, vocab_size)
                    targets_flat = targets.view(-1)
                    loss = loss_fn(logits, targets_flat)
                loss.backward()
            else:
                outputs = model(input_ids)
                logits = outputs.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                loss = loss_fn(logits, targets_flat)
                loss.backward()
            model.zero_grad()

        # Synchronize to ensure accurate timing
        if device == "cuda":
            torch.cuda.synchronize()

    # Time the execution
    timer = timeit.default_timer
    start_time = timer()

    for _ in tqdm(range(measure_steps), desc="Measuring", file=sys.stdout, ncols=80, miniters=1):
        run_step()

    end_time = timer()
    total_time = end_time - start_time

    # Save memory snapshot if requested
    if device == "cuda" and memory_snapshot:
        try:
            print(f"\nSaving memory snapshot to {memory_snapshot}...")
            torch.cuda.memory._dump_snapshot(memory_snapshot)
            print(f"Memory snapshot saved successfully!")
            print(f"\nTo visualize:")
            print(f"  1. Visit: https://pytorch.org/memory_viz")
            print(f"  2. Upload: {memory_snapshot}")
            print(f"  Or use: python -m torch.cuda.memory._visualizer {memory_snapshot}")
        except Exception as e:
            print(f"Warning: Could not save memory snapshot: {e}")
        finally:
            # Stop recording memory history
            torch.cuda.memory._record_memory_history(enabled=None)

    # Memory after measurement
    if device == "cuda" and profile_memory:
        mem_after = torch.cuda.memory_allocated() / 1024**3  # GB
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB

    # Calculate statistics
    avg_time_per_step = total_time / measure_steps
    throughput = batch_size / avg_time_per_step  # samples per second
    tokens_per_second = (batch_size * sequence_length) / avg_time_per_step

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per step: {avg_time_per_step * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Tokens per second: {tokens_per_second:.2f} tokens/sec")

    if device == "cuda" and profile_memory:
        print(f"\nMemory Usage:")
        print(f"  Allocated before: {mem_before:.3f} GB")
        print(f"  Allocated after: {mem_after:.3f} GB")
        print(f"  Peak memory: {peak_mem:.3f} GB")

    print("=" * 80)

    # Return results as dictionary
    results = {
        "total_time": total_time,
        "avg_time_per_step": avg_time_per_step,
        "throughput_samples_per_sec": throughput,
        "throughput_tokens_per_sec": tokens_per_second,
        "num_params": num_params,
    }

    if device == "cuda" and profile_memory:
        results.update({
            "memory_allocated_gb": mem_after,
            "peak_memory_gb": peak_mem,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer model forward and backward passes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model architecture hyperparameters
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--vocab-size", type=int, default=50257,
                            help="Vocabulary size")
    model_group.add_argument("--context-length", type=int, default=1024,
                            help="Maximum context length")
    model_group.add_argument("--d-model", type=int, default=768,
                            help="Model dimension")
    model_group.add_argument("--num-layers", type=int, default=12,
                            help="Number of transformer layers")
    model_group.add_argument("--num-heads", type=int, default=12,
                            help="Number of attention heads")
    model_group.add_argument("--d-ff", type=int, default=3072,
                            help="Feed-forward dimension")
    model_group.add_argument("--rope-theta", type=float, default=10000.0,
                            help="RoPE theta value")

    # Model size presets (matching the table in the assignment)
    model_group.add_argument("--model-size", type=str, choices=["small", "medium", "large", "xl", "2.7B"],
                            help="Use preset model size (overrides individual params)")

    # Batch configuration
    batch_group = parser.add_argument_group("Batch Configuration")
    batch_group.add_argument("--batch-size", type=int, default=8,
                            help="Batch size")
    batch_group.add_argument("--sequence-length", type=int, default=512,
                            help="Sequence length")

    # Benchmarking parameters
    bench_group = parser.add_argument_group("Benchmarking Parameters")
    bench_group.add_argument("--warmup-steps", type=int, default=10,
                            help="Number of warmup steps")
    bench_group.add_argument("--measure-steps", type=int, default=100,
                            help="Number of steps to measure")
    bench_group.add_argument("--pass-type", type=str, default="forward_backward",
                            choices=["forward", "forward_backward"],
                            help="Whether to measure forward only or forward+backward")

    # Device and precision
    device_group = parser.add_argument_group("Device and Precision")
    device_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                             choices=["cuda", "cpu", "mps"],
                             help="Device to run on")
    device_group.add_argument("--dtype", type=str, default="float32",
                             choices=["float32", "float16", "bfloat16"],
                             help="Data type to use")

    # Additional options
    other_group = parser.add_argument_group("Additional Options")
    other_group.add_argument("--use-amp", action="store_true",
                            help="Use Automatic Mixed Precision (AMP) training")
    other_group.add_argument("--use-compile", action="store_true",
                            help="Use torch.compile for optimization")
    other_group.add_argument("--no-memory-profile", action="store_true",
                            help="Disable memory profiling")
    other_group.add_argument("--memory-snapshot", type=str, default=None,
                            help="Save detailed memory snapshot to file (e.g., memory_snapshot.pickle)")

    args = parser.parse_args()

    # Apply model size presets if specified
    model_sizes = {
        "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }

    if args.model_size:
        preset = model_sizes[args.model_size]
        args.d_model = preset["d_model"]
        args.d_ff = preset["d_ff"]
        args.num_layers = preset["num_layers"]
        args.num_heads = preset["num_heads"]

    # Run benchmark
    results = benchmark_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        pass_type=args.pass_type,
        device=args.device,
        dtype=args.dtype,
        use_amp=args.use_amp,
        use_compile=args.use_compile,
        profile_memory=not args.no_memory_profile,
        memory_snapshot=args.memory_snapshot,
    )

    return results


if __name__ == "__main__":
    main()