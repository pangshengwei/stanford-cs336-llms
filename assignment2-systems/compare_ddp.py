"""
Comparison benchmark for naive vs. flattened DDP implementations.

This script compares the performance of:
1. Naive DDP: Individual all-reduce per parameter
2. Flattened DDP: Single batched all-reduce for all parameters

Usage:
    # Compare on CPU with Gloo
    python compare_ddp.py --backend gloo --device cpu --world-size 2

    # Compare on GPU with NCCL
    python compare_ddp.py --backend nccl --device cuda --world-size 2
"""
import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from cs336_systems.ddp_naive import NaiveDDP
from cs336_systems.ddp_flat import FlattenedDDP


class XLModel(nn.Module):
    """
    XL model size as described in the assignment.
    Similar to GPT-2 XL: ~1.5B parameters would be too large for testing,
    so we use a smaller representative model with multiple layers.
    """
    def __init__(self, d_model=512, n_layers=6, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x


def setup_process_group(rank, world_size, backend):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available for NCCL backend")
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise ValueError("No CUDA devices found")
        local_rank = rank % device_count
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group():
    """Clean up the distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def benchmark_implementation(
    rank, world_size, backend, implementation, num_iterations, batch_size, seq_length, model_params
):
    """
    Benchmark a single DDP implementation.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend (gloo or nccl)
        implementation: "naive" or "flattened"
        num_iterations: Number of training iterations
        batch_size: Batch size per rank
        seq_length: Sequence length for input
        model_params: Dict with model parameters
    """
    device = setup_process_group(rank, world_size, backend)
    dist.barrier()

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create model
    model = XLModel(**model_params).to(device)

    # Wrap with appropriate DDP implementation
    if implementation == "naive":
        ddp_model = NaiveDDP(model)
    elif implementation == "flattened":
        ddp_model = FlattenedDDP(model)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Generate random data
    vocab_size = model_params["vocab_size"]
    data = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    loss_fn = nn.CrossEntropyLoss()

    # Warmup iterations
    for _ in range(3):
        optimizer.zero_grad()
        outputs = ddp_model(data)
        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

    # Synchronize before benchmark
    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier()

    # Benchmark iterations
    iteration_times = []
    sync_times = []
    compute_times = []

    for i in range(num_iterations):
        iter_start = time.perf_counter()

        optimizer.zero_grad()

        # Forward and backward (computation)
        compute_start = time.perf_counter()
        outputs = ddp_model(data)
        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        compute_end = time.perf_counter()
        compute_times.append(compute_end - compute_start)

        # Gradient synchronization (communication)
        sync_start = time.perf_counter()
        ddp_model.finish_gradient_synchronization()
        if backend == "nccl":
            torch.cuda.synchronize()
        sync_end = time.perf_counter()
        sync_times.append(sync_end - sync_start)

        optimizer.step()

        iter_end = time.perf_counter()
        iteration_times.append(iter_end - iter_start)

    # Gather timing statistics from all ranks
    avg_iter_time = sum(iteration_times) / len(iteration_times)
    avg_sync_time = sum(sync_times) / len(sync_times)
    avg_compute_time = sum(compute_times) / len(compute_times)

    # Collect results from all ranks
    iter_time_tensor = torch.tensor(avg_iter_time, device=device)
    sync_time_tensor = torch.tensor(avg_sync_time, device=device)
    compute_time_tensor = torch.tensor(avg_compute_time, device=device)

    dist.all_reduce(iter_time_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(sync_time_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(compute_time_tensor, op=dist.ReduceOp.AVG)

    avg_iter_time = iter_time_tensor.item()
    avg_sync_time = sync_time_tensor.item()
    avg_compute_time = compute_time_tensor.item()

    if rank == 0:
        # Calculate model size
        total_params = sum(p.numel() for p in ddp_model.parameters())
        param_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32

        print(f"\n{implementation.upper()} DDP Results:")
        print(f"  Model size: {total_params:,} parameters ({param_size_mb:.2f} MB)")
        print(f"  Avg iteration time: {avg_iter_time*1000:.2f} ms")
        print(f"  Avg compute time: {avg_compute_time*1000:.2f} ms")
        print(f"  Avg communication time: {avg_sync_time*1000:.2f} ms")
        print(f"  Communication overhead: {(avg_sync_time/avg_iter_time)*100:.1f}%")

    cleanup_process_group()

    return {
        "implementation": implementation,
        "avg_iter_time": avg_iter_time,
        "avg_sync_time": avg_sync_time,
        "avg_compute_time": avg_compute_time,
        "total_params": total_params if rank == 0 else None,
    }


def run_comparison(rank, world_size, backend, args):
    """Run comparison between naive and flattened implementations."""
    model_params = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "vocab_size": args.vocab_size,
    }

    results = {}

    # Benchmark naive implementation
    if rank == 0:
        print(f"\n{'='*60}")
        print("Testing NAIVE DDP (individual parameter all-reduce)")
        print(f"{'='*60}")

    naive_result = benchmark_implementation(
        rank, world_size, backend, "naive",
        args.iterations, args.batch_size, args.seq_length, model_params
    )
    results["naive"] = naive_result

    # Benchmark flattened implementation
    if rank == 0:
        print(f"\n{'='*60}")
        print("Testing FLATTENED DDP (single batched all-reduce)")
        print(f"{'='*60}")

    flat_result = benchmark_implementation(
        rank, world_size, backend, "flattened",
        args.iterations, args.batch_size, args.seq_length, model_params
    )
    results["flattened"] = flat_result

    # Print comparison
    if rank == 0:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")

        naive_iter = results["naive"]["avg_iter_time"]
        flat_iter = results["flattened"]["avg_iter_time"]
        iter_speedup = naive_iter / flat_iter

        naive_sync = results["naive"]["avg_sync_time"]
        flat_sync = results["flattened"]["avg_sync_time"]
        sync_speedup = naive_sync / flat_sync

        print(f"\nIteration Time:")
        print(f"  Naive:     {naive_iter*1000:.2f} ms")
        print(f"  Flattened: {flat_iter*1000:.2f} ms")
        print(f"  Speedup:   {iter_speedup:.2f}x")

        print(f"\nCommunication Time:")
        print(f"  Naive:     {naive_sync*1000:.2f} ms")
        print(f"  Flattened: {flat_sync*1000:.2f} ms")
        print(f"  Speedup:   {sync_speedup:.2f}x")

        print(f"\nConclusion:")
        if sync_speedup > 1.5:
            print(f"  ✓ Flattened DDP provides {sync_speedup:.1f}x faster gradient communication")
            print(f"    by batching all-reduce operations into a single call, reducing")
            print(f"    communication overhead and better utilizing network bandwidth.")
        elif sync_speedup > 1.1:
            print(f"  ✓ Flattened DDP provides moderate {sync_speedup:.1f}x improvement")
            print(f"    in gradient communication time.")
        else:
            print(f"  ~ Minimal difference ({sync_speedup:.2f}x) - model may be too small")
            print(f"    to see significant benefits from batching.")


def main():
    parser = argparse.ArgumentParser(description="Compare naive vs. flattened DDP")
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Backend for distributed training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device type"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of processes"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per rank"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Model dimension"
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=6,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Vocabulary size"
    )

    args = parser.parse_args()

    # Validate backend and device
    if args.backend == "nccl" and args.device == "cpu":
        print("Warning: NCCL backend requires CUDA. Switching to gloo backend.")
        args.backend = "gloo"

    print(f"Comparing DDP Implementations:")
    print(f"  Backend: {args.backend}")
    print(f"  Device: {args.device}")
    print(f"  World size: {args.world_size}")
    print(f"  Model: d_model={args.d_model}, n_layers={args.n_layers}")
    print(f"  Batch size per rank: {args.batch_size}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Iterations: {args.iterations}")

    # Spawn processes
    mp.spawn(
        run_comparison,
        args=(args.world_size, args.backend, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()