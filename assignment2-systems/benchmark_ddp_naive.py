"""
Benchmark script for naive DDP implementation.

This script benchmarks the performance of the DDP implementation and compares
throughput and communication overhead across different configurations.

Usage:
    # CPU with Gloo backend
    python benchmark_ddp_naive.py --backend gloo --device cpu --world-size 2 4 6

    # GPU with NCCL backend (if CUDA is available)
    python benchmark_ddp_naive.py --backend nccl --device cuda --world-size 2 4
"""
import argparse
import os
import time
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from cs336_systems.ddp_naive import NaiveDDP


class ToyModel(nn.Module):
    """Simple toy model for benchmarking."""
    def __init__(self, input_size=100, hidden_size=200, output_size=50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def setup_process_group(rank, world_size, backend):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

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


def benchmark_ddp(rank, world_size, backend, num_iterations, batch_size, model_size):
    """
    Benchmark DDP training performance.

    Measures:
    - Total training time
    - Time per iteration
    - Gradient synchronization overhead
    """
    device = setup_process_group(rank, world_size, backend)
    dist.barrier()

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Model parameters
    input_size, hidden_size, output_size = model_size

    # Create DDP model
    model = ToyModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    ddp_model = NaiveDDP(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Generate random data
    data = torch.randn(batch_size, input_size, device=device)
    labels = torch.randn(batch_size, output_size, device=device)

    loss_fn = nn.MSELoss()

    # Warmup iterations
    for _ in range(3):
        optimizer.zero_grad()
        outputs = ddp_model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

    # Benchmark iterations
    dist.barrier()
    start_time = time.time()
    iteration_times = []
    sync_times = []

    for i in range(num_iterations):
        iter_start = time.time()

        optimizer.zero_grad()
        outputs = ddp_model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Measure gradient synchronization time
        sync_start = time.time()
        ddp_model.finish_gradient_synchronization()
        sync_end = time.time()
        sync_times.append(sync_end - sync_start)

        optimizer.step()

        iter_end = time.time()
        iteration_times.append(iter_end - iter_start)

    end_time = time.time()
    total_time = end_time - start_time

    # Compute statistics
    avg_iter_time = sum(iteration_times) / len(iteration_times)
    avg_sync_time = sum(sync_times) / len(sync_times)
    throughput = (num_iterations * batch_size * world_size) / total_time

    if rank == 0:
        print(f"\nBenchmark Results (world_size={world_size}, batch_size={batch_size}):")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg iteration time: {avg_iter_time*1000:.2f}ms")
        print(f"  Avg sync time: {avg_sync_time*1000:.2f}ms")
        print(f"  Sync overhead: {(avg_sync_time/avg_iter_time)*100:.1f}%")
        print(f"  Throughput: {throughput:.1f} samples/sec")

        # Calculate total parameters
        total_params = sum(p.numel() for p in ddp_model.parameters())
        param_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"  Model size: {total_params:,} parameters ({param_size_mb:.2f} MB)")

    cleanup_process_group()
    return total_time, avg_iter_time, avg_sync_time, throughput


def main():
    parser = argparse.ArgumentParser(description="Benchmark naive DDP implementation")
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
        nargs="+",
        default=[2, 4],
        help="Number of processes to test (can specify multiple)"
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
        default=32,
        help="Batch size per rank"
    )
    parser.add_argument(
        "--model-size",
        type=int,
        nargs=3,
        default=[100, 200, 50],
        metavar=("INPUT", "HIDDEN", "OUTPUT"),
        help="Model size (input hidden output)"
    )

    args = parser.parse_args()

    # Validate backend and device
    if args.backend == "nccl" and args.device == "cpu":
        print("Warning: NCCL backend requires CUDA. Switching to gloo backend.")
        args.backend = "gloo"

    print(f"Benchmarking naive DDP implementation:")
    print(f"  Backend: {args.backend}")
    print(f"  Device: {args.device}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Batch size per rank: {args.batch_size}")
    print(f"  Model: {args.model_size[0]} -> {args.model_size[1]} -> {args.model_size[2]}")
    print()

    results = {}
    for world_size in args.world_size:
        print(f"\n{'='*60}")
        print(f"Testing with world_size={world_size}")
        print(f"{'='*60}")

        mp.spawn(
            benchmark_ddp,
            args=(world_size, args.backend, args.iterations, args.batch_size, tuple(args.model_size)),
            nprocs=world_size,
            join=True
        )

    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()