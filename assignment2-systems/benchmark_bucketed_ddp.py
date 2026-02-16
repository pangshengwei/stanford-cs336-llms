"""
Benchmark bucketed DDP with varying bucket sizes.

This script benchmarks the bucketed DDP implementation with different bucket sizes
and compares with previous implementations (overlap, flattened).

Usage:
    # Benchmark different bucket sizes
    python benchmark_bucketed_ddp.py --backend gloo --device cpu --world-size 2

    # GPU with NCCL
    python benchmark_bucketed_ddp.py --backend nccl --device cuda --world-size 2
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from cs336_systems.ddp_overlap import DDPWithOverlap
from cs336_systems.ddp_flat import FlattenedDDP
from cs336_systems.ddp_bucketed import BucketedDDP


class LargeModel(nn.Module):
    """Large transformer model for benchmarking."""
    def __init__(self, d_model=768, n_layers=12, vocab_size=50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=12,
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
    os.environ["MASTER_PORT"] = "12360"

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


def benchmark_config(rank, world_size, backend, config_name, ddp_model, num_iterations, batch_size, seq_length, vocab_size):
    """Benchmark a single DDP configuration."""
    device = ddp_model.module.embedding.weight.device

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Generate random data
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

    # Compute averages
    avg_iter_time = sum(iteration_times) / len(iteration_times)
    avg_sync_time = sum(sync_times) / len(sync_times)
    avg_compute_time = sum(compute_times) / len(compute_times)

    # Collect results from all ranks
    iter_time_tensor = torch.tensor(avg_iter_time, device=device)
    sync_time_tensor = torch.tensor(avg_sync_time, device=device)
    compute_time_tensor = torch.tensor(avg_compute_time, device=device)

    dist.all_reduce(iter_time_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(sync_time_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(compute_time_tensor, op=dist.ReduceOp.SUM)

    avg_iter_time = iter_time_tensor.item() / world_size
    avg_sync_time = sync_time_tensor.item() / world_size
    avg_compute_time = compute_time_tensor.item() / world_size

    return {
        "config": config_name,
        "avg_iter_time": avg_iter_time,
        "avg_sync_time": avg_sync_time,
        "avg_compute_time": avg_compute_time,
    }


def run_benchmark(rank, world_size, backend, args):
    """Run benchmark with different bucket sizes."""
    device = setup_process_group(rank, world_size, backend)
    dist.barrier()

    torch.manual_seed(42)

    model_params = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "vocab_size": args.vocab_size,
    }

    results = {}

    # Test different bucket sizes
    bucket_sizes = args.bucket_sizes

    for bucket_size in bucket_sizes:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Bucketed DDP: bucket_size={bucket_size}MB")
            print(f"{'='*70}")

        # Create fresh model
        model = LargeModel(**model_params).to(device)
        ddp_model = BucketedDDP(model, bucket_size_mb=bucket_size)

        result = benchmark_config(
            rank, world_size, backend, f"Bucketed-{bucket_size}MB",
            ddp_model, args.iterations, args.batch_size, args.seq_length, model_params["vocab_size"]
        )

        if rank == 0:
            total_params = sum(p.numel() for p in ddp_model.parameters())
            param_size_mb = (total_params * 4) / (1024 * 1024)
            print(f"  Model: {total_params:,} params ({param_size_mb:.2f} MB)")
            print(f"  Iteration time: {result['avg_iter_time']*1000:.2f} ms")
            print(f"  Compute time:   {result['avg_compute_time']*1000:.2f} ms")
            print(f"  Comm time:      {result['avg_sync_time']*1000:.2f} ms")
            print(f"  Comm overhead:  {(result['avg_sync_time']/result['avg_iter_time'])*100:.1f}%")

        results[f"Bucketed-{bucket_size}MB"] = result

        cleanup_process_group()

        # Re-initialize for next config
        if bucket_size != bucket_sizes[-1]:
            device = setup_process_group(rank, world_size, backend)
            dist.barrier()
            torch.manual_seed(42)

    # Benchmark baseline implementations for comparison
    if args.compare_baseline:
        for impl_name, impl_desc, impl_class, impl_args in [
            ("Overlap", "Overlap DDP (async per-parameter)", DDPWithOverlap, {}),
            ("Flattened", "Flattened DDP (single batched all-reduce)", FlattenedDDP, {}),
        ]:
            device = setup_process_group(rank, world_size, backend)
            dist.barrier()
            torch.manual_seed(42)

            if rank == 0:
                print(f"\n{'='*70}")
                print(impl_desc)
                print(f"{'='*70}")

            model = LargeModel(**model_params).to(device)
            ddp_model = impl_class(model, **impl_args)

            result = benchmark_config(
                rank, world_size, backend, impl_name,
                ddp_model, args.iterations, args.batch_size, args.seq_length, model_params["vocab_size"]
            )

            if rank == 0:
                print(f"  Iteration time: {result['avg_iter_time']*1000:.2f} ms")
                print(f"  Compute time:   {result['avg_compute_time']*1000:.2f} ms")
                print(f"  Comm time:      {result['avg_sync_time']*1000:.2f} ms")
                print(f"  Comm overhead:  {(result['avg_sync_time']/result['avg_iter_time'])*100:.1f}%")

            results[impl_name] = result
            cleanup_process_group()

    # Print summary
    if rank == 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"\n{'Configuration':<30} {'Iter (ms)':<12} {'Comm (ms)':<12} {'Overhead':<10}")
        print("-" * 70)

        for config_name in sorted(results.keys()):
            r = results[config_name]
            iter_ms = r['avg_iter_time'] * 1000
            comm_ms = r['avg_sync_time'] * 1000
            overhead = (r['avg_sync_time'] / r['avg_iter_time']) * 100
            print(f"{config_name:<30} {iter_ms:<12.2f} {comm_ms:<12.2f} {overhead:<10.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Benchmark bucketed DDP")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per rank")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument(
        "--bucket-sizes",
        type=float,
        nargs="+",
        default=[1, 10, 25, 100, 1000],
        help="Bucket sizes in MB to test"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also compare with overlap and flattened implementations"
    )

    args = parser.parse_args()

    if args.backend == "nccl" and args.device == "cpu":
        print("Warning: NCCL requires CUDA. Switching to gloo.")
        args.backend = "gloo"

    print(f"Benchmarking Bucketed DDP:")
    print(f"  Backend: {args.backend}")
    print(f"  Device: {args.device}")
    print(f"  World size: {args.world_size}")
    print(f"  Model: d_model={args.d_model}, n_layers={args.n_layers}")
    print(f"  Bucket sizes: {args.bucket_sizes} MB")

    mp.spawn(
        run_benchmark,
        args=(args.world_size, args.backend, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
