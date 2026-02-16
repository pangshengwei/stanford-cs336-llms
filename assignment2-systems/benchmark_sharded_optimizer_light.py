"""
Lightweight benchmark for training speed with and without optimizer state sharding.

Usage:
    python benchmark_sharded_optimizer_light.py --world-size 2
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

from cs336_systems.sharded_optimizer import ShardedOptimizer


class SmallModel(nn.Module):
    """Small model for fast benchmarking."""
    def __init__(self, d_model=256, n_layers=4, vocab_size=10000):
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
    os.environ["MASTER_PORT"] = "12366"
    device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group():
    """Clean up the distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def benchmark_optimizer(rank, world_size, backend, use_sharded, num_iterations, batch_size, seq_length):
    """Benchmark with or without sharded optimizer."""
    device = setup_process_group(rank, world_size, backend)
    dist.barrier()

    torch.manual_seed(42)

    # Create model
    model = SmallModel().to(device)

    # Wrap with DDP
    ddp_model = nn.parallel.DistributedDataParallel(model)

    # Create optimizer (sharded or regular)
    if use_sharded:
        optimizer = ShardedOptimizer(
            ddp_model.parameters(),
            optimizer_cls=optim.AdamW,
            lr=1e-4,
            weight_decay=0.01
        )
        config_name = "Sharded AdamW"
    else:
        optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-4, weight_decay=0.01)
        config_name = "Regular AdamW"

    # Generate random data
    vocab_size = 10000
    data = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup iterations
    for _ in range(2):
        optimizer.zero_grad()
        outputs = ddp_model(data)
        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

    dist.barrier()

    # Benchmark iterations
    iteration_times = []

    for i in range(num_iterations):
        iter_start = time.perf_counter()

        optimizer.zero_grad()
        outputs = ddp_model(data)
        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

        iter_end = time.perf_counter()
        iteration_times.append(iter_end - iter_start)

    # Compute average
    avg_iter_time = sum(iteration_times) / len(iteration_times)

    # Collect results from all ranks
    iter_time_tensor = torch.tensor(avg_iter_time, device=device)
    dist.all_reduce(iter_time_tensor, op=dist.ReduceOp.SUM)
    avg_iter_time = iter_time_tensor.item() / world_size

    if rank == 0:
        total_params = sum(p.numel() for p in ddp_model.parameters())
        param_size_mb = (total_params * 4) / (1024 * 1024)
        print(f"\n{'='*70}")
        print(f"{config_name}")
        print(f"{'='*70}")
        print(f"  Model: {total_params:,} params ({param_size_mb:.2f} MB)")
        print(f"  Avg iteration time: {avg_iter_time*1000:.2f} ms")
        print(f"  Throughput: {1.0/avg_iter_time:.2f} iter/s")

    cleanup_process_group()
    return avg_iter_time


def run_benchmark(rank, world_size, backend, args):
    """Run benchmark for both configurations."""
    results = {}

    # Benchmark regular optimizer
    regular_time = benchmark_optimizer(
        rank, world_size, backend,
        use_sharded=False,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        seq_length=args.seq_length
    )
    results['regular'] = regular_time

    # Benchmark sharded optimizer
    sharded_time = benchmark_optimizer(
        rank, world_size, backend,
        use_sharded=True,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        seq_length=args.seq_length
    )
    results['sharded'] = sharded_time

    # Print comparison
    if rank == 0:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"  Regular AdamW:  {results['regular']*1000:.2f} ms/iter")
        print(f"  Sharded AdamW:  {results['sharded']*1000:.2f} ms/iter")
        overhead = ((results['sharded'] - results['regular']) / results['regular']) * 100
        print(f"  Overhead:       {overhead:+.2f}%")
        if abs(overhead) < 5:
            print(f"\n  Performance is essentially identical (within measurement noise)")
        elif overhead > 0:
            print(f"\n  Sharded optimizer is {overhead:.1f}% slower")
            print(f"  Overhead from serial broadcasts ({world_size} ranks)")
        else:
            print(f"\n  Sharded optimizer is {abs(overhead):.1f}% faster (measurement variance)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sharded optimizer")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per rank")
    parser.add_argument("--seq-length", type=int, default=64, help="Sequence length")

    args = parser.parse_args()

    print(f"Benchmarking Sharded Optimizer:")
    print(f"  Backend: {args.backend}")
    print(f"  World size: {args.world_size}")
    print(f"  Model: Small (256 dim, 4 layers, ~7M params)")

    mp.spawn(
        run_benchmark,
        args=(args.world_size, args.backend, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
