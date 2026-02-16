"""
Test script for naive DDP implementation.

This script verifies the correctness of the DDP implementation by comparing
results from distributed training with single-process training on the full dataset.

Usage:
    # CPU with Gloo backend
    python test_ddp_naive.py --backend gloo --device cpu --world-size 2

    # GPU with NCCL backend (if CUDA is available)
    python test_ddp_naive.py --backend nccl --device cuda --world-size 2
"""
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

# Add the cs336_systems directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from cs336_systems.ddp_naive import NaiveDDP


class ToyModel(nn.Module):
    """Simple toy model for testing DDP."""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
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
    os.environ["MASTER_PORT"] = "12355"

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


def test_ddp_training(rank, world_size, backend, num_iterations=5):
    """
    Test DDP training by comparing with single-process baseline.

    Each rank:
    1. Creates a model and wraps it with DDP
    2. Trains on a shard of the data
    3. Verifies that final weights match single-process training
    """
    device = setup_process_group(rank, world_size, backend)
    dist.barrier()

    # Set seed to ensure different initial models per rank (to test broadcast)
    torch.manual_seed(rank)

    # Model parameters
    batch_size = 16
    input_size = 10
    output_size = 5

    # Create non-parallel baseline model (all ranks need this for verification)
    non_parallel_model = ToyModel(input_size=input_size, output_size=output_size).to(device)
    baseline_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.01)

    # Create DDP model by deepcopying the baseline (this is key!)
    # This ensures they start from the same initial weights
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = NaiveDDP(ddp_base)
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Verify that after DDP initialization, all ranks have the same parameters
    # and they match rank 0's initial parameters
    if rank == 0:
        print(f"Rank {rank}: Verifying parameter broadcast...")
        for name, param in ddp_model.module.named_parameters():
            print(f"  {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")

    # Generate random data (same data on all ranks for reproducibility)
    torch.manual_seed(42)
    all_x = torch.randn(batch_size * world_size, input_size)
    all_y = torch.randn(batch_size * world_size, output_size)

    loss_fn = nn.MSELoss()

    if rank == 0:
        print(f"\nRank {rank}: Starting training for {num_iterations} iterations...")

    for i in range(num_iterations):
        # Non-parallel baseline (all ranks do this to stay in sync)
        baseline_optimizer.zero_grad()
        non_parallel_data = all_x.to(device)
        non_parallel_labels = all_y.to(device)
        non_parallel_outputs = non_parallel_model(non_parallel_data)
        non_parallel_loss = loss_fn(non_parallel_outputs, non_parallel_labels)
        non_parallel_loss.backward()
        baseline_optimizer.step()

        # DDP training (all ranks)
        ddp_optimizer.zero_grad()

        # Each rank gets a different shard of the data
        offset = rank * batch_size
        ddp_data = all_x[offset:offset + batch_size, :].to(device)
        ddp_labels = all_y[offset:offset + batch_size, :].to(device)

        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        ddp_loss.backward()

        # Wait for all gradient synchronization to complete
        ddp_model.finish_gradient_synchronization()

        ddp_optimizer.step()

        if rank == 0 and (i + 1) % 2 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: "
                  f"baseline_loss={non_parallel_loss.item():.6f}, "
                  f"ddp_loss={ddp_loss.item():.6f}")

    # Verify that DDP weights match baseline on rank 0
    if rank == 0:
        print(f"\nRank {rank}: Verifying final weights match baseline...")
        max_diff = 0.0
        all_match = True
        for (name, baseline_param), (_, ddp_param) in zip(
            non_parallel_model.named_parameters(),
            ddp_model.module.named_parameters()
        ):
            diff = (baseline_param - ddp_param).abs().max().item()
            max_diff = max(max_diff, diff)
            matches = torch.allclose(baseline_param, ddp_param, atol=1e-5)
            all_match = all_match and matches
            print(f"  {name}: max_diff={diff:.2e}, matches={matches}")

        if all_match:
            print(f"\n✓ SUCCESS: DDP weights match baseline (max_diff={max_diff:.2e})")
        else:
            print(f"\n✗ FAILURE: DDP weights do not match baseline (max_diff={max_diff:.2e})")

    # Verify that all ranks have the same parameters
    if rank == 0:
        print(f"\nRank {rank}: Verifying all ranks have synchronized weights...")

    for name, param in ddp_model.module.named_parameters():
        # Gather parameters from all ranks
        param_list = [torch.zeros_like(param) for _ in range(world_size)]
        dist.all_gather(param_list, param)

        # Check that all are equal
        if rank == 0:
            all_equal = all(torch.allclose(param_list[0], p, atol=1e-5) for p in param_list[1:])
            if not all_equal:
                print(f"  {name}: NOT SYNCHRONIZED across ranks!")
            else:
                max_diff_across_ranks = max((param_list[0] - p).abs().max().item() for p in param_list[1:])
                print(f"  {name}: synchronized (max_diff={max_diff_across_ranks:.2e})")

    if rank == 0:
        print(f"\n✓ Test completed successfully on rank {rank}")

    cleanup_process_group()


def main():
    parser = argparse.ArgumentParser(description="Test naive DDP implementation")
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Backend for distributed training (gloo for CPU, nccl for GPU)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device type (cpu or cuda)"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        choices=[2, 4, 6],
        help="Number of processes to spawn"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of training iterations"
    )

    args = parser.parse_args()

    # Validate backend and device combination
    if args.backend == "nccl" and args.device == "cpu":
        print("Warning: NCCL backend requires CUDA. Switching to gloo backend.")
        args.backend = "gloo"

    if args.backend == "gloo" and args.device == "cuda":
        print("Warning: Using gloo backend with CUDA (nccl is recommended for GPU)")

    print(f"Testing naive DDP implementation:")
    print(f"  Backend: {args.backend}")
    print(f"  Device: {args.device}")
    print(f"  World size: {args.world_size}")
    print(f"  Iterations: {args.iterations}")
    print()

    # Spawn processes for distributed training
    mp.spawn(
        test_ddp_training,
        args=(args.world_size, args.backend, args.iterations),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()