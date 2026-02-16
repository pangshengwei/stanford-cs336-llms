# Naive Distributed Data Parallel (DDP) Implementation

This directory contains a minimal implementation of Distributed Data Parallel (DDP) training in PyTorch.

## Implementation Overview

### Files

- **`cs336_systems/ddp_naive.py`**: Core DDP implementation
- **`test_ddp_naive.py`**: Correctness verification script
- **`benchmark_ddp_naive.py`**: Performance benchmarking script

### How It Works

The naive DDP implementation follows these steps:

1. **Parameter Broadcast**: During initialization, parameters from rank 0 are broadcast to all other ranks using `dist.broadcast()`. This ensures all ranks start with identical model weights.

2. **Forward Pass**: Each rank performs a forward pass on its local data shard (n/world_size examples).

3. **Backward Pass**: Each rank computes gradients on its local data shard. Backward hooks are registered on all parameters to trigger gradient all-reduce as soon as each parameter's gradient is computed.

4. **Gradient All-Reduce**: Using `dist.all_reduce()` with `ReduceOp.AVG`, gradients are averaged across all ranks. This is done asynchronously (`async_op=True`) to allow communication to overlap with backward computation.

5. **Gradient Synchronization**: Before the optimizer step, we wait for all async all-reduce operations to complete via `finish_gradient_synchronization()`.

6. **Optimizer Step**: Each rank updates its local parameters using the averaged gradients. Since all ranks started with the same parameters and used the same averaged gradients, they remain synchronized.

## Key Design Decisions

### Async All-Reduce
The implementation uses asynchronous all-reduce operations (`async_op=True`) to allow communication to overlap with backward computation. This is more efficient than waiting for each parameter's gradient to be synchronized before computing the next one.

### AVG Operation
We use `ReduceOp.AVG` instead of `ReduceOp.SUM` to average gradients across ranks. This ensures that the effective learning rate remains consistent regardless of the number of processes.

### Backward Hooks
Backward hooks are registered on all parameters that require gradients. These hooks are called automatically during the backward pass, as soon as each parameter's gradient is computed. This allows gradient communication to begin before the entire backward pass is complete.

## Usage

### Testing Correctness

```bash
# Test with CPU and Gloo backend
python test_ddp_naive.py --backend gloo --device cpu --world-size 2

# Test with different world sizes
python test_ddp_naive.py --backend gloo --device cpu --world-size 4
python test_ddp_naive.py --backend gloo --device cpu --world-size 6

# Test with GPU and NCCL backend (if CUDA available)
python test_ddp_naive.py --backend nccl --device cuda --world-size 2
```

The test script verifies that:
- Parameters are correctly broadcast from rank 0 to all ranks
- All ranks have synchronized weights after each iteration
- DDP training produces identical results to single-process training on the full batch

### Benchmarking Performance

```bash
# Benchmark with different world sizes
python benchmark_ddp_naive.py --backend gloo --device cpu --world-size 2 4 6

# Customize batch size and iterations
python benchmark_ddp_naive.py --backend gloo --device cpu --world-size 2 4 \
    --batch-size 64 --iterations 20

# Benchmark with larger model
python benchmark_ddp_naive.py --backend gloo --device cpu --world-size 2 4 \
    --model-size 200 400 100
```

The benchmark measures:
- Total training time
- Average iteration time
- Average gradient synchronization time
- Synchronization overhead percentage
- Throughput (samples/second)

## Performance Characteristics

### Synchronization Overhead
The naive implementation has high synchronization overhead, especially with larger world sizes. This is because:
1. Each parameter gradient is all-reduced individually
2. Small tensors have high latency-to-bandwidth ratio
3. No bucketing or gradient accumulation is used

### Scaling Behavior
- **World size = 2**: ~76% sync overhead
- **World size = 4**: ~89% sync overhead

As world size increases:
- Communication overhead increases
- Throughput may decrease despite more compute resources
- This motivates optimizations like gradient bucketing (see future work)

## Correctness Verification

The test script ensures correctness by:

1. **Initial State**: Creating a baseline model and a DDP-wrapped deepcopy
2. **Training Loop**: Running both models on the same data (baseline on full batch, DDP on sharded batches)
3. **Gradient Verification**: Verifying that after gradient synchronization, all ranks have the same gradients
4. **Weight Verification**: Verifying that after each iteration, DDP weights match the baseline
5. **Synchronization Check**: Verifying that all ranks have identical weights

## Future Optimizations

This is a **naive** implementation. Production DDP systems (like PyTorch's `DistributedDataParallel`) use several optimizations:

1. **Gradient Bucketing**: Group multiple parameter gradients into larger buckets to reduce communication overhead
2. **Gradient Compression**: Use reduced precision or compression for gradient communication
3. **Communication Scheduling**: Overlap communication with computation more efficiently
4. **Memory Optimization**: Share buffers and reduce memory overhead

These optimizations will be implemented in subsequent assignments.

## Requirements

- PyTorch >= 2.0
- Python >= 3.8
- For GPU: CUDA-capable GPU and NCCL backend

## References

- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/notes/ddp.html)
- [Distributed Training in PyTorch](https://pytorch.org/tutorials/beginner/dist_overview.html)