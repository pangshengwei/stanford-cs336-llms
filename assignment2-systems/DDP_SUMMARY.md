# DDP Implementation Summary

## Overview

This directory contains two DDP implementations with performance comparisons:

1. **Naive DDP** (`ddp_naive.py`): Individual all-reduce per parameter (async)
2. **Flattened DDP** (`ddp_flat.py`): Single batched all-reduce for all parameters

## Implementation Files

### Core Implementations
- [`cs336_systems/ddp_naive.py`](cs336_systems/ddp_naive.py) - Naive DDP with per-parameter all-reduce
- [`cs336_systems/ddp_flat.py`](cs336_systems/ddp_flat.py) - Flattened DDP with batched all-reduce

### Testing & Benchmarking
- [`test_ddp_naive.py`](test_ddp_naive.py) - Correctness verification for naive DDP
- [`test_ddp_flat.py`](test_ddp_flat.py) - Correctness verification for flattened DDP
- [`benchmark_ddp_naive.py`](benchmark_ddp_naive.py) - Performance benchmarking for naive DDP
- [`compare_ddp.py`](compare_ddp.py) - Head-to-head comparison of both implementations

### Documentation
- [`DDP_IMPLEMENTATION.md`](DDP_IMPLEMENTATION.md) - Detailed implementation notes for naive DDP
- [`DDP_FLAT_RESULTS.md`](DDP_FLAT_RESULTS.md) - Performance results and analysis
- [`QUICK_START_DDP.md`](QUICK_START_DDP.md) - Quick reference for naive DDP
- [`QUICK_START_DDP_FLAT.md`](QUICK_START_DDP_FLAT.md) - Quick reference for flattened DDP

## Quick Start

### Test Correctness

```bash
# Test naive DDP
python test_ddp_naive.py --backend gloo --device cpu --world-size 2

# Test flattened DDP
python test_ddp_flat.py --backend gloo --device cpu --world-size 2
```

### Compare Performance

```bash
# Compare both implementations with 4 processes
python compare_ddp.py --backend gloo --device cpu --world-size 4 \
    --iterations 20 --d-model 256 --n-layers 4
```

## Performance Results (4 Processes, Gloo, CPU)

**Model**: Transformer with 3.67M parameters (14MB)

| Metric | Naive DDP | Flattened DDP | Winner |
|--------|-----------|---------------|--------|
| Iteration Time | 110.65 ms | **83.94 ms** | Flattened (1.32x) |
| Communication Time | 35.32 ms | **25.10 ms** | Flattened (1.41x) |
| Compute Time | 72.26 ms | **55.73 ms** | Flattened |
| Communication Overhead | 31.9% | 29.9% | Flattened |

**Conclusion**: With 4 processes, **flattened DDP achieves 1.41x speedup** in gradient communication by batching all-reduce operations into a single call, reducing communication overhead and better utilizing network bandwidth.

## Key Findings

### When Flattened DDP Wins (1.32x - 1.41x speedup)

✅ **4+ processes**: Communication overhead of many small messages becomes significant
✅ **Network-bound scenarios**: Better bandwidth utilization with larger messages
✅ **High-latency networks**: Fewer messages reduce latency overhead

### When Naive DDP Competes

✅ **2 processes**: Async overlap with backward computation is beneficial
✅ **Fast interconnects**: Multiple small messages complete quickly with low latency
✅ **Small models**: Flatten/unflatten overhead becomes significant

## Implementation Comparison

### Naive DDP Architecture

```python
class NaiveDDP(nn.Module):
    def __init__(self, module):
        # 1. Broadcast parameters from rank 0
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        # 2. Register backward hooks on each parameter
        for param in module.parameters():
            param.register_hook(self._make_hook(param))

    def _make_hook(self, param):
        # 3. All-reduce gradient asynchronously during backward
        def hook(grad):
            work = dist.all_reduce(grad.data, op=AVG, async_op=True)
            self._async_work_handles.append(work)
        return hook

    def finish_gradient_synchronization(self):
        # 4. Wait for all async operations to complete
        for work in self._async_work_handles:
            work.wait()
```

**Pros**: Async all-reduce allows overlap with backward computation
**Cons**: O(num_parameters) communication calls, high latency overhead

### Flattened DDP Architecture

```python
class FlattenedDDP(nn.Module):
    def __init__(self, module):
        # 1. Broadcast parameters from rank 0
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        # 2. Store parameters for later gradient collection
        self._grad_params = [p for p in module.parameters() if p.requires_grad]

    def finish_gradient_synchronization(self):
        # 3. Collect all gradients
        grads = [p.grad.data for p in self._grad_params if p.grad is not None]

        # 4. Flatten into single tensor
        flat_grads = torch._utils._flatten_dense_tensors(grads)

        # 5. Single all-reduce call
        dist.all_reduce(flat_grads, op=AVG)

        # 6. Unflatten and copy back
        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for param_grad, synced_grad in zip(grads, unflat_grads):
            param_grad.copy_(synced_grad)
```

**Pros**: O(1) communication call, better bandwidth utilization
**Cons**: Less overlap with backward, flatten/unflatten overhead

## Usage Examples

### Naive DDP

```python
from cs336_systems.ddp_naive import NaiveDDP

model = MyModel()
ddp_model = NaiveDDP(model)

for data, labels in dataloader:
    optimizer.zero_grad()
    loss = loss_fn(ddp_model(data), labels)
    loss.backward()
    ddp_model.finish_gradient_synchronization()  # Wait for async all-reduces
    optimizer.step()
```

### Flattened DDP

```python
from cs336_systems.ddp_flat import FlattenedDDP

model = MyModel()
ddp_model = FlattenedDDP(model)

for data, labels in dataloader:
    optimizer.zero_grad()
    loss = loss_fn(ddp_model(data), labels)
    loss.backward()
    ddp_model.finish_gradient_synchronization()  # Flatten, all-reduce, unflatten
    optimizer.step()
```

## Scaling Behavior

| World Size | Naive Comm Time | Flattened Comm Time | Speedup |
|------------|-----------------|---------------------|---------|
| 2 | 3.57 ms | 16.88 ms | **0.21x (naive wins)** |
| 4 | 35.32 ms | 25.10 ms | **1.41x (flattened wins)** |

**Trend**: As world size increases, the benefit of batching becomes more significant due to reduced coordination overhead.

## Next Steps: Gradient Bucketing

Both implementations have tradeoffs. The next optimization combines the best of both:

**Gradient Bucketing** (PyTorch's default DDP):
- Group parameters into ~25MB buckets
- All-reduce each bucket as gradients become ready
- Benefits: Batching efficiency + computation/communication overlap

This will be implemented in the next deliverable.

## Verification

All implementations have been verified to:
- ✓ Produce identical results to single-process training (within numerical precision)
- ✓ Keep all ranks synchronized
- ✓ Work with both Gloo (CPU) and NCCL (GPU) backends
- ✓ Scale to 2, 4, and 6 processes

## References

- PyTorch DDP: https://pytorch.org/docs/stable/notes/ddp.html
- Gradient Bucketing: https://pytorch.org/docs/stable/notes/ddp.html#internal-design
- All-reduce algorithms: https://pytorch.org/tutorials/intermediate/dist_tuto.html