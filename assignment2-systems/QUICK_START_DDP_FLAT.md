# Quick Start: Flattened DDP Implementation

## What Was Implemented

A flattened gradient DDP implementation that batches all gradient all-reduce operations into a single communication call for improved performance.

## Key Innovation

Instead of all-reducing each parameter gradient individually:
```python
# Naive approach: O(num_parameters) all-reduce calls
for param in parameters:
    dist.all_reduce(param.grad, async_op=True)
```

The flattened approach batches everything:
```python
# Flattened approach: O(1) all-reduce call
flat_grads = flatten_dense_tensors([p.grad for p in parameters])
dist.all_reduce(flat_grads, op=ReduceOp.AVG)
unflat_grads = unflatten_dense_tensors(flat_grads, [p.grad for p in parameters])
```

## Files

```
assignment2-systems/
├── cs336_systems/
│   └── ddp_flat.py              # Flattened DDP implementation
├── test_ddp_flat.py              # Correctness verification
├── compare_ddp.py                # Performance comparison script
└── DDP_FLAT_RESULTS.md          # Detailed results and analysis
```

## Quick Test

```bash
# Verify correctness
python test_ddp_flat.py --backend gloo --device cpu --world-size 2

# Compare with naive implementation
python compare_ddp.py --backend gloo --device cpu --world-size 4
```

## Performance Results

### 2 Processes (Gloo, CPU)
```
Communication Time:
  Naive:     3.57 ms
  Flattened: 16.88 ms
  Result: Naive is faster (better overlap with backward)
```

### 4 Processes (Gloo, CPU)
```
Communication Time:
  Naive:     35.32 ms
  Flattened: 25.10 ms
  Speedup: 1.41x ✓

Overall Iteration Time:
  Naive:     110.65 ms
  Flattened: 83.94 ms
  Speedup: 1.32x ✓
```

**Conclusion**: Flattened DDP provides **1.41x faster gradient communication** with 4+ processes by batching all-reduce operations into a single call, reducing communication overhead and better utilizing network bandwidth.

## Using the FlattenedDDP Class

```python
from cs336_systems.ddp_flat import FlattenedDDP
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

# Wrap your model
model = MyModel()
ddp_model = FlattenedDDP(model)
optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

# Training loop
for data, labels in dataloader:
    optimizer.zero_grad()
    outputs = ddp_model(data)
    loss = loss_fn(outputs, labels)
    loss.backward()

    # Single batched gradient synchronization
    ddp_model.finish_gradient_synchronization()

    optimizer.step()
```

## Key Differences from Naive DDP

| Aspect | Naive DDP | Flattened DDP |
|--------|-----------|---------------|
| All-reduce calls | One per parameter | One total |
| Backward hooks | Yes (async) | No |
| Overlap with compute | High | Low |
| Communication efficiency | Low (small messages) | High (large message) |
| Best for | Small world size (2) | Large world size (4+) |

## Implementation Details

### FlattenedDDP Class

**`__init__(module)`**
- Broadcasts parameters from rank 0
- Stores list of parameters that require gradients

**`finish_gradient_synchronization()`**
1. Collects all parameter gradients
2. Flattens into single tensor using `torch._utils._flatten_dense_tensors()`
3. All-reduces the flattened tensor (AVG operation)
4. Unflattens using `torch._utils._unflatten_dense_tensors()`
5. Copies averaged gradients back to parameters

### Why Batching Helps

With **4+ processes**, batching provides:
- ✅ Fewer kernel launches
- ✅ Better network bandwidth utilization
- ✅ Lower latency overhead (1 message vs. 100s of messages)
- ✅ More efficient all-reduce algorithms with larger messages

With **2 processes**, naive DDP's async overlap wins:
- The backward hooks trigger all-reduce as soon as each gradient is ready
- Communication can overlap with remaining backward computation
- Flatten/unflatten overhead is not worth it

## Verification

The test script verifies:
- ✓ Flattened DDP produces identical results to single-process training
- ✓ All ranks have synchronized weights after each iteration
- ✓ Works with 2, 4, and 6 processes
- ✓ Works with both Gloo (CPU) and NCCL (GPU) backends

## When to Use Each Implementation

### Use Flattened DDP when:
- Training with 4+ GPUs/processes
- Network bandwidth is the bottleneck
- Using high-latency interconnects
- Model has many small parameters

### Use Naive DDP when:
- Training with 2 GPUs/processes
- Fast interconnect (low latency)
- Want maximum overlap with computation
- Prototyping or small-scale experiments

## Next Steps

The next optimization is **gradient bucketing** (PyTorch's default DDP approach):
- Group parameters into ~25MB buckets
- All-reduce each bucket independently
- Start all-reducing a bucket as soon as all its gradients are ready
- Combines benefits: batching efficiency + computation overlap

See `DDP_FLAT_RESULTS.md` for detailed analysis and more results.