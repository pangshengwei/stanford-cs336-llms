# Quick Start: Naive DDP Implementation

## What Was Implemented

A minimal Distributed Data Parallel (DDP) training implementation that:
1. Broadcasts parameters from rank 0 to all other ranks
2. All-reduces gradients across ranks after backward pass
3. Supports both CPU (Gloo) and GPU (NCCL) backends
4. Works with 2, 4, or 6 processes

## Files Created

```
assignment2-systems/
├── cs336_systems/
│   └── ddp_naive.py              # Core DDP implementation (NaiveDDP class)
├── test_ddp_naive.py              # Correctness verification script
├── benchmark_ddp_naive.py         # Performance benchmarking script
├── DDP_IMPLEMENTATION.md          # Detailed documentation
└── QUICK_START_DDP.md            # This file
```

## Quick Test

```bash
# Test correctness with 2 processes
python test_ddp_naive.py --backend gloo --device cpu --world-size 2

# Test with different world sizes
python test_ddp_naive.py --backend gloo --device cpu --world-size 4
python test_ddp_naive.py --backend gloo --device cpu --world-size 6
```

Expected output: ✓ SUCCESS messages indicating weights match baseline

## Quick Benchmark

```bash
# Benchmark performance
python benchmark_ddp_naive.py --backend gloo --device cpu --world-size 2 4

# Larger model
python benchmark_ddp_naive.py --backend gloo --device cpu --world-size 2 4 \
    --model-size 200 400 100 --iterations 20
```

## Using the DDP Class

```python
from cs336_systems.ddp_naive import NaiveDDP
import torch.distributed as dist

# Initialize process group (in each process)
dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

# Wrap your model
model = MyModel()
ddp_model = NaiveDDP(model)

# Training loop
for data, labels in dataloader:
    optimizer.zero_grad()
    outputs = ddp_model(data)
    loss = loss_fn(outputs, labels)
    loss.backward()

    # Wait for gradient synchronization
    ddp_model.finish_gradient_synchronization()

    optimizer.step()
```

## Key Implementation Details

### NaiveDDP Class
- **`__init__(module)`**: Wraps a PyTorch module, broadcasts parameters from rank 0
- **`forward(*args, **kwargs)`**: Delegates to the wrapped module
- **`finish_gradient_synchronization()`**: Waits for all async all-reduce operations to complete

### How It Works
1. During initialization, parameters are broadcast from rank 0 to all ranks
2. Backward hooks on each parameter trigger async all-reduce of gradients
3. `finish_gradient_synchronization()` waits for all all-reduce ops before optimizer step
4. All ranks update with the same averaged gradients, staying synchronized

## Performance Results (Example)

```
World size = 2:
  - Avg iteration time: 1.41ms
  - Sync overhead: 76.1%
  - Throughput: 45,506 samples/sec

World size = 4:
  - Avg iteration time: 5.31ms
  - Sync overhead: 89.0%
  - Throughput: 24,104 samples/sec
```

Note: Sync overhead increases with world size due to naive per-parameter all-reduce.

## GPU Testing (if CUDA available)

```bash
# Test with NCCL backend
python test_ddp_naive.py --backend nccl --device cuda --world-size 2

# Benchmark on GPU
python benchmark_ddp_naive.py --backend nccl --device cuda --world-size 2 4
```

## Verification

The test script verifies:
- ✓ Parameters broadcast correctly from rank 0
- ✓ All ranks have synchronized weights
- ✓ DDP results match single-process baseline (within numerical precision)
- ✓ Works with 2, 4, and 6 processes
- ✓ Works with both Gloo (CPU) and NCCL (GPU) backends

## Next Steps

Future optimizations to implement:
1. **Gradient bucketing**: Group parameters into buckets to reduce communication overhead
2. **Better overlap**: More efficient communication/computation overlap
3. **Memory optimization**: Reduce memory footprint
4. **Compression**: Use reduced precision for gradient communication

See `DDP_IMPLEMENTATION.md` for detailed implementation notes.