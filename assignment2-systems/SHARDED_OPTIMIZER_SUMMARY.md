# Sharded Optimizer Implementation Summary

## Overview

Implemented optimizer state sharding to reduce per-rank memory consumption by distributing optimizer state (momentum buffers, Adam variance, etc.) across ranks. Each rank maintains state for only ~1/world_size of parameters.

## Implementation

### Files Created

- **[cs336_systems/sharded_optimizer.py](cs336_systems/sharded_optimizer.py)** - ShardedOptimizer class
- **[tests/adapters.py](tests/adapters.py)** - Updated `get_sharded_optimizer()` adapter

### Key Design Decisions

#### 1. Parameter Sharding Strategy
- **Round-robin assignment**: Parameters are assigned to ranks in order (param_idx % world_size)
- Each rank gets approximately 1/world_size of parameters
- Assignment is deterministic and consistent across ranks
- All ranks know the complete parameter list and which rank owns each parameter

#### 2. Optimizer Wrapping
```python
class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        # Shard parameters across ranks
        self.all_params, self.param_to_rank, local_param_groups = self._shard_param_groups(params)

        # Initialize parent with only local params
        super().__init__(local_param_groups, {})

        # Create wrapped optimizer with only this rank's parameters
        self.optimizer = self.optimizer_cls(local_param_groups, **kwargs)
```

- **Wrapped optimizer**: Each rank creates an instance of the target optimizer class (e.g., AdamW) with only its shard of parameters
- **Memory savings**: Optimizer state (momentum, variance) only allocated for 1/world_size of parameters
- **Delegation**: Most operations (step, zero_grad, state_dict) delegate to wrapped optimizer

#### 3. Parameter Synchronization
```python
def step(self, closure=None, **kwargs):
    # Update this rank's shard
    loss = self.optimizer.step(closure=closure, **kwargs)

    # Broadcast updated parameters from each rank
    if dist.is_initialized():
        for param in self.all_params:
            owner_rank = self.param_to_rank[param]
            dist.broadcast(param.data, src=owner_rank)

    return loss
```

- After optimizer.step(), each rank has updated its shard of parameters
- Use `dist.broadcast` to send each rank's updated parameters to all other ranks
- Result: All ranks have synchronized model parameters
- Communication: O(total_params) data transfer, but distributed across ranks

#### 4. Parameter Group Support
```python
def add_param_group(self, param_group):
    # Assign new parameters to ranks
    # Add only local params to wrapped optimizer
```

- Supports parameter groups (different hyperparameters for different layers)
- Handles dynamic parameter addition (e.g., unfreezing layers during training)
- Maintains correct sharding for newly added parameters

## Test Results

All tests pass reliably (5/5 runs):
```bash
$ uv run pytest tests/test_sharded_optimizer.py -v
============================== 2 passed in 4.25s ===============================
```

Tests verify:
- ✓ Correct gradient accumulation and parameter updates
- ✓ Identical results to non-sharded baseline
- ✓ Works with tied weights (multiple parameters sharing same tensor)
- ✓ Works with different model architectures

## Memory Analysis

### Memory Breakdown (per rank)

**Without Sharding:**
- Model parameters: P bytes
- Gradients: P bytes
- Optimizer state: k×P bytes (k=1 for SGD, k=2 for Adam)
- **Total: (2+k)×P bytes**

**With Sharding:**
- Model parameters: P bytes (replicated)
- Gradients: P bytes (replicated)
- Optimizer state: k×P/N bytes (sharded, N=world_size)
- **Total: 2×P + k×P/N bytes**

### Memory Savings

For Adam optimizer (k=2):
- **Without sharding**: 4×P bytes per rank
- **With sharding**: 2×P + 2×P/N bytes per rank
- **Savings**: 2×P × (N-1)/N bytes

For N=2 ranks: **50% reduction** in optimizer state memory
For N=4 ranks: **75% reduction** in optimizer state memory

Example with 1B parameter model (P=4GB):
- Without sharding: 16 GB per rank
- With sharding (N=2): 12 GB per rank (4 GB saved)
- With sharding (N=4): 10 GB per rank (6 GB saved)

## Communication Analysis

### Per Iteration Communication

**Gradient synchronization (DDP):**
- All-reduce of P bytes (gradients)
- Bandwidth: P bytes
- Latency: log₂(N) message passes

**Parameter synchronization (Sharded Optimizer):**
- N broadcasts, each of P/N bytes
- Bandwidth: P bytes total
- Latency: N message passes (serial broadcasts)

**Total bandwidth**: 2×P bytes (same as non-sharded)
**Additional latency**: O(N) vs O(log N) for all-reduce

### Trade-offs

**Advantages:**
- ✓ Reduces optimizer state memory by k×P×(N-1)/N
- ✓ Same bandwidth as non-sharded (2×P per iteration)
- ✓ No change to model parameters memory
- ✓ Compatible with any optimizer

**Disadvantages:**
- ✗ Additional latency from serial broadcasts
- ✗ Model parameters still replicated across ranks
- ✗ Gradients still replicated across ranks

## Comparison with ZeRO Stage 1

### Our Implementation vs ZeRO-DP Pos

**Similarities:**
- Both shard optimizer state across ranks
- Both reduce optimizer state memory by ~1/world_size
- Both synchronize parameters after optimizer step

**Differences:**

1. **Parameter Synchronization:**
   - **Our implementation**: Serial broadcasts (O(N) latency)
   - **ZeRO Stage 1**: All-gather operation (O(log N) latency)
   - **Impact**: ZeRO is more efficient for large N

2. **Communication Granularity:**
   - **Our implementation**: Broadcast entire parameter tensor from each rank
   - **ZeRO Stage 1**: Can overlap communication with optimizer step
   - **Impact**: ZeRO has better overlap potential

3. **Memory Overhead:**
   - **Our implementation**: No additional buffers
   - **ZeRO Stage 1**: May use buffers for all-gather
   - **Impact**: Similar memory footprint

4. **Optimizer State Distribution:**
   - **Both**: Round-robin or contiguous chunk assignment
   - **Both**: Each rank maintains state for ~1/N parameters

5. **Communication Volume:**
   - **Both**: Same total bandwidth (P bytes for parameter sync)
   - **Difference**: Communication pattern (broadcast vs all-gather)

### Performance Implications

For small world sizes (N=2-4):
- Our implementation's O(N) broadcast latency is acceptable
- Simplicity advantage over more complex all-gather

For large world sizes (N≥8):
- ZeRO's O(log N) all-gather becomes significantly better
- Our O(N) broadcast latency grows linearly

## Usage Example

```python
from cs336_systems.sharded_optimizer import ShardedOptimizer
import torch.distributed as dist
import torch.optim as optim

# Initialize distributed
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Create model
model = MyModel()
ddp_model = torch.nn.parallel.DistributedDataParallel(model)

# Create sharded optimizer
optimizer = ShardedOptimizer(
    ddp_model.parameters(),
    optimizer_cls=optim.AdamW,
    lr=1e-4,
    weight_decay=0.01
)

# Training loop
for data, labels in dataloader:
    optimizer.zero_grad()
    loss = loss_fn(ddp_model(data), labels)
    loss.backward()  # DDP all-reduces gradients
    optimizer.step()  # Updates local shard + broadcasts parameters
```

## Running Tests

```bash
# Run tests
uv run pytest tests/test_sharded_optimizer.py -v

# Run multiple times for reliability
for i in {1..5}; do uv run pytest tests/test_sharded_optimizer.py; done
```

## Next Steps

Potential extensions:
1. **ZeRO Stage 2**: Shard gradients as well as optimizer states
2. **ZeRO Stage 3**: Shard model parameters (FSDP)
3. **Async parameter sync**: Overlap broadcast with next forward pass
4. **Bucketed sync**: Group broadcasts to reduce latency overhead
5. **All-gather sync**: Replace serial broadcasts with parallel all-gather

## References

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
