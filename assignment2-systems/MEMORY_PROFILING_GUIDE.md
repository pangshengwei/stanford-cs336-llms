# Memory Profiling Guide

## Overview

The benchmarking scripts now support **detailed memory profiling** using PyTorch's built-in memory profiler. This captures layer-by-layer memory allocations and helps identify memory bottlenecks.

## Two Types of Memory Profiling

### 1. Basic Memory Stats (Default)

Tracks overall memory usage:
- Allocated memory before/after
- Peak memory usage
- Automatic, minimal overhead

```bash
python benchmark.py --model-size small --device cuda
```

**Output:**
```
Memory Usage:
  Allocated before: 0.723 GB
  Allocated after: 0.723 GB
  Peak memory: 2.145 GB
```

### 2. Detailed Memory Snapshot (NEW! ✨)

Captures detailed allocation history:
- Layer-by-layer memory usage
- Allocation timeline
- Stack traces for each allocation
- Visualizable with PyTorch's memory visualizer

```bash
python benchmark.py \
    --model-size small \
    --device cuda \
    --memory-snapshot memory_snapshot.pickle
```

## Basic Usage

### Single Benchmark with Memory Snapshot

```bash
# Run benchmark and save memory snapshot
.venv/bin/python benchmark.py \
    --model-size small \
    --batch-size 8 \
    --sequence-length 512 \
    --device cuda \
    --memory-snapshot memory_snapshot.pickle
```

**What happens:**
1. Runs warmup steps (not profiled)
2. Starts memory recording
3. Runs measurement steps (profiled)
4. Saves snapshot to `memory_snapshot.pickle`
5. Provides visualization instructions

### Output Example

```
================================================================================
BENCHMARK CONFIGURATION
================================================================================
Model: 12 layers, 768 dim, 12 heads, 3072 ff_dim
Batch: 8 x 512
Device: cuda, Dtype: float32
Memory Snapshot: memory_snapshot.pickle  ← Enabled!
...
================================================================================

Starting detailed memory profiling...
Snapshot will be saved to: memory_snapshot.pickle

Measuring 100 steps...
Measuring: 100%|████████████████████| 100/100 [00:15<00:00, 6.5it/s]

Saving memory snapshot to memory_snapshot.pickle...
Memory snapshot saved successfully!

To visualize:
  1. Visit: https://pytorch.org/memory_viz
  2. Upload: memory_snapshot.pickle
  Or use: python -m torch.cuda.memory._visualizer memory_snapshot.pickle
```

## Visualizing Memory Snapshots

### Method 1: Online Visualizer (Recommended)

1. **Visit**: https://pytorch.org/memory_viz
2. **Upload** your `.pickle` file
3. **Explore**:
   - Timeline view of allocations
   - Memory breakdown by operation
   - Peak memory analysis
   - Layer-by-layer usage

### Method 2: Command Line Visualizer

```bash
# Start local visualizer
.venv/bin/python -m torch.cuda.memory._visualizer memory_snapshot.pickle

# Opens browser at http://localhost:8000
```

### Method 3: Programmatic Analysis

```python
import pickle

# Load snapshot
with open('memory_snapshot.pickle', 'rb') as f:
    snapshot = pickle.load(f)

# Analyze
for entry in snapshot:
    print(f"Time: {entry['time']}, Size: {entry['size']}")
```

## Advanced Usage

### Compare Different Model Sizes

```bash
# Generate snapshots for each model size
for size in small medium large; do
    .venv/bin/python benchmark.py \
        --model-size $size \
        --device cuda \
        --memory-snapshot snapshot_${size}.pickle
done

# Compare in visualizer
```

### Profile with Mixed Precision

```bash
# FP32 baseline
.venv/bin/python benchmark.py \
    --model-size small \
    --dtype float32 \
    --device cuda \
    --memory-snapshot fp32_memory.pickle

# BF16 comparison
.venv/bin/python benchmark.py \
    --model-size small \
    --dtype bfloat16 \
    --device cuda \
    --memory-snapshot bf16_memory.pickle

# AMP comparison
.venv/bin/python benchmark.py \
    --model-size small \
    --use-amp \
    --device cuda \
    --memory-snapshot amp_memory.pickle
```

### Sweep with Memory Snapshots

```bash
# Create snapshots directory
mkdir -p memory_snapshots

# Sweep model sizes with memory profiling
.venv/bin/python sweep_model_sizes.py \
    --model-sizes small medium large \
    --device cuda \
    --memory-snapshot-prefix memory_snapshots/model
```

**This creates:**
- `memory_snapshots/model_small.pickle`
- `memory_snapshots/model_medium.pickle`
- `memory_snapshots/model_large.pickle`

### Profile Specific Layers

For even more detailed profiling, you can use PyTorch Profiler:

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Run your benchmark
    outputs = model(input_ids)

# Export for visualization
prof.export_chrome_trace("trace.json")
```

## Understanding Memory Snapshots

### What's Captured

1. **Allocations**: Every `torch.malloc` call
2. **Deallocations**: Every `torch.free` call
3. **Stack traces**: Where allocations occurred
4. **Timestamps**: When allocations happened
5. **Sizes**: How much memory each allocation used

### Key Metrics to Look For

1. **Peak Memory**:
   - Highest memory usage during execution
   - Critical for batch size tuning

2. **Memory Fragmentation**:
   - Gaps between allocations
   - Can prevent larger allocations

3. **Layer Patterns**:
   - Which layers use most memory
   - Attention vs FFN memory usage

4. **Activation Memory**:
   - Forward pass activations
   - Stored for backward pass

5. **Gradient Memory**:
   - Gradient storage
   - Usually ~2x parameter memory

## Performance Impact

Memory profiling has some overhead:

| Profiling Type | Overhead | Use Case |
|---------------|----------|----------|
| None | 0% | Pure performance |
| Basic stats | <1% | Standard benchmarking |
| Detailed snapshot | 5-10% | Memory optimization |

**Recommendations:**
- Use **basic stats** for performance benchmarking
- Use **detailed snapshots** for memory optimization
- Profile **separately** from performance runs

## Common Use Cases

### 1. Find Memory Bottlenecks

```bash
# Profile large model
.venv/bin/python benchmark.py \
    --model-size large \
    --batch-size 1 \
    --device cuda \
    --memory-snapshot bottleneck.pickle

# Look for:
# - Which layers allocate most memory
# - Unexpected memory spikes
# - Memory not being freed
```

### 2. Optimize Batch Size

```bash
# Try different batch sizes
for bs in 1 2 4 8 16 32; do
    .venv/bin/python benchmark.py \
        --model-size small \
        --batch-size $bs \
        --device cuda \
        --memory-snapshot batch_${bs}.pickle \
        2>&1 | grep -A3 "Memory Usage"
done

# Find maximum batch size that fits in memory
```

### 3. Compare Precision Modes

```bash
# Compare memory usage
mkdir -p precision_profiles

for dtype in float32 bfloat16; do
    .venv/bin/python benchmark.py \
        --model-size small \
        --dtype $dtype \
        --device cuda \
        --memory-snapshot precision_profiles/${dtype}.pickle
done
```

### 4. Debug OOM (Out of Memory)

```bash
# Profile with small batch to understand memory pattern
.venv/bin/python benchmark.py \
    --model-size xl \
    --batch-size 1 \
    --sequence-length 128 \
    --device cuda \
    --memory-snapshot oom_debug.pickle

# Analyze snapshot to identify memory hogs
```

## Memory Optimization Tips

Based on profiling results, you can:

### 1. Reduce Activation Memory
- Use gradient checkpointing
- Smaller batch sizes
- Shorter sequence lengths

### 2. Reduce Model Memory
- Use mixed precision (BF16/FP16)
- Quantization
- Parameter sharing

### 3. Reduce Fragmentation
- Pre-allocate tensors
- Reuse buffers
- Clear cache: `torch.cuda.empty_cache()`

### 4. Enable Memory Efficient Features
```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Flash Attention (if available)
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    outputs = model(input_ids)
```

## File Sizes

Memory snapshots can be large:

| Model Size | Steps | Snapshot Size |
|-----------|-------|---------------|
| Small | 100 | 50-100 MB |
| Medium | 100 | 100-200 MB |
| Large | 100 | 200-500 MB |
| XL | 100 | 500+ MB |

**Tips:**
- Use fewer `--measure-steps` for profiling
- Compress snapshots: `gzip memory_snapshot.pickle`
- Clean up old snapshots regularly

## Troubleshooting

### "CUDA out of memory" during profiling

Memory profiling adds overhead. Solutions:
```bash
# Reduce batch size
--batch-size 1

# Reduce sequence length
--sequence-length 128

# Reduce measurement steps
--measure-steps 10

# Reduce max_entries
# (Edit benchmark.py: max_entries=10000)
```

### "Cannot start memory profiling"

Requires PyTorch 2.1+:
```bash
python -c "import torch; print(torch.__version__)"

# Upgrade if needed
pip install --upgrade torch
```

### Snapshot file too large

```bash
# Reduce entries captured
# In benchmark.py, change:
# max_entries=100000  →  max_entries=10000

# Or reduce measurement steps
--measure-steps 10
```

### Visualizer not working

Try different browser or local visualizer:
```bash
# Use command-line visualizer
.venv/bin/python -m torch.cuda.memory._visualizer memory_snapshot.pickle
```

## Example Workflow

Complete workflow for memory optimization:

```bash
# 1. Baseline measurement
.venv/bin/python benchmark.py \
    --model-size small \
    --device cuda \
    --memory-snapshot baseline.pickle

# 2. Profile with mixed precision
.venv/bin/python benchmark.py \
    --model-size small \
    --use-amp \
    --device cuda \
    --memory-snapshot amp.pickle

# 3. Visualize both
# Visit https://pytorch.org/memory_viz
# Upload baseline.pickle and amp.pickle

# 4. Compare peak memory
echo "=== Baseline ==="
grep "Peak memory" baseline_output.txt

echo "=== With AMP ==="
grep "Peak memory" amp_output.txt

# 5. Calculate savings
python -c "print(f'Savings: {(1 - 1.2/2.1)*100:.1f}%')"
```

## Integration with Other Tools

### With TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Log memory
writer.add_scalar('memory/allocated',
                  torch.cuda.memory_allocated() / 1024**3)
writer.add_scalar('memory/peak',
                  torch.cuda.max_memory_allocated() / 1024**3)
```

### With Weights & Biases

```python
import wandb

wandb.log({
    'memory/allocated_gb': torch.cuda.memory_allocated() / 1024**3,
    'memory/peak_gb': torch.cuda.max_memory_allocated() / 1024**3,
})
```

## References

- [PyTorch Memory Profiler](https://pytorch.org/blog/understanding-gpu-memory-1/)
- [Memory Visualizer](https://pytorch.org/memory_viz)
- [CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Memory Format Tutorial](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

## Quick Reference

```bash
# Basic profiling (default)
python benchmark.py --model-size small --device cuda

# Detailed snapshot
python benchmark.py --model-size small --device cuda --memory-snapshot snap.pickle

# Sweep with snapshots
python sweep_model_sizes.py --device cuda --memory-snapshot-prefix mem

# Visualize
# Visit: https://pytorch.org/memory_viz
# Or: python -m torch.cuda.memory._visualizer snap.pickle
```
