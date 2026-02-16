# Mixed Precision Benchmarking Guide

## Overview

The benchmarking scripts now support **two approaches** for reduced precision:

1. **Direct dtype casting** (`--dtype`)
2. **Automatic Mixed Precision (AMP)** (`--use-amp`)

## Approach 1: Direct dtype Casting

Cast the entire model to a specific precision:

```bash
# Run with BF16 (entire model in bfloat16)
python benchmark.py --model-size small --dtype bfloat16 --device cuda

# Run with FP16 (entire model in float16)
python benchmark.py --model-size small --dtype float16 --device cuda
```

**How it works:**
- Converts all model parameters to the specified dtype
- All operations run in that precision
- Simple and straightforward

**Use when:**
- You want consistent precision throughout
- Testing pure low-precision performance

## Approach 2: Automatic Mixed Precision (AMP)

Uses PyTorch's `torch.cuda.amp.autocast()` for selective precision:

```bash
# Run with AMP (automatic mixed precision)
python benchmark.py --model-size small --use-amp --device cuda

# AMP with specific dtype preference
python benchmark.py --model-size small --use-amp --dtype bfloat16 --device cuda
```

**How it works:**
- Model stays in FP32
- `autocast` context automatically casts operations to lower precision
- Critical operations (like reductions) stay in FP32 for numerical stability
- Automatically uses BF16 if supported, otherwise FP16

**Use when:**
- You want optimal balance of speed and accuracy
- Following production training best practices
- Need numerical stability (BF16 has better range than FP16)

## Performance Comparison Example

```bash
# Baseline: FP32
python benchmark.py --model-size small --dtype float32 --device cuda

# Direct BF16 casting
python benchmark.py --model-size small --dtype bfloat16 --device cuda

# AMP with BF16
python benchmark.py --model-size small --use-amp --dtype bfloat16 --device cuda
```

## Which Approach to Use?

### Use Direct dtype Casting When:
- ✅ Benchmarking pure low-precision performance
- ✅ Simpler setup
- ✅ Consistent precision everywhere
- ❌ Less numerical stability

### Use AMP When:
- ✅ Production-like training setup
- ✅ Better numerical stability
- ✅ Automatic optimization of precision per operation
- ✅ Industry standard approach
- ❌ Slightly more complex

## Key Differences

| Feature | Direct Casting | AMP |
|---------|---------------|-----|
| Model precision | Low (BF16/FP16) | High (FP32) |
| Operation precision | All low | Mixed (auto) |
| Numerical stability | Lower | Higher |
| Memory usage | Lower | Medium |
| Setup complexity | Simple | Simple |
| Production use | Less common | Standard |

## Sweep Examples

### Sweep over model sizes with AMP
```bash
python sweep_model_sizes.py \
    --model-sizes small medium large \
    --device cuda \
    --use-amp
```

### Compare precision modes
```bash
# Create a sweep script
for mode in "float32" "bfloat16" "amp"; do
    if [ "$mode" = "amp" ]; then
        python benchmark.py --model-size small --use-amp --device cuda
    else
        python benchmark.py --model-size small --dtype $mode --device cuda
    fi
done
```

### Sweep batch sizes with AMP
```bash
python sweep_batch_sizes.py \
    --model-size small \
    --batch-sizes 1 2 4 8 16 32 \
    --device cuda \
    --use-amp
```

## Technical Details

### BF16 (bfloat16)
- **Range**: Same as FP32 (8-bit exponent)
- **Precision**: Lower than FP16 (7-bit mantissa)
- **Best for**: Training large models
- **Hardware**: Requires Ampere or newer GPUs (A100, 3090, 4090, etc.)

### FP16 (float16)
- **Range**: Lower than FP32 (5-bit exponent)
- **Precision**: Higher than BF16 (10-bit mantissa)
- **Best for**: Inference, older GPUs
- **Hardware**: Volta and newer (V100, T4, etc.)

### AMP dtype Selection
The script automatically selects the best AMP dtype:
1. If `--dtype bfloat16` specified and BF16 supported → Use BF16
2. If BF16 supported by GPU → Use BF16
3. Otherwise → Use FP16

Check BF16 support:
```python
import torch
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
```

## Memory Savings

Approximate memory savings with reduced precision:

| Precision | Memory | Speed |
|-----------|--------|-------|
| FP32 | 1x (baseline) | 1x |
| BF16 | ~0.5x | 1.5-2x |
| FP16 | ~0.5x | 1.5-2x |
| AMP | ~0.6-0.7x | 1.3-1.8x |

## Common Issues

### "CUDA is required for AMP"
AMP only works on CUDA devices. Use direct dtype casting for CPU:
```bash
# CPU doesn't support AMP - use dtype instead
python benchmark.py --dtype bfloat16 --device cpu  # ❌ Won't help on CPU
python benchmark.py --dtype float32 --device cpu   # ✅ Use FP32 on CPU
```

### "BF16 not supported"
If your GPU doesn't support BF16:
```bash
# Will automatically fall back to FP16
python benchmark.py --use-amp --device cuda
```

Or explicitly use FP16:
```bash
python benchmark.py --dtype float16 --device cuda
```

## Recommended Workflow

1. **Start with baseline FP32:**
   ```bash
   python benchmark.py --model-size small --dtype float32 --device cuda
   ```

2. **Test AMP (recommended for training):**
   ```bash
   python benchmark.py --model-size small --use-amp --device cuda
   ```

3. **Test pure BF16 (maximum speed):**
   ```bash
   python benchmark.py --model-size small --dtype bfloat16 --device cuda
   ```

4. **Run sweeps with best configuration:**
   ```bash
   python sweep_model_sizes.py --use-amp --device cuda
   ```

## References

- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [BF16 for Training](https://arxiv.org/abs/2107.00119)
