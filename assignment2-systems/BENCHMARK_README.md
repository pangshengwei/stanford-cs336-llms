# Transformer Model Benchmarking

This directory contains scripts for benchmarking the Transformer model's forward and backward pass performance.

## Overview

The benchmarking infrastructure provides:
- Basic end-to-end profiling of forward and backward passes
- Support for various model configurations and hyperparameters
- Proper GPU synchronization for accurate timing
- Memory profiling
- Parameter sweep utilities
- Slurm integration for cluster-based benchmarking

## Files

- **`benchmark.py`**: Main benchmarking script
- **`sweep_model_sizes.py`**: Sweep over different model sizes
- **`sweep_batch_sizes.py`**: Sweep over different batch sizes
- **`sweep_context_lengths.py`**: Sweep over different sequence lengths
- **`slurm_benchmark.sh`**: Slurm job submission script
- **`submit_sweep.sh`**: Batch submit multiple Slurm jobs for sweeps

## Installation

First, install the dependencies:

```bash
cd assignment2-systems
uv sync
```

This will create a virtual environment at `.venv/` with all required dependencies.

## Basic Usage

### Running a Single Benchmark

The simplest way to run a benchmark:

```bash
# Using the virtual environment
.venv/bin/python benchmark.py

# Or activate the environment first
source .venv/bin/activate
python benchmark.py
```

### Command Line Arguments

The benchmark script supports extensive configuration via command-line arguments:

#### Model Architecture

```bash
# Use preset model sizes (from the assignment table)
python benchmark.py --model-size small    # 190M params
python benchmark.py --model-size medium   # 354M params
python benchmark.py --model-size large    # 774M params
python benchmark.py --model-size xl       # 1.3B params
python benchmark.py --model-size 2.7B     # 2.7B params

# Or specify individual parameters
python benchmark.py \
    --d-model 768 \
    --num-layers 12 \
    --num-heads 12 \
    --d-ff 3072
```

#### Batch Configuration

```bash
# Configure batch size and sequence length
python benchmark.py \
    --batch-size 16 \
    --sequence-length 1024
```

#### Benchmarking Parameters

```bash
# Control warmup and measurement steps
python benchmark.py \
    --warmup-steps 20 \
    --measure-steps 200 \
    --pass-type forward  # or "forward_backward"
```

#### Device and Precision

```bash
# Run on GPU with mixed precision
python benchmark.py \
    --device cuda \
    --dtype bfloat16  # or "float32", "float16"

# Use torch.compile for optimization
python benchmark.py --use-compile
```

### Example Commands

```bash
# Small model, forward+backward, float32
python benchmark.py \
    --model-size small \
    --batch-size 8 \
    --sequence-length 512 \
    --pass-type forward_backward \
    --device cuda

# Large model, forward only, bfloat16
python benchmark.py \
    --model-size large \
    --batch-size 4 \
    --sequence-length 1024 \
    --pass-type forward \
    --dtype bfloat16 \
    --device cuda

# Medium model with torch.compile
python benchmark.py \
    --model-size medium \
    --use-compile \
    --device cuda
```

## Parameter Sweeps

### Sweep Over Model Sizes

```bash
python sweep_model_sizes.py \
    --model-sizes small medium large \
    --batch-size 8 \
    --sequence-length 512 \
    --device cuda \
    --output model_sweep_results.json
```

### Sweep Over Batch Sizes

```bash
python sweep_batch_sizes.py \
    --model-size small \
    --batch-sizes 1 2 4 8 16 32 \
    --sequence-length 512 \
    --device cuda \
    --output batch_sweep_results.json
```

### Sweep Over Context Lengths

```bash
python sweep_context_lengths.py \
    --model-size small \
    --sequence-lengths 128 256 512 1024 2048 \
    --batch-size 8 \
    --device cuda \
    --output context_sweep_results.json
```

## Slurm Integration

For running benchmarks on a Slurm cluster:

### Single Job Submission

```bash
# Submit a single benchmark job
sbatch slurm_benchmark.sh <model_size> <batch_size> <seq_length> <pass_type> <dtype>

# Example:
sbatch slurm_benchmark.sh small 8 512 forward_backward float32
```

### Batch Submission for Sweeps

Edit `submit_sweep.sh` to customize your parameter sweep, then:

```bash
./submit_sweep.sh
```

This will submit multiple jobs to sweep over:
- Model sizes
- Batch sizes
- Sequence lengths
- Data types

Check job status:
```bash
squeue -u $USER
```

View results:
```bash
cat logs/benchmark_<job_id>.out
```

## Using with submitit

For programmatic job submission with submitit:

```python
import submitit

executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(
    timeout_min=60,
    slurm_partition="gpu",
    gpus_per_node=1,
    cpus_per_task=4,
    mem_gb=32,
)

# Submit jobs
jobs = []
for model_size in ["small", "medium", "large"]:
    job = executor.submit(
        benchmark_model,
        model_size=model_size,
        batch_size=8,
        sequence_length=512,
        device="cuda",
    )
    jobs.append(job)

# Wait for results
results = [job.result() for job in jobs]
```

## Output Format

The benchmark script prints:

```
================================================================================
BENCHMARK CONFIGURATION
================================================================================
Model: 12 layers, 768 dim, 12 heads, 3072 ff_dim
Batch: 8 x 512
Device: cuda, Dtype: float32
Pass type: forward_backward
Warmup: 10, Measure: 100
================================================================================

Initializing model...
Total parameters: 190.46M

...

================================================================================
BENCHMARK RESULTS
================================================================================
Total time: 12.3456 seconds
Average time per step: 123.46 ms
Throughput: 64.78 samples/sec
Tokens per second: 33168.64 tokens/sec

Memory Usage:
  Allocated before: 0.723 GB
  Allocated after: 0.723 GB
  Peak memory: 2.145 GB
================================================================================
```

## Key Implementation Details

### GPU Synchronization

The script properly handles CUDA's asynchronous execution:

```python
# After each forward/backward pass
if device == "cuda":
    torch.cuda.synchronize()
```

This ensures accurate timing by waiting for all GPU operations to complete.

### High-Resolution Timing

Uses `timeit.default_timer()` for the system's highest resolution clock:

```python
timer = timeit.default_timer
start_time = timer()
# ... run benchmarks ...
end_time = timer()
```

### Warmup Steps

Runs warmup iterations before measurement to:
- Allow GPU to reach steady state
- Trigger JIT compilation (if using torch.compile)
- Stabilize memory allocations

### Memory Profiling

Tracks GPU memory usage:
```python
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated()
# ... run benchmarks ...
peak_mem = torch.cuda.max_memory_allocated()
```

## Tips and Best Practices

1. **Always run warmup steps**: GPU performance can vary significantly in the first few iterations

2. **Use enough measurement steps**: At least 50-100 steps for stable measurements

3. **Profile memory separately**: Memory profiling can add overhead, use `--no-memory-profile` for pure speed tests

4. **Batch your sweeps**: Use the sweep scripts or Slurm to parallelize across different configurations

5. **Monitor GPU utilization**: Use `nvidia-smi` to check if you're saturating the GPU

6. **Consider torch.compile**: Can provide 1.5-2x speedups but adds compilation overhead

7. **Test on smaller models first**: Validate your benchmarking setup before running expensive sweeps

## Troubleshooting

### Out of Memory Errors

- Reduce `--batch-size` or `--sequence-length`
- Use mixed precision: `--dtype bfloat16` or `--dtype float16`
- Ensure no other processes are using GPU memory

### Slow Performance

- Verify GPU is being used: check that `device: cuda` appears in output
- Check if GPU is throttling due to temperature
- Ensure CUDA is installed correctly: `python -c "import torch; print(torch.cuda.is_available())"`

### Installation Issues

If dependencies are missing:
```bash
cd assignment2-systems
uv sync  # or: uv pip install -e .
```

## Further Reading

- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
