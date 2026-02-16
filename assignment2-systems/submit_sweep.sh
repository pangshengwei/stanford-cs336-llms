#!/bin/bash
# Script to submit multiple benchmark jobs to Slurm for parameter sweeps

# Create logs directory
mkdir -p logs

# Example 1: Sweep over model sizes
echo "Submitting model size sweep..."
for model_size in small medium large; do
    sbatch slurm_benchmark.sh $model_size 8 512 forward_backward float32
done

# Example 2: Sweep over batch sizes
echo "Submitting batch size sweep..."
for batch_size in 1 2 4 8 16 32; do
    sbatch slurm_benchmark.sh small $batch_size 512 forward_backward float32
done

# Example 3: Sweep over sequence lengths
echo "Submitting sequence length sweep..."
for seq_length in 128 256 512 1024; do
    sbatch slurm_benchmark.sh small 8 $seq_length forward_backward float32
done

# Example 4: Sweep over data types
echo "Submitting dtype sweep..."
for dtype in float32 float16 bfloat16; do
    sbatch slurm_benchmark.sh small 8 512 forward_backward $dtype
done

echo "All jobs submitted. Check 'squeue -u \$USER' to see job status."
