#!/bin/bash
# Example: Memory profiling workflow
#
# This script demonstrates how to profile memory usage for different configurations

set -e

# Create output directory
mkdir -p memory_profiles

echo "=== Memory Profiling Example ==="
echo ""
echo "Running 3 benchmarks with memory profiling:"
echo "  1. Baseline FP32"
echo "  2. BF16 (direct casting)"
echo "  3. AMP (automatic mixed precision)"
echo ""

# 1. Baseline FP32
echo "=== 1. Profiling FP32 baseline ==="
.venv/bin/python benchmark.py \
    --model-size small \
    --batch-size 4 \
    --sequence-length 256 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --device cuda \
    --dtype float32 \
    --memory-snapshot memory_profiles/fp32.pickle

echo ""
echo "=== 2. Profiling BF16 ==="
.venv/bin/python benchmark.py \
    --model-size small \
    --batch-size 4 \
    --sequence-length 256 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --device cuda \
    --dtype bfloat16 \
    --memory-snapshot memory_profiles/bf16.pickle

echo ""
echo "=== 3. Profiling AMP ==="
.venv/bin/python benchmark.py \
    --model-size small \
    --batch-size 4 \
    --sequence-length 256 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --device cuda \
    --use-amp \
    --memory-snapshot memory_profiles/amp.pickle

echo ""
echo "=== Profiling Complete! ==="
echo ""
echo "Memory snapshots saved:"
ls -lh memory_profiles/*.pickle
echo ""
echo "To visualize:"
echo "  1. Visit: https://pytorch.org/memory_viz"
echo "  2. Upload files from memory_profiles/"
echo ""
echo "Or use command-line visualizer:"
echo "  .venv/bin/python -m torch.cuda.memory._visualizer memory_profiles/fp32.pickle"
