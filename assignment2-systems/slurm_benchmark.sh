#!/bin/bash
#SBATCH --job-name=benchmark_transformer
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Create logs directory if it doesn't exist
mkdir -p logs

# Load any necessary modules (adjust for your cluster)
# module load python/3.11
# module load cuda/12.1

# Activate virtual environment
source .venv/bin/activate

# Parse command line arguments
MODEL_SIZE=${1:-"small"}
BATCH_SIZE=${2:-8}
SEQ_LENGTH=${3:-512}
PASS_TYPE=${4:-"forward_backward"}
DTYPE=${5:-"float32"}

# Run benchmark
python benchmark.py \
    --model-size $MODEL_SIZE \
    --batch-size $BATCH_SIZE \
    --sequence-length $SEQ_LENGTH \
    --pass-type $PASS_TYPE \
    --dtype $DTYPE \
    --device cuda \
    --warmup-steps 10 \
    --measure-steps 100

echo "Benchmark completed for model_size=$MODEL_SIZE, batch_size=$BATCH_SIZE, seq_length=$SEQ_LENGTH"
