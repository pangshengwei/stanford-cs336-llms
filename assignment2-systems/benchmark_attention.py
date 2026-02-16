"""
Benchmark script for attention implementations.

This script benchmarks:
1. PyTorch attention implementation (standard)
2. Compiled PyTorch attention
3. Full Transformer model (compiled vs uncompiled)

RESULTS:

================================================================================
Results for attention benchmarks (PyTorch):
================================================================================

Summary Table:
 batch_size  seq_len  d_model  forward_time_ms  backward_time_ms   memory_mb  compiled
          8      256       16         0.980484          2.909928    4.822266     False
          8     1024       16         6.388473         20.367854   67.757812     False
          8     4096       16        94.802045        318.267870 1066.687500     False
          8      256       32         2.549941          3.253242    6.697266     False
          8     1024       32        10.995657         25.222702   70.257812     False
          8     4096       32       112.757175        284.579395 1160.687500     False
          8      256       64         1.103163          2.191987    6.697266     False
          8     1024       64         6.584635         20.535190   75.257812     False
          8     4096       64       104.666360        302.514418 1168.687500     False
          8      256      128         1.121098          2.176053    9.197266     False
          8     1024      128         7.299780         20.975057   89.257812     False
          8     4096      128       114.303766        325.746189 1184.687500     False

================================================================================
Results for attention benchmarks (Compiled PyTorch):
================================================================================

Summary Table:
 batch_size  seq_len  d_model  forward_time_ms  backward_time_ms   memory_mb  compiled
          8      256       16         0.591745          0.652724    5.064453      True
          8     1024       16         4.169890          9.380215   67.882812      True
          8     4096       16        36.100321         94.415630 1050.812500      True
          8      256       32         0.702025          0.892638    6.939453      True
          8     1024       32         4.707417         17.040569   70.382812      True
          8     4096       32        41.561403        104.316397 1064.812500      True
          8      256       64         0.983005          1.309366    6.939453      True
          8     1024       64         7.051279         15.503181   75.382812      True
          8     4096       64        42.257354        117.056746 1088.812500      True
          8      256      128         0.907081          1.296775    9.439453      True
          8     1024      128         5.580027         19.002886   89.382812      True
          8     4096      128        53.671731        142.840427 1152.812500      True

================================================================================

Benchmarking uncompiled Transformer...
  Forward: 56.367 ms
  Full step: 174.744 ms

Benchmarking compiled Transformer...
  Forward: 25.905 ms
  Full step: 75.586 ms

================================================================================
Results saved to transformer_benchmarks.csv
================================================================================

Summary Table:
 compiled  forward_time_ms  full_step_time_ms
    False        56.367241         174.743560
     True        25.904693          75.585858

Speedup:
  Forward: 2.18x
  Full step: 2.31x
"""

import torch
import torch.nn as nn
import time
import itertools
from typing import Dict, List, Tuple
import pandas as pd
import sys
import os

# Add the cs336-basics module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cs336-basics'))

from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.nn_utils import softmax


def synchronize():
    """Synchronize device operations (MPS or CUDA)."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache():
    """Empty device cache (MPS or CUDA)."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024 / 1024
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_attention_forward_backward(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_compile: bool = False,
) -> Dict:
    """
    Benchmark attention forward and backward passes.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension (head dimension)
        num_iterations: Number of iterations to time
        warmup_iterations: Number of warmup iterations
        use_compile: Whether to use torch.compile

    Returns:
        Dictionary with timing and memory information
    """
    # Use MPS if available (Apple Silicon), otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if device.type == 'cpu':
        return {
            'error': 'No GPU available (neither MPS nor CUDA)'
        }

    try:
        # Create random inputs
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

        # Create causal mask
        seq = torch.arange(seq_len, device=device)
        causal_mask = seq.unsqueeze(0) >= seq.unsqueeze(1)  # (seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)

        # Define attention function
        attention_fn = scaled_dot_product_attention

        if use_compile:
            attention_fn = torch.compile(attention_fn)

        # Warmup
        for _ in range(warmup_iterations):
            output = attention_fn(Q, K, V, causal_mask)
            grad_output = torch.randn_like(output)
            output.backward(grad_output)
            Q.grad = None
            K.grad = None
            V.grad = None

        synchronize()

        # Benchmark forward pass
        forward_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            output = attention_fn(Q, K, V, causal_mask)
            synchronize()
            end = time.perf_counter()
            forward_times.append(end - start)

        # Measure memory before backward
        memory_before_backward = get_memory_usage()

        # Benchmark backward pass
        backward_times = []
        for _ in range(num_iterations):
            output = attention_fn(Q, K, V, causal_mask)
            grad_output = torch.randn_like(output)

            start = time.perf_counter()
            output.backward(grad_output)
            synchronize()
            end = time.perf_counter()

            backward_times.append(end - start)

            # Clear gradients
            Q.grad = None
            K.grad = None
            V.grad = None

        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'd_model': d_model,
            'forward_time_ms': sum(forward_times) / len(forward_times) * 1000,
            'backward_time_ms': sum(backward_times) / len(backward_times) * 1000,
            'memory_mb': memory_before_backward,
            'compiled': use_compile,
        }

    except RuntimeError as e:
        error_msg = str(e).lower()
        if 'out of memory' in error_msg:
            empty_cache()
            return {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'd_model': d_model,
                'error': 'OOM',
                'compiled': use_compile,
            }
        elif 'int_max' in error_msg or 'dims larger than' in error_msg:
            # MPS backend limitation with large tensor dimensions
            empty_cache()
            return {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'd_model': d_model,
                'error': 'MPS dimension limit exceeded',
                'compiled': use_compile,
            }
        else:
            raise


def run_attention_benchmarks():
    """Run attention benchmarks for various configurations."""

    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096]

    results = []

    # Print device information
    if torch.backends.mps.is_available():
        device_name = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_name = "CPU (no GPU available)"

    print("=" * 80)
    print("PART 1: PyTorch Attention Benchmarks")
    print(f"Device: {device_name}")
    print("=" * 80)

    for d_model, seq_len in itertools.product(d_models, seq_lens):
        print(f"\nBenchmarking: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")

        result = benchmark_attention_forward_backward(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            num_iterations=100,
            warmup_iterations=10,
            use_compile=False,
        )

        results.append(result)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Forward: {result['forward_time_ms']:.3f} ms")
            print(f"  Backward: {result['backward_time_ms']:.3f} ms")
            print(f"  Memory: {result['memory_mb']:.2f} MB")

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv('attention_benchmarks_pytorch.csv', index=False)
    print("\n" + "=" * 80)
    print("Results saved to attention_benchmarks_pytorch.csv")
    print("=" * 80)
    print("\nSummary Table:")
    print(df.to_string(index=False))

    return df


def run_compiled_attention_benchmarks():
    """Run compiled attention benchmarks."""

    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096]

    results = []

    # Print device information
    if torch.backends.mps.is_available():
        device_name = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_name = "CPU (no GPU available)"

    print("\n" + "=" * 80)
    print("PART 2: Compiled PyTorch Attention Benchmarks")
    print(f"Device: {device_name}")
    print("=" * 80)

    for d_model, seq_len in itertools.product(d_models, seq_lens):
        print(f"\nBenchmarking: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")

        result = benchmark_attention_forward_backward(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            num_iterations=100,
            warmup_iterations=10,
            use_compile=True,
        )

        results.append(result)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Forward: {result['forward_time_ms']:.3f} ms")
            print(f"  Backward: {result['backward_time_ms']:.3f} ms")
            print(f"  Memory: {result['memory_mb']:.2f} MB")

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv('attention_benchmarks_compiled.csv', index=False)
    print("\n" + "=" * 80)
    print("Results saved to attention_benchmarks_compiled.csv")
    print("=" * 80)
    print("\nSummary Table:")
    print(df.to_string(index=False))

    return df


def benchmark_transformer_model(
    batch_size: int = 8,
    seq_len: int = 512,
    vocab_size: int = 1000,
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    d_ff: int = 1024,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_compile: bool = False,
) -> Dict:
    """
    Benchmark full Transformer model.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of Transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_iterations: Number of iterations to time
        warmup_iterations: Number of warmup iterations
        use_compile: Whether to compile the model

    Returns:
        Dictionary with timing information
    """
    from cs336_basics.model import BasicsTransformerLM

    # Use MPS if available (Apple Silicon), otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if device.type == 'cpu':
        return {'error': 'No GPU available (neither MPS nor CUDA)'}

    try:
        # Create model
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=seq_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=10000.0,
        ).to(device)

        if use_compile:
            model = torch.compile(model)

        # Create random inputs
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Warmup
        for _ in range(warmup_iterations):
            optimizer.zero_grad()
            logits = model(x)
            loss = logits.sum()
            loss.backward()
            optimizer.step()

        synchronize()

        # Benchmark forward only
        forward_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            logits = model(x)
            synchronize()
            end = time.perf_counter()
            forward_times.append(end - start)

        # Benchmark forward + backward + optimizer step
        full_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            optimizer.zero_grad()
            logits = model(x)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            synchronize()
            end = time.perf_counter()
            full_times.append(end - start)

        return {
            'compiled': use_compile,
            'forward_time_ms': sum(forward_times) / len(forward_times) * 1000,
            'full_step_time_ms': sum(full_times) / len(full_times) * 1000,
        }

    except RuntimeError as e:
        error_msg = str(e).lower()
        if 'out of memory' in error_msg:
            empty_cache()
            return {'error': 'OOM', 'compiled': use_compile}
        elif 'int_max' in error_msg or 'dims larger than' in error_msg:
            # MPS backend limitation with large tensor dimensions
            empty_cache()
            return {'error': 'MPS dimension limit exceeded', 'compiled': use_compile}
        else:
            raise


def run_transformer_benchmarks():
    """Run Transformer model benchmarks."""

    # Print device information
    if torch.backends.mps.is_available():
        device_name = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_name = "CPU (no GPU available)"

    print("\n" + "=" * 80)
    print("PART 3: Transformer Model Benchmarks")
    print(f"Device: {device_name}")
    print("=" * 80)

    results = []

    # Benchmark uncompiled
    print("\nBenchmarking uncompiled Transformer...")
    result_uncompiled = benchmark_transformer_model(use_compile=False)
    results.append(result_uncompiled)

    if 'error' not in result_uncompiled:
        print(f"  Forward: {result_uncompiled['forward_time_ms']:.3f} ms")
        print(f"  Full step: {result_uncompiled['full_step_time_ms']:.3f} ms")
    else:
        print(f"  ERROR: {result_uncompiled['error']}")

    # Benchmark compiled
    print("\nBenchmarking compiled Transformer...")
    result_compiled = benchmark_transformer_model(use_compile=True)
    results.append(result_compiled)

    if 'error' not in result_compiled:
        print(f"  Forward: {result_compiled['forward_time_ms']:.3f} ms")
        print(f"  Full step: {result_compiled['full_step_time_ms']:.3f} ms")
    else:
        print(f"  ERROR: {result_compiled['error']}")

    # Save and display
    df = pd.DataFrame(results)
    df.to_csv('transformer_benchmarks.csv', index=False)
    print("\n" + "=" * 80)
    print("Results saved to transformer_benchmarks.csv")
    print("=" * 80)
    print("\nSummary Table:")
    print(df.to_string(index=False))

    # Calculate speedup
    if 'error' not in result_uncompiled and 'error' not in result_compiled:
        forward_speedup = result_uncompiled['forward_time_ms'] / result_compiled['forward_time_ms']
        full_speedup = result_uncompiled['full_step_time_ms'] / result_compiled['full_step_time_ms']
        print(f"\nSpeedup:")
        print(f"  Forward: {forward_speedup:.2f}x")
        print(f"  Full step: {full_speedup:.2f}x")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark attention and Transformer models')
    parser.add_argument('--part', type=str, choices=['attention', 'compiled', 'transformer', 'all'],
                       default='all', help='Which benchmarks to run')

    args = parser.parse_args()

    if args.part in ['attention', 'all']:
        run_attention_benchmarks()

    if args.part in ['compiled', 'all']:
        run_compiled_attention_benchmarks()

    if args.part in ['transformer', 'all']:
        run_transformer_benchmarks()