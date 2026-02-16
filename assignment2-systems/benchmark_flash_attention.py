"""
Benchmark script comparing FlashAttention-2 with standard PyTorch attention.

This script compares:
1. FlashAttention-2 implementation (PyTorch version for MPS, Triton for CUDA)
2. Standard PyTorch attention

Measures forward, backward, and end-to-end latencies across various configurations.
"""

import torch
import time
import itertools
from typing import Dict, List, Tuple
import pandas as pd
from cs336_systems.flash_attention_pytorch import FlashAttentionPyTorch

# Try to import Triton version for CUDA
try:
    from cs336_systems.flash_attention_triton import FlashAttentionTriton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def synchronize():
    """Synchronize device operations."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache():
    """Empty device cache."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_function(fn, *args, warmup=10, rep=100):
    """
    Benchmark a function similar to triton.testing.do_bench.

    Args:
        fn: Function to benchmark
        *args: Arguments to pass to the function
        warmup: Number of warmup iterations
        rep: Number of timing iterations

    Returns:
        Average time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)

    synchronize()

    # Benchmark
    times = []
    for _ in range(rep):
        start = time.perf_counter()
        fn(*args)
        synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return sum(times) / len(times)


def standard_attention_forward(Q, K, V, is_causal=True):
    """Standard PyTorch attention implementation."""
    batch_size, n_queries, d = Q.shape
    _, n_keys, _ = K.shape

    scale = 1.0 / (d ** 0.5)

    # S = Q @ K^T / sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Apply causal mask
    if is_causal:
        q_indices = torch.arange(n_queries, device=Q.device)[:, None]
        k_indices = torch.arange(n_keys, device=Q.device)[None, :]
        causal_mask = q_indices >= k_indices
        S = torch.where(causal_mask, S, torch.tensor(-1e6, device=Q.device, dtype=S.dtype))

    # Softmax
    P = torch.softmax(S, dim=-1)

    # O = P @ V
    O = torch.matmul(P, V)

    return O


def benchmark_attention_comparison(
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    is_causal: bool = True,
) -> Dict:
    """
    Benchmark FlashAttention vs standard PyTorch attention.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        dtype: Data type (bfloat16 or float32)
        device: Device to run on
        is_causal: Whether to use causal masking

    Returns:
        Dictionary with benchmark results
    """
    try:
        empty_cache()

        # Create random inputs
        Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

        # Create gradient for backward
        dO = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

        # Choose FlashAttention implementation
        if device.type == 'cuda' and TRITON_AVAILABLE:
            FlashAttn = FlashAttentionTriton
            impl_name = "FlashAttention (Triton)"
        else:
            FlashAttn = FlashAttentionPyTorch
            impl_name = "FlashAttention (PyTorch)"

        results = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'd_model': d_model,
            'dtype': str(dtype).split('.')[-1],
            'implementation': impl_name,
        }

        # === Benchmark Standard PyTorch Attention ===

        # Forward only
        def pytorch_forward():
            Q_clone = Q.detach().requires_grad_(True)
            K_clone = K.detach().requires_grad_(True)
            V_clone = V.detach().requires_grad_(True)
            return standard_attention_forward(Q_clone, K_clone, V_clone, is_causal)

        pytorch_forward_time = benchmark_function(pytorch_forward, warmup=5, rep=50)

        # Backward only
        def pytorch_backward():
            Q_clone = Q.detach().requires_grad_(True)
            K_clone = K.detach().requires_grad_(True)
            V_clone = V.detach().requires_grad_(True)
            O = standard_attention_forward(Q_clone, K_clone, V_clone, is_causal)
            O.backward(dO)

        pytorch_backward_time = benchmark_function(pytorch_backward, warmup=5, rep=50)

        # End-to-end (forward + backward)
        pytorch_e2e_time = pytorch_forward_time + pytorch_backward_time

        # === Benchmark FlashAttention ===

        # Forward only
        def flash_forward():
            Q_clone = Q.detach().requires_grad_(True)
            K_clone = K.detach().requires_grad_(True)
            V_clone = V.detach().requires_grad_(True)
            return FlashAttn.apply(Q_clone, K_clone, V_clone, is_causal)

        flash_forward_time = benchmark_function(flash_forward, warmup=5, rep=50)

        # Backward only
        def flash_backward():
            Q_clone = Q.detach().requires_grad_(True)
            K_clone = K.detach().requires_grad_(True)
            V_clone = V.detach().requires_grad_(True)
            O = FlashAttn.apply(Q_clone, K_clone, V_clone, is_causal)
            O.backward(dO)

        flash_backward_time = benchmark_function(flash_backward, warmup=5, rep=50)

        # End-to-end (forward + backward)
        flash_e2e_time = flash_forward_time + flash_backward_time

        # Store results
        results.update({
            'pytorch_forward_ms': pytorch_forward_time,
            'pytorch_backward_ms': pytorch_backward_time,
            'pytorch_e2e_ms': pytorch_e2e_time,
            'flash_forward_ms': flash_forward_time,
            'flash_backward_ms': flash_backward_time,
            'flash_e2e_ms': flash_e2e_time,
            'forward_speedup': pytorch_forward_time / flash_forward_time,
            'backward_speedup': pytorch_backward_time / flash_backward_time,
            'e2e_speedup': pytorch_e2e_time / flash_e2e_time,
        })

        return results

    except RuntimeError as e:
        error_msg = str(e).lower()
        if 'out of memory' in error_msg or 'int_max' in error_msg or 'dims larger than' in error_msg:
            empty_cache()
            return {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'd_model': d_model,
                'dtype': str(dtype).split('.')[-1],
                'error': 'OOM or dimension limit',
            }
        else:
            raise


def run_flash_attention_benchmarks():
    """Run comprehensive FlashAttention benchmarks."""

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device('cpu')
        device_name = "CPU"

    print("=" * 100)
    print("FlashAttention-2 Benchmark")
    print(f"Device: {device_name}")
    print("=" * 100)

    # Configuration
    batch_size = 1  # As specified
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_models = [16, 32, 64, 128]

    # Use float32 for MPS (bfloat16 support is limited)
    if device.type == 'mps':
        dtypes = [torch.float32]
        print("Note: Using float32 only (bfloat16 has limited MPS support)")
    else:
        dtypes = [torch.bfloat16, torch.float32]

    is_causal = True

    results = []
    total_configs = len(seq_lens) * len(d_models) * len(dtypes)
    current = 0

    # Run benchmarks
    for seq_len, d_model, dtype in itertools.product(seq_lens, d_models, dtypes):
        current += 1

        print(f"\n[{current}/{total_configs}] Benchmarking: seq_len={seq_len}, d_model={d_model}, dtype={dtype}")

        result = benchmark_attention_comparison(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            dtype=dtype,
            device=device,
            is_causal=is_causal,
        )

        results.append(result)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  PyTorch - Forward: {result['pytorch_forward_ms']:.3f} ms, "
                  f"Backward: {result['pytorch_backward_ms']:.3f} ms, "
                  f"E2E: {result['pytorch_e2e_ms']:.3f} ms")
            print(f"  Flash   - Forward: {result['flash_forward_ms']:.3f} ms, "
                  f"Backward: {result['flash_backward_ms']:.3f} ms, "
                  f"E2E: {result['flash_e2e_ms']:.3f} ms")
            print(f"  Speedup - Forward: {result['forward_speedup']:.2f}x, "
                  f"Backward: {result['backward_speedup']:.2f}x, "
                  f"E2E: {result['e2e_speedup']:.2f}x")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_file = 'flash_attention_benchmarks.csv'
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 100)
    print(f"Results saved to {output_file}")
    print("=" * 100)

    # Display summary table
    print("\nSummary Table (First 20 rows):")
    print("=" * 100)

    # Select columns to display
    display_cols = ['seq_len', 'd_model', 'dtype',
                   'pytorch_forward_ms', 'flash_forward_ms', 'forward_speedup',
                   'pytorch_backward_ms', 'flash_backward_ms', 'backward_speedup',
                   'pytorch_e2e_ms', 'flash_e2e_ms', 'e2e_speedup']

    # Filter out error rows for display
    df_success = df[~df.get('error', pd.Series([False]*len(df))).notna()]

    if len(df_success) > 0:
        display_df = df_success[display_cols].head(20)

        # Format floats for better readability
        pd.options.display.float_format = '{:.2f}'.format
        print(display_df.to_string(index=False))

        # Print average speedups
        print("\n" + "=" * 100)
        print("Average Speedups:")
        print("=" * 100)
        print(f"Forward:  {df_success['forward_speedup'].mean():.2f}x")
        print(f"Backward: {df_success['backward_speedup'].mean():.2f}x")
        print(f"E2E:      {df_success['e2e_speedup'].mean():.2f}x")

        # Print best cases
        print("\n" + "=" * 100)
        print("Best Speedup Cases:")
        print("=" * 100)
        best_e2e = df_success.loc[df_success['e2e_speedup'].idxmax()]
        print(f"Best E2E: {best_e2e['e2e_speedup']:.2f}x at seq_len={int(best_e2e['seq_len'])}, "
              f"d_model={int(best_e2e['d_model'])}, dtype={best_e2e['dtype']}")
    else:
        print("No successful benchmarks to display.")

    # Print any errors
    df_errors = df[df.get('error', pd.Series([False]*len(df))).notna()]
    if len(df_errors) > 0:
        print("\n" + "=" * 100)
        print(f"Errors encountered: {len(df_errors)} configurations")
        print("=" * 100)
        print(df_errors[['seq_len', 'd_model', 'dtype', 'error']].to_string(index=False))

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark FlashAttention-2 vs PyTorch attention')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with fewer configurations')

    args = parser.parse_args()

    if args.quick:
        print("Running quick benchmark (limited configurations)...")
        # Override globals for quick test
        import __main__
        __main__.seq_lens = [128, 512, 2048]
        __main__.d_models = [32, 64]

    run_flash_attention_benchmarks()