"""
Memory accounting analysis for attention mechanism.

This script analyzes the memory usage of the attention mechanism
and provides detailed accounting of where memory is allocated.
"""

import math


def analyze_attention_memory(batch_size: int, seq_len: int, d_model: int, dtype_bytes: int = 2):
    """
    Analyze memory usage for attention mechanism.

    Args:
        batch_size: Batch size (B)
        seq_len: Sequence length (N)
        d_model: Model/head dimension (d)
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)

    Returns:
        Dictionary with detailed memory breakdown
    """

    B, N, d = batch_size, seq_len, d_model

    # Forward pass activations (stored for backward pass)
    activations = {
        'Q': B * N * d,
        'K': B * N * d,
        'V': B * N * d,
        'attention_scores': B * N * N,  # QK^T / sqrt(d)
        'attention_weights': B * N * N,  # softmax(scores)
        'output': B * N * d,  # attention_weights @ V
    }

    # Gradients (for backward pass)
    gradients = {
        'dQ': B * N * d,
        'dK': B * N * d,
        'dV': B * N * d,
        'd_attention_weights': B * N * N,
        'd_attention_scores': B * N * N,
        'd_output': B * N * d,
    }

    # Total memory
    total_activations = sum(activations.values())
    total_gradients = sum(gradients.values())
    total_elements = total_activations + total_gradients
    total_mb = total_elements * dtype_bytes / 1024 / 1024

    # Breakdown by component
    breakdown = {
        'Q, K, V (activations)': 3 * B * N * d * dtype_bytes / 1024 / 1024,
        'Attention scores (N x N)': B * N * N * dtype_bytes / 1024 / 1024,
        'Attention weights (N x N)': B * N * N * dtype_bytes / 1024 / 1024,
        'Output': B * N * d * dtype_bytes / 1024 / 1024,
        'Gradients (dQ, dK, dV)': 3 * B * N * d * dtype_bytes / 1024 / 1024,
        'Gradient attention matrices': 2 * B * N * N * dtype_bytes / 1024 / 1024,
    }

    # Memory dominated by N^2 terms
    n_squared_memory = 4 * B * N * N * dtype_bytes / 1024 / 1024  # scores + weights + 2 gradients
    linear_memory = 6 * B * N * d * dtype_bytes / 1024 / 1024  # Q,K,V,output + gradients

    return {
        'batch_size': B,
        'seq_len': N,
        'd_model': d,
        'dtype_bytes': dtype_bytes,
        'total_mb': total_mb,
        'n_squared_mb': n_squared_memory,
        'linear_mb': linear_memory,
        'breakdown': breakdown,
        'activations': activations,
        'gradients': gradients,
    }


def print_memory_analysis(batch_size: int, seq_len: int, d_model: int, dtype_bytes: int = 2):
    """Print detailed memory analysis."""

    result = analyze_attention_memory(batch_size, seq_len, d_model, dtype_bytes)

    print("=" * 80)
    print(f"Memory Analysis for Attention")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size (B): {batch_size}")
    print(f"  Sequence length (N): {seq_len}")
    print(f"  Model dimension (d): {d_model}")
    print(f"  Dtype: {'fp16' if dtype_bytes == 2 else 'fp32'} ({dtype_bytes} bytes)")
    print()

    print(f"Total Memory: {result['total_mb']:.2f} MB")
    print(f"  O(N²) terms: {result['n_squared_mb']:.2f} MB ({result['n_squared_mb']/result['total_mb']*100:.1f}%)")
    print(f"  O(N*d) terms: {result['linear_mb']:.2f} MB ({result['linear_mb']/result['total_mb']*100:.1f}%)")
    print()

    print("Detailed Breakdown:")
    for name, mb in result['breakdown'].items():
        print(f"  {name:35s}: {mb:8.2f} MB")
    print()

    print("Key Insights:")
    print(f"  - Attention score matrix (N x N): {seq_len} x {seq_len} = {seq_len**2:,} elements")
    print(f"  - For backward pass, we store attention scores AND weights (both N x N)")
    print(f"  - Memory grows as O(B * N²) for large N, dominating O(B * N * d) terms")
    print()

    # Compare different sequence lengths
    print("How memory scales with sequence length:")
    print(f"{'Seq Len':>10s} {'N² Memory (MB)':>20s} {'Total Memory (MB)':>20s}")
    for test_len in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        test_result = analyze_attention_memory(batch_size, test_len, d_model, dtype_bytes)
        print(f"{test_len:10d} {test_result['n_squared_mb']:20.2f} {test_result['total_mb']:20.2f}")
    print()

    print("Solution to eliminate O(N²) memory cost:")
    print("  - Flash Attention recomputes attention scores during backward pass")
    print("  - Instead of storing NxN attention matrices, only store O(N) statistics")
    print("  - Uses tiling and recomputation to avoid materializing full attention matrix")
    print("  - Reduces memory from O(N²) to O(N), enabling much longer sequences")


if __name__ == '__main__':
    # Analyze some example configurations
    configs = [
        (8, 4096, 64, 2),   # Example that might OOM
        (8, 8192, 64, 2),   # Larger sequence
        (8, 16384, 64, 2),  # Even larger
    ]

    for batch_size, seq_len, d_model, dtype_bytes in configs:
        print_memory_analysis(batch_size, seq_len, d_model, dtype_bytes)
        print("\n" + "=" * 80 + "\n")