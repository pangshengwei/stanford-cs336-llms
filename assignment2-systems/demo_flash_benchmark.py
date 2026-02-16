"""
Quick demo of FlashAttention benchmarking.

This runs a minimal benchmark to demonstrate the comparison between
FlashAttention and standard PyTorch attention.
"""

import torch
import time
from cs336_systems.flash_attention_pytorch import FlashAttentionPyTorch
import pandas as pd


def synchronize():
    """Synchronize device operations."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_function(fn, *args, warmup=3, rep=10):
    """Simple benchmark function."""
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
    """Standard PyTorch attention."""
    batch_size, n_queries, d = Q.shape
    scale = 1.0 / (d ** 0.5)

    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if is_causal:
        _, n_keys, _ = K.shape
        q_indices = torch.arange(n_queries, device=Q.device)[:, None]
        k_indices = torch.arange(n_keys, device=Q.device)[None, :]
        causal_mask = q_indices >= k_indices
        S = torch.where(causal_mask, S, torch.tensor(-1e6, device=Q.device, dtype=S.dtype))

    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    return O


def demo_benchmark():
    """Run a quick demo benchmark."""

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

    print("=" * 80)
    print("FlashAttention-2 Quick Demo Benchmark")
    print(f"Device: {device_name}")
    print("=" * 80)

    # Minimal configuration
    batch_size = 1
    configs = [
        (256, 32),
        (512, 64),
        (1024, 64),
        (2048, 128),
    ]

    results = []

    for seq_len, d_model in configs:
        print(f"\nBenchmarking: seq_len={seq_len}, d_model={d_model}")

        # Create inputs
        Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True)
        dO = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        # Benchmark Standard PyTorch
        def pytorch_fwd():
            Q_c = Q.detach().requires_grad_(True)
            K_c = K.detach().requires_grad_(True)
            V_c = V.detach().requires_grad_(True)
            return standard_attention_forward(Q_c, K_c, V_c, True)

        pytorch_fwd_time = benchmark_function(pytorch_fwd, warmup=2, rep=10)

        def pytorch_bwd():
            Q_c = Q.detach().requires_grad_(True)
            K_c = K.detach().requires_grad_(True)
            V_c = V.detach().requires_grad_(True)
            O = standard_attention_forward(Q_c, K_c, V_c, True)
            O.backward(dO)

        pytorch_bwd_time = benchmark_function(pytorch_bwd, warmup=2, rep=10)

        # Benchmark FlashAttention
        def flash_fwd():
            Q_c = Q.detach().requires_grad_(True)
            K_c = K.detach().requires_grad_(True)
            V_c = V.detach().requires_grad_(True)
            return FlashAttentionPyTorch.apply(Q_c, K_c, V_c, True)

        flash_fwd_time = benchmark_function(flash_fwd, warmup=2, rep=10)

        def flash_bwd():
            Q_c = Q.detach().requires_grad_(True)
            K_c = K.detach().requires_grad_(True)
            V_c = V.detach().requires_grad_(True)
            O = FlashAttentionPyTorch.apply(Q_c, K_c, V_c, True)
            O.backward(dO)

        flash_bwd_time = benchmark_function(flash_bwd, warmup=2, rep=10)

        # Calculate speedups
        fwd_speedup = pytorch_fwd_time / flash_fwd_time
        bwd_speedup = pytorch_bwd_time / flash_bwd_time
        e2e_speedup = (pytorch_fwd_time + pytorch_bwd_time) / (flash_fwd_time + flash_bwd_time)

        results.append({
            'seq_len': seq_len,
            'd_model': d_model,
            'pytorch_fwd_ms': pytorch_fwd_time,
            'flash_fwd_ms': flash_fwd_time,
            'fwd_speedup': fwd_speedup,
            'pytorch_bwd_ms': pytorch_bwd_time,
            'flash_bwd_ms': flash_bwd_time,
            'bwd_speedup': bwd_speedup,
            'e2e_speedup': e2e_speedup,
        })

        print(f"  PyTorch - Fwd: {pytorch_fwd_time:.2f}ms, Bwd: {pytorch_bwd_time:.2f}ms")
        print(f"  Flash   - Fwd: {flash_fwd_time:.2f}ms, Bwd: {flash_bwd_time:.2f}ms")
        print(f"  Speedup - Fwd: {fwd_speedup:.2f}x, Bwd: {bwd_speedup:.2f}x, E2E: {e2e_speedup:.2f}x")

    # Display results
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    pd.options.display.float_format = '{:.2f}'.format
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Average Speedups:")
    print("=" * 80)
    print(f"Forward:  {df['fwd_speedup'].mean():.2f}x")
    print(f"Backward: {df['bwd_speedup'].mean():.2f}x")
    print(f"E2E:      {df['e2e_speedup'].mean():.2f}x")


if __name__ == '__main__':
    demo_benchmark()