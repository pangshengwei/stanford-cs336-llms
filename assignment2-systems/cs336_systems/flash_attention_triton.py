"""
FlashAttention-2 implementation using Triton kernels.

This implements the FlashAttention-2 algorithm using custom Triton kernels
for efficient GPU execution.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    FlashAttention-2 forward pass kernel in Triton.

    This kernel processes one query tile for one batch element.
    The launch grid is (Tq, batch_size).

    Args:
        Q_ptr, K_ptr, V_ptr: Pointers to input tensors
        O_ptr, L_ptr: Pointers to output tensors
        stride_*: Strides for each dimension
        N_QUERIES, N_KEYS: Sequence lengths
        scale: 1/sqrt(d) scaling factor
        D: Model dimension (constexpr)
        Q_TILE_SIZE, K_TILE_SIZE: Tile sizes (constexpr)
        is_causal: Whether to apply causal masking (constexpr)
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load query tile
    Qi = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)

    # Initialize accumulators (using float32 for precision)
    Oi = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    li = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    mi = tl.full([Q_TILE_SIZE], float('-inf'), dtype=tl.float32)

    # Compute number of key tiles
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # Query indices for causal masking
    q_offset = query_tile_index * Q_TILE_SIZE
    q_indices = q_offset + tl.arange(0, Q_TILE_SIZE)

    # Iterate over key tiles
    for j in range(Tk):
        # Load key and value tiles
        Kj = tl.load(K_block_ptr)  # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr)  # (K_TILE_SIZE, D)

        # Compute attention scores: Sij = Qi @ Kj^T * scale
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Apply causal mask if needed
        if is_causal:
            k_offset = j * K_TILE_SIZE
            k_indices = k_offset + tl.arange(0, K_TILE_SIZE)
            # Create mask: q_idx >= k_idx
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            # Mask out positions where q_idx < k_idx
            Sij = tl.where(causal_mask, Sij, float('-1e6'))

        # Compute row-wise maximum
        mij = tl.maximum(mi, tl.max(Sij, axis=1))  # (Q_TILE_SIZE,)

        # Compute unnormalized softmax: P_tilde_ij = exp(Sij - mij)
        P_tilde_ij = tl.exp(Sij - mij[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Update running sum: lij = exp(mi - mij) * li + rowsum(P_tilde_ij)
        lij = tl.exp(mi - mij) * li + tl.sum(P_tilde_ij, axis=1)  # (Q_TILE_SIZE,)

        # Update output: Oij = diag(exp(mi - mij)) @ Oi + P_tilde_ij @ Vj
        correction = tl.exp(mi - mij)
        Oi = Oi * correction[:, None]
        # Cast P_tilde_ij to match V dtype before matmul
        P_tilde_ij = P_tilde_ij.to(Vj.dtype)
        Oi = tl.dot(P_tilde_ij, Vj, acc=Oi)

        # Update running values
        li = lij
        mi = mij

        # Advance key and value pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Final normalization: O = O / l
    Oi = Oi / li[:, None]

    # Compute final logsumexp: L = m + log(l)
    Li = mi + tl.log(li)

    # Cast output to appropriate dtype and store
    Oi = Oi.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, Oi)
    tl.store(L_block_ptr, Li)


class FlashAttentionTriton(torch.autograd.Function):
    """
    Triton-based FlashAttention-2 implementation.

    Uses custom Triton kernels for the forward pass and compiled PyTorch
    for the backward pass.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass using Triton kernel.

        Args:
            ctx: autograd context
            Q: Query tensor (batch_size, n_queries, d)
            K: Key tensor (batch_size, n_keys, d)
            V: Value tensor (batch_size, n_keys, d)
            is_causal: Whether to apply causal masking

        Returns:
            O: Output tensor (batch_size, n_queries, d)
        """
        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        # Tile sizes
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        # Scale factor
        scale = 1.0 / (d ** 0.5)

        # Allocate output tensors
        O = torch.empty_like(Q)
        L = torch.empty(batch_size, n_queries, dtype=torch.float32, device=Q.device)

        # Launch grid: (num_query_tiles, batch_size)
        Tq = triton.cdiv(n_queries, Q_TILE_SIZE)
        grid = (Tq, batch_size)

        # Launch kernel
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale

        return O

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass using compiled PyTorch.

        Args:
            ctx: autograd context
            dO: Gradient of output

        Returns:
            Gradients for Q, K, V, and None for is_causal
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        # Use compiled backward function (or eager for MPS)
        if Q.device.type == 'mps':
            dQ, dK, dV = flash_backward_triton_eager(Q, K, V, O, dO, L, is_causal, scale)
        else:
            dQ, dK, dV = flash_backward_triton_compiled(Q, K, V, O, dO, L, is_causal, scale)

        return dQ, dK, dV, None


def flash_backward_triton_eager(Q, K, V, O, dO, L, is_causal, scale):
    """
    Eager backward pass (no compilation) for MPS compatibility.

    Args:
        Q, K, V: Input tensors
        O: Output from forward
        dO: Gradient of output
        L: Logsumexp from forward
        is_causal: Causal masking flag
        scale: Scale factor

    Returns:
        dQ, dK, dV: Gradients
    """
    batch_size, n_queries, d = Q.shape
    _, n_keys, _ = K.shape

    # Recompute S = Q @ K^T / sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if is_causal:
        q_indices = torch.arange(n_queries, device=Q.device)[:, None]
        k_indices = torch.arange(n_keys, device=Q.device)[None, :]
        causal_mask = q_indices >= k_indices
        S = torch.where(causal_mask, S, torch.tensor(-1e6, device=Q.device, dtype=S.dtype))

    # Recompute P = exp(S - L)
    P = torch.exp(S - L.unsqueeze(-1))

    # Compute D = rowsum(O * dO)
    D = torch.sum(O * dO, dim=-1)

    # Compute gradients
    dV = torch.matmul(P.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = P * (dP - D.unsqueeze(-1))
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale

    return dQ, dK, dV


@torch.compile
def flash_backward_triton_compiled(Q, K, V, O, dO, L, is_causal, scale):
    """
    Compiled backward pass for Triton FlashAttention.

    Implements the same backward computation as the PyTorch version.

    Args:
        Q, K, V: Input tensors
        O: Output from forward
        dO: Gradient of output
        L: Logsumexp from forward
        is_causal: Causal masking flag
        scale: Scale factor

    Returns:
        dQ, dK, dV: Gradients
    """
    batch_size, n_queries, d = Q.shape
    _, n_keys, _ = K.shape

    # Recompute S = Q @ K^T / sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if is_causal:
        q_indices = torch.arange(n_queries, device=Q.device)[:, None]
        k_indices = torch.arange(n_keys, device=Q.device)[None, :]
        causal_mask = q_indices >= k_indices
        S = torch.where(causal_mask, S, torch.tensor(-1e6, device=Q.device, dtype=S.dtype))

    # Recompute P = exp(S - L)
    P = torch.exp(S - L.unsqueeze(-1))

    # Compute D = rowsum(O * dO)
    D = torch.sum(O * dO, dim=-1)

    # Compute gradients
    dV = torch.matmul(P.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = P * (dP - D.unsqueeze(-1))
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale

    return dQ, dK, dV