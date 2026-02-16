"""
FlashAttention-2 implementation in pure PyTorch.

This implements the FlashAttention-2 algorithm using tiling, recomputation,
and operator fusion techniques to reduce memory usage and improve performance.
"""

import torch
import torch.nn.functional as F


class FlashAttentionPyTorch(torch.autograd.Function):
    """
    PyTorch autograd.Function implementing FlashAttention-2 forward pass.

    This is a tiled implementation that computes attention in blocks to avoid
    materializing the full attention matrix in memory.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass of FlashAttention-2.

        Args:
            ctx: autograd context for saving tensors
            Q: Query tensor of shape (batch_size, n_queries, d)
            K: Key tensor of shape (batch_size, n_keys, d)
            V: Value tensor of shape (batch_size, n_keys, d)
            is_causal: Whether to apply causal masking

        Returns:
            O: Output tensor of shape (batch_size, n_queries, d)
        """
        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        # Tile sizes - using at least 16x16 as required
        Q_TILE_SIZE = 64  # Bq
        K_TILE_SIZE = 64  # Bk

        # Scale factor
        scale = 1.0 / (d ** 0.5)

        # Number of tiles
        Tq = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        Tk = (n_keys + K_TILE_SIZE - 1) // K_TILE_SIZE

        # Initialize output and logsumexp
        O = torch.zeros_like(Q, dtype=torch.float32)
        L = torch.zeros(batch_size, n_queries, dtype=torch.float32, device=Q.device)

        # Process each query tile
        for i in range(Tq):
            # Query tile indices
            q_start = i * Q_TILE_SIZE
            q_end = min((i + 1) * Q_TILE_SIZE, n_queries)

            # Load query tile
            Qi = Q[:, q_start:q_end, :]  # (batch_size, Bq, d)

            # Initialize tile accumulators
            Oi = torch.zeros(batch_size, q_end - q_start, d, dtype=torch.float32, device=Q.device)
            li = torch.zeros(batch_size, q_end - q_start, dtype=torch.float32, device=Q.device)
            mi = torch.full((batch_size, q_end - q_start), -torch.inf, dtype=torch.float32, device=Q.device)

            # Process each key tile
            for j in range(Tk):
                # Key/Value tile indices
                k_start = j * K_TILE_SIZE
                k_end = min((j + 1) * K_TILE_SIZE, n_keys)

                # Load key and value tiles
                Kj = K[:, k_start:k_end, :]  # (batch_size, Bk, d)
                Vj = V[:, k_start:k_end, :]  # (batch_size, Bk, d)

                # Compute attention scores for this tile
                # Sij = Qi @ Kj^T / sqrt(d)
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale  # (batch_size, Bq, Bk)

                # Apply causal mask if needed
                if is_causal:
                    # Create causal mask for this tile
                    q_indices = torch.arange(q_start, q_end, device=Q.device)[:, None]
                    k_indices = torch.arange(k_start, k_end, device=Q.device)[None, :]
                    causal_mask = q_indices >= k_indices  # (Bq, Bk)
                    Sij = torch.where(causal_mask, Sij, torch.tensor(-1e6, device=Q.device, dtype=Sij.dtype))

                # Compute new maximum
                mij = torch.max(mi, torch.max(Sij, dim=-1).values)  # (batch_size, Bq)

                # Compute unnormalized softmax (numerator only)
                # P_tilde_ij = exp(Sij - mij)
                P_tilde_ij = torch.exp(Sij - mij.unsqueeze(-1))  # (batch_size, Bq, Bk)

                # Update running sum l
                # lij = exp(mi - mij) * li + rowsum(P_tilde_ij)
                lij = torch.exp(mi - mij) * li + torch.sum(P_tilde_ij, dim=-1)  # (batch_size, Bq)

                # Update output
                # Oij = diag(exp(mi - mij)) @ Oi + P_tilde_ij @ Vj
                correction = torch.exp(mi - mij).unsqueeze(-1)  # (batch_size, Bq, 1)
                Oi = Oi * correction + torch.matmul(P_tilde_ij, Vj)

                # Update running values
                li = lij
                mi = mij

            # Final normalization: O = diag(1/l) @ O
            O[:, q_start:q_end, :] = (Oi / li.unsqueeze(-1)).to(Q.dtype)

            # Compute final logsumexp: L = m + log(l)
            L[:, q_start:q_end] = mi + torch.log(li)

        # Save tensors for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale

        return O.to(Q.dtype)

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass using recomputation.

        Args:
            ctx: autograd context with saved tensors
            dO: Gradient of output, shape (batch_size, n_queries, d)

        Returns:
            Gradients for Q, K, V, and None for is_causal
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        # Use compiled backward function (disable for MPS due to compilation issues)
        if Q.device.type == 'mps':
            dQ, dK, dV = flash_backward_eager(Q, K, V, O, dO, L, is_causal, scale)
        else:
            dQ, dK, dV = flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale)

        return dQ, dK, dV, None


def flash_backward_eager(Q, K, V, O, dO, L, is_causal, scale):
    """
    Eager backward pass (no compilation) for MPS compatibility.

    Args:
        Same as flash_backward_compiled

    Returns:
        dQ, dK, dV: Gradients for Q, K, V
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
def flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale):
    """
    Compiled backward pass for FlashAttention.

    Implements Equations 13-19 from the FlashAttention-2 paper using
    recomputation to avoid storing the attention matrix.

    Args:
        Q: Query tensor (batch_size, n_queries, d)
        K: Key tensor (batch_size, n_keys, d)
        V: Value tensor (batch_size, n_keys, d)
        O: Output from forward pass (batch_size, n_queries, d)
        dO: Gradient of output (batch_size, n_queries, d)
        L: Logsumexp from forward pass (batch_size, n_queries)
        is_causal: Whether causal masking was used
        scale: Scale factor (1/sqrt(d))

    Returns:
        dQ, dK, dV: Gradients for Q, K, V
    """
    batch_size, n_queries, d = Q.shape
    _, n_keys, _ = K.shape

    # Recompute S = Q @ K^T / sqrt(d)  (Eq 13)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch_size, n_queries, n_keys)

    # Apply causal mask if needed
    if is_causal:
        q_indices = torch.arange(n_queries, device=Q.device)[:, None]
        k_indices = torch.arange(n_keys, device=Q.device)[None, :]
        causal_mask = q_indices >= k_indices
        S = torch.where(causal_mask, S, torch.tensor(-1e6, device=Q.device, dtype=S.dtype))

    # Recompute P = exp(S - L)  (Eq 14)
    # Note: L was computed as log(sum(exp(S))) so exp(S - L) = exp(S) / sum(exp(S)) = softmax(S)
    P = torch.exp(S - L.unsqueeze(-1))  # (batch_size, n_queries, n_keys)

    # Compute D = rowsum(O * dO)  (used in Eq 17)
    D = torch.sum(O * dO, dim=-1)  # (batch_size, n_queries)

    # Compute dV = P^T @ dO  (Eq 15)
    dV = torch.matmul(P.transpose(-2, -1), dO)  # (batch_size, n_keys, d)

    # Compute dP = dO @ V^T  (Eq 16)
    dP = torch.matmul(dO, V.transpose(-2, -1))  # (batch_size, n_queries, n_keys)

    # Compute dS = P * (dP - D)  (Eq 17)
    # Note: element-wise multiplication
    dS = P * (dP - D.unsqueeze(-1))  # (batch_size, n_queries, n_keys)

    # Compute dQ = dS @ K / sqrt(d)  (Eq 18)
    dQ = torch.matmul(dS, K) * scale  # (batch_size, n_queries, d)

    # Compute dK = dS^T @ Q / sqrt(d)  (Eq 19)
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale  # (batch_size, n_keys, d)

    return dQ, dK, dV