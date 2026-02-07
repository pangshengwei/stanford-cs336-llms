import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """
    Linear transformation module without bias.
    
    Performs y = xW^T where W is the weight matrix.
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weight as W (not W^T) for memory ordering reasons
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        # Initialize weights using truncated normal distribution
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using truncated normal distribution."""
        # Standard deviation for truncated normal: 1/sqrt(in_features)
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2*std, b=2*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize weights using normal distribution
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    From Zhang and Sennrich (2019), equation 4:
        RMSNorm(a_i) = (a_i / RMS(a)) * g_i
    where:
        RMS(a) = sqrt(1/d_model * sum(a_i^2) + eps)
        g_i is a learnable gain parameter
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) # learnable gain parameters
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Compute RMS: sqrt(mean(x^2) + eps)
        # rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        rms = torch.sqrt((1/self.d_model) * torch.sum(x ** 2, dim=-1, keepdim=True) + self.eps)
        normalized = (x / rms) * self.weight
        
        # Cast back to original dtype
        return normalized.to(input_dtype)

class SiLU(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    Combines SiLU (Swish) activation with Gated Linear Units (GLU):
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1*x) ⊙ W3*x)
    
    where:
        - SiLU(x) = x * σ(x) = x / (1 + e^(-x))
        - ⊙ represents element-wise multiplication
        - W1, W3 ∈ R^(d_ff × d_model): up-projection matrices
        - W2 ∈ R^(d_model × d_ff): down-projection matrix
    """
    
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        
        # Calculate d_ff as approximately (8/3) * d_model, rounded to nearest multiple of 64
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            # Round to nearest multiple of 64 for hardware efficiency
            d_ff = ((d_ff + 63) // 64) * 64
        
        self.d_ff = d_ff
        
        # Three linear transformations (no bias as per standard transformer implementations)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu = SiLU()
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) from Su et al. [2021].
    
    Applies pairwise rotation to query/key embeddings based on token positions.
    For a query token q^(i) at position i, applies rotation matrix R^i:
        q'^(i) = R^i * q^(i)
    
    The rotation matrix R^i is block-diagonal with 2×2 rotation blocks:
        R^i_k = [[cos(θ_{i,k}), -sin(θ_{i,k})],
                 [sin(θ_{i,k}),  cos(θ_{i,k})]]
    
    where θ_{i,k} = i / (Θ^(2k/d)) for k ∈ {1, ..., d/2}
    
    This layer has no learnable parameters - cos and sin values are precomputed.
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Initialize RoPE module with precomputed cos/sin buffers.
        
        Args:
            theta: Θ (base) value for the RoPE frequency calculation
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be input
            device: Device to store buffers on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Precompute frequency values for each pair of dimensions
        # θ_{i,k} = i / (Θ^(2k/d)) where k ∈ {0, 1, ..., d/2-1}
        # Compute inverse frequencies: 1 / (Θ^(2k/d))
        k = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (2 * k / d_k))
        
        # Register as buffer (not a parameter - we don't want to learn these)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos and sin values for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # Compute θ_{i,k} = position * inv_freq
        angles = torch.outer(positions, inv_freq)
        
        # Precompute cos and sin
        # Shape: (max_seq_len, d_k // 2)
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)
        
        # Register as buffers
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Get cos and sin values for the given token positions
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # Split x into pairs: (x0, x1), (x2, x3), ..., (x_{d-2}, x_{d-1})
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]  # even indices: x0, x2, x4, ...
        x2 = x_reshaped[..., 1]  # odd indices: x1, x3, x5, ...
        
        # Apply 2D rotation:
        x1_rotated = x1 * cos - x2 * sin # [x1']   [cos(θ)  -sin(θ)] [x1]
        x2_rotated = x1 * sin + x2 * cos # [x2'] = [sin(θ)   cos(θ)] [x2]
        
        # Stack back together and reshape to original shape back to (..., seq_len, d_k)
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)
        return x_rotated.reshape(*x.shape)

class Softmax(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # subtract max for numerical stability
        return torch.exp(x - torch.max(x, dim=self.dim, keepdim=True)[0]) / torch.sum(torch.exp(x - torch.max(x, dim=self.dim, keepdim=True)[0]), dim=self.dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Scaled dot-product attention from Vaswani et al. [2017].
    
    Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Query tensor of shape (batch_size, ..., seq_len_q, d_k)
        K: Key tensor of shape (batch_size, ..., seq_len_k, d_k)
        V: Value tensor of shape (batch_size, ..., seq_len_k, d_v)
        mask: Optional boolean mask of shape (..., seq_len_q, seq_len_k)
              True = attend, False = don't attend (will be masked to -inf)
    
    Returns:
        Output tensor of shape (batch_size, ..., seq_len_q, d_v)
    """    
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1]) # Q.shape[-1] is d_k
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf')) # Where mask is False, set scores to -inf so softmax gives 0 probability
    
    attn_probs = Softmax(dim=-1)(scores)
    # Handle case where entire row is masked (would give NaN from -inf), Replace NaN with 0 (no attention to any key)
    attn_probs = torch.where(torch.isnan(attn_probs), torch.zeros_like(attn_probs), attn_probs)
    
    # Apply attention to values
    output = attn_probs @ V
    
    return output
        
class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking and optional RoPE.
    
    Implements:
        MultiHeadSelfAttention(x) = W_O * MultiHead(W_Q*x, W_K*x, W_V*x)
    
    where MultiHead concatenates outputs from multiple attention heads:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
        head_i = Attention(Q_i, K_i, V_i)
    
    Following Vaswani et al. [2017], we set d_k = d_v = d_model / num_heads.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device=None,
        dtype=None
    ):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Dimensionality of the model (input/output dimension)
            num_heads: Number of attention heads
            rope: Optional RoPE module for positional encoding
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # d_k = d_v = d_model / h
        self.rope = rope
        
        # Projection matrices
        # W_Q, W_K, W_V ∈ R^(d_model × d_model)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # Output projection W_O ∈ R^(d_model × d_model)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Project to Q, K, V. Each has shape: (batch_size, seq_len, d_model)
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # Step 2: Reshape for multi-head attention: Split d_model into (num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to: (batch_size, num_heads, seq_len, head_dim)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Step 3: Apply RoPE to Q and K (not V!)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            
            # Reshape to apply RoPE: (batch_size * num_heads, seq_len, head_dim)
            Q_flat = Q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
            K_flat = K.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
            
            # Expand token_positions for each head: (batch_size * num_heads, seq_len)
            token_positions_expanded = token_positions.unsqueeze(1).expand(batch_size, self.num_heads, seq_len).reshape(batch_size * self.num_heads, seq_len)
            
            # Apply RoPE
            Q_flat = self.rope(Q_flat, token_positions_expanded)
            K_flat = self.rope(K_flat, token_positions_expanded)
            
            # Reshape back: (batch_size, num_heads, seq_len, head_dim)
            Q = Q_flat.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            K = K_flat.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Step 4: Create causal mask: mask[i, j] = True if position i can attend to position j (i.e., j <= i)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        
        # Step 5: Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        
        # Step 6: Concatenate heads: Transpose back: (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)
        
        # Reshape to concatenate heads: (batch_size, seq_len, d_model)
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        
        # Step 7: Apply output projection, Shape: (batch_size, seq_len, d_model)
        output = self.output_proj(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with two sublayers:
    1. Multi-head self-attention with RoPE
    2. Position-wise feed-forward network (SwiGLU)

    Each sublayer follows: output = input + Sublayer(RMSNorm(input))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None
    ):
        """
        Args:
            d_model: Dimensionality of the Transformer block inputs.
            num_heads: Number of heads to use in multi-head self-attention.
            d_ff: Dimensionality of the position-wise feed-forward inner layer.
            theta: Base value for RoPE frequency calculation.
            max_seq_len: Maximum sequence length for RoPE precomputation.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # RMSNorm layers for each sublayer (named ln1, ln2 to match test expectations)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        # Initialize RoPE for positional encoding
        head_dim = d_model // num_heads
        rope = RoPE(theta=theta, d_k=head_dim, max_seq_len=max_seq_len, device=device)

        # Multi-head self-attention with RoPE
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype
        )

        # Position-wise feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Optional token positions for RoPE

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Sublayer 1: y = x + MultiHeadSelfAttention(RMSNorm(x))
        y = x + self.attn(self.ln1(x), token_positions=token_positions)

        # Sublayer 2: output = y + FeedForward(RMSNorm(y))
        output = y + self.ffn(self.ln2(y))

        return output