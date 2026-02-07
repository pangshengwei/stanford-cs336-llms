import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np

class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss: -log(softmax(logits)[target])
    Uses log-sum-exp trick for numerical stability.
    """
    
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss from scratch
        Formula: -log(softmax(logits)[target]) = -log(exp(logits[target]) / sum(exp(logits))) = -logits[target] + log(sum(exp(logits)))
        
        Args:
            logits: Unnormalized logits of shape (..., vocab_size)
            targets: Target class indices of shape (...)
        
        Returns:
            Scalar loss averaged across all examples
        """
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        batch_size = logits_flat.shape[0]
        
        # Subtract max for numerical stability
        max_logits = torch.max(logits_flat, dim=-1, keepdim=True)[0]
        logits_shifted = logits_flat - max_logits
        
        # Compute log(sum(exp(logits)))
        log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1, keepdim=True))
        
        # formula: -log(softmax(logits)[target])
        batch_indices = torch.arange(batch_size, device=logits.device)
        target_logits = logits_shifted[batch_indices, targets_flat].unsqueeze(-1)
        
        # Cross-entropy: -target_logit + log_sum_exp
        losses = -target_logits + log_sum_exp
        
        return torch.mean(losses)

class Perplexity(nn.Module):
    """
    Perplexity metric: exp(average cross-entropy loss)
    
    For a sequence with cross-entropy losses ℓ₁, ..., ℓₘ:
        perplexity = exp(1/m * Σ ℓᵢ)
    """
    
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.loss_fn = CrossEntropyLoss(device=device, dtype=dtype)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute perplexity from logits and targets.
        
        Args:
            logits: Unnormalized logits of shape (..., vocab_size)
            targets: Target class indices of shape (...)
        
        Returns:
            Scalar perplexity value
        """
        avg_loss = self.loss_fn(logits, targets)
        return torch.exp(avg_loss)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
        
        for p in group["params"]:
            if p.grad is None:
                continue # Skip if no gradient
            
            state = self.state[p] # Get state associated with p.
            t = state.get("t", 0) # Get iteration number from the state, or initial value.
            grad = p.grad.data # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
            state["t"] = t + 1 # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer from Loshchilov and Hutter (2019).
    
    Maintains first and second moment estimates for each parameter:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        
    With bias correction and weight decay.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate α
            betas: (β₁, β₂) coefficients for moment estimates
            eps: ε for numerical stability
            weight_decay: λ weight decay coefficient
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure to recompute the loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Initialize state for this parameter if needed
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                # Get state
                t = state["t"]
                m = state["m"]
                v = state["v"]
                
                # Increment timestep (t starts at 1)
                t += 1
                state["t"] = t
                
                # Update first moment: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update second moment: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected learning rate: α_t = α * sqrt(1 - β₂^t) / (1 - β₁^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                lr_t = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters: θ_t = θ_{t-1} - α_t * m_t / (sqrt(v_t) + ε)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                
                # Apply weight decay: θ_t = θ_t - α * λ * θ_t
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                
        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    """
    Cosine annealing learning rate schedule with warmup.
    
    Schedule:
        - Warmup (t < T_w): α_t = (t / T_w) * α_max
        - Cosine annealing (T_w ≤ t ≤ T_c): α_t = α_min + 0.5 * (1 + cos((t - T_w) / (T_c - T_w) * π)) * (α_max - α_min)
        - Post-annealing (t > T_c): α_t = α_min
    
    Args:
        it: Current iteration (timestep)
        max_learning_rate: Maximum learning rate α_max
        min_learning_rate: Minimum (final) learning rate α_min
        warmup_iters: Number of warmup iterations T_w
        cosine_cycle_iters: Number of cosine annealing iterations T_c
    
    Returns:
        Learning rate for iteration t
    """
    # Warmup phase: linear increase from 0 to max_learning_rate
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    # Cosine annealing phase
    elif it <= cosine_cycle_iters:
        # Progress through cosine cycle: 0 to 1
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # Cosine annealing: starts at 1, goes to 0
        cosine_decay = 0.5 * (1.0 + math.cos(progress * math.pi))
        # Scale from max to min
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)
    
    # Post-annealing phase: constant at min_learning_rate
    else:
        return min_learning_rate


def gradient_clipping(parameters, max_l2_norm: float, eps: float = 1e-6):
    """
    Clip gradients to have maximum L2 norm.
    
    If ||g||_2 > M, scale gradients by M / (||g||_2 + ε)
    
    Args:
        parameters: Iterable of parameters with gradients
        max_l2_norm: Maximum L2 norm M
        eps: Small value for numerical stability (default: 1e-6)
    """
    # Collect all gradients
    gradients = []
    for p in parameters:
        if p.grad is not None:
            gradients.append(p.grad)
    
    if len(gradients) == 0:
        return
    
    # Compute total L2 norm of all gradients
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in gradients))
    
    # Clip if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for g in gradients:
            g.mul_(clip_coef)

def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of training sequences from tokenized data.
    
    Given a sequence of tokens x = (x1, ..., xn), samples B sequences of length m,
    where each sequence starts at a random position i ∈ [0, n - m - 1].
    
    Returns:
        - inputs: (batch_size, context_length) - input token sequences
        - targets: (batch_size, context_length) - corresponding next tokens
    
    Example:
        For data = [x1, x2, x3, x4, x5, x6] and context_length = 3:
        If we sample starting at position 1:
            inputs  = [x2, x3, x4]
            targets = [x3, x4, x5]
    
    Args:
        data: Numpy array of token IDs, shape (n,)
        batch_size: Number of sequences to sample (B)
        context_length: Length of each sequence (m)
        device: PyTorch device string ('cpu', 'cuda:0', 'mps', etc.)
    
    Returns:
        Tuple of (inputs, targets), both tensors of shape (batch_size, context_length)
    """
    n = len(data)
    
    # Sample random starting positions for each sequence in the batch
    # Valid range: [0, n - context_length - 1] to ensure we can get context_length + 1 tokens
    max_start_idx = n - context_length - 1
    if max_start_idx < 0:
        raise ValueError(f"Data length {n} is too short for context_length {context_length}")
    
    # Sample batch_size random starting positions
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # Create input and target sequences
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    targets = np.zeros((batch_size, context_length), dtype=np.int64)
    
    for i, start_idx in enumerate(start_indices):
        # Input: [x_start, x_start+1, ..., x_start+context_length-1]
        inputs[i] = data[start_idx : start_idx + context_length]
        # Target: [x_start+1, x_start+2, ..., x_start+context_length]
        targets[i] = data[start_idx + 1 : start_idx + context_length + 1]
    
    # Convert to PyTorch tensors and move to device
    inputs_tensor = torch.from_numpy(inputs).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)
    
    return inputs_tensor, targets_tensor
    
    
def save_checkpoint(model, optimizer, iteration, out):
    """
    Save the model and optimizer state to a file.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }, out)


def load_checkpoint(src, model, optimizer):
    """
    Load the model and optimizer state from a file.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = 'cpu',
    eos_token_id: int = None
):
    """
    Generate text from the language model.
    
    Args:
        model: Transformer language model
        tokenizer: BPE tokenizer with encode/decode methods
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Softmax temperature (τ). Lower = more deterministic.
                    τ → 0: argmax (greedy), τ = 1: standard sampling, τ > 1: more random
        top_p: Nucleus sampling threshold. Only sample from top tokens with cumulative prob >= p.
               p = 1.0: sample from full distribution, p < 1.0: truncate low-prob tokens
        device: Device to run on
        eos_token_id: End-of-sequence token ID. If None, uses tokenizer's eos_token_id
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Get EOS token ID
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    
    # Generate tokens
    generated_ids = input_ids.copy()
    
    for _ in range(max_new_tokens):
        # Get model predictions
        # Only use the last context_length tokens if sequence is too long
        context_length = model.context_length
        if input_tensor.size(1) > context_length:
            input_tensor = input_tensor[:, -context_length:]
        
        # Forward pass
        logits = model(input_tensor)  # Shape: (1, seq_len, vocab_size)
        
        # Get logits for the last position (next token prediction)
        next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
        
        # Apply temperature scaling
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Convert logits to probabilities
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # Compute cumulative probabilities
            cumsum_probs = torch.cumsum(sorted_probs, dim=0)
            
            # Find the smallest set of tokens with cumulative prob >= top_p
            # Remove tokens with cumsum > top_p (keep first token that exceeds threshold)
            sorted_indices_to_remove = cumsum_probs > top_p
            
            # Shift right to keep at least one token (the first one)
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            # Zero out probabilities of removed tokens
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0
            
            # Renormalize
            probs = probs / probs.sum()
        
        # Sample next token
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        # Check for EOS token
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        
        # Append to generated sequence
        generated_ids.append(next_token_id)
        
        # Update input tensor
        input_tensor = torch.cat([
            input_tensor,
            torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        ], dim=1)
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, num_eval_batches=100):
    """
    Evaluate model on validation data.
    
    Args:
        model: The model to evaluate
        data: Validation dataset (numpy array)
        batch_size: Batch size for evaluation
        context_length: Context length
        device: Device to run on
        num_eval_batches: Number of batches to evaluate on
    
    Returns:
        Average loss and perplexity
    """
    model.eval()
    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    
    for _ in range(num_eval_batches):
        inputs, targets = get_batch(data, batch_size, context_length, device)
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    
    avg_loss = total_loss / num_eval_batches
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def train(
    model,
    optimizer,
    train_data,
    val_data,
    batch_size,
    context_length,
    device,
    max_steps,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    max_grad_norm=1.0,
    eval_interval=500,
    save_interval=1000,
    log_interval=100,
    checkpoint_dir='checkpoints',
    start_step=0,
    use_wandb=False,
    wandb_project=None,
    wandb_run_name=None,
    wandb_config=None
):
    """
    Train the language model.
    
    Args:
        model: Transformer model
        optimizer: Optimizer (e.g., AdamW)
        train_data: Training dataset (numpy array of token IDs)
        val_data: Validation dataset (numpy array of token IDs)
        batch_size: Batch size
        context_length: Context length (sequence length)
        device: Device ('cpu', 'cuda:0', 'mps')
        max_steps: Total number of training steps
        max_learning_rate: Maximum learning rate
        min_learning_rate: Minimum learning rate
        warmup_iters: Number of warmup iterations
        max_grad_norm: Maximum gradient norm for clipping
        eval_interval: Evaluate every N steps
        save_interval: Save checkpoint every N steps
        log_interval: Log training loss every N steps
        checkpoint_dir: Directory to save checkpoints
        start_step: Starting step (for resuming training)
    
    Returns:
        Training history dictionary
    """
    import os
    import time
    
    # Initialize Weights & Biases if requested
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project or "tinystories-lm",
                name=wandb_run_name,
                config=wandb_config or {},
                resume="allow" if start_step > 0 else False
            )
            # Log model architecture
            wandb.watch(model, log="all", log_freq=log_interval)
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            use_wandb = False
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Loss function
    loss_fn = CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_perplexity': [],
        'learning_rates': [],
        'steps': [],
        'wall_times': []
    }
    
    # Training loop
    model.train()
    start_time = time.time()
    running_loss = 0.0
    
    print(f"Starting training from step {start_step} to {max_steps}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Context length: {context_length}")
    print(f"Total tokens per step: {batch_size * context_length}")
    print(f"Total tokens to process: {batch_size * context_length * max_steps:,}")
    print("-" * 80)
    
    for step in range(start_step, max_steps):
        # Get learning rate for this step
        lr = get_lr_cosine_schedule(
            step,
            max_learning_rate,
            min_learning_rate,
            warmup_iters,
            max_steps  # Cosine decay ends at max_steps
        )
        
        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Sample batch
        inputs, targets = get_batch(train_data, batch_size, context_length, device)
        
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(model.parameters(), max_grad_norm)
        
        # Calculate gradient norm before clipping (for logging)
        total_grad_norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
        ).item()
        
        # Optimizer step
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        
        # Log to W&B (every step)
        if use_wandb:
            wandb.log({
                'train/loss_step': loss.item(),
                'train/learning_rate': lr,
                'train/grad_norm': total_grad_norm,
                'train/step': step + 1,
            }, step=step + 1)
        
        # Logging
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (batch_size * context_length * log_interval) / (time.time() - start_time + 1e-10)
            
            print(f"Step {step + 1}/{max_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Tokens/sec: {tokens_per_sec:.0f} | "
                  f"Time: {elapsed:.1f}s")
            
            history['train_loss'].append(avg_loss)
            history['learning_rates'].append(lr)
            history['steps'].append(step + 1)
            history['wall_times'].append(elapsed)
            
            # Log to W&B
            if use_wandb:
                wandb.log({
                    'train/loss_avg': avg_loss,
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/wall_time': elapsed,
                }, step=step + 1)
            
            running_loss = 0.0
            start_time = time.time()
        
        # Evaluation
        if (step + 1) % eval_interval == 0:
            val_loss, val_perplexity = evaluate(
                model, val_data, batch_size, context_length, device
            )
            
            elapsed_total = sum(history['wall_times']) if history['wall_times'] else 0
            print(f"\n{'='*80}")
            print(f"Evaluation at step {step + 1}")
            print(f"Validation Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
            print(f"Total time: {elapsed_total:.1f}s")
            print(f"{'='*80}\n")
            
            history['val_loss'].append(val_loss)
            history['val_perplexity'].append(val_perplexity)
            
            # Log to W&B
            if use_wandb:
                wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': val_perplexity,
                }, step=step + 1)
            
            model.train()
        
        # Save checkpoint
        if (step + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step + 1}.pt')
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final evaluation
    val_loss, val_perplexity = evaluate(model, val_data, batch_size, context_length, device)
    print(f"\n{'='*80}")
    print(f"Final Evaluation")
    print(f"Validation Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
    print(f"{'='*80}\n")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
    save_checkpoint(model, optimizer, max_steps, final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    
    # Finish W&B run
    if use_wandb:
        wandb.finish()
    
    return history
    