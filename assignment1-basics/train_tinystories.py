#!/usr/bin/env python3
"""
Training script for TinyStories dataset.

Usage:
    python train_tinystories.py --batch_size 32 --max_lr 3e-4 --device auto
"""

import argparse
import json
import os
import numpy as np
import torch

from cs336_basics.transformer import Transformer
from cs336_basics.train import AdamW, train, load_checkpoint


def load_tokenized_data(tokenized_dir: str = 'data/tokenized'):
    """
    Load pre-tokenized data from disk using memory mapping.
    
    Args:
        tokenized_dir: Directory containing tokenized .npy files
    
    Returns:
        train_data, val_data (memory-mapped numpy arrays)
    """
    train_tokens_path = os.path.join(tokenized_dir, 'train_tokens.npy')
    val_tokens_path = os.path.join(tokenized_dir, 'val_tokens.npy')
    
    # Check if files exist
    if not os.path.exists(train_tokens_path):
        raise FileNotFoundError(
            f"Training tokens not found at {train_tokens_path}\n"
            f"Please run: python tokenize_data.py"
        )
    if not os.path.exists(val_tokens_path):
        raise FileNotFoundError(
            f"Validation tokens not found at {val_tokens_path}\n"
            f"Please run: python tokenize_data.py"
        )
    
    print("Loading pre-tokenized data...")
    
    # Load with memory mapping for efficient large file handling
    train_data = np.load(train_tokens_path, mmap_mode='r')
    val_data = np.load(val_tokens_path, mmap_mode='r')
    
    print(f"  Train: {len(train_data):,} tokens ({train_data.nbytes / 1e6:.2f} MB)")
    print(f"  Val:   {len(val_data):,} tokens ({val_data.nbytes / 1e6:.2f} MB)")
    print(f"  Dtype: {train_data.dtype}")
    
    # Verify data integrity
    print(f"  Train token range: [{train_data.min()}, {train_data.max()}]")
    print(f"  Val token range:   [{val_data.min()}, {val_data.max()}]")
    
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description='Train Transformer on TinyStories')
    
    # Data arguments
    parser.add_argument('--tokenized_dir', type=str, default='data/tokenized',
                        help='Directory containing pre-tokenized .npy files')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256,
                        help='Context length (sequence length)')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=1344,
                        help='Feed-forward dimension (should be ~8/3 * d_model and multiple of 64)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--rope_theta', type=float, default=10000.0,
                        help='RoPE theta parameter')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Maximum number of training steps')
    parser.add_argument('--max_lr', type=float, default=3e-4,
                        help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=3e-5,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=500,
                        help='Number of warmup iterations')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95,
                        help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Adam epsilon')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Logging and checkpointing
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluate every N steps')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log training loss every N steps')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda:0, mps)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster training')
    parser.add_argument('--compile_backend', type=str, default='aot_eager',
                        help='Compile backend (aot_eager for mps, default for cuda)')
    
    # Weights & Biases
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='tinystories-lm',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Set device with validation
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
            print("Auto-detected: CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("Auto-detected: Apple Silicon GPU (MPS)")
        else:
            device = 'cpu'
            print("Auto-detected: CPU")
    else:
        device = args.device
        # Validate requested device
        if device == 'mps':
            if not hasattr(torch.backends, 'mps'):
                print(f"⚠️  WARNING: MPS not supported in this PyTorch version")
                print(f"   Your PyTorch: {torch.__version__}")
                print(f"   Required: 2.0+")
                print(f"   Falling back to CPU")
                device = 'cpu'
            elif not torch.backends.mps.is_available():
                print(f"⚠️  WARNING: MPS not available on this system")
                print(f"   Falling back to CPU")
                device = 'cpu'
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                print(f"⚠️  WARNING: CUDA not available")
                print(f"   Falling back to CPU")
                device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load pre-tokenized data
    print("\n" + "="*80)
    print("Loading data...")
    print("="*80)
    train_data, val_data = load_tokenized_data(args.tokenized_dir)
    
    # Calculate total tokens to process
    total_tokens = args.batch_size * args.context_length * args.max_steps
    print(f"\nTotal tokens to process: {total_tokens:,}")
    
    # Create model
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_params_no_embed = sum(p.numel() for p in model.parameters() 
                               if p is not model.token_embeddings.weight and p is not model.lm_head.weight)
    print(f"Total parameters: {num_params:,}")
    print(f"Non-embedding parameters: {num_params_no_embed:,}")
    
    # Compile model if requested
    if args.compile:
        print(f"Compiling model with backend: {args.compile_backend}")
        if device == 'mps':
            model = torch.compile(model, backend=args.compile_backend)
        else:
            model = torch.compile(model)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from step {start_step}")
    
    # Print configuration
    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.max_lr} -> {args.min_lr}")
    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Adam betas: ({args.beta1}, {args.beta2})")
    print(f"Max gradient norm: {args.max_grad_norm}")
    print("="*80 + "\n")
    
    # Prepare W&B config
    wandb_config = {
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'rope_theta': args.rope_theta,
        'batch_size': args.batch_size,
        'max_steps': args.max_steps,
        'max_lr': args.max_lr,
        'min_lr': args.min_lr,
        'warmup_iters': args.warmup_iters,
        'weight_decay': args.weight_decay,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'eps': args.eps,
        'max_grad_norm': args.max_grad_norm,
        'total_tokens': args.batch_size * args.context_length * args.max_steps,
        'num_params': num_params,
        'num_params_no_embed': num_params_no_embed,
        'device': device,
    }
    
    # Auto-generate run name if not provided
    if args.use_wandb and args.wandb_run_name is None:
        args.wandb_run_name = f"bs{args.batch_size}_lr{args.max_lr:.0e}_steps{args.max_steps}"
    
    # Train
    history = train(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
        max_steps=args.max_steps,
        max_learning_rate=args.max_lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        max_grad_norm=args.max_grad_norm,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        start_step=start_step,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config=wandb_config
    )
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
