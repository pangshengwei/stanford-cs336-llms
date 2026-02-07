#!/usr/bin/env python3
"""
Tokenize TinyStories data and save to disk.

This script tokenizes the raw text data once and saves it as numpy arrays.
Subsequent training runs can load the pre-tokenized data directly.

Usage:
    python tokenize_data.py \
        --train_data data/TinyStoriesV2-GPT4-train.txt \
        --val_data data/TinyStoriesV2-GPT4-valid.txt \
        --tokenizer_vocab vocab_tinystories_train.json \
        --tokenizer_merges merges_tinystories_train.txt \
        --output_dir data/tokenized
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from cs336_basics.bpe_tokenizer import Tokenizer


def tokenize_file_with_progress(
    filepath: str,
    tokenizer: Tokenizer
) -> list[int]:
    """
    Tokenize a text file with progress bar.
    
    Args:
        filepath: Path to text file
        tokenizer: BPE tokenizer
    
    Returns:
        List of token IDs
    """
    print(f"Tokenizing {filepath}...")
    
    # Get file size for progress bar
    file_size = os.path.getsize(filepath)
    
    # Read entire file with progress bar
    print("  Reading file...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"  File size: {len(text):,} characters ({file_size / 1e6:.2f} MB)")
    
    # Tokenize with progress bar
    # We'll tokenize in chunks to show progress
    print("  Tokenizing...")
    chunk_size = 100_000  # Characters per chunk
    all_tokens = []
    
    with tqdm(total=len(text), unit='chars', unit_scale=True) as pbar:
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            tokens = tokenizer.encode(chunk)
            all_tokens.extend(tokens)
            pbar.update(len(chunk))
    
    print(f"  Total tokens: {len(all_tokens):,}")
    return all_tokens


def main():
    parser = argparse.ArgumentParser(description='Tokenize TinyStories data')
    
    parser.add_argument('--train_data', type=str, default='data/TinyStoriesV2-GPT4-train.txt',
                        help='Path to training text file')
    parser.add_argument('--val_data', type=str, default='data/TinyStoriesV2-GPT4-valid.txt',
                        help='Path to validation text file')
    parser.add_argument('--tokenizer_vocab', type=str, default='vocab_tinystories_train.json',
                        help='Path to vocabulary JSON')
    parser.add_argument('--tokenizer_merges', type=str, default='merges_tinystories_train.txt',
                        help='Path to BPE merges file')
    parser.add_argument('--output_dir', type=str, default='data/tokenized',
                        help='Directory to save tokenized data')
    parser.add_argument('--special_tokens', type=str, nargs='*', default=['<|endoftext|>'],
                        help='Special tokens')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Tokenizing TinyStories Dataset")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_files(
        args.tokenizer_vocab,
        args.tokenizer_merges,
        special_tokens=args.special_tokens
    )
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Number of merges: {len(tokenizer.merges)}")
    print(f"Special tokens: {args.special_tokens}")
    
    # Tokenize validation data FIRST (smaller, catches issues faster)
    print("\n" + "-"*80)
    print("Processing validation set first (smaller, catches issues faster)...")
    print("-"*80)
    val_tokens = tokenize_file_with_progress(args.val_data, tokenizer)
    
    # Check if tokenization worked
    if len(val_tokens) == 0:
        print("\n❌ ERROR: Validation tokenization produced 0 tokens!")
        print("   Please check your tokenizer and data files.")
        return
    
    # Save validation tokens
    val_output_path = os.path.join(args.output_dir, 'val_tokens.npy')
    print(f"Saving to {val_output_path}...")
    val_array = np.array(val_tokens, dtype=np.uint16)
    np.save(val_output_path, val_array)
    print(f"  Saved {len(val_tokens):,} tokens ({val_array.nbytes / 1e6:.2f} MB)")
    print("  ✓ Validation tokenization successful!")
    
    # Tokenize training data (larger file)
    print("\n" + "-"*80)
    print("Processing training set...")
    print("-"*80)
    train_tokens = tokenize_file_with_progress(args.train_data, tokenizer)
    
    # Save training tokens
    train_output_path = os.path.join(args.output_dir, 'train_tokens.npy')
    print(f"Saving to {train_output_path}...")
    train_array = np.array(train_tokens, dtype=np.uint16)
    np.save(train_output_path, train_array)
    print(f"  Saved {len(train_tokens):,} tokens ({train_array.nbytes / 1e6:.2f} MB)")
    print("  ✓ Training tokenization successful!")
    
    # Check training tokenization
    if len(train_tokens) == 0:
        print("\n❌ ERROR: Training tokenization produced 0 tokens!")
        print("   Please check your tokenizer and data files.")
        return
    
    # Summary
    print("\n" + "="*80)
    print("✓ Tokenization Complete!")
    print("="*80)
    print(f"Training tokens:   {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")
    print(f"Total tokens:      {len(train_tokens) + len(val_tokens):,}")
    print(f"\nOutput files:")
    print(f"  - {train_output_path}")
    print(f"  - {val_output_path}")
    print("\nYou can now use these files for training:")
    print(f"  python train_tinystories.py --device mps")
    print("="*80)
    
    # Verify data
    print("\nVerifying tokenized data...")
    train_loaded = np.load(train_output_path, mmap_mode='r')
    val_loaded = np.load(val_output_path, mmap_mode='r')
    
    print(f"  Train data shape: {train_loaded.shape}, dtype: {train_loaded.dtype}")
    print(f"  Val data shape: {val_loaded.shape}, dtype: {val_loaded.dtype}")
    
    if len(train_loaded) > 0 and len(val_loaded) > 0:
        print(f"  Train min/max token ID: {train_loaded.min()}/{train_loaded.max()}")
        print(f"  Val min/max token ID: {val_loaded.min()}/{val_loaded.max()}")
        
        # Check for issues
        vocab_size = len(tokenizer.vocab)
        if train_loaded.max() >= vocab_size or val_loaded.max() >= vocab_size:
            print(f"\n⚠️  WARNING: Found token IDs >= vocab_size ({vocab_size})")
        else:
            print(f"\n✓ All token IDs are valid (< {vocab_size})")
    else:
        print(f"\n⚠️  ERROR: Tokenization produced 0 tokens!")
        print(f"     This likely means the tokenizer encode method has issues.")
        print(f"     Check that vocab and merges files are in the correct format.")


if __name__ == '__main__':
    main()
