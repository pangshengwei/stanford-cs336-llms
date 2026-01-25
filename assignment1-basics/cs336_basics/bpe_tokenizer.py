from collections import defaultdict, Counter
from typing import Iterable, TextIO
import regex as re 
from multiprocessing import Pool, cpu_count
import os
import regex as re
from collections import Counter
from typing import BinaryIO
import argparse
import json
from tqdm import tqdm

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        """
        Initialize the BPE tokenizer 

        Args:
        vocab: dict[int, bytes]: The vocabulary of the tokenizer. example: {0: b'h', 1: b'e', 2: b'l', 3: b'o', 4: b'w', 5: b'r', 6: b'd'}
        merges: list[tuple[bytes, bytes]]: The merges of the tokenizer. example: [(b'he', b'h'), (b'lo', b'l'), (b'wor', b'w'), (b'ld', b'd')]
        special_tokens: list[str]: The special tokens of the tokenizer. example: ['<|endoftext|>']
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or [] 
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.byte_to_id = {v:k for k, v in vocab.items()}
        self.merge_priority = {pair: i for i, pair in enumerate(merges)} # lower = earlier

        if self.special_tokens:
            # create a regex pattern to split on the special tokens
            # Sort by length (descending) to match longer tokens first (greedy matching)
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(token) for token in sorted_tokens)
            self.special_token_pattern = re.compile(f"({special_pattern})")
        else:
            self.special_token_pattern = None

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> 'Tokenizer':
        """
        constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. 
        params:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        returns:
            Tokenizer: A new Tokenizer object
        """
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges = [tuple(line.strip().split()) for line in f]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode text into a list of token IDs

        Args:
        text: str: The text to encode.

        Returns:
        list[int]: The list of token IDs.
        """
        if not text:
            return []

        token_ids = []

        if self.special_token_pattern:
            # split text on special tokens into separate segments
            segments = self.special_token_pattern.split(text)
        else:
            segments = [text]
        
        for segment in segments:
            if not segment: continue 

            if segment in self.special_tokens:
                special_token_bytes = segment.encode('utf-8')
                if special_token_bytes in self.byte_to_id:
                    token_ids.append(self.byte_to_id[special_token_bytes])
                continue 

            # split segment by PAT regex pattern
            words = re.findall(self.PAT, segment)

            for word in words: 
                word_bytes = word.encode('utf-8') 
                bpe_tokens = self._apply_bpe(word_bytes)
                for token in bpe_tokens: 
                    if token in self.byte_to_id:
                        token_ids.append(self.byte_to_id[token])

        return token_ids

    def encode_iterable(self, iterable: TextIO) -> list[int]:
        """
        Encode text from an iterable (like a file) without loading everything into memory.

        Args:
            iterable: Text iterable (e.g., file object)

        Yields:
            Token IDs one at a time
        """
        # Process the file line by line or in chunks to be memory-efficient
        buffer = ""
        
        for line in iterable:
            buffer += line
            
            # Process buffer when it gets large enough (e.g., 10KB)
            # But keep some overlap to handle tokens that span chunk boundaries
            if len(buffer) > 10000:
                # Find a good breaking point (e.g., after a space or newline)
                # to avoid splitting in the middle of a word
                break_point = buffer.rfind('\n', 0, len(buffer) - 1000)
                if break_point == -1:
                    break_point = buffer.rfind(' ', 0, len(buffer) - 1000)
                if break_point == -1:
                    break_point = len(buffer) - 1000
                
                # Process the chunk up to the break point
                chunk_to_process = buffer[:break_point + 1]
                buffer = buffer[break_point + 1:]
                
                # Encode and yield tokens
                for token_id in self.encode(chunk_to_process):
                    yield token_id
        
        # Process any remaining buffer
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id


    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs into a string

        Args:
        token_ids: list[int]: The list of token IDs.

        Returns:
        str: The decoded string.
        """
        if not token_ids: return ""
        byte_list = [] 
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_list.append(self.vocab[token_id])

        all_bytes = b"".join(byte_list)
        return all_bytes.decode('utf-8', errors='replace')

    def _apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        """
        Apply BPE merges to a sequence of bytes 

        Args:
        token_bytes: bytes: The bytes to apply BPE merges to. example: b'hello'

        Returns:
        list[bytes]: The list of BPE merged tokens. example: [b'he', b'lo']
        """
        if len(token_bytes) == 0: return [] 
        if len(token_bytes) == 1: return [token_bytes]

        # start with each byte as a separate token 
        tokens = [bytes([b]) for b in token_bytes]

        while True:
            # find all pairs of adjacent tokens that are in the merges list
            pairs = [] 
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merge_priority: 
                    pairs.append((self.merge_priority[pair], i, pair))

            if not pairs: break
            pairs.sort() 
            _, merge_idx, (first, second) = pairs[0] 

            # Apply the merge 
            new_tokens = [] 
            i = 0 
            while i < len(tokens): 
                if i == merge_idx: 
                    new_tokens.append(first + second)
                    i += 2 
                else:
                    new_tokens.append(tokens[i])
                    i += 1
                
            tokens = new_tokens

        return tokens


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

        
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input corpus

    Args:
    input_path: str: The path to the input corpus.
    vocab_size: int: The size of the vocabulary.
    special_tokens: list[str]: The special tokens to add to the vocabulary.
    """
    # Step 1: Initialize vocab with all individual bytes 
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # Step 2: Add special tokens to vocab
    for st in special_tokens:
        vocab[next_token_id] = st.encode('utf-8')
        next_token_id += 1

    # Step 3: Read the corpus and pre-tokenize it 
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Create regex pattern to split on special tokens
    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        special_token_pattern = re.compile(f"({special_pattern})")
    else:
        special_token_pattern = None
    
    word_freqs = Counter() 
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        corpus_text = f.read()
        # Split on special tokens to prevent merging across boundaries
        if special_token_pattern:
            segments = special_token_pattern.split(corpus_text)
        else:
            segments = [corpus_text]

        # Process each segment separately
        for segment in segments:
            if not segment or segment in special_tokens:
                continue
            # split each segment into tokens to build the token_byte frequency counter
            words = re.findall(PAT, segment)
            for word in words:
                word_freqs[word.encode('utf-8')] += 1

    # convert words to list of bytes for BPE processing
    word_tokens = {word: [bytes([b]) for b in word] for word in word_freqs.keys()}

    # Step 4: Iteratively merge most frequent pairs with incremental updates
    merges = []
    
    # Initial count of all pairs
    pair_freqs = {}
    for word, tokens in word_tokens.items():
        freq = word_freqs[word]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    
    while len(vocab) < vocab_size: 
        # If no pairs left then exit 
        if not pair_freqs: break

        # find the most frequent pair
        # Break ties by preferring lexicographically greater pair
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        first, second = best_pair 

        # add merged token back to vocab 
        merged_token = first + second 
        vocab[next_token_id] = merged_token
        next_token_id += 1
        merges.append((first, second))

        # Remove the merged pair from pair_freqs
        del pair_freqs[best_pair]

        # apply the merge and incrementally update pair counts
        for word, tokens in word_tokens.items():
            if len(tokens) < 2:
                continue
            
            # Check if this word contains the merged pair
            has_pair = False
            for i in range(len(tokens) - 1):
                if tokens[i] == first and tokens[i+1] == second:
                    has_pair = True
                    break
            
            if not has_pair:
                continue
            
            freq = word_freqs[word]
            
            # Build new tokens and track changes to pair counts
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i+1] == second:
                    # Before merge: decrement counts for affected pairs
                    # Left neighbor pair: (tokens[i-1], first) if exists
                    if i > 0:
                        old_pair = (tokens[i-1], first)
                        pair_freqs[old_pair] = pair_freqs.get(old_pair, 0) - freq
                        if pair_freqs[old_pair] <= 0:
                            del pair_freqs[old_pair]
                    
                    # Right neighbor pair: (second, tokens[i+2]) if exists
                    if i + 2 < len(tokens):
                        old_pair = (second, tokens[i+2])
                        pair_freqs[old_pair] = pair_freqs.get(old_pair, 0) - freq
                        if pair_freqs[old_pair] <= 0:
                            del pair_freqs[old_pair]
                    
                    # Add merged token
                    new_tokens.append(merged_token)
                    
                    # After merge: increment counts for new pairs
                    # Left neighbor pair: (tokens[i-1], merged_token) if exists
                    if i > 0:
                        new_pair = (new_tokens[-2], merged_token)
                        pair_freqs[new_pair] = pair_freqs.get(new_pair, 0) + freq
                    
                    # Right neighbor pair: (merged_token, tokens[i+2]) if exists
                    if i + 2 < len(tokens):
                        new_pair = (merged_token, tokens[i+2])
                        pair_freqs[new_pair] = pair_freqs.get(new_pair, 0) + freq
                    
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            word_tokens[word] = new_tokens

    return vocab, merges

def process_chunk(args: tuple[str, int, int, str, list[str]]) -> Counter:
    """
    Process a single chunk of the file and return word frequencies.
    
    Args:
        args: Tuple of (filepath, start_byte, end_byte, PAT, special_tokens)
    
    Returns:
        Counter of word frequencies for this chunk
    """
    filepath, start, end, PAT, special_tokens = args
    
    # Create regex pattern to split on special tokens
    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        special_token_pattern = re.compile(f"({special_pattern})")
    else:
        special_token_pattern = None
    
    word_freqs = Counter()
    
    # Read the chunk
    with open(filepath, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
        
        # Split on special tokens to prevent merging across boundaries
        if special_token_pattern:
            segments = special_token_pattern.split(chunk_text)
        else:
            segments = [chunk_text]
        
        # Process each segment separately
        for segment in segments:
            if not segment or segment in special_tokens:
                continue
            
            # Apply pre-tokenization pattern
            words = re.findall(PAT, segment)
            for word in words:
                word_freqs[word.encode('utf-8')] += 1
    
    return word_freqs

def train_bpe_parallel(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input corpus using parallel processing.
    
    Args:
        input_path: Path to the input corpus
        vocab_size: Size of the vocabulary
        special_tokens: Special tokens to add to the vocabulary
        num_processes: Number of processes to use (default: cpu_count())
    
    Returns:
        Tuple of (vocab, merges)
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Training BPE with {num_processes} processes...")
    
    # Step 1: Initialize vocab with all individual bytes
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    
    # Step 2: Add special tokens to vocab
    for st in special_tokens:
        vocab[next_token_id] = st.encode('utf-8')
        next_token_id += 1
    
    # Step 3: Find chunk boundaries at special token locations
    print("Finding chunk boundaries...")
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(
            f,
            num_processes,
            b"<|endoftext|>"  # Assuming this is your special token
        )
    
    print(f"Split file into {len(boundaries) - 1} chunks")
    
    # Step 4: Prepare arguments for parallel processing
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    chunk_args = [
        (input_path, start, end, PAT, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    # Step 5: Process chunks in parallel to get word frequencies
    print("Pre-tokenizing corpus in parallel...")
    with Pool(processes=num_processes) as pool:
        chunk_results = list(tqdm(
            pool.imap(process_chunk, chunk_args),
            total=len(chunk_args),
            desc="Processing chunks",
            unit="chunk"
        ))
    
    # Step 6: Combine all word frequency counters
    print("Combining results...")
    word_freqs = Counter()
    for chunk_counter in chunk_results:
        word_freqs.update(chunk_counter)
    
    print(f"Found {len(word_freqs)} unique pre-tokens")
    
    # Step 7: Convert words to list of bytes for BPE processing
    word_tokens = {word: [bytes([b]) for b in word] for word in word_freqs.keys()}
    
    # Step 8: Iteratively merge most frequent pairs (sequential part)
    print("Performing BPE merges...")
    merges = []
    
    # Initial count of all pairs
    pair_freqs = {}
    for word, tokens in word_tokens.items():
        freq = word_freqs[word]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    
    # Progress tracking
    num_merges_needed = vocab_size - len(vocab)
    
    # Create progress bar for merges
    pbar = tqdm(total=num_merges_needed, desc="BPE merges", unit="merge")
    
    while len(vocab) < vocab_size:
        # If no pairs left then exit
        if not pair_freqs:
            print(f"\nNo more pairs to merge. Final vocab size: {len(vocab)}")
            break
        
        # Find the most frequent pair
        # Break ties by preferring lexicographically greater pair
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        first, second = best_pair
        
        # Add merged token to vocab
        merged_token = first + second
        vocab[next_token_id] = merged_token
        next_token_id += 1
        merges.append((first, second))
        
        # Update progress bar
        pbar.update(1)
        
        # Remove the merged pair from pair_freqs
        del pair_freqs[best_pair]
        
        # Apply the merge and incrementally update pair counts
        for word, tokens in word_tokens.items():
            if len(tokens) < 2:
                continue
            
            # Check if this word contains the merged pair
            has_pair = False
            for i in range(len(tokens) - 1):
                if tokens[i] == first and tokens[i+1] == second:
                    has_pair = True
                    break
            
            if not has_pair:
                continue
            
            freq = word_freqs[word]
            
            # Build new tokens and track changes to pair counts
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i+1] == second:
                    # Before merge: decrement counts for affected pairs
                    if i > 0:
                        old_pair = (tokens[i-1], first)
                        pair_freqs[old_pair] = pair_freqs.get(old_pair, 0) - freq
                        if pair_freqs[old_pair] <= 0:
                            del pair_freqs[old_pair]
                    
                    if i + 2 < len(tokens):
                        old_pair = (second, tokens[i+2])
                        pair_freqs[old_pair] = pair_freqs.get(old_pair, 0) - freq
                        if pair_freqs[old_pair] <= 0:
                            del pair_freqs[old_pair]
                    
                    # Add merged token
                    new_tokens.append(merged_token)
                    
                    # After merge: increment counts for new pairs
                    if i > 0:
                        new_pair = (new_tokens[-2], merged_token)
                        pair_freqs[new_pair] = pair_freqs.get(new_pair, 0) + freq
                    
                    if i + 2 < len(tokens):
                        new_pair = (merged_token, tokens[i+2])
                        pair_freqs[new_pair] = pair_freqs.get(new_pair, 0) + freq
                    
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            word_tokens[word] = new_tokens
    
    # Close progress bar
    pbar.close()
    
    print(f"Training complete! Final vocab size: {len(vocab)}")
    return vocab, merges

# python cs336_basics/bpe_tokenizer.py --input_path="data/TinyStoriesV2-GPT4-valid.txt" --vocab_size=10000 --special_tokens="<|endoftext|>" --num_processes=4 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer on a corpus')
    parser.add_argument('--input_path', default="../data/TinyStoriesV2-GPT4-valid.txt", type=str, help='The path to the input corpus')
    parser.add_argument('--vocab_size', default=10000, type=int, help='The size of the vocabulary')
    parser.add_argument('--special_tokens', default=["<|endoftext|>"], type=list[str], help='The special tokens to add to the vocabulary')
    parser.add_argument('--num_processes', default=4, type=int, help='The number of processes to use')
    args = parser.parse_args()

    input_path = args.input_path
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens
    num_processes = args.num_processes

    args = parser.parse_args()
    special_tokens = ["<|endoftext|>"]
    
    # Train with parallel processing
    vocab, merges = train_bpe_parallel(
        input_path,
        vocab_size,
        special_tokens,
        num_processes=num_processes
    )
    
    # Save vocabulary
    with open("vocab.json", "w", encoding="utf-8") as f:
        vocab_serializable = {k: v.hex() for k, v in vocab.items()}
        json.dump(vocab_serializable, f, indent=2)
    
    # Save merges
    with open("merges.txt", "w", encoding="utf-8") as f:
        for first, second in merges:
            f.write(f"{first.hex()} {second.hex()}\n")
    
    print(f"Saved vocabulary ({len(vocab)} tokens) and merges ({len(merges)} merges)")
