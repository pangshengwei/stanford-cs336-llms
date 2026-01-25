from collections import defaultdict, Counter
from typing import Iterable, TextIO
import regex as re 

class BPETokenizer:
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
            special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self.special_token_pattern = re.compile(f"({special_pattern})")
        else:
            self.special_token_pattern = None

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

