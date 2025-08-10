"""Byte-level BPE tokenizer implementation."""

from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from collections import defaultdict
import re


class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[int, int]] = []
        self.vocab: Dict[int, int] = {}
        self.inverse_vocab: Dict[int, int] = {}
        # Initialize with all bytes
        for i in range(256):
            self.vocab[i] = i
            self.inverse_vocab[i] = i

    def _get_pairs(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """Get all adjacent pairs and their counts."""
        pairs = defaultdict(int)
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def train(self, text: str) -> None:
        """Train BPE tokenizer on text corpus."""
        # Convert text to bytes
        tokens = list(text.encode("utf-8"))

        # Build initial vocabulary (all bytes already added in __init__)
        next_token = 256

        # Perform merges
        while next_token < self.vocab_size:
            # Get pair frequencies
            pairs = self._get_pairs(tokens)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Add merge to list
            self.merges.append(best_pair)

            # Add new token to vocabulary
            self.vocab[next_token] = next_token
            self.inverse_vocab[next_token] = next_token

            # Apply merge to tokens
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(next_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            next_token += 1

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        # Convert to bytes
        tokens = list(text.encode("utf-8"))

        # Apply merges in order
        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == merge[0]
                    and tokens[i + 1] == merge[1]
                ):
                    # Map the merge to its token ID
                    merge_idx = self.merges.index(merge)
                    new_tokens.append(256 + merge_idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        # Build merge mapping
        merge_map = {}
        for idx, (a, b) in enumerate(self.merges):
            merge_map[256 + idx] = (a, b)

        # Recursively decode tokens
        def decode_token(token: int) -> List[int]:
            if token < 256:
                return [token]
            elif token in merge_map:
                a, b = merge_map[token]
                return decode_token(a) + decode_token(b)
            else:
                return [token]  # Unknown token

        # Decode all tokens to bytes
        bytes_list = []
        for token in tokens:
            bytes_list.extend(decode_token(token))

        # Convert bytes to string
        return bytes(bytes_list).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        """Save tokenizer to JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": [[int(a), int(b)] for a, b in self.merges],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load tokenizer from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = [(a, b) for a, b in data["merges"]]
        # Rebuild vocab
        self.vocab = {}
        self.inverse_vocab = {}
        for i in range(256):
            self.vocab[i] = i
            self.inverse_vocab[i] = i
        for i in range(len(self.merges)):
            token_id = 256 + i
            self.vocab[token_id] = token_id
            self.inverse_vocab[token_id] = token_id


def train_bpe(
    input_file: Path, output_file: Path, vocab_size: int = 1000
) -> BPETokenizer:
    """Train BPE tokenizer on file and save to output."""
    tokenizer = BPETokenizer(vocab_size=vocab_size)

    # Read text from file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Train tokenizer
    tokenizer.train(text)

    # Save to output
    tokenizer.save(output_file)

    return tokenizer
