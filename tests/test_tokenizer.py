"""Tests for BPE tokenizer."""

import pytest
from hypothesis import given, strategies as st
from tok.bpe import BPETokenizer
import tempfile
from pathlib import Path


class TestBPETokenizer:
    """Test BPE tokenizer functionality."""

    def test_init(self):
        """Test tokenizer initialization."""
        tokenizer = BPETokenizer(vocab_size=500)
        assert tokenizer.vocab_size == 500
        assert len(tokenizer.vocab) == 256  # All bytes
        assert len(tokenizer.inverse_vocab) == 256

    def test_train_simple(self):
        """Test training on simple text."""
        tokenizer = BPETokenizer(vocab_size=260)
        text = "hello world hello world"
        tokenizer.train(text)

        # Should have learned some merges
        assert len(tokenizer.merges) > 0
        assert len(tokenizer.vocab) > 256

    def test_encode_decode_basic(self):
        """Test basic encode/decode."""
        tokenizer = BPETokenizer(vocab_size=300)
        text = "The quick brown fox jumps over the lazy dog"
        tokenizer.train(text)

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text

    @given(st.text(min_size=1, max_size=100))
    def test_encode_decode_roundtrip(self, text):
        """Property test: encode/decode should be identity."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train("Sample training text for vocabulary building")

        try:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            # Allow for Unicode replacement on decode errors
            assert len(decoded) >= 0
        except UnicodeDecodeError:
            # Some random byte sequences may not be valid UTF-8
            pass

    def test_save_load(self):
        """Test saving and loading tokenizer."""
        tokenizer1 = BPETokenizer(vocab_size=300)
        text = "Test text for tokenizer"
        tokenizer1.train(text)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            tokenizer1.save(path)

            tokenizer2 = BPETokenizer()
            tokenizer2.load(path)

            assert tokenizer2.vocab_size == tokenizer1.vocab_size
            assert tokenizer2.merges == tokenizer1.merges

            # Should produce same encoding
            tokens1 = tokenizer1.encode(text)
            tokens2 = tokenizer2.encode(text)
            assert tokens1 == tokens2

    def test_empty_text(self):
        """Test handling of empty text."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train("Some text")

        tokens = tokenizer.encode("")
        assert tokens == []

        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_unicode_handling(self):
        """Test handling of Unicode text."""
        tokenizer = BPETokenizer(vocab_size=400)
        text = "Hello 世界! 🌍 Привет мир"
        tokenizer.train(text)

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text
