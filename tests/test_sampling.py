"""Tests for text sampling and generation functionality."""

from unittest.mock import patch

import pytest
import torch

from model.gpt import TinyGPT
from sample import sample
from tok.bpe import BPETokenizer


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, text):
        # Simple mock: convert text to list of character codes modulo vocab_size
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens):
        # Simple mock: convert back to characters
        return "".join(
            [chr(token % 128 + 32) for token in tokens if 32 <= token % 128 + 32 < 127]
        )


class TestSampleFunction:
    """Test the sample function."""

    def test_sample_basic(self):
        """Test basic sampling functionality."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = "Hello"
        device = torch.device("cpu")

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            temperature=1.0,
            device=device,
        )

        assert isinstance(result, str)
        assert len(result) >= len(prompt)  # Should include original prompt

    def test_sample_with_seed(self):
        """Test that sampling is deterministic with seed."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = "Test"
        device = torch.device("cpu")
        seed = 42

        # Generate twice with same seed
        result1 = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            seed=seed,
            device=device,
        )

        result2 = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            seed=seed,
            device=device,
        )

        assert result1 == result2, "Results should be identical with same seed"

    def test_sample_different_seeds(self):
        """Test that different seeds give different results."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = "Test"
        device = torch.device("cpu")

        result1 = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            seed=42,
            device=device,
        )

        result2 = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            seed=123,
            device=device,
        )

        # Results should likely be different (very small chance they're the same)
        assert result1 != result2, "Different seeds should produce different results"

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
    def test_sample_different_temperatures(self, temperature):
        """Test sampling with different temperatures."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = "Test"
        device = torch.device("cpu")

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            temperature=temperature,
            seed=42,
            device=device,
        )

        assert isinstance(result, str)
        assert len(result) >= len(prompt)

    def test_sample_with_top_k(self):
        """Test sampling with top-k."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = "Test"
        device = torch.device("cpu")

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            top_k=10,
            seed=42,
            device=device,
        )

        assert isinstance(result, str)
        assert len(result) >= len(prompt)

    def test_sample_with_top_p(self):
        """Test sampling with nucleus (top-p)."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = "Test"
        device = torch.device("cpu")

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            top_p=0.9,
            seed=42,
            device=device,
        )

        assert isinstance(result, str)
        assert len(result) >= len(prompt)

    def test_sample_with_top_k_and_top_p(self):
        """Test sampling with both top-k and top-p."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = "Test"
        device = torch.device("cpu")

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            top_k=20,
            top_p=0.8,
            seed=42,
            device=device,
        )

        assert isinstance(result, str)
        assert len(result) >= len(prompt)

    def test_sample_max_new_tokens(self):
        """Test that max_new_tokens parameter works correctly."""
        model = TinyGPT(vocab_size=50, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=50)

        prompt = "Hi"
        device = torch.device("cpu")

        # Mock the tokenizer to return predictable lengths
        original_encode = tokenizer.encode

        def mock_encode(text):
            return [1, 2] if text == "Hi" else original_encode(text)

        def mock_decode(tokens):
            return "Hi" + "X" * (len(tokens) - 2)  # Predictable output

        tokenizer.encode = mock_encode
        tokenizer.decode = mock_decode

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            seed=42,
            device=device,
        )

        # Should have original prompt + 5 new tokens
        expected_length = len("Hi") + 5  # "Hi" + 5 X's
        assert (
            len(result) == expected_length
        ), f"Expected length {expected_length}, got {len(result)}"

    def test_sample_empty_prompt(self):
        """Test sampling with empty prompt."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = ""
        device = torch.device("cpu")

        # Mock tokenizer to handle empty prompt
        def mock_encode(text):
            return [] if text == "" else [1, 2, 3]

        def mock_decode(tokens):
            return "Generated" + "X" * max(0, len(tokens) - len("Generated"))

        tokenizer.encode = mock_encode
        tokenizer.decode = mock_decode

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            seed=42,
            device=device,
        )

        assert isinstance(result, str)
        assert len(result) > 0  # Should generate something

    def test_sample_model_in_eval_mode(self):
        """Test that model is put in eval mode during sampling."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.5)
        tokenizer = MockTokenizer(vocab_size=100)

        # Start in training mode
        model.train()
        assert model.training

        prompt = "Test"
        device = torch.device("cpu")

        with patch.object(model, "eval", wraps=model.eval) as mock_eval:
            sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=3,
                device=device,
            )

            # Should have called eval()
            mock_eval.assert_called_once()

    def test_sample_device_handling(self):
        """Test that tensors are moved to correct device."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = "Test"
        device = torch.device("cpu")

        with patch("torch.tensor") as mock_tensor:
            # Mock tensor creation to verify device parameter
            mock_tensor.return_value = torch.tensor([1, 2, 3])

            sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=3,
                device=device,
            )

            # Should have called torch.tensor with device parameter
            mock_tensor.assert_called()
            call_kwargs = mock_tensor.call_args[1]
            assert "device" in call_kwargs
            assert call_kwargs["device"] == device


class TestSamplingIntegration:
    """Integration tests for sampling with real tokenizer."""

    def test_sample_with_bpe_tokenizer(self):
        """Test sampling with actual BPE tokenizer."""
        # Create a simple BPE tokenizer
        tokenizer = BPETokenizer(vocab_size=300)
        test_text = "The quick brown fox jumps over the lazy dog. " * 5
        tokenizer.train(test_text)

        model = TinyGPT(vocab_size=300, d_model=64, n_layers=2, n_heads=4, dropout=0.0)

        prompt = "The quick"
        device = torch.device("cpu")

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            temperature=1.0,
            seed=42,
            device=device,
        )

        assert isinstance(result, str)
        assert prompt in result  # Original prompt should be in result
        assert len(result) > len(prompt)  # Should have generated additional text

    def test_sample_consistency_with_generation(self):
        """Test that sample function gives consistent results with model.generate."""
        model = TinyGPT(vocab_size=50, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=50)

        prompt = "Test"
        encoded_prompt = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded_prompt])

        # Generate using model.generate directly
        torch.manual_seed(42)
        with torch.no_grad():
            model_output = model.generate(input_ids, max_new_tokens=5, temperature=1.0)

        direct_result = tokenizer.decode(model_output[0].tolist())

        # Generate using sample function
        sample_result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            temperature=1.0,
            seed=42,
            device=torch.device("cpu"),
        )

        # Results should be identical
        assert sample_result == direct_result

    @pytest.mark.parametrize("max_tokens", [1, 5, 10, 20])
    def test_sample_different_lengths(self, max_tokens):
        """Test sampling with different output lengths."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        prompt = "Start"
        device = torch.device("cpu")

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            seed=42,
            device=device,
        )

        assert isinstance(result, str)
        # Length should be reasonable (at least the prompt)
        assert len(result) >= len(prompt)

    def test_sample_with_model_variants(self):
        """Test sampling with different model configurations."""
        configs = [
            {"vocab_size": 100, "d_model": 32, "n_layers": 1, "n_heads": 2},
            {"vocab_size": 200, "d_model": 64, "n_layers": 2, "n_heads": 4},
            {
                "vocab_size": 100,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 2,
                "use_rope": True,
            },
        ]

        for config in configs:
            model = TinyGPT(dropout=0.0, **config)
            tokenizer = MockTokenizer(vocab_size=config["vocab_size"])

            result = sample(
                model=model,
                tokenizer=tokenizer,
                prompt="Test",
                max_new_tokens=5,
                seed=42,
                device=torch.device("cpu"),
            )

            assert isinstance(result, str)
            assert len(result) > 0


class TestSamplingEdgeCases:
    """Test edge cases and error handling in sampling."""

    def test_sample_with_very_high_temperature(self):
        """Test sampling with very high temperature."""
        model = TinyGPT(vocab_size=50, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=50)

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_new_tokens=5,
            temperature=10.0,
            seed=42,
            device=torch.device("cpu"),
        )

        assert isinstance(result, str)
        # Should not crash with high temperature

    def test_sample_with_very_low_temperature(self):
        """Test sampling with very low temperature."""
        model = TinyGPT(vocab_size=50, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=50)

        result = sample(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_new_tokens=5,
            temperature=0.01,
            seed=42,
            device=torch.device("cpu"),
        )

        assert isinstance(result, str)
        # Should not crash with very low temperature

    def test_sample_with_extreme_top_k(self):
        """Test sampling with extreme top_k values."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        # Very large top_k (larger than vocab_size)
        result1 = sample(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_new_tokens=3,
            top_k=1000,
            seed=42,
            device=torch.device("cpu"),
        )

        # Very small top_k
        result2 = sample(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_new_tokens=3,
            top_k=1,
            seed=42,
            device=torch.device("cpu"),
        )

        assert isinstance(result1, str)
        assert isinstance(result2, str)

    def test_sample_with_extreme_top_p(self):
        """Test sampling with extreme top_p values."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
        tokenizer = MockTokenizer(vocab_size=100)

        # Very high top_p
        result1 = sample(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_new_tokens=3,
            top_p=0.99,
            seed=42,
            device=torch.device("cpu"),
        )

        # Very low top_p
        result2 = sample(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_new_tokens=3,
            top_p=0.01,
            seed=42,
            device=torch.device("cpu"),
        )

        assert isinstance(result1, str)
        assert isinstance(result2, str)
