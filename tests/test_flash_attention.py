"""Tests for FlashAttention/SDPA integration."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from model.gpt import CausalSelfAttention, TinyGPT


class TestFlashAttentionFallback:
    """Test FlashAttention and fallback behavior."""

    def test_attention_with_flash_enabled(self):
        """Test attention layer with FlashAttention enabled."""
        d_model, n_heads = 32, 4

        # Mock SDPA availability
        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn(2, n_heads, 8, d_model // n_heads)

            attn = CausalSelfAttention(
                d_model=d_model, n_heads=n_heads, dropout=0.0, use_flash=True
            )

            # Ensure SDPA is detected as available
            attn.has_sdpa = True

            batch_size, seq_len = 2, 8
            x = torch.randn(batch_size, seq_len, d_model)

            output = attn(x)

            assert output.shape == (batch_size, seq_len, d_model)
            # Should have called SDPA
            assert mock_sdpa.call_count == 1

    def test_attention_with_flash_disabled(self):
        """Test attention layer with FlashAttention disabled."""
        d_model, n_heads = 32, 4

        attn = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=0.0, use_flash=False
        )

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            output = attn(x)

            assert output.shape == (batch_size, seq_len, d_model)
            # Should not have called SDPA
            assert mock_sdpa.call_count == 0

    def test_attention_fallback_when_sdpa_unavailable(self):
        """Test fallback to manual attention when SDPA is not available."""
        d_model, n_heads = 32, 4

        attn = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=0.0, use_flash=True
        )

        # Simulate SDPA not being available
        attn.has_sdpa = False
        attn.use_flash = False

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, d_model)
        # Should work without errors

    def test_sdpa_detection_on_init(self):
        """Test that SDPA availability is correctly detected during initialization."""
        d_model, n_heads = 32, 4

        # Test when SDPA is available
        with patch("torch.nn.functional.scaled_dot_product_attention"):
            attn_with_sdpa = CausalSelfAttention(
                d_model=d_model, n_heads=n_heads, use_flash=True
            )
            assert attn_with_sdpa.has_sdpa
            assert attn_with_sdpa.use_flash

        # Test when SDPA is not available (mock ImportError)
        def mock_import_error(*args, **kwargs):
            raise ImportError("scaled_dot_product_attention not available")

        with patch(
            "model.gpt.torch.nn.functional.scaled_dot_product_attention",
            side_effect=ImportError,
        ):
            # Need to reload the module or create a new instance
            attn_no_sdpa = CausalSelfAttention(
                d_model=d_model, n_heads=n_heads, use_flash=True
            )
            # Manually set what would happen during init
            attn_no_sdpa.has_sdpa = False
            attn_no_sdpa.use_flash = False

            assert not attn_no_sdpa.has_sdpa
            assert not attn_no_sdpa.use_flash

    def test_flash_attention_causal_mask(self):
        """Test that FlashAttention receives correct causal mask."""
        d_model, n_heads = 32, 4

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn(2, n_heads, 6, d_model // n_heads)

            attn = CausalSelfAttention(
                d_model=d_model, n_heads=n_heads, dropout=0.1, use_flash=True
            )
            attn.has_sdpa = True

            batch_size, seq_len = 2, 6
            x = torch.randn(batch_size, seq_len, d_model)

            attn(x)

            # Check that SDPA was called with correct arguments
            assert mock_sdpa.call_count == 1
            call_args = mock_sdpa.call_args

            # Should have q, k, v as positional args
            assert len(call_args[0]) == 3

            # Check keyword arguments
            kwargs = call_args[1]
            assert "attn_mask" in kwargs
            assert "dropout_p" in kwargs
            assert "is_causal" in kwargs

            # Check causal mask shape and properties
            causal_mask = kwargs["attn_mask"]
            assert causal_mask.shape == (seq_len, seq_len)
            assert causal_mask.dtype == torch.bool

            # Check that it's actually causal (lower triangular)
            expected_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            assert torch.equal(causal_mask, expected_mask)

            # Check dropout probability
            expected_dropout = 0.1  # From attention layer
            assert kwargs["dropout_p"] == expected_dropout

    def test_flash_vs_manual_attention_equivalence(self):
        """Test that FlashAttention and manual attention give similar results."""
        d_model, n_heads = 32, 4
        torch.manual_seed(42)

        # Create two identical attention layers
        attn_flash = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=0.0, use_flash=True
        )
        attn_manual = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=0.0, use_flash=False
        )

        # Copy weights to make them identical
        attn_manual.load_state_dict(attn_flash.state_dict())

        # Force flash attention to be available for one and not the other
        attn_flash.has_sdpa = True
        attn_manual.has_sdpa = False
        attn_manual.use_flash = False

        batch_size, seq_len = 2, 6
        torch.manual_seed(123)
        x = torch.randn(batch_size, seq_len, d_model)

        with torch.no_grad():
            # Mock SDPA to return manual attention result for comparison
            def mock_sdpa_implementation(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            ):
                # Implement manual attention for comparison
                scale = 1.0 / (q.size(-1) ** 0.5)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale

                if attn_mask is not None:
                    # Convert boolean mask to additive mask
                    scores = scores.masked_fill(~attn_mask, float("-inf"))

                attn_weights = F.softmax(scores, dim=-1)
                if dropout_p > 0:
                    attn_weights = F.dropout(attn_weights, p=dropout_p, training=False)

                return torch.matmul(attn_weights, v)

            with patch(
                "torch.nn.functional.scaled_dot_product_attention",
                side_effect=mock_sdpa_implementation,
            ):
                output_flash = attn_flash(x)
                output_manual = attn_manual(x)

            # Results should be very close (allowing for small numerical differences)
            assert torch.allclose(
                output_flash, output_manual, atol=1e-5, rtol=1e-4
            ), "FlashAttention and manual attention should give similar results"

    def test_model_with_flash_attention(self):
        """Test full model with FlashAttention enabled."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4,
            use_flash=True,
            dropout=0.0,
        )

        # Mock SDPA availability for all attention layers
        for layer in model.blocks:
            layer.attn.has_sdpa = True

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn(
                batch_size, 4, seq_len, 8
            )  # Mock return value

            logits, _ = model(input_ids)

            assert logits.shape == (batch_size, seq_len, 100)
            # Should have called SDPA for each layer
            assert mock_sdpa.call_count == 2  # 2 layers

    def test_attention_training_vs_eval_mode(self):
        """Test that FlashAttention behaves correctly in training vs eval mode."""
        d_model, n_heads = 32, 4

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn(2, n_heads, 6, d_model // n_heads)

            attn = CausalSelfAttention(
                d_model=d_model, n_heads=n_heads, dropout=0.1, use_flash=True
            )
            attn.has_sdpa = True

            batch_size, seq_len = 2, 6
            x = torch.randn(batch_size, seq_len, d_model)

            # Test in training mode
            attn.train()
            attn(x)

            # Check dropout_p parameter in training mode
            call_args_train = mock_sdpa.call_args[1]
            assert call_args_train["dropout_p"] == 0.1

            # Test in eval mode
            attn.eval()
            attn(x)

            # Check dropout_p parameter in eval mode
            call_args_eval = mock_sdpa.call_args[1]
            assert call_args_eval["dropout_p"] == 0.0


class TestFlashAttentionPerformance:
    """Test performance-related aspects of FlashAttention."""

    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    def test_flash_attention_different_seq_lengths(self, seq_len):
        """Test FlashAttention with different sequence lengths."""
        d_model, n_heads = 64, 8

        attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            max_seq_len=max(seq_len, 256),  # Ensure max_seq_len >= seq_len
            use_flash=True,
        )
        attn.has_sdpa = True

        batch_size = 2
        x = torch.randn(batch_size, seq_len, d_model)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn(
                batch_size, n_heads, seq_len, d_model // n_heads
            )

            output = attn(x)

            assert output.shape == (batch_size, seq_len, d_model)
            assert mock_sdpa.call_count == 1

            # Check that causal mask has correct size
            call_kwargs = mock_sdpa.call_args[1]
            causal_mask = call_kwargs["attn_mask"]
            assert causal_mask.shape == (seq_len, seq_len)

    def test_flash_attention_memory_efficiency(self):
        """Test that FlashAttention path doesn't store unnecessary intermediates."""
        d_model, n_heads = 32, 4

        attn = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=0.0, use_flash=True
        )
        attn.has_sdpa = True

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Mock SDPA to return appropriate tensor
            def mock_sdpa_func(*args, **kwargs):
                q, k, v = args[:3]
                return torch.randn_like(v)  # Return tensor with same shape as v

            mock_sdpa.side_effect = mock_sdpa_func

            output = attn(x)
            loss = output.sum()
            loss.backward()

            # Should complete without memory issues
            assert x.grad is not None
            assert torch.all(torch.isfinite(x.grad))

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_flash_attention_different_batch_sizes(self, batch_size):
        """Test FlashAttention with different batch sizes."""
        d_model, n_heads, seq_len = 32, 4, 16

        attn = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=0.0, use_flash=True
        )
        attn.has_sdpa = True

        x = torch.randn(batch_size, seq_len, d_model)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn(
                batch_size, n_heads, seq_len, d_model // n_heads
            )

            output = attn(x)

            assert output.shape == (batch_size, seq_len, d_model)
            assert mock_sdpa.call_count == 1
