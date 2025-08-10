"""Tests for RoPE (Rotary Position Embedding) implementation."""

import pytest
import torch

from model.gpt import CausalSelfAttention, TinyGPT, apply_rope


class TestApplyRope:
    """Test the apply_rope function."""

    def test_rope_shape_preservation(self):
        """Test that RoPE preserves tensor shapes."""
        batch_size, n_heads, seq_len, head_dim = 2, 4, 8, 16

        # Create test tensors
        x = torch.randn(batch_size, n_heads, seq_len, head_dim)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)

        result = apply_rope(x, cos, sin)

        assert result.shape == x.shape, f"Expected shape {x.shape}, got {result.shape}"

    def test_rope_even_head_dimension(self):
        """Test RoPE with even head dimension."""
        batch_size, n_heads, seq_len, head_dim = 1, 1, 4, 8

        x = torch.randn(batch_size, n_heads, seq_len, head_dim)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)

        result = apply_rope(x, cos, sin)

        # Check that the transformation is applied correctly
        # RoPE splits the head_dim in half and applies rotation
        assert result.shape == (batch_size, n_heads, seq_len, head_dim)
        assert not torch.equal(result, x), "RoPE should modify the input"

    def test_rope_rotation_property(self):
        """Test that RoPE implements proper rotation."""
        # Simple case: 1 batch, 1 head, 1 sequence position, 4-dim head
        _batch_size, _n_heads, _seq_len, _head_dim = 1, 1, 1, 4

        # Create specific test vectors
        x = torch.tensor([[[[1.0, 0.0, 2.0, 0.0]]]])  # [batch, heads, seq, head_dim]

        # Simple rotation: cos=0, sin=1 should rotate 90 degrees
        cos = torch.tensor([[0.0, 0.0]])  # [seq, head_dim//2]
        sin = torch.tensor([[1.0, 1.0]])  # [seq, head_dim//2]

        result = apply_rope(x, cos, sin)

        # For cos=0, sin=1: [x1*0 - x2*1, x1*1 + x2*0] = [-x2, x1]
        expected = torch.tensor([[[[0.0, 1.0, 0.0, 2.0]]]])

        assert torch.allclose(
            result, expected, atol=1e-6
        ), f"Expected {expected}, got {result}"

    def test_rope_sequence_length_handling(self):
        """Test that RoPE handles different sequence lengths correctly."""
        batch_size, n_heads, head_dim = 1, 2, 8
        max_seq_len = 10
        actual_seq_len = 6

        x = torch.randn(batch_size, n_heads, actual_seq_len, head_dim)
        cos = torch.randn(max_seq_len, head_dim // 2)  # Longer than actual sequence
        sin = torch.randn(max_seq_len, head_dim // 2)

        result = apply_rope(x, cos, sin)

        assert result.shape == x.shape
        # Should only use first actual_seq_len positions of cos/sin

    def test_rope_identity_case(self):
        """Test RoPE when cos=1, sin=0 (should be approximately identity)."""
        batch_size, n_heads, seq_len, head_dim = 2, 2, 3, 6

        x = torch.randn(batch_size, n_heads, seq_len, head_dim)
        cos = torch.ones(seq_len, head_dim // 2)  # cos=1
        sin = torch.zeros(seq_len, head_dim // 2)  # sin=0

        result = apply_rope(x, cos, sin)

        # When cos=1, sin=0: rotation becomes [x1*1 - x2*0, x1*0 + x2*1] = [x1, x2]
        # This should preserve the original values
        assert torch.allclose(
            result, x, atol=1e-6
        ), "RoPE with cos=1, sin=0 should preserve input"


class TestRopeInAttention:
    """Test RoPE integration in attention layer."""

    def test_attention_with_rope_enabled(self):
        """Test attention layer with RoPE enabled."""
        d_model, n_heads, max_seq_len = 32, 4, 16

        attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,  # No dropout for deterministic testing
            max_seq_len=max_seq_len,
            use_rope=True,
        )

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.equal(output, x), "Attention should modify input"

    def test_attention_rope_vs_no_rope(self):
        """Test that RoPE and non-RoPE attention give different results."""
        d_model, n_heads, max_seq_len = 32, 4, 16

        # Create identical models except for RoPE
        attn_rope = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            max_seq_len=max_seq_len,
            use_rope=True,
        )

        attn_no_rope = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            max_seq_len=max_seq_len,
            use_rope=False,
        )

        # Copy weights to make them identical except for RoPE
        attn_no_rope.load_state_dict(attn_rope.state_dict(), strict=False)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)

        with torch.no_grad():
            output_rope = attn_rope(x)
            output_no_rope = attn_no_rope(x)

        # Outputs should be different due to RoPE
        assert not torch.allclose(
            output_rope, output_no_rope, atol=1e-5
        ), "RoPE and non-RoPE attention should give different results"

    def test_rope_cos_sin_initialization(self):
        """Test that RoPE cos/sin buffers are initialized correctly."""
        d_model, n_heads, max_seq_len = 64, 8, 32
        head_dim = d_model // n_heads

        attn = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len, use_rope=True
        )

        # Check that cos/sin buffers exist and have correct shapes
        assert hasattr(attn, "cos"), "RoPE attention should have cos buffer"
        assert hasattr(attn, "sin"), "RoPE attention should have sin buffer"
        assert hasattr(attn, "inv_freq"), "RoPE attention should have inv_freq buffer"

        assert attn.cos.shape == (max_seq_len, head_dim // 2)
        assert attn.sin.shape == (max_seq_len, head_dim // 2)
        assert attn.inv_freq.shape == (head_dim // 2,)

        # Check that cos/sin values are reasonable
        assert torch.all(attn.cos >= -1) and torch.all(
            attn.cos <= 1
        ), "cos values should be in [-1, 1]"
        assert torch.all(attn.sin >= -1) and torch.all(
            attn.sin <= 1
        ), "sin values should be in [-1, 1]"

    def test_rope_frequency_calculation(self):
        """Test that RoPE frequencies are calculated correctly."""
        d_model, n_heads = 64, 8
        head_dim = d_model // n_heads  # 8

        attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, use_rope=True)

        # Check inverse frequency calculation
        expected_inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        assert torch.allclose(
            attn.inv_freq, expected_inv_freq, atol=1e-6
        ), f"Expected inv_freq {expected_inv_freq}, got {attn.inv_freq}"


class TestRopeInModel:
    """Test RoPE integration in full model."""

    def test_model_with_rope_forward(self):
        """Test full model forward pass with RoPE."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4,
            use_rope=True,
            dropout=0.0,
        )

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        logits, loss = model(input_ids)

        assert logits.shape == (batch_size, seq_len, 100)
        assert loss is None  # No targets provided

    def test_model_rope_vs_learned_positions(self):
        """Test that RoPE model behaves differently from learned positions."""
        config = {
            "vocab_size": 100,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
        }

        model_rope = TinyGPT(use_rope=True, **config)
        model_learned = TinyGPT(use_rope=False, **config)

        # Check that models have different structures
        assert (
            model_rope.pos_emb is None
        ), "RoPE model should not have positional embeddings"
        assert (
            model_learned.pos_emb is not None
        ), "Learned position model should have positional embeddings"

        batch_size, seq_len = 2, 6
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            logits_rope, _ = model_rope(input_ids)
            logits_learned, _ = model_learned(input_ids)

        # Models should give different outputs (different positional encodings)
        assert not torch.allclose(
            logits_rope, logits_learned, atol=1e-3
        ), "RoPE and learned position models should give different outputs"

    def test_rope_model_generation(self):
        """Test text generation with RoPE model."""
        model = TinyGPT(
            vocab_size=50, d_model=32, n_layers=1, n_heads=2, use_rope=True, dropout=0.0
        )

        # Test generation
        input_ids = torch.randint(0, 50, (1, 4))

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=6, temperature=1.0)

        assert output.shape == (1, 10)  # 4 + 6 tokens
        assert torch.all(output[:, :4] == input_ids), "Should preserve input tokens"
        assert torch.all(output >= 0) and torch.all(
            output < 50
        ), "Generated tokens should be valid"

    @pytest.mark.parametrize("seq_len", [4, 8, 16, 32])
    def test_rope_different_sequence_lengths(self, seq_len):
        """Test RoPE model with different sequence lengths."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=1,
            n_heads=4,
            max_seq_len=64,  # Larger than test sequences
            use_rope=True,
            dropout=0.0,
        )

        batch_size = 2
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        logits, _ = model(input_ids)

        assert logits.shape == (batch_size, seq_len, 100)

    def test_rope_model_parameter_count(self):
        """Test that RoPE model has fewer parameters than learned position model."""
        config = {
            "vocab_size": 1000,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 8,
            "max_seq_len": 256,
        }

        model_rope = TinyGPT(use_rope=True, **config)
        model_learned = TinyGPT(use_rope=False, **config)

        params_rope = sum(p.numel() for p in model_rope.parameters())
        params_learned = sum(p.numel() for p in model_learned.parameters())

        # RoPE model should have fewer parameters (no positional embeddings)
        expected_diff = (
            config["max_seq_len"] * config["d_model"]
        )  # Positional embedding size
        assert (
            params_learned - params_rope == expected_diff
        ), f"Parameter difference should be {expected_diff}, got {params_learned - params_rope}"


class TestRopeNumericalStability:
    """Test numerical stability of RoPE implementation."""

    def test_rope_large_values(self):
        """Test RoPE with large input values."""
        _batch_size, _n_heads, _seq_len, _head_dim = 1, 1, 2, 4

        # Large input values
        x = torch.tensor(
            [[[[1000.0, -1000.0, 500.0, -500.0], [2000.0, -2000.0, 1000.0, -1000.0]]]]
        )
        cos = torch.tensor([[0.5, 0.5], [0.8, 0.8]])
        sin = torch.tensor([[0.866, 0.866], [0.6, 0.6]])

        result = apply_rope(x, cos, sin)

        # Should not produce NaN or infinity
        assert torch.all(
            torch.isfinite(result)
        ), "RoPE should handle large values without overflow"

    def test_rope_small_values(self):
        """Test RoPE with very small input values."""
        _batch_size, _n_heads, _seq_len, _head_dim = 1, 1, 1, 4

        # Very small values
        x = torch.tensor([[[[1e-10, -1e-10, 1e-15, -1e-15]]]])
        cos = torch.tensor([[0.9999, 0.9999]])
        sin = torch.tensor([[0.0141, 0.0141]])

        result = apply_rope(x, cos, sin)

        # Should preserve small values without underflow
        assert torch.all(torch.isfinite(result)), "RoPE should handle small values"
        assert not torch.allclose(
            result, torch.zeros_like(result)
        ), "Should not underflow to zero"

    def test_rope_gradient_flow(self):
        """Test that gradients flow properly through RoPE."""
        batch_size, n_heads, seq_len, head_dim = 2, 2, 4, 8

        x = torch.randn(batch_size, n_heads, seq_len, head_dim, requires_grad=True)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)

        result = apply_rope(x, cos, sin)
        loss = result.sum()
        loss.backward()

        # Check that gradients exist and are finite
        assert x.grad is not None, "Gradients should flow through RoPE"
        assert torch.all(torch.isfinite(x.grad)), "Gradients should be finite"
        assert not torch.allclose(
            x.grad, torch.zeros_like(x.grad)
        ), "Gradients should be non-zero"
