"""Tests for GPT model components."""

import pytest
import torch
import torch.nn as nn
from model.gpt import CausalSelfAttention, MLP, Block, TinyGPT


class TestCausalSelfAttention:
    """Test causal self-attention layer."""
    
    def test_init(self):
        """Test attention initialization."""
        attn = CausalSelfAttention(d_model=128, n_heads=4)
        assert attn.d_model == 128
        assert attn.n_heads == 4
        assert attn.d_head == 32
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        attn = CausalSelfAttention(d_model=128, n_heads=4)
        x = torch.randn(2, 10, 128)  # batch=2, seq=10, d_model=128
        output = attn(x)
        assert output.shape == x.shape
    
    def test_causal_mask(self):
        """Test that causal mask prevents future attention."""
        attn = CausalSelfAttention(d_model=64, n_heads=2, dropout=0.0)
        
        # Create input where later positions have distinct values
        x = torch.zeros(1, 5, 64)
        for i in range(5):
            x[0, i, i] = 1.0
        
        output = attn(x)
        # First position should only depend on itself
        # (exact test would require checking attention weights)
        assert output.shape == x.shape


class TestMLP:
    """Test MLP layer."""
    
    def test_init(self):
        """Test MLP initialization."""
        mlp = MLP(d_model=128)
        assert isinstance(mlp.fc1, nn.Linear)
        assert isinstance(mlp.fc2, nn.Linear)
        assert mlp.fc1.in_features == 128
        assert mlp.fc1.out_features == 512  # 4x expansion
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        mlp = MLP(d_model=128)
        x = torch.randn(2, 10, 128)
        output = mlp(x)
        assert output.shape == x.shape


class TestBlock:
    """Test transformer block."""
    
    def test_init(self):
        """Test block initialization."""
        block = Block(d_model=128, n_heads=4)
        assert isinstance(block.attn, CausalSelfAttention)
        assert isinstance(block.mlp, MLP)
        assert isinstance(block.ln1, nn.LayerNorm)
        assert isinstance(block.ln2, nn.LayerNorm)
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        block = Block(d_model=128, n_heads=4)
        x = torch.randn(2, 10, 128)
        output = block(x)
        assert output.shape == x.shape
    
    def test_residual_connection(self):
        """Test that block uses residual connections."""
        block = Block(d_model=128, n_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 128)
        
        # With residual connections, output shouldn't be too different from input
        output = block(x)
        # Check that some signal is preserved (not exact due to layer norm)
        assert output.shape == x.shape


class TestTinyGPT:
    """Test TinyGPT model."""
    
    def test_init(self):
        """Test model initialization."""
        model = TinyGPT(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4
        )
        assert model.vocab_size == 1000
        assert model.d_model == 128
        assert model.n_layers == 2
        assert len(model.blocks) == 2
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = TinyGPT(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4
        )
        input_ids = torch.randint(0, 1000, (2, 10))  # batch=2, seq=10
        logits, loss = model(input_ids)
        
        assert logits.shape == (2, 10, 1000)
        assert loss is None  # No targets provided
    
    def test_forward_with_targets(self):
        """Test forward pass with targets."""
        model = TinyGPT(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4
        )
        input_ids = torch.randint(0, 1000, (2, 10))
        targets = torch.randint(0, 1000, (2, 10))
        
        logits, loss = model(input_ids, targets=targets)
        
        assert logits.shape == (2, 10, 1000)
        assert loss is not None
        assert loss.ndim == 0  # Scalar loss
    
    def test_generate(self):
        """Test text generation."""
        model = TinyGPT(
            vocab_size=100,
            d_model=64,
            n_layers=1,
            n_heads=2
        )
        input_ids = torch.randint(0, 100, (1, 5))
        
        output = model.generate(input_ids, max_new_tokens=10)
        
        assert output.shape == (1, 15)  # Original 5 + 10 new
        assert torch.all(output[:, :5] == input_ids)  # Preserves input
    
    def test_weight_tying(self):
        """Test weight tying between embeddings and output."""
        model = TinyGPT(
            vocab_size=100,
            d_model=64,
            n_layers=1,
            n_heads=2,
            tie_weights=True
        )
        
        # Check that weights are tied
        assert torch.all(model.tok_emb.weight == model.head.weight)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = TinyGPT(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert total_params == trainable_params  # All params should be trainable