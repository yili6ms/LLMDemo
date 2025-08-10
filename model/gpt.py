"""Tiny GPT model implementation."""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to query or key tensors."""
    # x: (batch, n_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim//2)
    seq_len, head_dim = x.size(-2), x.size(-1)
    
    # Split into even and odd dimensions
    x1 = x[..., :head_dim//2]  # Even dimensions
    x2 = x[..., head_dim//2:]  # Odd dimensions
    
    # Apply rotation
    # cos and sin are broadcast over batch and head dimensions
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_rope: bool = False,
        use_flash: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        self.use_flash = use_flash
        
        # Check if SDPA is available (PyTorch >= 2.0)
        try:
            from torch.nn.functional import scaled_dot_product_attention
            self.has_sdpa = True
        except ImportError:
            self.has_sdpa = False
            self.use_flash = False
        
        # Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            )
        )
        
        # RoPE embeddings
        if use_rope:
            # Create rotation frequencies
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
            self.register_buffer("inv_freq", inv_freq)
            
            # Precompute cos/sin for max sequence length
            t = torch.arange(max_seq_len, dtype=torch.float)
            freqs = torch.outer(t, inv_freq)
            self.register_buffer("cos", torch.cos(freqs))
            self.register_buffer("sin", torch.sin(freqs))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of attention layer."""
        B, T, C = x.size()  # batch, sequence, embedding
        
        # Calculate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q = apply_rope(q, self.cos, self.sin)
            k = apply_rope(k, self.cos, self.sin)
        
        # Use Flash Attention / SDPA if available and enabled
        if self.use_flash and self.has_sdpa:
            # Create causal mask for SDPA
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            
            # SDPA expects different tensor layout
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False  # We're providing explicit mask
            )
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # Apply attention to values
            y = att @ v  # (B, nh, T, hs)
        
        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP."""
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_rope: bool = False,
        use_flash: bool = True
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len, use_rope, use_flash)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer block."""
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Tiny GPT model."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        tie_weights: bool = False,
        use_rope: bool = False,
        use_flash: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if not use_rope:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        else:
            self.pos_emb = None
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, dropout, max_seq_len, use_rope, use_flash)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        if tie_weights:
            self.head.weight = self.tok_emb.weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of model, returns logits and optional loss."""
        B, T = input_ids.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"
        
        # Token and position embeddings
        tok_emb = self.tok_emb(input_ids)
        if self.use_rope:
            x = self.drop(tok_emb)  # No position embeddings with RoPE
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            pos_emb = self.pos_emb(pos)
            x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and output
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            
        return logits, loss
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
                
            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
                
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            
        return input_ids