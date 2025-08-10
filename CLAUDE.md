# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Tiny GPT/Transformer implementation project focused on building a minimal language model from scratch. The project follows a 6-week development plan with clearly defined milestones for implementing tokenization (BPE), model architecture, training pipeline, and evaluation.

## Development Commands

### Core Development Tasks

```bash
# Tokenizer operations
make bpe-train VOCAB=1000       # Train BPE tokenizer, produces tok/bpe.json
python -m tok.bpe train|encode|decode  # CLI for tokenizer operations

# Model training and evaluation
make train CFG=configs/tiny.yaml  # Train model with config, saves checkpoint.pt
make eval                          # Compute validation loss and perplexity
make sample PROMPT="Hello"         # Generate text from trained model

# Code quality and testing
make format                        # Format code with Black/Ruff
make lint                          # Run linting checks
make test                          # Run test suite
pytest -k tokenizer               # Run specific test group

# Performance and benchmarking
make bench                         # Measure tokens/sec and memory usage
make plot                          # Generate loss/perplexity plots from runs/
```

### Python Environment
- Use Python 3.10-3.12
- Dependencies: `python -m pip install -r requirements.txt`
- Type checking: `mypy` should pass on all modules

## Architecture and Key Components

### Module Structure (Planned)
- **tok/bpe.py**: Byte-level BPE tokenizer implementation
  - Training algorithm with configurable vocab size
  - Encode/decode with special token handling
  - JSON serialization for vocab persistence
  
- **model/gpt.py**: Core Transformer architecture
  - CausalSelfAttention with precomputed masks
  - MLP blocks with GELU activation
  - LayerNorm and residual connections
  - TinyGPT assembly with embeddings and LM head
  
- **train.py**: Training pipeline
  - AdamW optimizer with weight decay filtering
  - Warmup + cosine LR scheduling
  - Mixed precision training with GradScaler
  - Gradient clipping at 1.0
  - Checkpoint saving based on validation loss
  
- **sample.py**: Text generation
  - Temperature sampling
  - Top-k and nucleus (top-p) sampling
  - Deterministic seeding support

- **utils/metrics.py**: Evaluation utilities
  - Perplexity calculation
  - Moving average tracking
  - Console logging

### Data Pipeline
- Overlapping chunk dataset for efficient batching
- Train/validation split
- Target shifting (inputs shifted by 1 for next-token prediction)

### Advanced Features (Week 6)
- Optional RoPE positional encoding (config flag)
- Weight tying between embeddings and output layer
- FlashAttention/SDPA optimization when available (PyTorch ≥2.1)

## Development Milestones

- **Week 2**: BPE tokenizer with round-trip tests passing
- **Week 3**: Model forward pass with correct shapes and attention masking
- **Week 4**: End-to-end training loop with checkpointing
- **Week 5**: Text generation and evaluation harness
- **Week 6**: Performance optimizations and extensions

## Testing Requirements

- Overall code coverage target: ≥80%
- Critical paths (tokenizer, attention): ≥90% coverage
- Property-based tests for tokenizer round-trip
- Gradient flow verification for model components
- Determinism tests for sampling with fixed seeds

## Key Implementation Notes

1. **Numerical Stability**: Use AMP grad-scaler, gradient clipping, and LR warmup to prevent instabilities
2. **Tokenization**: Implement property-based tests to ensure round-trip encode/decode correctness
3. **Attention Masking**: Verify no future-token leakage with unit tests
4. **Checkpointing**: Support resume training with identical subsequent metrics
5. **Performance**: Target >15% throughput improvement with SDPA over baseline attention

## Configuration

- Training configs in YAML format (configs/tiny.yaml)
- CLI arguments override config values
- Experiment results logged to runs/ directory
- Best model saved as checkpoint.pt