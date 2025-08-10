# LLM Tiny - Minimal Transformer Implementation

A minimal GPT/Transformer implementation for educational purposes, built from scratch with a focus on clarity and understanding.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train tokenizer
make bpe-train VOCAB=1000

# Train model
make train CFG=configs/tiny.yaml

# Generate text
make sample PROMPT="Hello world"

# Evaluate model
make eval
```

## Project Structure

```
llm-tiny/
├── tok/            # Tokenizer implementation (BPE)
├── model/          # GPT model architecture
├── data/           # Training data
├── utils/          # Utilities and metrics
├── configs/        # Training configurations
├── docs/           # Theory and documentation
├── tests/          # Test suite
├── train.py        # Training script
└── sample.py       # Text generation script
```

## Features

- **Tokenizer**: Byte-level BPE with configurable vocabulary size and CLI interface
- **Model Architecture**: Minimal GPT with multi-head causal attention and GELU MLP
- **Positional Encoding**: Support for both learned embeddings and RoPE
- **Attention**: FlashAttention/SDPA support for improved performance (PyTorch ≥2.1)
- **Training**: Mixed precision, gradient clipping, AdamW optimizer, warmup+cosine LR scheduling
- **Generation**: Temperature, top-k, and nucleus sampling with deterministic seeding
- **Monitoring**: CSV logging, matplotlib plotting, and comprehensive benchmarking
- **Extensions**: Weight tying, hyperparameter sweeping, and performance analysis
- **Testing**: Comprehensive test suite with property-based testing and CI/CD

## Development

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Benchmark performance
python benchmark.py

# Plot training metrics
python plot.py --log runs/training.csv

# Run hyperparameter sweep
python sweep.py
```

## Requirements

- Python 3.10-3.12
- PyTorch ≥2.1.0
- CUDA (optional, for GPU training)

## License

MIT License - see LICENSE file for details.