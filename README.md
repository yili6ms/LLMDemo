# TinyGPT - Educational Transformer Implementation

A minimal yet complete GPT/Transformer implementation for educational purposes, built from scratch with a focus on clarity, performance, and modern ML best practices.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0%2B-orange.svg)

## 🚀 Quick Start

```bash
# Clone and install
git clone <repository-url>
cd llm
pip install -r requirements.txt

# Train a tokenizer
python -m tok --train data/tiny.txt --vocab-size 1000 --out tokenizer.json

# Train a model
python train.py --config configs/tiny.yaml

# Generate text
python sample.py --model checkpoints/model.pt --tokenizer tokenizer.json --prompt "Once upon a time"
```

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Testing](#-testing)
- [Performance](#-performance)
- [Theory](#-theory)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Components
- **🔤 Tokenizer**: Byte-Pair Encoding (BPE) with configurable vocabulary
- **🧠 Model**: Minimal GPT with multi-head causal self-attention
- **📊 Training**: Mixed precision, gradient clipping, modern optimizers
- **🎯 Generation**: Temperature, top-k, and nucleus (top-p) sampling
- **📈 Monitoring**: CSV logging, plotting utilities, benchmarking

### Advanced Features
- **🔄 Position Encodings**: Both learned embeddings and RoPE (Rotary Position Embedding)
- **⚡ FlashAttention**: SDPA integration for improved memory efficiency (PyTorch ≥2.1)
- **🎛️ Weight Tying**: Optional parameter sharing between embedding layers
- **🔍 Hyperparameter Sweeping**: Automated parameter search
- **🧪 Comprehensive Testing**: Unit tests, integration tests, property-based testing

## 🏗️ Architecture

### Model Components
```
TinyGPT
├── Token Embeddings (vocab_size, d_model)
├── Position Embeddings (max_seq_len, d_model) OR RoPE
├── Transformer Blocks × n_layers
│   ├── Layer Normalization
│   ├── Causal Self-Attention
│   │   ├── Multi-head attention with causal masking
│   │   └── Optional FlashAttention/SDPA
│   ├── Residual Connection
│   ├── Layer Normalization  
│   ├── Feed-Forward Network (GELU activation)
│   └── Residual Connection
├── Final Layer Normalization
└── Language Model Head (tied weights optional)
```

### Tokenizer
- **Algorithm**: Byte-Pair Encoding (BPE)
- **Byte-level**: Handles any Unicode text
- **Configurable vocabulary**: 256 base tokens + learned merges
- **Special tokens**: Beginning-of-sequence, end-of-sequence support

## 🔧 Installation

### Requirements
- Python 3.10+
- PyTorch 2.1.0+
- Optional: CUDA for GPU acceleration

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

## 📖 Usage

### 1. Tokenizer Training
```bash
# Train BPE tokenizer
python -m tok --train data/tiny.txt --vocab-size 1000 --out tokenizer.json

# Test tokenizer
python -m tok --tokenizer tokenizer.json --encode "Hello world!"
```

### 2. Model Training
```bash
# Basic training
python train.py --config configs/tiny.yaml

# Custom configuration
python train.py \
    --vocab-size 1000 \
    --d-model 256 \
    --n-layers 4 \
    --n-heads 8 \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --max-steps 10000
```

### 3. Text Generation
```bash
# Basic generation
python sample.py \
    --model checkpoints/model.pt \
    --tokenizer tokenizer.json \
    --prompt "The future of AI" \
    --max-new-tokens 100

# Advanced sampling
python sample.py \
    --model checkpoints/model.pt \
    --tokenizer tokenizer.json \
    --prompt "Once upon a time" \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9 \
    --seed 42
```

### 4. Evaluation and Analysis
```bash
# Benchmark performance
python benchmark.py --model checkpoints/model.pt --tokenizer tokenizer.json

# Plot training metrics
python plot.py --log logs/training.csv --output plots/

# Hyperparameter sweep
python sweep.py --config configs/sweep.yaml
```

## ⚙️ Configuration

### Model Configuration (YAML)
```yaml
# Model architecture
vocab_size: 1000
d_model: 256
n_layers: 4
n_heads: 8
max_seq_len: 512
dropout: 0.1

# Position encoding
use_rope: false  # Set to true for RoPE instead of learned positions

# Attention
use_flash: true  # Enable FlashAttention/SDPA if available

# Training
batch_size: 32
seq_len: 128
learning_rate: 0.0003
max_steps: 10000
warmup_steps: 1000
weight_decay: 0.01

# Optimization
use_mixed_precision: true
gradient_clip: 1.0

# Logging
eval_every: 500
save_every: 1000
log_every: 100
```

### Available Configurations
- `configs/tiny.yaml` - Small model for quick experiments
- `configs/rope_config.yaml` - Model with RoPE position encodings

## 📁 Project Structure

```
llm/
├── 📄 README.md           # This file
├── 📄 requirements.txt    # Python dependencies
├── 📄 pyproject.toml     # Project metadata and tool configs
├── 📄 Makefile           # Development commands
├── 🔄 .gitignore         # Git ignore patterns
├── 📊 benchmark.py       # Performance benchmarking
├── 📈 plot.py            # Training visualization
├── 🔄 sweep.py           # Hyperparameter sweeping
├── 🏃 train.py           # Main training script
├── 🎯 sample.py          # Text generation script
├── 📁 configs/           # Training configurations
│   ├── tiny.yaml
│   └── rope_config.yaml
├── 📁 data/              # Training data
│   ├── README.md
│   ├── tiny.txt          # Sample dataset
│   └── .gitkeep
├── 📁 docs/              # Documentation
│   └── theory.md         # Theoretical background
├── 📁 model/             # Model implementation
│   ├── __init__.py
│   └── gpt.py           # TinyGPT architecture
├── 📁 tok/               # Tokenizer implementation
│   ├── __init__.py
│   ├── __main__.py      # CLI interface
│   └── bpe.py           # BPE tokenizer
├── 📁 utils/             # Utilities
│   ├── __init__.py
│   ├── csv_logger.py    # Training metrics logging
│   └── metrics.py       # Evaluation metrics
├── 📁 tests/             # Test suite
│   ├── __init__.py
│   ├── test_*.py        # Unit and integration tests
│   └── ...
├── 📁 logs/              # Training logs (created)
├── 📁 checkpoints/       # Model checkpoints (created)
├── 📁 outputs/           # Generated text (created)
└── 📁 plots/             # Training plots (created)
```

## 🛠️ Development

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Type checking
make typecheck

# Run all quality checks
make check
```

### Testing
```bash
# Run all tests
make test

# Run specific test
python -m pytest tests/test_model.py -v

# Run with coverage
make coverage
```

### Makefile Commands
- `make install` - Install dependencies
- `make train` - Train model with default config
- `make sample` - Generate text sample
- `make eval` - Evaluate model
- `make format` - Format code with black
- `make lint` - Run flake8 linting
- `make typecheck` - Run mypy type checking
- `make test` - Run pytest
- `make coverage` - Run tests with coverage
- `make clean` - Clean temporary files

## 🧪 Testing

Comprehensive test suite covering:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Property-Based Tests**: Input validation with Hypothesis
- **Performance Tests**: Memory and speed benchmarks

### Test Categories
- `test_tokenizer.py` - BPE tokenizer functionality
- `test_model.py` - Model architecture and forward pass
- `test_training.py` - Training utilities and optimization
- `test_sampling.py` - Text generation and sampling methods
- `test_rope.py` - RoPE position encoding implementation
- `test_flash_attention.py` - FlashAttention integration
- `test_csv_logger.py` - Metrics logging functionality
- `test_integration.py` - End-to-end workflow testing

## ⚡ Performance

### Optimization Features
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) with GradScaler
- **FlashAttention**: Memory-efficient attention when available
- **Gradient Clipping**: Stable training with large models
- **Efficient Data Loading**: Optimized batch generation
- **RoPE**: Parameter-efficient position encoding

### Benchmarks
Run `python benchmark.py` to measure:
- Training throughput (tokens/sec)
- Memory usage
- Generation speed
- Attention mechanism performance

## 📚 Theory

### Mathematical Foundation
The model implements the standard Transformer architecture with causal masking:

**Attention Mechanism:**
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Multi-Head Attention:**
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Position Encoding Options:**
1. **Learned Embeddings**: `PE(pos) = W_pos[pos]`
2. **RoPE**: `RoPE(x,pos) = rotate(x, θ·pos)` where θ = 1/10000^(2i/d)

For detailed theory, see `docs/theory.md`.

### Key Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (Radford et al., 2019)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install dev dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Run quality checks: `make check`
6. Submit a pull request

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings for public APIs
- Write tests for new features
- Keep functions focused and modular

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI GPT papers for the architecture inspiration
- PyTorch team for the excellent deep learning framework
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) for implementation insights
- Hugging Face Transformers for tokenization reference

---

**Educational Purpose**: This implementation prioritizes clarity and understanding over production performance. It's designed to help you learn how modern language models work from first principles.

For questions or issues, please open a GitHub issue or contribute to the project!