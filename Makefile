.PHONY: help format lint test train sample eval bench clean bpe-train

PYTHON := python3
VOCAB ?= 1000
CFG ?= configs/tiny.yaml
PROMPT ?= "Once upon a time"
STEPS ?= 100

help:
	@echo "Available targets:"
	@echo "  format      - Format code with black"
	@echo "  lint        - Run linting with ruff and mypy"
	@echo "  test        - Run test suite with pytest"
	@echo "  train       - Train model with config (CFG=configs/tiny.yaml)"
	@echo "  sample      - Generate text from model (PROMPT=\"text\")"
	@echo "  eval        - Evaluate model on validation set"
	@echo "  bench       - Run performance benchmarks"
	@echo "  bpe-train   - Train BPE tokenizer (VOCAB=1000)"
	@echo "  clean       - Remove generated files and caches"

format:
	@echo "Formatting code with black..."
	@$(PYTHON) -m black .
	@echo "✓ Code formatted"

lint:
	@echo "Running ruff..."
	@$(PYTHON) -m ruff check .
	@echo "Running mypy..."
	@$(PYTHON) -m mypy --ignore-missing-imports .
	@echo "✓ Linting complete"

test:
	@echo "Running tests..."
	@$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=term-missing
	@echo "✓ Tests complete"

train:
	@echo "Training model with config: $(CFG)"
	@$(PYTHON) train.py --config $(CFG)
	@echo "✓ Training complete"

sample:
	@echo "Generating text..."
	@$(PYTHON) sample.py --prompt $(PROMPT) --steps $(STEPS)

eval:
	@echo "Evaluating model..."
	@$(PYTHON) -c "import train; train.evaluate()"
	@echo "✓ Evaluation complete"

bench:
	@echo "Running benchmarks..."
	@$(PYTHON) benchmark.py
	@echo "✓ Benchmarks complete"

bpe-train:
	@echo "Training BPE tokenizer with vocab size: $(VOCAB)"
	@$(PYTHON) -m tok.bpe train --vocab $(VOCAB) --input data/tiny.txt --output tok/bpe.json
	@echo "✓ Tokenizer trained"

clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name ".coverage" -delete
	@rm -rf runs/ checkpoints/ 2>/dev/null || true
	@echo "✓ Cleanup complete"