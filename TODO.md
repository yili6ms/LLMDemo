# TODO List - LLM Tiny Transformer Project

## Week 0 - Repository & Workflow Setup
- [x] **LLM-001** | Bootstrap repo - Create llm-tiny/ skeleton, pyproject.toml/requirements.txt, .editorconfig, README.md, LICENSE
- [x] **LLM-002** | Makefile + tasks - Create make format, lint, test, train, sample targets
- [x] **LLM-003** | CI pipeline - Setup GitHub Actions with Python matrix (3.10-3.12), pip cache, lint/tests

## Week 1 - Foundations (Docs, Math, Stubs)
- [x] **LLM-101** | Theory notes - Create docs/theory.md with LM concepts, equations
- [x] **LLM-102** | Module stubs & types - Create tok/bpe.py, model/gpt.py, utils.py, train.py, sample.py with stubs
- [x] **LLM-103** | Dataset placeholder - Add data/tiny.txt (<200KB) with README
- [x] **LLM-104** | Metrics scaffolding - Create utils/metrics.py with perplexity helper, console logger

## Week 2 - Tokenizer (BPE)
- [x] **LLM-201** | BPE training - Implement minimal byte-level BPE trainer, tok/bpe.json save/load
- [x] **LLM-202** | Encode/Decode - Implement greedy merge application, reversible decode
- [ ] **LLM-203** | Tokenizer tests - Add property tests with hypothesis for round-trip
- [x] **LLM-204** | Tokenizer CLI - Create python -m tok.bpe train|encode|decode commands

## Week 3 - Model Core (Tiny GPT)
- [x] **LLM-301** | Attention layer - Implement CausalSelfAttention with precomputed mask, dropout
- [x] **LLM-302** | MLP + Block - Implement GELU MLP, residual + LayerNorm, Block forward
- [x] **LLM-303** | TinyGPT assembly - Create embeddings, blocks, final LayerNorm, LM head
- [x] **LLM-304** | Generate() sampler - Add temperature, top-k, nucleus sampling options

## Week 4 - Data & Training
- [x] **LLM-401** | Data batching - Create overlapping chunk dataset, get_batch util
- [x] **LLM-402** | Optimizer & scheduler - Implement AdamW with weight decay filtering, warmup + cosine LR
- [x] **LLM-403** | Mixed precision & clip - Add AMP scaler, grad clip at 1.0
- [x] **LLM-404** | Checkpointing - Implement save best by val loss, resume support
- [x] **LLM-405** | Training script - Create train.py with YAML/CLI config

## Week 5 - Sampling & Evaluation
- [x] **LLM-501** | Sampler CLI - Create sample.py with prompt, steps, top_k, temp args
- [x] **LLM-502** | Eval harness - Create validation loop, compute cross-entropy + perplexity
- [x] **LLM-503** | Hyperparam sweep - Script to sweep d_model, n_layer, lr, dropout

## Week 6 - Extensions & Infrastructure
- [x] **LLM-601** | RoPE positional encoding - Replace learned pos emb with RoPE as config flag
- [x] **LLM-602** | Weight tying - Share tok_emb and head weights (opt-in)
- [x] **LLM-603** | FlashAttention/SDPA - Use scaled_dot_product_attention when available
- [x] **LLM-604** | Logging & plots - Create CSV logger + Matplotlib plot script

## Cross-cutting Tasks
- [ ] **LLM-701** | Tests & coverage - Setup pytest config, achieve ≥80% overall coverage
- [x] **LLM-702** | Style & formatting - Configure Ruff/Black, ensure make lint clean
- [x] **LLM-703** | Performance smoke tests - Create benchmark script for tokens/sec, memory usage

---

## Milestones
- **M1 (End Week 2)**: Tokenizer round-trip passes property tests; CLI works
- **M2 (Mid Week 3)**: Forward pass shapes correct; attention mask verified
- **M3 (End Week 4)**: Training loop runs end-to-end, checkpoints best model
- **M4 (Week 5)**: Sampler produces coherent domain text; eval harness reports ppl
- **M5 (Week 6)**: At least one extension (RoPE/SDPA/weight tying) merged; plots/logging in place

## Dependencies
- LLM-301 depends on LLM-102 (module stubs)
- LLM-405 depends on LLM-401..404
- LLM-501..503 depend on LLM-405 checkpoint
- LLM-603 requires PyTorch ≥2.1

## Estimation Summary
- Week 0: 1.5 days
- Week 1: 1.75 days
- Week 2: 2.25 days
- Week 3: 2.5 days
- Week 4: 2.5 days
- Week 5: 1.25 days
- Week 6: 2.0 days
- **Total**: ~13.75 ideal dev-days (solo)