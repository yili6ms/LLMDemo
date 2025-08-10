LLM Learning Plan — Dev Task Breakdown

Executable backlog mapped to 6 weeks. Each ticket includes objective, deliverables, acceptance criteria (AC), and estimate. IDs use the prefix LLM- for easy import into Jira/Linear.

⸻

0) Repo & Workflow (Week 0 pre-work)
	•	LLM-001 | Bootstrap repo
Deliverables: llm-tiny/ skeleton; pyproject.toml (Poetry) or requirements.txt; .editorconfig; README.md with quickstart; LICENSE (MIT).
AC: Fresh clone runs python -m pip install -r requirements.txt and make test successfully.
Est: 0.5d
	•	LLM-002 | Makefile + tasks
Deliverables: make format, make lint, make test, make train, make sample.
AC: Each target runs locally; documented in README.
Est: 0.5d
	•	LLM-003 | CI pipeline
Deliverables: GitHub Actions with Python matrix (3.10–3.12), cache for pip, run lint/tests.
AC: CI green on main and PRs; status badge in README.
Est: 0.5d

⸻

1) Week 1 — Foundations (Docs, Math, Stubs)
	•	LLM-101 | Theory notes
Deliverables: docs/theory.md summarizing LM objective, perplexity, attention, masking, LayerNorm, optimization.
AC: All concepts explained with 1–2 equations each; cross-checked against implementation stubs.
Est: 0.5d
	•	LLM-102 | Module stubs & types
Deliverables: tok/bpe.py, model/gpt.py, utils.py, train.py, sample.py with function/class stubs and type hints.
AC: mypy passes; unit tests compile.
Est: 0.5d
	•	LLM-103 | Dataset placeholder
Deliverables: data/tiny.txt (small corpus); data README with licensing.
AC: File present (<200KB), UTF‑8 clean.
Est: 0.25d
	•	LLM-104 | Metrics scaffolding
Deliverables: utils/metrics.py with moving average, perplexity helper; simple console logger.
AC: Unit tests for perplexity correct vs. cross‑entropy.
Est: 0.5d

⸻

2) Week 2 — Tokenizer (BPE)
	•	LLM-201 | BPE training
Deliverables: Minimal byte-level BPE trainer; tok/bpe.json save/load.
AC: Train with --vocab 1000 finishes <60s on laptop for data/tiny.txt.
Est: 1d
	•	LLM-202 | Encode/Decode
Deliverables: Greedy merge application; reversible decode.
AC: Round-trip decode(encode(s)) == s for 100 random lines (ignoring un-decodable errors).
Est: 0.5d
	•	LLM-203 | Tokenizer tests
Deliverables: Property tests (hypothesis) for round-trip; fixtures for whitespace handling.
AC: pytest -k tokenizer green; coverage >90% for tok/.
Est: 0.5d
	•	LLM-204 | CLI
Deliverables: python -m tok.bpe train|encode|decode with args.
AC: Commands work; help text present.
Est: 0.25d

⸻

3) Week 3 — Model Core (Tiny GPT)
	•	LLM-301 | Attention layer
Deliverables: CausalSelfAttention with precomputed causal mask; dropout.
AC: Shape tests; no future-token leakage verified by mask unit test.
Est: 1d
	•	LLM-302 | MLP + Block
Deliverables: GELU MLP; residual + LayerNorm; Block forward.
AC: Gradient flow sanity (non-zero grads) on toy batch.
Est: 0.5d
	•	LLM-303 | TinyGPT assembly
Deliverables: Embeddings, blocks, final LayerNorm, LM head; init.
AC: Forward on random tokens returns logits of shape (B,T,V).
Est: 0.5d
	•	LLM-304 | Generate() sampler
Deliverables: Temperature, top-k, nucleus options.
AC: Determinism with fixed seed; top-k/p unit tests.
Est: 0.5d

⸻

4) Week 4 — Data & Training
	•	LLM-401 | Data batching
Deliverables: Overlapping chunk dataset; get_batch util.
AC: Randomized batches; targets are inputs shifted by 1.
Est: 0.5d
	•	LLM-402 | Optimizer & sched
Deliverables: AdamW with weight decay filtering; warmup + cosine LR.
AC: LR follows schedule (assert over first 10k steps in unit test).
Est: 0.5d
	•	LLM-403 | Mixed precision & clip
Deliverables: AMP scaler; grad clip at 1.0.
AC: Training step runs on CPU/GPU; no inf/nan with AMP on GPU.
Est: 0.5d
	•	LLM-404 | Checkpointing
Deliverables: Save best by val loss; resume support.
AC: Stop/restart yields identical subsequent metrics.
Est: 0.5d
	•	LLM-405 | Training script
Deliverables: train.py with YAML/CLI config (Hydra or argparse).
AC: Single command trains to stable loss on data/tiny.txt.
Est: 0.5d

⸻

5) Week 5 — Sampling & Evaluation
	•	LLM-501 | Sampler CLI
Deliverables: sample.py --prompt "..." --steps 100 --top_k 50 --temp 0.9.
AC: Produces text; reproducible with --seed.
Est: 0.25d
	•	LLM-502 | Eval harness
Deliverables: Validation loop; compute cross-entropy + perplexity.
AC: make eval prints val_loss and ppl; logs saved to runs/.
Est: 0.5d
	•	LLM-503 | Hyperparam sweep (small)
Deliverables: Script to sweep d_model, n_layer, lr, dropout (2–3 values each).
AC: CSV of results; pick best by val ppl.
Est: 0.5d

⸻

6) Week 6 — Extensions & Infra
	•	LLM-601 | RoPE positional encoding (opt)
Deliverables: Replace learned pos emb with RoPE as config flag.
AC: Unit test for rotary application; functional parity on small run.
Est: 0.75d
	•	LLM-602 | Weight tying
Deliverables: Share tok_emb and head weights (opt-in).
AC: Small improvement in val loss on fixed seed benchmark.
Est: 0.25d
	•	LLM-603 | FlashAttention / SDPA path
Deliverables: Use scaled_dot_product_attention when available.
AC: Throughput +>15% on GPU vs. baseline.
Est: 0.75d
	•	LLM-604 | Logging & plots
Deliverables: Simple CSV logger + Matplotlib plot script for loss/ppl.
AC: make plot outputs runs/plot.png.
Est: 0.25d

⸻

Cross-cutting Tickets
	•	LLM-701 | Tests & coverage
Deliverables: Pytest config; coverage report.
AC: Overall coverage ≥80%; critical paths (tokenizer, attention) ≥90%.
Est: 0.75d
	•	LLM-702 | Style & formatting
Deliverables: Ruff/Black configured; contest-style minimal comments allowed in model/ per preference; docstrings in public APIs.
AC: make lint clean.
Est: 0.25d
	•	LLM-703 | Performance smoke tests
Deliverables: Benchmark script: tokens/sec for train/inf; memory usage.
AC: Baseline recorded in BENCH.md.
Est: 0.5d

⸻

Milestones & Exit Criteria
	•	M1 (End Week 2): Tokenizer round-trip passes property tests; CLI works.
	•	M2 (Mid Week 3): Forward pass shapes correct; attention mask verified.
	•	M3 (End Week 4): Training loop runs end-to-end, checkpoints best model.
	•	M4 (Week 5): Sampler produces coherent domain text; eval harness reports ppl.
	•	M5 (Week 6): At least one extension (RoPE/SDPA/weight tying) merged; plots/logging in place.

⸻

Dependencies
	•	LLM-301 depends on LLM-102 (module stubs).
	•	LLM-405 depends on LLM-401..404.
	•	LLM-501..503 depend on LLM-405 checkpoint.
	•	LLM-603 requires PyTorch ≥2.1.

⸻

Risk Register & Mitigations
	•	Data scarcity → Overfit tests + regularization knobs (dropout, weight decay).
	•	Numerical instability → AMP grad-scaler, grad clipping, LR warmup.
	•	Tokenization bugs → Property-based tests, round-trip asserts.
	•	GPU variability → SDPA fallback to naive attention.

⸻

Definition of Done (DoD)
	1.	Code merged on main, CI green, coverage threshold met.
	2.	README updated with usage instructions.
	3.	Benchmarks recorded for any performance-sensitive change.
	4.	If user-facing (CLI), help text + example included.
	5.	Changelog entry added.

⸻

Commands (suggested)
	•	make bpe-train VOCAB=1000 → produces tok/bpe.json
	•	make train CFG=configs/tiny.yaml → trains and saves checkpoint.pt
	•	make sample PROMPT="Hello" → prints generated text
	•	make eval → reports val loss and ppl
	•	make bench → tokens/sec + memory

⸻

Estimation Summary (ideal days)
	•	Week 0: 1.5d
	•	Week 1: 1.75d
	•	Week 2: 2.25d
	•	Week 3: 2.5d
	•	Week 4: 2.5d
	•	Week 5: 1.25d
	•	Week 6: 2.0d
Total: ~13.75 ideal dev-days (solo), excluding long training runs.

⸻

Notes for your preferences
	•	Python-first; performance-conscious.
	•	Keep model code terse (contest style) but maintain tests + API docstrings.
	•	Add optional flags for RoPE/SDPA early to make later experiments simple.