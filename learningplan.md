LLM Learning Plan: From Theory to Tiny Transformer

⸻

Week 1: Foundations
	•	Day 1–2: Read about language modeling objectives (cross-entropy, perplexity) and tokenization basics (BPE).
	•	Day 3: Learn multi-head attention math, causal masking, and LayerNorm.
	•	Day 4–5: Study Transformer block structure (attention + MLP + residuals + positional embeddings).
	•	Day 6–7: Review optimization techniques (AdamW, warmup/cosine decay, dropout).

⸻

Week 2: Tokenizer Implementation
	•	Day 8: Implement minimal BPE training algorithm.
	•	Day 9: Add encode/decode functions, special tokens.
	•	Day 10: Train BPE on tiny corpus, experiment with vocab size.
	•	Day 11: Save/load tokenizer JSON.
	•	Day 12–14: Tokenization experiments: byte-level vs subword.

⸻

Week 3: Model Core
	•	Day 15–16: Implement CausalSelfAttention (masking & multi-head split).
	•	Day 17–18: Build MLP, LayerNorm, and Block modules.
	•	Day 19: Assemble TinyGPT class (embeddings, blocks, LM head).
	•	Day 20: Initialize weights, verify forward pass shapes.
	•	Day 21: Add generate method with temperature, top-k, top-p sampling.

⸻

Week 4: Data Pipeline & Training Loop
	•	Day 22: Build data batching with overlapping chunks.
	•	Day 23: Split dataset into train/validation.
	•	Day 24–25: Write training loop (AdamW, gradient clipping, LR scheduler).
	•	Day 26: Add mixed precision with GradScaler.
	•	Day 27: Track perplexity and save best checkpoint.
	•	Day 28: Overfit tiny batch for debugging.

⸻

Week 5: Sampling & Evaluation
	•	Day 29: Implement standalone sampling script.
	•	Day 30: Test on various prompts.
	•	Day 31–32: Evaluate on validation set (perplexity, qualitative output).
	•	Day 33: Tune hyperparameters.
	•	Day 34–35: Compare outputs for different temperatures/top-k.

⸻

Week 6: Extensions & Experiments
	•	Day 36–37: Add RoPE or alternative positional encodings.
	•	Day 38: Implement weight tying.
	•	Day 39–40: Try FlashAttention or scaled_dot_product_attention.
	•	Day 41: Train with larger corpus (Tiny Shakespeare, WikiText-2).
	•	Day 42: Add checkpointing, logging, and plots.

⸻

Ongoing: Reading & Theory
	•	Weekly: Read key papers (Attention Is All You Need, Scaling Laws, Chinchilla).
	•	Biweekly: Explore efficient Transformers and RLHF overviews.
	•	Monthly: Re-implement core in NumPy for deeper understanding.

⸻

Milestones
	•	Milestone 1 (Week 2): BPE tokenizer encodes/decodes text correctly.
	•	Milestone 2 (Week 3): TinyGPT forward pass runs without errors.
	•	Milestone 3 (Week 4): Model trains to reasonable perplexity.
	•	Milestone 4 (Week 5): Generates coherent text.
	•	Milestone 5 (Week 6): Implements at least one advanced feature.

⸻

Tip: Keep modules small and test independently. Use overfit tests to validate each component before scaling up.