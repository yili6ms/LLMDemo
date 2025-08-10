# Language Modeling Theory

This document covers the key theoretical concepts underlying language modeling and the Transformer architecture implemented in TinyGPT.

## Language Modeling Objective

### Cross-Entropy Loss

The fundamental objective of language modeling is to maximize the likelihood of the training data under the model. This is equivalent to minimizing the cross-entropy loss:

```
L = -∑(i=1 to N) log P(x_i | x_<i)
```

Where:
- `N` is the sequence length
- `x_i` is the token at position i
- `x_<i` represents all tokens before position i
- `P(x_i | x_<i)` is the conditional probability of token `x_i` given the context

### Perplexity

Perplexity is a common evaluation metric for language models, defined as:

```
PPL = exp(L)
```

Where `L` is the average cross-entropy loss. Lower perplexity indicates better performance. Perplexity can be interpreted as the weighted average number of choices the model has when predicting the next token.

## Tokenization

### Byte-Pair Encoding (BPE)

BPE is a subword tokenization algorithm that iteratively merges the most frequent pairs of adjacent symbols:

1. Initialize vocabulary with all bytes (0-255)
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size

**Algorithm:**
```
vocab = {0, 1, ..., 255}  # All bytes
while |vocab| < vocab_size:
    pairs = count_pairs(corpus)
    best_pair = argmax(pairs)
    corpus = merge(corpus, best_pair)
    vocab.add(new_token)
```

The encoding process applies merges in order, while decoding recursively splits merged tokens back to bytes.

## Transformer Architecture

### Multi-Head Self-Attention

The core component of the Transformer is scaled dot-product attention:

```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

Where:
- `Q` (queries), `K` (keys), `V` (values) are linear projections of the input
- `d_k` is the dimension of the key vectors
- The scaling factor `1/√d_k` prevents vanishing gradients

**Multi-head attention** runs multiple attention heads in parallel:

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Causal Masking

For language modeling, we use causal (autoregressive) masking to prevent positions from attending to future tokens:

```
mask[i,j] = {
  0  if j > i  (future positions)
  1  if j ≤ i  (current and past positions)
}
```

This ensures the model can only use information from previous tokens when predicting the next token.

### Position Embeddings

Since attention is permutation-invariant, we add positional information:

**Learned Positional Embeddings:**
```
x = token_embedding(input) + position_embedding(position)
```

**RoPE (Rotary Position Embedding):**
Applies rotation matrices to query and key vectors:
```
q_m = R_m q
k_n = R_n k
```
Where `R_θ` rotates vectors by angle θ proportional to position.

### Layer Normalization

LayerNorm normalizes across the feature dimension:

```
LayerNorm(x) = γ * (x - μ) / σ + β
```

Where:
- `μ = mean(x)` across features
- `σ = std(x)` across features
- `γ, β` are learned parameters

### MLP Block

The feed-forward network applies two linear transformations with GELU activation:

```
MLP(x) = W_2 * GELU(W_1 * x + b_1) + b_2
```

**GELU activation:**
```
GELU(x) = x * Φ(x)
```
Where Φ(x) is the CDF of the standard normal distribution.

## Transformer Block

Each transformer block combines attention and MLP with residual connections:

```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

This is the "Pre-LN" variant, which is more stable than Post-LN.

## Optimization

### AdamW

AdamW (Adam with decoupled weight decay) updates parameters as:

```
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t^2
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
```

Where:
- `g_t` is the gradient
- `m_t, v_t` are first and second moment estimates
- `α` is the learning rate
- `λ` is the weight decay coefficient

### Learning Rate Scheduling

**Warmup + Cosine Decay:**
```
lr(t) = {
  lr_max * t / warmup_steps                    if t < warmup_steps
  lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(π * progress))  otherwise
}
```
Where `progress = (t - warmup_steps) / (max_steps - warmup_steps)`

## Training Techniques

### Gradient Clipping

Prevents exploding gradients by scaling down when norm exceeds threshold:

```
if ||g|| > clip_value:
    g = g * clip_value / ||g||
```

### Mixed Precision Training

Uses FP16 for forward/backward passes and FP32 for parameter updates:
- Reduces memory usage by ~50%
- Increases training speed on modern GPUs
- Requires gradient scaling to prevent underflow

### Dropout

Regularization technique that randomly sets neurons to zero during training:
```
dropout(x) = {
  x / (1 - p)  with probability (1 - p)
  0            with probability p
}
```

## Model Scaling

### Parameter Count

For a transformer with:
- Vocabulary size V
- Model dimension d
- Number of layers L
- Feed-forward dimension 4d

Approximate parameter count:
```
Params ≈ V*d + L*(4*d² + 2*d) + d*V
       ≈ 2*V*d + L*4*d²  (for large models)
```

### Compute Requirements

Training compute scales approximately as:
```
Compute ∝ N * D * T
```
Where:
- N = number of parameters
- D = dataset size (tokens)
- T = number of training epochs

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)
- RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)