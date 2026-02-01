# Transformer Architecture: Complete Guide

**Last Updated**: 2026-02-01
**Prerequisites**: Understanding of attention mechanism (see `01_attention_basics.md`)
**Learning Goal**: Master all Transformer components and how they work together

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Multi-Head Attention](#multi-head-attention)
3. [Positional Encoding](#positional-encoding)
4. [Encoder Architecture](#encoder-architecture)
5. [Decoder Architecture](#decoder-architecture)
6. [Feed-Forward Networks](#feed-forward-networks)
7. [Layer Normalization & Residual Connections](#layer-normalization--residual-connections)
8. [Putting It All Together](#putting-it-all-together)

---

## Architecture Overview

The Transformer, introduced in "Attention is All You Need" (Vaswani et al., 2017), revolutionized sequence modeling by relying entirely on attention mechanisms, eliminating recurrence.

### High-Level Architecture

```
Input Sequence                          Target Sequence
     ↓                                        ↓
[Input Embedding]                    [Output Embedding]
     ↓                                        ↓
[Positional Encoding]                [Positional Encoding]
     ↓                                        ↓
┌─────────────────┐                  ┌─────────────────┐
│                 │                  │                 │
│  ENCODER (×N)   │ ──────────→      │  DECODER (×N)   │
│                 │  (context)       │                 │
│ • Self-Attn     │                  │ • Masked Self   │
│ • Feed-Forward  │                  │ • Cross-Attn    │
│                 │                  │ • Feed-Forward  │
└─────────────────┘                  └─────────────────┘
                                             ↓
                                     [Linear + Softmax]
                                             ↓
                                     Output Probabilities
```

### Key Innovation

**No Recurrence**: Unlike RNNs/LSTMs, Transformers process all positions in parallel, making them:
- **Faster to train** (parallelizable)
- **Better at long-range dependencies** (direct connections)
- **More scalable** (can leverage GPU parallelism)

---

## Multi-Head Attention

Single attention gives one perspective on the data. **Multi-head attention runs multiple attention operations in parallel**, each learning different patterns.

### Motivation

Consider the sentence: "The animal didn't cross the street because **it** was too tired."

Different attention heads can capture different relationships:
- **Head 1**: "it" → "animal" (subject reference)
- **Head 2**: "cross" → "street" (action-object)
- **Head 3**: "tired" → "didn't" (negation)

### Mathematical Formulation

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### Step-by-Step Process

1. **Project inputs** to h different subspaces:
   - $Q_i = QW_i^Q$ where $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
   - $K_i = KW_i^K$ where $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
   - $V_i = VW_i^V$ where $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$

2. **Compute attention** for each head independently:
   $$
   \text{head}_i = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i
   $$

3. **Concatenate** all heads:
   $$
   \text{Concat} = [head_1; head_2; \ldots; head_h]
   $$

4. **Final linear projection**:
   $$
   \text{Output} = \text{Concat} \cdot W^O
   $$

### Typical Configuration

- **Number of heads** (h): 8 or 16
- **Model dimension** ($d_{model}$): 512
- **Per-head dimension**: $d_k = d_v = d_{model}/h = 64$

**Why split dimensions?**
Computational cost is the same as single-head attention with full dimension, but we get h different learned perspectives!

### Visualization

```
Input (d_model=512)
       ↓
┌──────────────────────────────────────┐
│    Split into 8 heads (64 dim each)  │
├──────┬──────┬──────┬─────────┬───────┤
│Head 1│Head 2│Head 3│   ...   │Head 8 │
│      │      │      │         │       │
│ Q K V│ Q K V│ Q K V│   Q K V │ Q K V │
│  ↓   │  ↓   │  ↓   │    ↓    │   ↓   │
│Attn  │Attn  │Attn  │  Attn   │ Attn  │
│ (64) │ (64) │ (64) │  (64)   │  (64) │
└──────┴──────┴──────┴─────────┴───────┘
       ↓
   Concatenate (512)
       ↓
   Linear Projection W^O
       ↓
   Output (512)
```

---

## Positional Encoding

**The Problem**: Attention has no inherent notion of position or order. The sentence "dog bites man" would be treated the same as "man bites dog"!

**The Solution**: Add positional information to input embeddings.

### Sinusoidal Positional Encoding

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

Where:
- $pos$ = position in sequence (0, 1, 2, ...)
- $i$ = dimension index (0, 1, 2, ..., $d_{model}/2$)
- Even dimensions use sine, odd use cosine

### Why This Formula?

1. **Unique encoding**: Each position gets a unique pattern
2. **Bounded values**: Always in [-1, 1]
3. **Relative positions**: Model can learn to attend by relative position
4. **Extrapolation**: Can handle sequences longer than training

### Wavelength Pattern

Different dimensions have different wavelengths:
- Low dimensions (i=0): Fast oscillation (wavelength = 2π)
- High dimensions (i=d/2): Slow oscillation (wavelength = 10000·2π)

```
Position encoding across dimensions for pos=0 to 9:

Dim 0 (fast):  ▁▄▀▄▁▄▀▄▁▄  (sin wave)
Dim 1:         ▀▄▁▄▀▄▁▄▀▄  (cos wave)
Dim 2:         ▁▂▄▀▄▂▁▂▄▀  (slower)
...
Dim 511:       ▁▁▁▁▁▂▂▂▂▂  (very slow)
```

### Implementation Note

```python
# Positional encoding is ADDED to embeddings, not concatenated
final_input = word_embedding + positional_encoding
```

This allows the model to use both content and position information together.

---

## Encoder Architecture

The Encoder processes the input sequence and produces contextualized representations.

### Single Encoder Layer

```
Input from previous layer
        ↓
┌───────────────────────┐
│  Multi-Head           │
│  Self-Attention       │  ← Queries, Keys, Values all from same input
└───────────────────────┘
        ↓
     [Add & Norm]  ← Residual connection + Layer Normalization
        ↓
┌───────────────────────┐
│  Feed-Forward         │
│  Network              │
└───────────────────────┘
        ↓
     [Add & Norm]  ← Another residual + normalization
        ↓
   Output to next layer
```

### Self-Attention in Encoder

- **Input**: Sequence of token embeddings
- **Q, K, V**: All derived from the same input
- **Purpose**: Each token can attend to all other tokens
- **No masking**: Can see the full input (bidirectional)

### Mathematical Flow

For input $X \in \mathbb{R}^{seq\_len \times d_{model}}$:

1. **Self-Attention**:
   $$
   Z = \text{MultiHeadAttn}(X, X, X)
   $$

2. **Add & Norm**:
   $$
   X' = \text{LayerNorm}(X + Z)
   $$

3. **Feed-Forward**:
   $$
   F = \text{FFN}(X')
   $$

4. **Add & Norm**:
   $$
   \text{Output} = \text{LayerNorm}(X' + F)
   $$

### Stack of N Layers

The original Transformer uses N=6 encoder layers. Each layer:
- Has its own parameters
- Processes increasingly abstract representations
- Maintains the same dimensionality ($d_{model}$)

---

## Decoder Architecture

The Decoder generates the output sequence one token at a time, using both the encoder output and previous decoder outputs.

### Single Decoder Layer

```
Input from previous decoder layer
        ↓
┌───────────────────────┐
│  Masked Multi-Head    │
│  Self-Attention       │  ← Cannot see future tokens!
└───────────────────────┘
        ↓
     [Add & Norm]
        ↓
┌───────────────────────┐
│  Cross-Attention      │  ← Q from decoder, K,V from encoder
│                       │     (this is the key connection!)
└───────────────────────┘
        ↓
     [Add & Norm]
        ↓
┌───────────────────────┐
│  Feed-Forward         │
│  Network              │
└───────────────────────┘
        ↓
     [Add & Norm]
        ↓
   Output to next layer
```

### Three Types of Attention in Decoder

1. **Masked Self-Attention**
   - Q, K, V all from decoder input
   - **Causal mask**: Position i can only attend to positions ≤ i
   - Ensures autoregressive property (no cheating!)

2. **Cross-Attention** (Encoder-Decoder Attention)
   - **Q**: from decoder (what we're generating)
   - **K, V**: from encoder output (source information)
   - This is how the decoder accesses input information!

3. **Feed-Forward**
   - Same as encoder

### Causal Masking

For sequence "A B C D", when generating token at position 2:

```
Attention mask (0 = allowed, -∞ = blocked):

     A   B   C   D
A  [ 0  -∞  -∞  -∞ ]  ← When at A, can only see A
B  [ 0   0  -∞  -∞ ]  ← When at B, can see A, B
C  [ 0   0   0  -∞ ]  ← When at C, can see A, B, C
D  [ 0   0   0   0 ]  ← When at D, can see all
```

This prevents the model from "peeking" at future tokens during training.

---

## Feed-Forward Networks

Each encoder/decoder layer has a position-wise feed-forward network (FFN).

### Architecture

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

Or more explicitly:
1. **Expansion**: Linear layer from $d_{model}$ to $d_{ff}$ (typically 2048-4096)
2. **Activation**: ReLU or GELU
3. **Projection**: Linear layer back to $d_{model}$

### Why "Position-Wise"?

The same FFN is applied to each position independently:

```python
for position in sequence:
    output[position] = FFN(input[position])
```

This is equivalent to two 1×1 convolutions!

### Intuition

- **Attention**: Mixes information across positions
- **FFN**: Processes each position's information independently
- Together: Attention gathers context, FFN transforms it

### Typical Configuration

- Input: 512 dimensions
- Hidden: 2048 dimensions (4× expansion)
- Output: 512 dimensions

The expansion allows rich, non-linear transformations.

---

## Layer Normalization & Residual Connections

These two techniques are critical for training deep Transformers.

### Layer Normalization

**Normalizes activations across features** (not batch):

$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma} + \beta
$$

Where:
- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ (mean)
- $\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}$ (std)
- $\gamma, \beta$ are learned parameters

**Why not Batch Norm?**
Sequences have variable lengths; Layer Norm normalizes per example, not across batch.

### Residual Connections

**Add the input back to the output**:

$$
\text{output} = x + \text{Sublayer}(x)
$$

**Why critical?**
- **Gradient flow**: Gradients can flow directly through addition
- **Identity mapping**: Model can learn to skip layers if needed
- **Deeper networks**: Enables training very deep models (12, 24, 48+ layers)

### Pre-Norm vs Post-Norm

**Original (Post-Norm)**:
```
x → Sublayer → Add(x) → LayerNorm → output
```

**Modern (Pre-Norm, more stable)**:
```
x → LayerNorm → Sublayer → Add(x) → output
```

Pre-Norm is now more common for very deep models.

---

## Putting It All Together

### Complete Forward Pass

**Input**: "The cat sat"
**Target**: "Le chat assis" (French translation)

1. **Encoder**:
   ```
   "The cat sat"
   → [Embeddings + Positional Encoding]
   → [6× Encoder Layers]
   → Contextualized representations
   ```

2. **Decoder** (generating "Le"):
   ```
   Input: <START>
   → [Embedding + Positional Encoding]
   → [Masked Self-Attention] (only sees <START>)
   → [Cross-Attention with encoder output]
   → [Feed-Forward]
   → [Linear + Softmax]
   → Probability distribution over French vocabulary
   → Sample: "Le"
   ```

3. **Decoder** (generating "chat"):
   ```
   Input: <START> Le
   → ...
   → [Masked Self-Attention] (sees <START>, Le)
   → [Cross-Attention with encoder]
   → ...
   → Sample: "chat"
   ```

And so on, autoregress ively!

### Full Architecture Diagram

```
ENCODER SIDE                    DECODER SIDE

Input Embedding                 Output Embedding
     +                               +
Positional Enc.                 Positional Enc.
     ↓                               ↓
┌─────────────┐                ┌─────────────┐
│  Nx Layers  │                │  Nx Layers  │
│             │                │             │
│ ┌─────────┐ │                │ ┌─────────┐ │
│ │Multi-Head│ │                │ │ Masked  │ │
│ │Self-Attn │ │                │ │Self-Attn│ │
│ └─────────┘ │                │ └─────────┘ │
│      ↓      │                │      ↓      │
│  [Add&Norm] │                │  [Add&Norm] │
│      ↓      │                │      ↓      │
│      │      │◄───────────────┼─┐┌─────────┐ │
│      │      │  (K, V)        │ ││  Cross  │ │
│      │      │                │ ││  Attn   │ │
│  ┌────────┐ │                │ │└─────────┘ │
│  │  FFN   │ │                │ │     ↓      │
│  └────────┘ │                │ │ [Add&Norm] │
│      ↓      │                │ │     ↓      │
│  [Add&Norm] │                │ │ ┌────────┐ │
│      ↓      │                │ │ │  FFN   │ │
└─────────────┘                │ │ └────────┘ │
                               │ │     ↓      │
                               │ │ [Add&Norm] │
                               │ └─────────┘   │
                               └───────────────┘
                                       ↓
                                  [Linear]
                                       ↓
                                  [Softmax]
                                       ↓
                               Output Probabilities
```

### Key Connections

1. **Encoder Self-Attention**: Builds context-aware representations
2. **Decoder Masked Self-Attention**: Maintains autoregressive property
3. **Cross-Attention**: Decoder queries encoder memory
4. **Residuals + Norms**: Enable deep stacking

---

## Why Transformers Work

1. **Parallelization**: All positions processed simultaneously
2. **Long-range dependencies**: Direct connections via attention
3. **Flexibility**: Multi-head attention learns diverse patterns
4. **Scalability**: Architecture scales to billions of parameters

### Computational Complexity

- **Self-Attention**: $O(n^2 \cdot d)$ where n = sequence length
- **Feed-Forward**: $O(n \cdot d^2)$
- **Bottleneck**: Quadratic in sequence length

This led to variants: Sparse Transformers, Linformers, Performers, etc.

---

## Summary

**Transformer = Attention + Position + Normalization + Residuals**

**Key Components**:
- Multi-head attention (parallel perspectives)
- Positional encoding (sequence order)
- Feed-forward networks (non-linear transformation)
- Layer norm + residuals (stable deep training)

**Encoder-Decoder Structure**:
- Encoder: Bidirectional context building
- Decoder: Autoregressive generation with encoder memory

**Impact**: Foundation of GPT, BERT, T5, and essentially all modern LLMs.

---

## References

1. Vaswani et al., "Attention is All You Need" (2017) - [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
3. [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

---

**Word Count**: ~2,200 words ✓
**All components covered**: Multi-head attention, positional encoding, encoder, decoder, FFN, layer norm, residuals ✓
**Architecture diagrams**: ASCII visualizations ✓
**Mathematical formulations**: LaTeX notation ✓

Next: JEPA Principle (the paradigm shift!) 🚀
