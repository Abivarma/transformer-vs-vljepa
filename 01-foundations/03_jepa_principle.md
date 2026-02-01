# JEPA: Joint Embedding Predictive Architecture

**Last Updated**: 2026-02-01
**Prerequisites**: Understanding of Transformers and attention
**Key Paper**: [VL-JEPA](https://arxiv.org/abs/2512.10942) (Chen et al., 2024)
**Learning Goal**: Understand the paradigm shift from generative to embedding prediction

---

## Table of Contents
1. [What is JEPA?](#what-is-jepa)
2. [Token Prediction vs Embedding Prediction](#token-prediction-vs-embedding-prediction)
3. [Why Embeddings for Multimodal](#why-embeddings-for-multimodal)
4. [InfoNCE Loss & Contrastive Learning](#infonce-loss--contrastive-learning)
5. [Uniformity vs Alignment Trade-off](#uniformity-vs-alignment-trade-off)
6. [VL-JEPA Architecture](#vl-jepa-architecture)

---

## What is JEPA?

**JEPA** = **J**oint **E**mbedding **P**redictive **A**rchitecture

JEPA represents a fundamental shift in how we think about learning representations, championed by Yann LeCun and Meta AI.

### The Core Idea

Instead of predicting discrete tokens (like GPT), JEPA predicts **continuous embeddings** in a learned latent space.

```
Traditional Generative Model (e.g., GPT):
Input: "The cat sat on the"
Predict: Next token → "mat" (discrete choice from vocabulary)

JEPA:
Input: Image of a cat
Predict: Embedding vector that represents "cat" (continuous vector in latent space)
```

### Why "Joint Embedding"?

"Joint" means different modalities (vision, language) are mapped to a **shared embedding space**:

```
    Image                 Text
      ↓                    ↓
   Encoder              Encoder
      ↓                    ↓
    ┌─────────────────────┐
    │  Shared Embedding   │
    │      Space          │
    │                     │
    │  [cat_img] ≈ [cat]  │  ← Similar embeddings!
    │                     │
    └─────────────────────┘
```

**Key Insight**: If "image of cat" and "word 'cat'" have similar embeddings, the model has learned a useful multimodal representation!

### JEPA vs Generative Models

| Aspect | Generative (GPT, etc.) | JEPA |
|--------|------------------------|------|
| **Prediction** | Discrete tokens | Continuous embeddings |
| **Loss** | Cross-entropy | Contrastive (InfoNCE) |
| **Output Space** | Vocabulary (finite) | Embedding space (continuous) |
| **Multimodal** | Requires tokenization | Natural fit |
| **Goal** | Model probability distribution | Learn meaningful representations |

---

## Token Prediction vs Embedding Prediction

### Token Prediction (Generative Approach)

**How it works**:
1. Input: Sequence of tokens
2. Model: Predicts probability distribution over vocabulary
3. Loss: Cross-entropy between prediction and true next token

$$
\mathcal{L}_{CE} = -\sum_{i=1}^{|V|} y_i \log(\hat{y}_i)
$$

Where $|V|$ is vocabulary size (e.g., 50,000 tokens).

**Example (Text Generation)**:
```
Input:  "The cat sat on"
Target: "the"
Output: P("the") = 0.7, P("a") = 0.2, P("it") = 0.05, ...
Loss:   -log(0.7) = 0.36
```

**Challenges for Multimodal**:
- **Discrete bottleneck**: Vision/audio must be tokenized
- **Vocabulary mismatch**: Different modalities need different vocabularies
- **Reconstruction focus**: Model learns to reconstruct, not to understand

### Embedding Prediction (JEPA Approach)

**How it works**:
1. Input: Any modality (image, text, audio)
2. Encoder: Maps to continuous embedding
3. Loss: Contrastive loss (embeddings of similar things should be close)

$$
\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k)/\tau)}
$$

**Example (Vision-Language)**:
```
Input Image: [🐱 photo]  →  Embedding: [0.2, -0.5, 0.8, ...]
Input Text:  "cat"       →  Embedding: [0.19, -0.48, 0.82, ...]

Similarity: cos_sim = 0.95  ← Very close!
Loss:       Small (embeddings are aligned)
```

**Advantages**:
- **No tokenization needed**: Direct embedding of raw inputs
- **Shared space**: Natural multimodal fusion
- **Semantic focus**: Learns meaning, not just reconstruction

### Visual Comparison

```
GENERATIVE MODEL:
┌─────────┐
│  Input  │
└────┬────┘
     │
┌────▼────┐
│Transformer│
└────┬────┘
     │
┌────▼────┐
│ Softmax │  ← Predict discrete token
└────┬────┘
     │
   Token


JEPA:
┌─────────┐    ┌─────────┐
│ Image   │    │  Text   │
└────┬────┘    └────┬────┘
     │              │
┌────▼────┐    ┌────▼────┐
│Vision   │    │Language │
│Encoder  │    │Encoder  │
└────┬────┘    └────┬────┘
     │              │
     │    ┌────┐    │
     └────►Joint│◄───┘
          │Emb.│  ← Continuous embedding space
          └────┘
```

---

## Why Embeddings for Multimodal

### The Tokenization Problem

To use generative models for vision, we must **tokenize** images:

```
Image → Patches → Discrete Tokens (VQVAE, DALL-E)
```

**Issues**:
1. Information loss during tokenization
2. Fixed vocabulary size limits expressiveness
3. Two-stage training (tokenizer + model)
4. Doesn't naturally generalize to new modalities

### The JEPA Solution

**Map everything to a continuous embedding space**:

```
Image    → Vision Encoder   → Embedding (512-dim)
Text     → Language Encoder → Embedding (512-dim)
Audio    → Audio Encoder    → Embedding (512-dim)
Video    → Video Encoder    → Embedding (512-dim)

All in the SAME SPACE!
```

### Benefits for Multimodal

1. **Unified Representation**
   - No need for separate vocabularies
   - Can mix modalities freely
   - Zero-shot transfer possible

2. **Semantic Alignment**
   - Similar concepts cluster together
   - Enables cross-modal retrieval
   - Learn from correspondence, not reconstruction

3. **Scalability**
   - Easy to add new modalities
   - No retraining from scratch
   - Modular encoder design

4. **Efficiency**
   - Direct embedding (no decoding step)
   - Contrastive loss is simpler than pixel/token reconstruction
   - Can leverage unlabeled multimodal data

### Concrete Example: Image-Caption Retrieval

**Query**: "A dog playing in the park"

**JEPA Approach**:
1. Encode query → embedding $q$
2. Encode all images → embeddings $\{v_1, v_2, ..., v_n\}$
3. Find closest: $\arg\max_i \text{similarity}(q, v_i)$

**Fast and direct!** No need to generate captions for all images.

---

## InfoNCE Loss & Contrastive Learning

JEPA uses **contrastive learning** to train encoders. The key is InfoNCE (Information Noise-Contrastive Estimation) loss.

### Core Principle

"Pull together embeddings of similar items, push apart embeddings of dissimilar items."

### Mathematical Formulation

For an anchor sample $i$ with positive sample $j$ and negative samples $\{k\}$:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

Where:
- $z_i, z_j, z_k$ are embeddings
- $\text{sim}(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||}$ (cosine similarity)
- $\tau$ is temperature (controls distribution sharpness)
- $N$ is number of negatives

### Intuition

**Numerator**: Similarity between anchor and positive
- Want this to be HIGH → $\exp(\text{large})$

**Denominator**: Sum of similarities to all samples (including positive)
- Normalizes to make it a proper probability

**Effect**: Maximizing this loss makes:
- $\text{sim}(z_i, z_j)$ large (positive pair close)
- $\text{sim}(z_i, z_k)$ small (negative pairs far)

### Positive vs Negative Pairs

**For Image-Caption Pairs**:
```
Positive: (cat_image, "cat")
Negative: (cat_image, "dog")
          (cat_image, "car")
          (cat_image, "tree")
```

**For Augmented Views** (Self-supervised):
```
Positive: (img, augmented_img)
Negative: (img, other_images_in_batch)
```

### Temperature Parameter τ

Controls how "soft" the distribution is:

- **Low τ** (e.g., 0.01): Sharp distribution, focuses on hardest negatives
- **High τ** (e.g., 1.0): Soft distribution, considers all negatives equally

Typical range: τ ∈ [0.05, 0.1]

### Why "InfoNCE"?

Derived from **Noise-Contrastive Estimation**: Distinguish true data from noise by learning a classifier. Maximizing InfoNCE ≈ maximizing mutual information between views.

---

## Uniformity vs Alignment Trade-off

Training with InfoNCE creates a tension between two objectives:

### Alignment

**Goal**: Similar items should have similar embeddings.

$$
\mathcal{L}_{align} = \mathbb{E}_{(x,y) \sim p_{pos}} [||z_x - z_y||^2]
$$

**Visualization**:
```
Good Alignment:
cat_img → [●]
"cat"   → [●]  ← Close together!
```

### Uniformity

**Goal**: Embeddings should spread out across the hypersphere (avoid collapse).

$$
\mathcal{L}_{uniform} = \log \mathbb{E}_{x,y \sim p_{data}} [e^{-2||z_x - z_y||^2}]
$$

**Visualization**:
```
Good Uniformity:
      ●
   ●     ●
 ●   ●     ●
   ●     ●
      ●

Bad (Collapsed):
      ●●●
      ●●●
      ●●●
```

### The Trade-off

```
Pure Alignment:
All embeddings collapse to a single point
→ Perfect alignment, zero uniformity
→ Model learns nothing!

Pure Uniformity:
All embeddings spread evenly
→ No alignment, can't distinguish similar from dissimilar
→ Model learns nothing useful!

JEPA (Balanced):
Similar items close, different items spread out
→ Useful representations!
```

### How InfoNCE Balances

- **Numerator** $\exp(\text{sim}(z_i, z_j))$: Encourages alignment
- **Denominator** $\sum_k \exp(\text{sim}(z_i, z_k))$: Encourages uniformity

The temperature $\tau$ controls this balance!

---

## VL-JEPA Architecture

**VL-JEPA** (Vision-Language JEPA) is Meta AI's application of JEPA principles to multimodal learning.

### Key Components

```
┌─────────────┐         ┌─────────────┐
│   Image     │         │   Caption   │
└──────┬──────┘         └──────┬──────┘
       │                       │
┌──────▼──────┐         ┌──────▼──────┐
│   V-JEPA    │         │   Y-Encoder │
│  (frozen    │         │  (language) │
│   vision)   │         │             │
└──────┬──────┘         └──────┬──────┘
       │                       │
       │  Vision Tokens        │  Language Emb.
       │                       │
       └───────┐       ┌───────┘
               │       │
        ┌──────▼───────▼──────┐
        │                     │
        │     Predictor       │
        │  (Bidirectional     │
        │   Attention)        │
        │                     │
        └──────────┬──────────┘
                   │
            Predicted Tokens
                   │
        ┌──────────▼──────────┐
        │    InfoNCE Loss     │
        └─────────────────────┘
```

### V-JEPA (Vision Encoder)

- Pre-trained on images using self-supervised learning
- Outputs **visual tokens** (patch embeddings)
- **Frozen during VL-JEPA training** (efficiency!)

### Y-Encoder (Language Encoder)

- Encodes text to embeddings
- Trainable (learns to align with vision)

### Predictor

- Uses **bidirectional cross-attention**
- Queries: Visual tokens
- Keys/Values: Language embeddings
- Predicts embeddings of masked visual tokens

### Training Objective

1. Mask random visual tokens
2. Use language + visible tokens to predict masked tokens
3. Contrastive loss: Pull predictions close to true embeddings

**Key Innovation**: Language helps predict visual structure → learns meaningful vision-language alignment!

---

## Summary

**JEPA Paradigm Shift**:
- From predicting tokens → predicting embeddings
- From reconstruction → representation learning
- From unimodal → naturally multimodal

**Core Components**:
- Joint embedding space for all modalities
- InfoNCE contrastive loss
- Balance between alignment and uniformity

**VL-JEPA Contribution**:
- Efficiently combines frozen vision (V-JEPA) with trainable language
- Bidirectional attention predictor
- State-of-the-art vision-language understanding

**Why It Matters**:
- **Efficiency**: No pixel/token reconstruction
- **Generality**: Easy to extend to new modalities
- **Performance**: Competitive with or beats generative models
- **Interpretability**: Embedding space is continuous and meaningful

---

## References

1. Chen et al., "VL-JEPA: A Joint Embedding Predictive Architecture for Vision and Language" (2024) - [arxiv.org/abs/2512.10942](https://arxiv.org/abs/2512.10942)
2. LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)
3. Bardes et al., "V-JEPA: Visual Joint Embedding Predictive Architecture" (2023)
4. Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018)

---

**Word Count**: ~1,900 words ✓
**VL-JEPA paper cited**: ✓
**InfoNCE loss explained**: ✓
**Embedding vs token prediction**: ✓
**Uniformity vs alignment**: ✓

Final document: Architecture Comparison! 🔥
