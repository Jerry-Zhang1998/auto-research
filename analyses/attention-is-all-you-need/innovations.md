---
paper: "Attention Is All You Need"
name: attention-is-all-you-need
analyzed: 2026-06-16
github: https://github.com/tensorflow/tensor2tensor
paper_type: Architecture
sections_detected:
  theoretical_analysis: true
  model_architecture: true
  loss_design: false
  training_strategy: true
  efficiency_analysis: true
  ablation_study: true
  implementation_notes: true
---

# Innovation Analysis: Attention Is All You Need

## Paper Profile

**Type**: Architecture
**Domain**: NLP
**Primary Contribution**: The Transformer — first sequence transduction model relying solely on self-attention, eliminating all recurrence and convolution

Sections present in this analysis:
- [x] Problem Statement
- [x] Theoretical Analysis
- [x] Core Contributions
- [x] Model Architecture
- [ ] Loss Design — not present (uses standard cross-entropy with label smoothing, no novel loss)
- [x] Training Strategy
- [x] Efficiency Analysis
- [x] Key Results
- [x] Ablation Study
- [x] Implementation Notes
- [x] Paper Significance

---

## 0. Repository

- **GitHub**: https://github.com/tensorflow/tensor2tensor
- **Status**: official (Google Brain / Google Research — original tensor2tensor implementation)
- **Notes**: The original repo is TensorFlow-based. PyTorch reproductions (e.g., harvard-nlp/annotated-transformer) are widely used for reference.

---

## 1. Problem Statement

### 1.1 Core Problem
"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder." Recurrent models (LSTM, GRU) factor computation along sequence positions, generating hidden states h_t as a function of h_{t-1} and the input at position t. This **inherently sequential computation precludes parallelization within training examples**, which becomes critical at longer sequence lengths as memory constraints limit batching.

### 1.2 Prior Art Limitations
- **RNN/LSTM bottleneck**: O(n) sequential operations per layer means training is fundamentally serial; gradient paths between distant positions traverse O(n) steps, making long-range dependency learning difficult
- **CNN alternatives (ByteNet, ConvS2S)**: parallelize computation but require O(log_k(n)) or O(n) operations to relate two distant positions — not constant
- **Attention + RNN hybrids**: attention mechanisms were used in conjunction with recurrent layers, not as standalone sequence models

### 1.3 Proposed Solution (one-line)
This paper proposes the Transformer, a model architecture relying entirely on multi-head self-attention to draw global dependencies between input and output, achieving O(1) sequential operations and O(1) path length between any two positions.

---

## Theoretical Analysis

### T.1 Theoretical Framework

The paper uses **computational complexity analysis** over sequence modeling operations as its theoretical framework. The setting is: sequences of length n, representation dimension d. Three layer types are compared along three dimensions: (1) total complexity per layer, (2) minimum sequential operations (parallelizability), and (3) maximum path length between any two positions (long-range learnability). The framework provides formal bounds that directly justify the architectural choices in the Transformer.

### T.2 Core Claim / Main Result

**Claim type**: informal claim (no labeled theorem)
**Proof status**: derivation from first principles — O(·) expressions stated with counting justification; no formal proof in the theorem-proving sense

Central claim (Section 4, Table 1):
> "A self-attention layer connects all positions with a constant number of sequentially-executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translation."

Secondary results:
- **Dot-product scaling** (footnote 4): For q, k with i.i.d. components of mean 0 and variance 1, q·k = Σᵢ qᵢkᵢ has mean 0 and variance d_k — hence dividing by √d_k restores unit variance and prevents softmax saturation.
- **Positional encoding relative-position property** (Section 3.5, implicit): PE(pos+k) is a linear function of PE(pos) for any fixed offset k, via trigonometric addition formulas. [stated as motivation; derivation not shown in paper]

### T.3 Key Derivation Steps

Starting point: sequence length n, representation dim d; count operations for one layer

   ↓  [counting attention operations]  each of n positions attends to all n positions; each dot product costs O(d)
Self-Attention per-layer: O(n² · d) total, O(1) sequential operations (all positions computed in parallel)

   ↓  [counting RNN operations]  each of n positions requires previous hidden state; d×d matrix multiply per step
Recurrent per-layer: O(n · d²) total, O(n) sequential operations (inherently serial)

   ↓  [algebraic comparison — n < d for standard NMT sentence encodings]
Self-attention is faster when n < d: O(n²·d) < O(n·d²) iff n < d
[standard algebraic comparison; n < d holds for BPE/word-piece encodings with typical n ≤ 512, d = 512]

Core result (Table 1):
```
Layer Type         Complexity/Layer   Sequential Ops   Max Path Length
Self-Attention     O(n²·d)            O(1)             O(1)
Recurrent          O(n·d²)            O(n)             O(n)
Convolutional      O(k·n·d²)          O(1)             O(log_k(n))
Self-Attn (restr.) O(r·n·d)           O(1)             O(n/r)
```

Dot-product scaling derivation (footnote 4):

Starting point: q, k ∈ ℝ^{d_k}, components q_i, k_i ~ i.i.d.(mean=0, var=1)

   ↓  [variance of sum of products]  Var(q_i·k_i) = 1, and terms are independent
Var(q·k) = Var(Σᵢ qᵢkᵢ) = d_k

   ↓  [normalization to unit variance]  divide by √d_k
Attention(Q,K,V) = softmax(QKᵀ / √d_k) V   (Eq. 1)
[Core result: scaling by 1/√d_k ensures softmax inputs have unit variance, preventing vanishing gradients]

LR schedule derivation (Eq. 6):

Starting point: need linear warmup then inverse-sqrt decay

   ↓  [min of two regimes — piecewise smooth]
lrate = d_model^{-0.5} · min(step_num^{-0.5}, step_num · warmup_steps^{-1.5})

   ↓  [inspecting crossover point]
Peak occurs when step_num = warmup_steps: lrate_peak = d_model^{-0.5} · warmup_steps^{-0.5}
With d_model=512, warmup_steps=4000: lrate_peak ≈ 7.07×10⁻⁴

### T.4 Assumptions & Conditions

| Assumption | Type | Impact if violated |
|------------|------|--------------------|
| n < d for sentence encodings (e.g., n ≤ 512, d = 512) | [restrictive] | Self-attention becomes computationally slower than recurrence; O(n²) memory becomes prohibitive for very long sequences |
| q, k components are i.i.d. mean-0 variance-1 | [implicit — reader must verify] | √d_k scaling may be suboptimal if embeddings are poorly initialized or layer-normed before projection |
| Attention weights are not all zero (learning meaningful attention) | [implicit — reader must verify] | O(1) path length holds structurally but long-range gradient flow may still vanish if attention collapses |
| Sinusoidal frequencies span the range of sequence positions seen at test time | [restrictive] | Extrapolation to sequence lengths longer than training length is not guaranteed |

### T.5 Theory → Design Connection

**1. Removing recurrence entirely** (the paper's defining architectural decision) is directly justified by the complexity result in T.3: self-attention achieves O(1) sequential operations vs O(n) for RNNs, enabling full parallelization of training. The O(1) maximum path length (vs O(n) for RNNs) formally reduces the gradient path between any two positions to a constant — directly addressing the vanishing gradient problem for long-range dependencies. Without the n < d assumption from T.4, this tradeoff would break down (e.g., at character level or for audio); the paper implicitly scopes its claim to NMT with BPE/word-piece encodings.

**2. The 1/√d_k scaling in Eq. (1)** is directly justified by the footnote 4 variance derivation (T.3): without scaling, dot products grow in magnitude as O(d_k), pushing softmax into regions with near-zero gradients. This motivated the specific choice of dividing by √d_k rather than d_k or a learned scalar. An implementer who omits this scaling will observe training instability at large d_k.

**3. Multi-head attention with h=8** is motivated indirectly by the T.3 finding: with a single attention head, all d_model dimensions are pooled together, which "averaging inhibits" attending to different representation subspaces. h heads each operating in d_k=64 dimensions enables the model to learn diverse attention patterns simultaneously. The T.4 assumption about n < d still holds at the per-head level (n ≤ 512, d_k=64 — this assumption is violated, but the paper notes that in practice the attention scores still work well).

**4. Sinusoidal positional encoding** is motivated by the relative-position linear combination property (T.3): since PE(pos+k) is a linear function of PE(pos), a single attention head can attend to relative positions by learning a linear combination of absolute PE vectors. This property is specific to sinusoidal encoding and would not hold for random or learned encodings. However, the ablation (Section 6.2) found sinusoidal and learned encodings perform similarly, so this theoretical motivation is partially post-hoc.

---

## 2. Core Contributions

- **Contribution**: The Transformer — fully attention-based sequence transduction
  **What it is**: First encoder-decoder architecture using only self-attention and cross-attention layers, with no recurrence (LSTM/GRU) and no convolution
  **Why it matters**: Enables full parallelization of training (O(1) sequential ops vs O(n) for RNNs); reduces maximum dependency path length from O(n) to O(1); achieves SOTA on WMT EN-DE and EN-FR at a fraction of prior training cost

- **Contribution**: Scaled Dot-Product Attention with theoretical motivation
  **What it is**: Attention(Q,K,V) = softmax(QKᵀ/√d_k)V — dividing by √d_k is derived from a variance argument (footnote 4), not ad-hoc
  **Why it matters**: Enables stable training at large d_k; makes dot-product attention competitive with additive attention while being much faster (matrix multiplication vs feed-forward)

- **Contribution**: Multi-Head Attention
  **What it is**: h=8 parallel attention heads each in d_k=d_v=64 dimensions, outputs concatenated and projected
  **Why it matters**: Single-head attention collapses diverse position-based patterns; multi-head allows simultaneous attention to different aspects (syntax, coreference, semantics) — confirmed by ablation showing h=1 drops BLEU ~0.9

- **Contribution**: Learned LR warmup schedule (Eq. 6)
  **What it is**: Linear increase for warmup_steps=4000 steps, then inverse-sqrt decay, scaled by d_model^{-0.5}
  **Why it matters**: Prior work used fixed LR or simple decay; this schedule is shown empirically to be critical and is now standard in Transformer training

---

## 3. Model Architecture

### 3.1 High-Level Design

Encoder-decoder architecture with stacked self-attention and point-wise FFN layers. Input tokens are embedded, summed with sinusoidal positional encodings, and processed through N=6 identical encoder layers. The decoder attends to encoder output via cross-attention and generates output tokens auto-regressively.

```
Input Tokens                  Output Tokens (shifted right)
     ↓                                ↓
  Embedding (× √d_model)         Embedding (× √d_model)
     + PE                              + PE
     ↓                                ↓
 ┌─────────────────┐           ┌────────────────────────┐
 │  Encoder Layer  │ × 6       │    Decoder Layer        │ × 6
 │                 │           │                         │
 │  MH-Self-Attn   │           │  Masked MH-Self-Attn    │
 │  Add & Norm     │           │  Add & Norm             │
 │  FFN            │           │  MH-Cross-Attn (→Enc)   │
 │  Add & Norm     │           │  Add & Norm             │
 └────────┬────────┘           │  FFN                    │
          │                    │  Add & Norm             │
          └──────────────────→ └───────────┬─────────────┘
                                           ↓
                                   Linear + Softmax
                                   Output Probabilities
```

### 3.2 Key Components

#### Token Embedding + Positional Encoding
- **Role**: Convert discrete tokens to continuous vectors with positional information
- **Design**: Learned embedding matrix E ∈ ℝ^{vocab × d_model}, weights multiplied by √d_model. Sinusoidal PE added element-wise:
  ```
  PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
  PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
  ```
- **Input/Output shape**: [B, T] tokens → [B, T, 512]
- **Innovation**: Weight sharing between encoder embedding, decoder embedding, and pre-softmax linear projection (W_E = W_decoder_out^T); not used in prior NMT models uniformly
- **Prior work**: Embedding lookup is standard; PE is new to this paper

#### Scaled Dot-Product Attention
- **Role**: Compute weighted sum of values given query-key compatibility
- **Design**:
  ```
  Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
  ```
  Queries Q ∈ ℝ^{n×d_k}, Keys K ∈ ℝ^{m×d_k}, Values V ∈ ℝ^{m×d_v}
  For causal (decoder self-attention): mask future positions with −∞ before softmax
- **Input/Output shape**: Q [B, n, d_k], K [B, m, d_k], V [B, m, d_v] → [B, n, d_v]
- **Innovation**: Scaling by 1/√d_k (theoretically motivated, see T.5); prior additive attention [Bahdanau 2014] used a single-layer FFN for compatibility

#### Multi-Head Attention
- **Role**: Attend to different representation subspaces simultaneously
- **Design**:
  ```
  MultiHead(Q, K, V) = Concat(head_1, ..., head_8) W^O
  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
  ```
  Projections: W_i^Q ∈ ℝ^{512×64}, W_i^K ∈ ℝ^{512×64}, W_i^V ∈ ℝ^{512×64}, W^O ∈ ℝ^{512×512}
  h=8 heads, d_k=d_v=64
- **Input/Output shape**: Q/K/V [B, T, 512] → [B, T, 512]
- **Innovation**: Parallel multi-head projection enables diverse attention patterns; h×d_k = d_model is required

#### Encoder Layer (×6)
- **Role**: Encode input sequence with full bidirectional attention
- **Design**: Two sub-layers with residual connections + **post-LayerNorm**:
  1. `x = LayerNorm(x + MultiHeadSelfAttn(x, x, x))`
  2. `x = LayerNorm(x + FFN(x))`
  Important: **post-norm** (normalize AFTER adding residual) — differs from pre-norm variants used in later work; this specific ordering affects training stability
- **Input/Output shape**: [B, T, 512] → [B, T, 512]

#### Decoder Layer (×6)
- **Role**: Generate output sequence with access to encoder context and past outputs
- **Design**: Three sub-layers:
  1. Masked MH self-attention (causal — prevent attending to future positions)
  2. MH cross-attention (Q from decoder, K/V from encoder output)
  3. FFN
  All with residual + post-LayerNorm
- **Input/Output shape**: decoder input [B, T', 512], encoder output [B, T, 512] → [B, T', 512]

#### Position-wise Feed-Forward Network
- **Role**: Non-linear transformation applied independently to each position
- **Design**:
  ```
  FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
  ```
  W_1 ∈ ℝ^{512×2048}, W_2 ∈ ℝ^{2048×512}; ReLU activation
- **Input/Output shape**: [B, T, 512] → [B, T, 512]
- **Prior work**: Standard two-layer FFN; the d_ff=2048 = 4×d_model ratio became a convention

### 3.3 Architecture Diagram (text)

```
Encoder (×6):                    Decoder (×6):
┌─────────────────────────┐      ┌────────────────────────────────┐
│ x: [B, T, 512]          │      │ y: [B, T', 512]                │
│         ↓               │      │         ↓                      │
│  MH-Self-Attn(x,x,x)    │      │  Masked-MH-Self-Attn(y,y,y)   │
│         ↓               │      │         ↓                      │
│  Add(x, _) + LayerNorm  │      │  Add(y, _) + LayerNorm         │
│         ↓               │      │         ↓                      │
│  FFN (512→2048→512)      │      │  MH-Cross-Attn(_, enc, enc)    │
│         ↓               │      │         ↓                      │
│  Add + LayerNorm         │      │  Add + LayerNorm               │
│         ↓               │      │         ↓                      │
│  out: [B, T, 512] ──────┼──→   │  FFN (512→2048→512)            │
└─────────────────────────┘      │         ↓                      │
                                 │  Add + LayerNorm               │
                                 │  out: [B, T', 512]             │
                                 └────────────────┬───────────────┘
                                                  ↓
                                  Linear [512→vocab] + Softmax
```

---

## 5. Training Strategy

| Aspect | Detail |
|--------|--------|
| Dataset (EN-DE) | WMT 2014 English-German, ~4.5M sentence pairs, BPE with 37K shared vocab |
| Dataset (EN-FR) | WMT 2014 English-French, 36M sentences, 32K word-piece vocab |
| Batching | ~25,000 source tokens + ~25,000 target tokens per batch (by approximate length) |
| Optimizer | Adam, β₁=0.9, β₂=0.98, ε=10⁻⁹ |
| Learning Rate | Eq. 6: d_model^{-0.5}·min(step^{-0.5}, step·warmup^{-1.5}), warmup_steps=4000 |
| LR Peak (base) | ~7.07×10⁻⁴ at step 4000 |
| Epochs / Steps | Base: 100K steps (~12 hours); Big: 300K steps (3.5 days) |
| Hardware | 8 NVIDIA P100 GPUs; base step: 0.4s; big step: 1.0s |
| Dropout (residual) | P_drop=0.1 applied to sub-layer output before Add & Norm |
| Dropout (embedding) | P_drop=0.1 applied to embedding+PE sums in both encoder and decoder |
| Label Smoothing | ε_ls=0.1 (degrades perplexity but improves BLEU and accuracy) |
| Inference | Beam search, beam size=4, length penalty α=0.6 |
| Checkpoint averaging | Base: average last 5 ckpts; Big: average last 20 ckpts (at 10-min intervals) |

---

## Efficiency Analysis

### E.1 Computational Complexity

| Layer Type | Complexity/Layer | Sequential Ops | Max Path Length |
|------------|-----------------|----------------|-----------------|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional (k) | O(k·n·d²) | O(1) | O(log_k(n)) |

Self-attention is faster when n < d. For BPE-encoded sentences: typical n ≤ 100–512, d = 512. Memory cost of attention matrix is O(n²) which becomes significant for n > 512.

### E.2 Parameter Count

Transformer base model: ~65M parameters (not stated explicitly; derived from architecture)
- Each encoder/decoder layer: 4 attention projection matrices (4×512×512=1M) + 2 FFN matrices (512×2048+2048×512≈2M) + LayerNorm params → ~3M/layer
- 12 layers total (6 enc + 6 dec) → ~36M
- Embeddings + softmax: vocab × 512 (shared, ~25M for 37K vocab)

### E.3 Training Cost (FLOPs)

| Model | EN-DE BLEU | Training FLOPs |
|-------|------------|----------------|
| ConvS2S | 25.16 | 9.6×10¹⁸ |
| GNMT+RL | 24.6 | 2.3×10¹⁹ |
| **Transformer (base)** | **27.3** | **3.3×10¹⁸** (3× less than ConvS2S!) |
| **Transformer (big)** | **28.4** | **2.3×10¹⁹** |

### E.4 Memory Footprint

Attention matrix per layer: O(n²·d) — for n=512, d=512, h=8: 512²×8 = 2M floats ≈ 8MB per layer per batch item. Not explicitly stated in the paper.

### E.5 Scalability

The Transformer (big) doubles most dimensions vs base (d_model=1024, h=16, d_ff=4096, P_drop=0.3) and adds 2 BLEU on EN-DE. Training cost scales 7× (3.3×10¹⁸ → 2.3×10¹⁹). No formal scaling law presented; Table 3 ablation shows roughly log-linear improvement with model size.

---

## 6. Key Results

### 6.1 Main Benchmarks

| Benchmark | Metric | Transformer (big) | Prior SOTA | Improvement |
|-----------|--------|-------------------|------------|-------------|
| WMT 2014 EN-DE | BLEU | **28.4** | 26.36 (ConvS2S Ensemble) | +2.04 BLEU |
| WMT 2014 EN-FR | BLEU | **41.8** | 41.29 (ConvS2S Ensemble) | +0.51 BLEU |
| WSJ Parsing | F1 | 91.3 | 91.21 (semi-supervised) | +0.09 |

### 6.2 Most Important Findings

- Transformer (big) outperforms ALL prior models including ensembles on EN-DE by >2 BLEU — first single model to do so
- Transformer (base) surpasses all prior single models at 10× lower training cost than ConvS2S (3.3×10¹⁸ vs 9.6×10¹⁸ FLOPs)
- EN-FR: 41.8 BLEU at <1/4 the training cost of the previous SOTA
- Strong generalization: 91.3 F1 on WSJ parsing with no task-specific tuning of architecture

### 6.3 Ablation Insights

Key findings from Table 3 ablation on EN-DE (newstest2013):

| Variation | BLEU | Insight |
|-----------|------|---------|
| Base model (h=8, d_k=64) | 25.8 | Baseline |
| h=1 (single head) | 24.9 | −0.9: multi-head attention is important |
| h=16 (more heads, d_k=32) | 25.5 | Diminishing returns; d_k=32 too small |
| d_k=16 | 24.7 | −1.1: **quality of compatibility function is critical** |
| Bigger (d_model=1024) | 26.4 | +0.6: model size helps |
| Dropout=0.0 | 25.3 | −0.5: dropout important for regularization |
| Learned PE | 25.7 | −0.1: sinusoidal PE works as well as learned |

Most surprising: reducing d_k from 64 to 16 hurts more than reducing h from 8 to 1 — the quality of the dot-product compatibility function per head matters more than the number of heads.

---

## 7. Implementation Notes

### 7.1 Critical Details

- **Post-norm, not pre-norm**: The paper uses LayerNorm(x + Sublayer(x)) — normalization AFTER residual addition. Many modern implementations switched to pre-norm (LayerNorm(x) then Sublayer) for more stable training, but this is a deviation from the paper.
- **Shared weight matrix**: Embedding matrix W_E is shared between encoder input embedding, decoder input embedding, AND decoder output linear projection. Weights multiplied by √d_model in embedding layers. Must use the same matrix or results will differ.
- **Causal masking**: Decoder self-attention uses additive masking of −∞ to future positions before softmax. Implemented as an upper-triangular mask; must be applied before (not after) softmax.
- **Adam β₂=0.98** (not 0.999): This non-standard β₂ is critical; with β₂=0.999 training will diverge with the warmup schedule.
- **Label smoothing with ε=0.1**: Applied to the target distribution; hurts perplexity metric but required for reported BLEU scores.
- **Dropout on residual paths**: Applied to sub-layer output BEFORE adding residual, not after LayerNorm. Also applied to embedding+PE sums.

### 7.2 Potential Pitfalls

- Using standard Adam (β₂=0.999) instead of β₂=0.98 will destabilize training with the warmup schedule
- Forgetting √d_model scaling on embedding outputs causes the embedding scale to mismatch PE scale
- Not sharing the embedding/output projection matrices will increase parameters and degrade performance vs the paper
- Pre-norm vs post-norm: if using modern pre-norm, omit the warmup (pre-norm is stable without it) — but then results differ from the paper

### 7.3 Hyperparameter Sensitivity

From ablation Table 3: d_k (quality of compatibility function) is the most sensitive hyperparameter. Dropout rate P_drop is important (removing it drops BLEU 0.5). Number of attention heads h matters but is less sensitive than d_k.

### 7.4 Reproduction Checklist

- [ ] Adam optimizer with β₁=0.9, β₂=0.98, ε=10⁻⁹ (not default β₂=0.999)
- [ ] LR schedule: d_model^{-0.5}·min(step^{-0.5}, step·4000^{-1.5}), peaks at step 4000
- [ ] Dropout P_drop=0.1 on sub-layer outputs AND embedding+PE sums
- [ ] Label smoothing ε_ls=0.1
- [ ] Shared weight matrix across encoder embedding, decoder embedding, decoder output projection
- [ ] Embedding outputs multiplied by √512 = 22.6
- [ ] Post-LayerNorm: LayerNorm(x + Sublayer(x)), not pre-norm
- [ ] Causal mask: −∞ to upper triangle of decoder self-attention score matrix
- [ ] Multi-head: h=8, d_k=d_v=64, W^O ∈ ℝ^{512×512}
- [ ] FFN: ReLU activation, d_ff=2048

---

## 8. Paper Significance

### 8.1 Why This Paper Matters

The Transformer displaced RNNs as the dominant sequence modeling architecture and became the foundation for essentially all large language models (BERT, GPT, T5, etc.). It demonstrated that recurrence is not necessary for sequence transduction — self-attention alone suffices and is superior in quality, training speed, and parallelizability. The architectural template (stacked attention + FFN + residuals + LayerNorm) has been applied to vision, audio, protein structure, code, and multimodal tasks with minimal modification.

### 8.2 Limitations Acknowledged

- O(n²) memory requirement makes attention expensive for very long sequences (images, audio, documents)
- Restricted to n < d regime for efficiency advantage over RNNs
- Authors note local attention (neighborhood of size r) as a direction for long-sequence efficiency

### 8.3 Follow-up Directions

- **Efficiency**: Sparse attention (Longformer, BigBird), linear attention (Performer), Flash Attention — addressing the O(n²) memory bottleneck
- **Pre-training**: BERT (bidirectional encoder), GPT (decoder-only) — showed unsupervised pre-training on Transformers yields massive gains
- **Scaling**: Scaling laws, GPT-3, PaLM — Transformer scales favorably with compute
- **Pre-norm**: Many subsequent works switched to pre-LayerNorm for training stability without warmup
