---
title: "Attention Is All You Need"
authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin"
year: 2017
arxiv_id: "1706.03762"
venue: "NeurIPS 2017"
pdf: "papers/attention-is-all-you-need.pdf"
arch_figure:
  file: "analyses/attention-is-all-you-need/figures/fig_001.png"
  caption: "Figure 1: The Transformer - model architecture"
---

# Attention Is All You Need

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

## 1 Introduction

Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures.

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states h_t, as a function of the previous hidden state h_{t-1} and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

## 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations.

To the best of our knowledge, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.

## 3 Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure. The encoder maps an input sequence of symbol representations (x_1,...,x_n) to a sequence of continuous representations z = (z_1,...,z_n). Given z, the decoder then generates an output sequence (y_1,...,y_m) of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

### 3.1 Encoder and Decoder Stacks

**Encoder:** The encoder is composed of a stack of N=6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)). All sub-layers in the model produce outputs of dimension d_model = 512.

**Decoder:** The decoder is also composed of a stack of N=6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer which performs multi-head attention over the output of the encoder stack. We modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions (masking).

### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

**3.2.1 Scaled Dot-Product Attention:**

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V          (Eq. 1)

The input consists of queries and keys of dimension d_k, and values of dimension d_v. We scale the dot products by 1/sqrt(d_k) to counteract the effect of large magnitudes pushing softmax into regions with extremely small gradients.

**3.2.2 Multi-Head Attention:**

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O      (Eq. 2)
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

With h=8 parallel attention heads, d_k = d_v = d_model/h = 64. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

**3.2.3 Applications of Attention in the Model:**
- Encoder-decoder attention: queries from decoder, keys/values from encoder
- Encoder self-attention: all keys, values, queries from previous encoder layer
- Decoder self-attention: masked to prevent leftward information flow (causal masking)

### 3.3 Position-wise Feed-Forward Networks

Each layer contains a fully connected FFN applied to each position separately:

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2                   (Eq. 3)

Inner-layer dimensionality d_ff = 2048.

### 3.4 Embeddings and Softmax

Learned embeddings convert tokens to vectors of dimension d_model=512. The same weight matrix is shared between the two embedding layers and the pre-softmax linear transformation. In the embedding layers, weights are multiplied by sqrt(d_model).

### 3.5 Positional Encoding

Since the model contains no recurrence or convolution, positional encodings are added to the input embeddings:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))             (Eq. 4)
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))             (Eq. 5)

## 4 Why Self-Attention

Comparison of self-attention layers vs recurrent and convolutional layers:

| Layer Type | Complexity per Layer | Sequential Ops | Max Path Length |
|------------|---------------------|----------------|-----------------|
| Self-Attention | O(n^2 · d) | O(1) | O(1) |
| Recurrent | O(n · d^2) | O(n) | O(n) |
| Convolutional | O(k · n · d^2) | O(1) | O(log_k(n)) |
| Self-Attention (restricted) | O(r · n · d) | O(1) | O(n/r) |

Self-attention connects all positions with O(1) sequential operations and O(1) maximum path length, making it easier to learn long-range dependencies.

## 5 Training

### 5.1 Training Data and Batching
- WMT 2014 English-German: ~4.5 million sentence pairs, BPE with 37,000 shared vocabulary
- WMT 2014 English-French: 36M sentences, 32,000 word-piece vocabulary
- Batching by approximate sequence length: ~25,000 source + 25,000 target tokens per batch

### 5.2 Hardware and Schedule
- 8 NVIDIA P100 GPUs
- Base models: ~0.4s/step, trained 100,000 steps (~12 hours)
- Big models: ~1.0s/step, trained 300,000 steps (3.5 days)

### 5.3 Optimizer
Adam optimizer with β₁=0.9, β₂=0.98, ε=10⁻⁹

Learning rate schedule:
    lrate = d_model^(-0.5) · min(step_num^(-0.5), step_num · warmup_steps^(-1.5))   (Eq. 6)

warmup_steps = 4000 (linear warmup then inverse square root decay)

### 5.4 Regularization
- **Residual Dropout**: P_drop = 0.1 applied to sub-layer output before add & norm; also to embedding+PE sums
- **Label Smoothing**: ε_ls = 0.1 (hurts perplexity but improves accuracy and BLEU)

## 6 Results

### 6.1 Machine Translation (Table 2)

| Model | EN-DE BLEU | EN-FR BLEU | Training FLOPs (EN-DE) |
|-------|------------|------------|------------------------|
| ByteNet | 23.75 | — | — |
| GNMT+RL | 24.6 | 39.92 | 2.3×10¹⁹ |
| ConvS2S | 25.16 | 40.46 | 9.6×10¹⁸ |
| MoE | 26.03 | 40.56 | 2.0×10¹⁹ |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7×10¹⁹ |
| **Transformer (base)** | **27.3** | **38.1** | **3.3×10¹⁸** |
| **Transformer (big)** | **28.4** | **41.8** | **2.3×10¹⁹** |

### 6.2 Model Variations (Ablation — Table 3)

Key findings from ablation study on EN-DE translation:
- Reducing attention heads (from 8 to 1) drops BLEU by ~0.9
- Reducing d_k from 64 to 16 drops BLEU by ~1.1 (quality of compatibility function matters)
- Bigger models perform better; dropout is important for regularization
- Sinusoidal positional encoding performs similarly to learned positional embeddings

### 6.3 English Constituency Parsing

Transformer (4 layers, d_model=1024) achieves 91.3 F1 on WSJ Section 23, outperforming or matching RNN grammar and LSTM models, demonstrating strong generalization to non-translation tasks.

## 7 Conclusion

The Transformer is the first sequence transduction model based entirely on attention, replacing recurrent layers with multi-headed self-attention. On WMT 2014 EN-DE and EN-FR tasks, it achieves new state-of-the-art BLEU scores. The code is available at: https://github.com/tensorflow/tensor2tensor
