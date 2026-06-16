# Attention Is All You Need — Reproduction

**Paper**: Vaswani et al. (2017) — https://arxiv.org/abs/1706.03762
**Analysis**: analyses/attention-is-all-you-need/innovations.md
**Official code**: https://github.com/tensorflow/tensor2tensor (TensorFlow — not adapted here)

## Quick Start

```bash
# 1. Install dependencies
pip install torch sentencepiece sacrebleu

# 2. Prepare WMT 2014 EN-DE dataset with BPE tokenisation
#    (37K shared vocabulary, as in the paper)
#    Place tokenised files in:
#      datasets/wmt14-en-de/processed/train.src  train.tgt
#      datasets/wmt14-en-de/processed/val.src    val.tgt
#      datasets/wmt14-en-de/processed/test.src   test.tgt
#      datasets/wmt14-en-de/processed/vocab.json  (token → ID mapping)

# 3. Train (base model)
cd outputs/attention-is-all-you-need/reproduction
python train.py --run-name base_run

# 4. Evaluate perplexity on test split
python test.py --run-name base_run

# 5. Visualise training curves
python3 ../../../scripts/generate_viz.py \
    --log-dir ../../../logs/attention-is-all-you-need/base_run \
    --output-dir ../html
```

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters (base model values from paper Table 2) |
| `model.py` | Transformer: sinusoidal PE, scaled dot-product attention, MH-attention, FFN, encoder, decoder |
| `loss.py` | Label-smoothing cross-entropy (ε_ls=0.1, Section 5.4) |
| `dataset.py` | WMT14 EN-DE parallel corpus loader; `collate_fn` for variable-length batching |
| `train.py` | `TransformerTrainer(BaseTrainer)` with Noam LR schedule (Eq. 6) |
| `test.py` | Perplexity + token accuracy evaluation; saves `test_results.json` |

## Architecture (Base Model)

- **Encoder**: 6 × (Multi-head self-attention + FFN), d_model=512, h=8, d_ff=2048
- **Decoder**: 6 × (Masked self-attn + Cross-attn + FFN)
- **Positional encoding**: sinusoidal, fixed
- **Shared weights**: encoder embed = decoder embed = output projection
- **Post-LayerNorm**: `x = LayerNorm(x + Dropout(Sublayer(x)))` — paper-faithful (not pre-norm)
- **~65M parameters**

## Training Details

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| d_model | 512 | Table 2 |
| n_heads | 8 | Table 2 |
| d_ff | 2048 | Table 2 |
| N (layers) | 6 | Table 2 |
| d_k = d_v | 64 | Table 2 |
| optimizer | Adam | Section 5.3 |
| β₁, β₂, ε | 0.9, **0.98**, **1e-9** | Section 5.3 |
| warmup_steps | 4000 | Eq. 6 |
| dropout | 0.1 | Section 5.4 |
| label_smoothing | 0.1 | Section 5.4 |

## Logs

Each run writes to `logs/attention-is-all-you-need/{run_name}/`:
```
config.json        — config snapshot
metrics.jsonl      — per-step metrics (train_loss, val_loss, lr)
train.log          — human-readable log
metrics.csv        — exported at training end
ckpt_best.pt       — best validation checkpoint
ckpt_latest.pt     — most recent checkpoint
test_results.json  — filled after running test.py
```

## Key Implementation Decisions

- **Noam LR schedule** is per-step (not per-epoch); `optimizer.lr` is set directly in `train_step`
- **β₂=0.98** (not default 0.999) — critical for training stability with warmup (Section 5.3)
- **Post-LayerNorm** — matches the paper exactly; pre-norm (common in later work) would differ
- **Weight tying** — `tgt_embed.weight = src_embed.weight = output_proj.weight` reduces parameters by ~25M for a 37K vocab
- Gradient clipping at 1.0 — not specified in paper but standard practice

## Expected Results

| Model | WMT EN-DE BLEU | WMT EN-FR BLEU |
|-------|---------------|---------------|
| Base (paper) | 27.3 | 38.1 |
| Big (paper) | **28.4** | **41.8** |

## Known Gaps

- **BLEU evaluation** requires beam search decoding (beam=4, α=0.6); not included (only perplexity)
- **Token batching** — paper batches by ~25K source tokens per batch; this reproduction uses fixed batch_size=32 sequences
- **Checkpoint averaging** — paper averages last 5/20 checkpoints; not implemented
- Dataset requires external BPE preprocessing with SentencePiece
