from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Model dimension (Section 3, Table 2 — base model)
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048            # 4 × d_model (Section 3.3)
    dropout: float = 0.1        # P_drop (Section 5.4, applied to sub-layers + embeddings)
    max_seq_len: int = 512

    # Vocabulary — WMT 2014 EN-DE BPE shared vocab (Section 5.2)
    src_vocab_size: int = 37_000
    tgt_vocab_size: int = 37_000
    share_embeddings: bool = True   # encoder embed = decoder embed = output proj (Section 3.4)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class TrainConfig:
    max_epochs: int = 100       # base model ≈ 100K steps on 8×P100 (Section 5.3)
    batch_size: int = 32        # ~25K src+tgt tokens/batch; 32 seqs is a proxy
    lr: float = 0.0             # set dynamically by Noam schedule (Eq. 6)
    weight_decay: float = 0.0

    # Adam params (Section 5.3) — β₂=0.98 and ε=1e-9 are NON-STANDARD and critical
    beta1: float = 0.9
    beta2: float = 0.98         # paper: diverges with default β₂=0.999 + warmup
    eps: float = 1e-9

    warmup_steps: int = 4000    # Eq. 6: peak LR at step 4000
    label_smoothing: float = 0.1    # ε_ls (Section 5.4)
    grad_clip: float = 1.0          # not specified in paper, standard default
    log_every: int = 100

    # Dataset
    dataset_name: str = "wmt14-en-de"
    max_src_len: int = 128          # truncation for faster training; paper uses ~512
    max_tgt_len: int = 128


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
