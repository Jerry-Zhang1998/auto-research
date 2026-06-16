"""
WMT 2014 EN-DE translation dataset loader.

Expected directory layout (relative to project root):
    datasets/wmt14-en-de/processed/
        vocab.json          — {"<pad>":0, "<bos>":1, "<eos>":2, "the":3, ...}
        train.src           — tokenized source, one sentence per line (space-separated tokens)
        train.tgt           — tokenized target, one sentence per line
        val.src / val.tgt
        test.src / test.tgt

BPE pre-processing (run before training):
    pip install sentencepiece
    spm_train --input=train.en,train.de --model_prefix=bpe --vocab_size=37000
              --character_coverage=1.0 --model_type=bpe
    spm_encode --model=bpe.model --output_format=piece < train.en > train.src
    (then build vocab.json from the SentencePiece vocabulary)

The loader pads sequences to the same length within each batch using pad_token_id=0.
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict

from config import TrainConfig

# ── Project root — datasets/ lives 3 levels above this file ───────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))


class Vocabulary:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, token2id: Dict[str, int]):
        self.token2id = token2id
        self.id2token = {v: k for k, v in token2id.items()}

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def encode(self, tokens: List[str], add_bos: bool = False, add_eos: bool = True) -> List[int]:
        ids = [self.token2id.get(t, self.UNK) for t in tokens]
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(
            self.id2token.get(i, "<unk>")
            for i in ids
            if i not in (self.PAD, self.BOS, self.EOS)
        )

    def __len__(self) -> int:
        return len(self.token2id)


class WMT14Dataset(Dataset):
    """
    Reads pre-tokenized WMT 2014 EN-DE parallel text.
    Returns (src_ids, tgt_ids) as LongTensors without padding
    (collate_fn handles padding per batch).
    """

    def __init__(
        self,
        config: TrainConfig,
        vocab: Vocabulary,
        split: str = "train",
    ):
        self.config  = config
        self.vocab   = vocab
        self.max_src = config.max_src_len
        self.max_tgt = config.max_tgt_len

        data_dir  = os.path.join(_PROJ_ROOT, "datasets", config.dataset_name, "processed")
        src_path  = os.path.join(data_dir, f"{split}.src")
        tgt_path  = os.path.join(data_dir, f"{split}.tgt")

        if not os.path.exists(src_path):
            raise FileNotFoundError(
                f"Dataset not found at {src_path}\n"
                "Run BPE pre-processing and place tokenized files in "
                f"datasets/{config.dataset_name}/processed/"
            )

        self.src_sentences: List[List[str]] = []
        self.tgt_sentences: List[List[str]] = []

        with open(src_path, "r", encoding="utf-8") as sf, \
             open(tgt_path, "r", encoding="utf-8") as tf:
            for s_line, t_line in zip(sf, tf):
                s_toks = s_line.strip().split()
                t_toks = t_line.strip().split()
                if not s_toks or not t_toks:
                    continue
                # Truncate (leave room for BOS/EOS)
                self.src_sentences.append(s_toks[: self.max_src - 2])
                self.tgt_sentences.append(t_toks[: self.max_tgt - 2])

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids = self.vocab.encode(self.src_sentences[idx], add_bos=False, add_eos=True)
        tgt_ids = self.vocab.encode(self.tgt_sentences[idx], add_bos=True,  add_eos=True)
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad (src, tgt) pairs to the longest sequence in the batch."""
    srcs, tgts = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_id)
    return src_padded, tgt_padded   # [B, max_src_len], [B, max_tgt_len]


def load_vocab(dataset_name: str) -> Optional[Vocabulary]:
    vocab_path = os.path.join(_PROJ_ROOT, "datasets", dataset_name, "processed", "vocab.json")
    if os.path.exists(vocab_path):
        return Vocabulary.load(vocab_path)
    return None


def get_dataloader(config: TrainConfig, split: str = "train") -> DataLoader:
    vocab = load_vocab(config.dataset_name)
    if vocab is None:
        raise FileNotFoundError(
            f"Vocabulary not found at datasets/{config.dataset_name}/processed/vocab.json\n"
            "Build a shared BPE vocabulary first (37K tokens for WMT EN-DE)."
        )
    dataset = WMT14Dataset(config, vocab, split=split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id=0),
    )
