"""PyTorch Dataset for DST training pairs."""

import json
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class NLGDataset(Dataset):
    def __init__(self, data_path, enc_vocab, dec_vocab):
        with open(data_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        src = self.enc_vocab.encode(pair["input"])
        trg = (
            [self.dec_vocab.word2idx["<SOS>"]]
            + self.dec_vocab.encode(pair["target"])
            + [self.dec_vocab.word2idx["<EOS>"]]
        )

        return torch.tensor(src, dtype=torch.long), torch.tensor(trg, dtype=torch.long)


def collate_fn(batch):
    """Pad sequences in a batch."""
    srcs, trgs = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in srcs])
    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
    trgs_padded = pad_sequence(trgs, batch_first=True, padding_value=0)

    return srcs_padded, src_lengths, trgs_padded
