"""Vocabulary builder for the seq2seq DST model."""

import json
import os
from collections import Counter

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN,
                  "[USR]", "[SYS]", "[INTENT]", "[KB]"]


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()

        for token in SPECIAL_TOKENS:
            self._add_word(token)

    def _add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_from_texts(self, texts, min_freq=1):
        """Build vocabulary from a list of texts."""
        for text in texts:
            for word in text.lower().split():
                self.word_count[word] += 1

        for word, count in self.word_count.items():
            if count >= min_freq:
                self._add_word(word)

    def encode(self, text):
        """Convert text to list of indices."""
        return [
            self.word2idx.get(w, self.word2idx[UNK_TOKEN])
            for w in text.lower().split()
        ]

    def decode(self, indices):
        """Convert list of indices to text."""
        return " ".join(
            self.idx2word.get(idx, UNK_TOKEN) for idx in indices
            if idx != self.word2idx[PAD_TOKEN]
        )

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}


def build_vocabs(min_freq=2):
    """Build encoder and decoder vocabularies from NLG training data."""
    train_path = os.path.join(PROCESSED_DIR, "train.json")
    with open(train_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    enc_vocab = Vocabulary()
    dec_vocab = Vocabulary()

    enc_vocab.build_from_texts([p["input"] for p in pairs], min_freq=min_freq)
    dec_vocab.build_from_texts([p["target"] for p in pairs], min_freq=1)

    enc_vocab.save(os.path.join(PROCESSED_DIR, "enc_vocab.json"))
    dec_vocab.save(os.path.join(PROCESSED_DIR, "dec_vocab.json"))

    print(f"Encoder vocab: {len(enc_vocab)} tokens")
    print(f"Decoder vocab: {len(dec_vocab)} tokens")

    return enc_vocab, dec_vocab


if __name__ == "__main__":
    build_vocabs()
