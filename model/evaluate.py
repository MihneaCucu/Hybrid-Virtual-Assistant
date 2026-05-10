"""Evaluation utilities — BLEU score for NLG."""

import math
import torch
from collections import Counter


def decode_response(output_indices, vocab):
    """Decode model output indices to a response string."""
    tokens = []
    for idx in output_indices:
        word = vocab.idx2word.get(idx, "<UNK>")
        if word == "<EOS>":
            break
        if word not in ("<PAD>", "<SOS>"):
            tokens.append(word)
    return tokens


def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _bleu_precision(hypothesis, reference, n):
    hyp_counts = _ngram_counts(hypothesis, n)
    ref_counts = _ngram_counts(reference, n)
    clipped = sum(min(c, ref_counts[ng]) for ng, c in hyp_counts.items())
    total = max(len(hypothesis) - n + 1, 0)
    return clipped, total


def sentence_bleu(hypothesis, reference, max_n=4):
    """Compute sentence-level BLEU-4 with smoothing."""
    if not hypothesis:
        return 0.0

    bp = min(1.0, math.exp(1 - len(reference) / len(hypothesis))) if hypothesis else 0.0

    log_avg = 0.0
    for n in range(1, max_n + 1):
        clipped, total = _bleu_precision(hypothesis, reference, n)
        # Add-one smoothing for higher-order n-grams
        precision = (clipped + 1) / (total + 1)
        log_avg += math.log(precision) / max_n

    return bp * math.exp(log_avg)


def compute_bleu(model, dataloader, dec_vocab, device, max_len=200):
    """Compute corpus BLEU score on a dataset."""
    model.eval()
    scores = []

    with torch.no_grad():
        for src, src_lengths, trg in dataloader:
            src, src_lengths, trg = src.to(device), src_lengths.to(device), trg.to(device)

            encoder_outputs, hidden, cell = model.encoder(src, src_lengths)
            batch_size = src.shape[0]

            input_token = torch.full((batch_size,), dec_vocab.word2idx["<SOS>"], device=device)
            predictions = []

            for _ in range(max_len):
                pred, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
                top1 = pred.argmax(1)
                predictions.append(top1)
                input_token = top1

            predictions = torch.stack(predictions, dim=1)

            for i in range(batch_size):
                hyp = decode_response(predictions[i].tolist(), dec_vocab)
                ref = decode_response(trg[i, 1:].tolist(), dec_vocab)
                scores.append(sentence_bleu(hyp, ref))

    return sum(scores) / len(scores) if scores else 0.0
