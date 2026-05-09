"""Run inference on random validation examples and compare to references.

Usage: python -m inference [--n 10]
"""

import os
import sys
import json
import random
import argparse
import torch

from data.vocab import Vocabulary
from model.seq2seq import Encoder, Decoder, Seq2Seq
from model.train import EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
from model.evaluate import decode_response

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")
MAX_DECODE_LEN = 200


def load_model(device):
    enc_vocab = Vocabulary()
    dec_vocab = Vocabulary()
    enc_vocab.load(os.path.join(DATA_DIR, "enc_vocab.json"))
    dec_vocab.load(os.path.join(DATA_DIR, "dec_vocab.json"))

    encoder = Encoder(len(enc_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(dec_vocab), EMBED_DIM, HIDDEN_DIM, HIDDEN_DIM, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    return model, enc_vocab, dec_vocab


def run_inference(model, enc_vocab, dec_vocab, device, input_text):
    src = torch.tensor([enc_vocab.encode(input_text)], dtype=torch.long).to(device)
    src_lengths = torch.tensor([src.shape[1]]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src, src_lengths)
        input_token = torch.full((1,), dec_vocab.word2idx["<SOS>"], device=device)
        predictions = []
        for _ in range(MAX_DECODE_LEN):
            pred, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = pred.argmax(1)
            word = dec_vocab.idx2word.get(top1.item(), "<UNK>")
            if word == "<EOS>":
                break
            predictions.append(top1.item())
            input_token = top1

    return " ".join(decode_response(predictions, dec_vocab))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model, enc_vocab, dec_vocab, = load_model(device)

    with open(os.path.join(DATA_DIR, "val.json"), encoding="utf-8") as f:
        val_data = json.load(f)

    examples = random.sample(val_data, min(args.n, len(val_data)))

    print(f"\nRunning inference on {len(examples)} random validation examples\n")
    print("=" * 70)

    for i, ex in enumerate(examples, 1):
        predicted = run_inference(model, enc_vocab, dec_vocab, device, ex["input"])

        print(f"[{i}/{len(examples)}]")
        print(f"Intent:    {ex['intent']}")
        print(f"Input:     {ex['input']}")
        print(f"Reference: {ex['target']}")
        print(f"Predicted: {predicted}")
        print("-" * 70)


if __name__ == "__main__":
    main()
