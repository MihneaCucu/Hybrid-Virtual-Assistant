"""Training script for the NLG seq2seq model."""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.vocab import Vocabulary
from model.seq2seq import Encoder, Decoder, Seq2Seq
from model.dataset import NLGDataset, collate_fn
from model.evaluate import compute_bleu

# Hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
TEACHER_FORCING = 0.5
CLIP = 1.0
LOG_EVERY = 50  

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
LOG_PATH = os.path.join(SAVE_DIR, "training_log.csv")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    enc_vocab = Vocabulary()
    dec_vocab = Vocabulary()
    enc_vocab.load(os.path.join(DATA_DIR, "enc_vocab.json"))
    dec_vocab.load(os.path.join(DATA_DIR, "dec_vocab.json"))

    train_dataset = NLGDataset(os.path.join(DATA_DIR, "train.json"), enc_vocab, dec_vocab)
    val_dataset = NLGDataset(os.path.join(DATA_DIR, "val.json"), enc_vocab, dec_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(len(enc_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(dec_vocab), EMBED_DIM, HIDDEN_DIM, HIDDEN_DIM, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_bleu = 0.0
    num_batches = len(train_loader)

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_bleu", "best_bleu"])

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (src, src_lengths, trg) in enumerate(train_loader, 1):
            src, src_lengths, trg = src.to(device), src_lengths.to(device), trg.to(device)

            output = model(src, src_lengths, trg, TEACHER_FORCING)
            output = output[:, 1:].contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % LOG_EVERY == 0 or batch_idx == num_batches:
                avg_so_far = total_loss / batch_idx
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{num_batches} | Loss: {avg_so_far:.4f}")

        avg_loss = total_loss / num_batches
        bleu = compute_bleu(model, val_loader, dec_vocab, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Val BLEU: {bleu:.4f}")

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  Saved best model (BLEU={best_bleu:.4f})")

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_loss:.4f}", f"{bleu:.4f}", f"{best_bleu:.4f}"])

    print(f"\nTraining complete. Best BLEU: {best_bleu:.4f}")
    print(f"Log saved to {LOG_PATH}")


if __name__ == "__main__":
    train()
