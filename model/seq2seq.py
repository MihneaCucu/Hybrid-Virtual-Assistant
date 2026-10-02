"""LSTM Seq2Seq with Attention for Natural Language Generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Combine bidirectional states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell = torch.cat((cell[-2], cell[-1]), dim=1)
        hidden = torch.tanh(self.fc_h(hidden)).unsqueeze(0)
        cell = torch.tanh(self.fc_c(cell)).unsqueeze(0)

        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        self.lstm = nn.LSTM(
            embed_dim + enc_hidden_dim * 2, dec_hidden_dim,
            num_layers=1, batch_first=True,
        )
        self.fc_out = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))

        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=2).squeeze(1))

        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        input_token = trg[:, 0]  # <SOS>

        for t in range(1, trg_len):
            prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = prediction

            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = trg[:, t]
            else:
                input_token = prediction.argmax(1)

        return outputs
