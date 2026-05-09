"""Top-level Dialogue Manager — NLG interface.

Receives intent (from S1) and KB results (from S3), generates a natural
language response using the trained NLG seq2seq model.

Usage:
    dm = DialogueManager(model=nlg_model, enc_vocab=enc_v, dec_vocab=dec_v, device=device)
    response = dm.process_turn(
        user_text="I want Italian food in the centre",
        intent="takeaway_query",
        kb_results={"food": "italian", "area": "centre", "name": "Pizza Hut", "pricerange": "cheap"}
    )
"""

import torch


MAX_HISTORY = 4
MAX_DECODE_LEN = 200


class DialogueManager:
    def __init__(self, model, enc_vocab, dec_vocab, device):
        """
        Args:
            model:     Trained NLG DSTSeq2Seq model.
            enc_vocab: Encoder vocabulary (enc_vocab.json).
            dec_vocab: Decoder vocabulary (dec_vocab.json).
            device:    torch device.
        """
        self.model = model
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab
        self.device = device
        self.history = []

    def process_turn(self, user_text, intent, kb_results=None):
        """Generate a response for one dialogue turn.

        Args:
            user_text:  Raw user utterance.
            intent:     MASSIVE intent string from S1 (e.g. "takeaway_query").
            kb_results: Dict of slot->value pairs from S3, or None.

        Returns:
            str: Generated system response.
        """
        self.history.append(f"[USR] {user_text.strip()}")

        kb_text = self._format_kb(kb_results)
        context = self.history[-MAX_HISTORY:]
        input_text = " ".join(context) + f" [INTENT] {intent} [KB] {kb_text}"

        response = self._run_nlg(input_text)

        self.history.append(f"[SYS] {response}")
        return response

    def _format_kb(self, kb_results):
        if not kb_results:
            return "none"
        parts = [f"{k}={v}" for k, v in kb_results.items() if v]
        return " ".join(parts) if parts else "none"

    def _run_nlg(self, input_text):
        src_indices = self.enc_vocab.encode(input_text)
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        src_lengths = torch.tensor([len(src_indices)]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.model.encoder(src_tensor, src_lengths)
            input_token = torch.full(
                (1,), self.dec_vocab.word2idx["<SOS>"], device=self.device
            )
            tokens = []
            for _ in range(MAX_DECODE_LEN):
                pred, hidden, cell, _ = self.model.decoder(
                    input_token, hidden, cell, encoder_outputs
                )
                top1 = pred.argmax(1)
                word = self.dec_vocab.idx2word.get(top1.item(), "<UNK>")
                if word == "<EOS>":
                    break
                tokens.append(word)
                input_token = top1

        return " ".join(tokens) if tokens else "I'm not sure how to help with that."

    def reset(self):
        self.history = []
