import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer
from torchcrf import CRF

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class JointModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            max_position_embeddings=256,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1
        )
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.slot_classifier = nn.Linear(hidden_size, num_slots)
        self.crf = CRF(num_slots, batch_first=True)
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

    def forward(self, input_ids, attention_mask, labels=None, intent=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state

        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        pooled_output = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        sequence_output = outputs.last_hidden_state
        sequence_output, _ = self.bilstm(sequence_output)
        pooled_output = self.dropout(pooled_output)

        slot_logits = self.slot_classifier(sequence_output)
        intent_logits = self.intent_classifier(pooled_output)

        loss = None

        if labels is not None:
            slot_mask = attention_mask.bool()
            crf_labels = labels.clone()
            crf_labels[~slot_mask] = 0

            slot_loss = -self.crf(
                slot_logits,
                crf_labels,
                mask=slot_mask,
                reduction='mean'
            )

            intent_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(
                intent_logits,
                intent
            )

            loss = 0.5 * slot_loss + 2.0 * intent_loss

        slot_predictions = self.crf.decode(
            slot_logits,
            mask=attention_mask.bool()
        )

        return loss, slot_predictions, intent_logits
