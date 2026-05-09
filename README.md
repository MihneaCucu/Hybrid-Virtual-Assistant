# Dialogue Manager

Dialogue Manager component of task-oriented dialogue system for city tourism.


## Supported Intents

| Intent | Example response |
|---|---|
| `recommendation_locations` | "The Botanical Garden is a park in the west of the city." |
| `takeaway_query` | "There are several Italian restaurants in the centre." |
| `takeaway_order` | "I have booked Pizza Hut for 2 people at 7pm on Thursday." |
| `calendar_set` / `calendar_query` | "Your reservation is confirmed. Reference: ABC123." |
| `transport_taxi` / `transport_query` | "I have arranged a blue Toyota taxi for you." |
| `transport_ticket` | "There is a train from Cambridge to London at 09:00, costing 23.60 pounds." |
| `qa_factoid` | "The City Museum is located in the centre area." |
| `general_greet` / `general_quirky` | "Hello! How can I help?" / "Is there anything else I can help with?" |

## Setup

```bash
pip install torch numpy
python -m data.download          # download MultiWOZ 2.4
python -m data.preprocess_nlg    # build training pairs
python -m data.vocab             # build vocabularies
```

## Training

```bash
python -m model.train
```

## Inference — random validation examples

```bash
python -m inference          # 10 examples (default)
python -m inference --n 20   # 20 examples
```

## Integration

```python
import torch
from data.vocab import Vocabulary
from model.seq2seq import Encoder, Decoder, Seq2Seq
from model.train import EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
from manager import DialogueManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc_vocab = Vocabulary()
dec_vocab = Vocabulary()
enc_vocab.load("data/processed/enc_vocab.json")
dec_vocab.load("data/processed/dec_vocab.json")

encoder = Encoder(len(enc_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(len(dec_vocab), EMBED_DIM, HIDDEN_DIM, HIDDEN_DIM, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))

dm = DialogueManager(model=model, enc_vocab=enc_vocab, dec_vocab=dec_vocab, device=device)

# Each turn: pass user text + intent from S1 + KB results from S3
response = dm.process_turn(
    user_text="I want a cheap Italian restaurant in the centre",
    intent="takeaway_query",
    kb_results={"name": "Pizza Hut", "food": "italian", "area": "centre", "pricerange": "cheap"}
)
print(response)

dm.reset()  # start a new conversation
```

## Model

| Component | Details |

| Architecture | BiLSTM Encoder + LSTM Decoder |
| Encoder vocab | 4,429 tokens |
| Decoder vocab | 5,734 tokens |
| Hidden dim | 512 |
| Parameters | ~18.7M |
| Training data | MultiWOZ 2.4 — 7,708 examples across 11 intents |
| Best val BLEU | 0.1320 (epoch 8) |
