from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup
import preprocessing
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
import os
import json
import NLU_model
from seqeval.metrics import f1_score

dataset = load_from_disk("massive_processed")
_, tokenizer, intent_names, unique_slots, label2id, id2label = preprocessing.get_attributes()
num_intents = len(intent_names)
num_slots = len(label2id)

nr_epochs = 10
data_collator = DataCollatorForTokenClassification(tokenizer)

train_loader = DataLoader(
    dataset["train"],
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator
)

val_loader = DataLoader(
    dataset["validation"],
    batch_size=16,
    collate_fn=data_collator
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NLU_model.JointModel(
    num_intents=num_intents,
    num_slots=num_slots
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)

num_training_steps = len(train_loader) * nr_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

for epoch in range(nr_epochs):

    #training
    model.train()

    total_loss = 0

    for batch in train_loader:

        batch = {k: v.to(device) for k, v in batch.items()}

        loss, _, _ = model(**batch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    #validation
    model.eval()

    intent_correct = 0
    intent_total = 0

    joint_correct = 0
    joint_total = 0

    true_slots = []
    pred_slots = []

    with torch.no_grad():

        for batch in val_loader:

            batch = {k: v.to(device) for k, v in batch.items()}

            loss, slot_preds, intent_logits = model(**batch)

            #predicting intents
            intent_preds = intent_logits.argmax(dim=1)

            intent_correct += (
                intent_preds == batch["intent"]
            ).sum().item()

            intent_total += len(intent_preds)

            #predicting slots
            labels = batch["labels"]

            for i in range(len(slot_preds)):

                pred_seq = slot_preds[i]
                true_seq = labels[i].cpu().tolist()

                true_labels = []
                pred_labels = []

                pred_idx = 0
                tokens = tokenizer.convert_ids_to_tokens(
                    batch["input_ids"][i]
                )

                for j in range(len(tokens)):

                    token = tokens[j]

                    if token in ["[CLS]", "[SEP]", "[PAD]"]:
                        continue

                    true_label = id2label[true_seq[j]]
                    pred_label = id2label[pred_seq[j]]

                    true_labels.append(true_label)
                    pred_labels.append(pred_label)

                true_slots.append(true_labels)
                pred_slots.append(pred_labels)

                #joint accuracy
                intent_ok = (
                    intent_preds[i].item()
                    == batch["intent"][i].item()
                )

                slots_ok = pred_labels == true_labels

                if intent_ok and slots_ok:
                    joint_correct += 1

                joint_total += 1

    #metrics
    intent_accuracy = intent_correct / intent_total
    slot_f1 = f1_score(true_slots, pred_slots)
    joint_accuracy = joint_correct / joint_total
    print(f"\nEpoch {epoch}")
    print(f"Train Loss:      {avg_train_loss:.4f}")
    print(f"Intent Accuracy: {intent_accuracy:.4f}")
    print(f"Slot F1:         {slot_f1:.4f}")
    print(f"Joint Accuracy:  {joint_accuracy:.4f}")


#saving model state
save_dir = "BiLSTM_model_10epochs"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), f"{save_dir}/model.pt")
config = {
    "num_intents": num_intents,
    "num_slots": num_slots,
    "hidden_size": 384,
    "num_layers": 6,
    "num_heads": 6,
    "max_length": 128
}
with open(f"{save_dir}/config.json", "w") as f:
    json.dump(config, f)
with open(f"{save_dir}/label2id.json", "w") as f:
    json.dump(label2id, f)
with open(f"{save_dir}/id2label.json", "w") as f:
    json.dump(id2label, f)
with open(f"{save_dir}/intent_names.json", "w") as f:
    json.dump(intent_names, f)
tokenizer.save_pretrained(save_dir)
