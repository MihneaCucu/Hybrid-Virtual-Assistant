import torch
import NLU_model
import json
from transformers import AutoTokenizer

model_name = "BiLSTM_model_10epochs"
with open(f"{model_name}/config.json") as f:
    config = json.load(f)
model = NLU_model.JointModel(
    num_intents=config["num_intents"],
    num_slots=config["num_slots"]
)

model.load_state_dict(torch.load(f"{model_name}/model.pt"))
model.eval()
tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")

with open(f"{model_name}/label2id.json") as f:
    label2id = json.load(f)

with open(f"{model_name}/id2label.json") as f:
    id2label = json.load(f)

with open(f"{model_name}/intent_names.json") as f:
    intent_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

intent_responses = {
    "datetime_query": "Here is the current date and time information.",
    "iot_hue_lightchange": "The Hue light settings have been updated.",
    "transport_ticket": "Your transport ticket information has been retrieved.",
    "takeaway_query": "Here are the available takeaway options.",
    "qa_stock": "Here is the latest stock market information.",
    "general_greet": "Hello! How can I help you today?",
    "recommendation_events": "Here are some events you might enjoy.",
    "music_dislikeness": "I’ll avoid playing similar music in the future.",
    "iot_wemo_off": "The WeMo device has been turned off.",
    "cooking_recipe": "Here is a recipe you can try.",
    "qa_currency": "Here is the latest currency exchange information.",
    "transport_traffic": "Here is the current traffic update.",
    "general_quirky": "Here’s something fun and quirky for you.",
    "weather_query": "Here is the latest weather forecast.",
    "audio_volume_up": "The volume has been increased.",
    "email_addcontact": "The contact has been added successfully.",
    "takeaway_order": "Your takeaway order has been placed.",
    "email_querycontact": "Here is the contact information you requested.",
    "iot_hue_lightup": "The Hue lights have been brightened.",
    "recommendation_locations": "Here are some locations you may like.",
    "play_audiobook": "Playing your audiobook now.",
    "lists_createoradd": "The item has been added to your list.",
    "news_query": "Here are the latest news updates.",
    "alarm_query": "Here are your current alarms.",
    "iot_wemo_on": "The WeMo device has been turned on.",
    "general_joke": "Here’s a joke for you.",
    "qa_definition": "Here is the definition you requested.",
    "social_query": "Here are the latest social media updates.",
    "music_settings": "Your music settings have been updated.",
    "audio_volume_other": "The volume settings have been adjusted.",
    "calendar_remove": "The event has been removed from your calendar.",
    "iot_hue_lightdim": "The Hue lights have been dimmed.",
    "calendar_query": "Here are your upcoming calendar events.",
    "email_sendemail": "Your email has been sent successfully.",
    "iot_cleaning": "The cleaning device has been activated.",
    "audio_volume_down": "The volume has been lowered.",
    "play_radio": "Playing the radio now.",
    "cooking_query": "Here is the cooking information you requested.",
    "datetime_convert": "The date and time have been converted.",
    "qa_maths": "Here is the solution to your math query.",
    "iot_hue_lightoff": "The Hue lights have been turned off.",
    "iot_hue_lighton": "The Hue lights have been turned on.",
    "transport_query": "Here is the transport information you requested.",
    "music_likeness": "Glad you like the music!",
    "email_query": "Here are your recent emails.",
    "play_music": "Playing music now.",
    "audio_volume_mute": "The audio has been muted.",
    "social_post": "Your social media post has been published.",
    "alarm_set": "Your alarm has been set.",
    "qa_factoid": "Here is the information you requested.",
    "calendar_set": "The event has been added to your calendar.",
    "play_game": "Launching the game now.",
    "alarm_remove": "The alarm has been removed.",
    "lists_remove": "The item has been removed from your list.",
    "transport_taxi": "Your taxi request has been processed.",
    "recommendation_movies": "Here are some movie recommendations for you.",
    "iot_coffee": "Your coffee machine has been started.",
    "music_query": "Here is the music information you requested.",
    "play_podcasts": "Playing your podcast now.",
    "lists_query": "Here are the items on your list."
}


def predict(text):
    # tokenize
    inputs = tokenizer(
        text.lower(),
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # forward pass
    with torch.no_grad():
        _, slot_preds, intent_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # intent prediction
    intent_pred = torch.argmax(intent_logits, dim=1).item()
    intent_label = intent_names[intent_pred]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    pred_sequence = slot_preds[0]
    slot_labels = [
        id2label[str(p)] if isinstance(list(id2label.keys())[0], str)
        else id2label[p]
        for p in pred_sequence
    ]
    # decoding BIO into slot spans
    slots = []
    current_slot = None
    current_value_tokens = []

    #removing [CLS] and [SEP] tokens
    tokens = tokens[1:-1]
    slot_labels = slot_labels[1:-1]

    for token, label in zip(tokens, slot_labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if label.startswith("B-"):
            # save previous slot
            if current_slot is not None:
                slots.append((current_slot, tokenizer.convert_tokens_to_string(current_value_tokens)))

            current_slot = label[2:]
            current_value_tokens = [token]

        elif label.startswith("I-") and current_slot is not None:
            current_value_tokens.append(token)

        else:
            if current_slot is not None:
                slots.append((current_slot, tokenizer.convert_tokens_to_string(current_value_tokens)))
                current_slot = None
                current_value_tokens = []

    # catch last slot
    if current_slot is not None:
        slots.append((current_slot, tokenizer.convert_tokens_to_string(current_value_tokens)))

    return intent_label, slots

while True:
    text = input("\nEnter a sentence (or 'quit'): ")
    if text.lower() == "quit":
        break

    intent, slots = predict(text)

    print(f"\n{intent_responses[intent]}")
    print("Parameters:")

    if not slots:
        print("  None")
    else:
        for slot, value in slots:
            raw_value = value.strip(",.?!-/:; ")
            print(f"  {slot}: {raw_value}")