# Knowledge/Q&A Local Question Bank

Use this file when you want more local testing examples than the short six-question story demo. The questions are grouped by capability so the demo does not look like a simple FAQ.

Run the extended scripted demo:

```bash
python scripts/demo_qa.py --queries data/demo_queries_extended.jsonl --show-sources
```

Run interactive Q&A:

```bash
python scripts/chat_cli.py
```

## Definition Questions

- `What is the Romanian Athenaeum?`
- `What is King Michael I Park?`
- `What are the Cismigiu Gardens?`
- `What is Arcul de Triumf?`

## Historical Fact Questions

- `When was the Romanian Athenaeum opened?`
- `When was Stavropoleos Monastery built?`
- `When did Bucharest become the capital?`
- `When was Bucharest first mentioned in documents?`
- `Over what years was the Palace of the Parliament constructed?`

## Person, Culture, And Symbolism

- `Who ordered the Palace of the Parliament?`
- `Which festival is associated with the Romanian Athenaeum?`
- `What does Arcul de Triumf symbolize?`
- `What style is Stavropoleos Monastery built in?`
- `What style is the Palace of the Parliament built in?`

## Measurements And Counts

- `How tall is the Palace of the Parliament?`
- `How large is King Michael I Park?`
- `How large are Cismigiu Gardens?`
- `How many sectors does Bucharest have?`
- `How long is the Metrorex system?`
- `How many metro lines does Bucharest Metro have?`

## Museum Content

- `What collections does the National Museum of Art of Romania feature?`
- `What does the National History Museum of Romania contain?`
- `What kind of museum is the Village Museum?`
- `What period do structures in the Village Museum range from?`
- `What type of museum is the Grigore Antipa National Museum?`

## Addresses And Locations

- `Where is Stavropoleos Monastery?`
- `What is the exact address of Antipa Museum?`
- `Where is the main entrance to Cismigiu Gardens?`
- `What is the exact address of the National Museum of Art of Romania?`
- `Where is University Square located?`

## Metro And Transport

- `What metro station is Antipa Museum at?`
- `What metro station is Romanian Athenaeum at?`
- `What metro station is Arcul de Triumf at?`
- `What metro station is Cismigiu Gardens at?`
- `Show metro as well as STB stations near Romanian Athenaeum`
- `What transit route serves Aeroport Henri Coanda Sosiri?`

## Travel Guidance

- `What season is best to visit Bucharest?`
- `What is the budget for visiting Bucharest?`
- `How many days should I stay in Bucharest?`
- `When is a comfortable time to visit Bucharest?`
- `What daily money range should I plan for Bucharest?`

## Fallback And Handoff

- `How much is a metro ticket in Bucharest?`
- `How much does parking cost in Bucharest city center?`
- `Who is the president of France?`
- `Tell me a joke about Bucharest traffic.`
- `Set a reminder for tomorrow and tell me where University Square is.`
- `Book dinner near the Athenaeum and tell me the metro station.`

## What This Demonstrates

- The module supports more than address lookup.
- It handles narrative Q&A, structured local facts, travel guidance, fallback, and handoff.
- It remains bounded to Bucharest tourist-guide knowledge instead of becoming a general chatbot.
