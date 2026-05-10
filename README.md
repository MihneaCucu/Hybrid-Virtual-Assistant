---
title: Hybrid Virtual Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
---

# Hybrid Virtual Assistant

A city tourism and task assistant for Bucharest. Ask factual questions about
the city, or give commands like bookings and transport.

## Modules

- **NLU** (Stefan) - BiLSTM intent classifier and slot tagger
- **QA** (Mihnea) - BM25 retrieval + MiniLM reader over a Bucharest knowledge base
- **Dialogue Manager** (George) - seq2seq response generator for task-oriented turns
