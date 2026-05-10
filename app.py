"""Gradio chat interface for the Hybrid Virtual Assistant."""

import gradio as gr
from pipeline import load_all, process, reset_dm

# load all models at startup
load_all()

ROUTE_LABELS = {
    "dm":       "Dialogue Manager",
    "qa":       "Knowledge Base Q&A",
    "fallback": "Fallback",
}

ROUTE_COLORS = {
    "dm":       "#3b82f6",   # blue
    "qa":       "#10b981",   # green
    "fallback": "#f59e0b",   # amber
}


def _format_metadata(result: dict) -> str:
    intent     = result.get("intent", "")
    slots      = result.get("slots", [])
    route      = result.get("route", "")
    confidence = result.get("confidence", 0.0)
    source     = result.get("source_doc")

    lines = []
    lines.append(f"**Intent:** `{intent}`")

    if slots:
        slot_str = "  \n".join(f"- `{k}`: {v}" for k, v in slots)
        lines.append(f"**Slots:**  \n{slot_str}")
    else:
        lines.append("**Slots:** none")

    route_label = ROUTE_LABELS.get(route, route)
    lines.append(f"**Routed to:** {route_label}")

    if route == "qa":
        lines.append(f"**Confidence:** {confidence:.0%}")
    if source:
        lines.append(f"**Source:** `{source}`")

    return "  \n".join(lines)


def respond(message: str, history: list[dict]) -> tuple[list[dict], str]:
    result   = process(message)
    response = result["response"]
    history  = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": response},
    ]
    meta = _format_metadata(result)
    return history, meta


def clear_chat():
    reset_dm()
    return [], "", "No message yet."


# Gradio UI
with gr.Blocks(title="Hybrid Virtual Assistant") as demo:

    gr.Markdown(
        """
# Hybrid Virtual Assistant
A city tourism & task assistant for Bucharest.
Ask factual questions about the city, or give commands like bookings and transport.
        """
    )

    with gr.Row():
        # left column: chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=500)

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type a message and press Enter...",
                    show_label=False,
                    scale=5,
                    autofocus=True,
                    submit_btn=True,
                )

            with gr.Row():
                clear_btn = gr.Button("Clear conversation", variant="secondary", scale=1)

        # right column: metadata
        with gr.Column(scale=1, min_width=240):
            gr.Markdown("### Last turn")
            meta_box = gr.Markdown("No message yet.")

    # optional speech tab
    with gr.Accordion("Speech input (optional)", open=False):
        gr.Markdown(
            "Record your voice - the transcript will be sent as a message."
        )
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
        transcribe_btn = gr.Button("Transcribe & Send")
        transcribe_status = gr.Textbox(label="Transcription", interactive=False)

        def transcribe_and_send(audio_path, history):
            if audio_path is None:
                return history, "No audio recorded.", "No message yet."
            try:
                import whisper
                model = whisper.load_model("base")
                result_w = model.transcribe(audio_path)
                text = result_w["text"].strip()
            except ImportError:
                return history, "(openai-whisper not installed)", "No message yet."
            except Exception as e:
                return history, f"(Transcription error: {e})", "No message yet."

            new_history, meta = respond(text, history)
            return new_history, text, meta

        transcribe_btn.click(
            transcribe_and_send,
            inputs=[audio_in, chatbot],
            outputs=[chatbot, transcribe_status, meta_box],
        )

    # example queries
    gr.Examples(
        examples=[
            "Hello!",
            "Where is the Romanian Athenaeum?",
            "I want a cheap Italian restaurant in the centre",
            "Book a table for 2 at 7pm on Friday",
            "How do I get to the Palace of the Parliament?",
            "What metro line goes to University Square?",
            "Book me a taxi from the airport to the city centre",
        ],
        inputs=msg_box,
        label="Example queries",
    )

    # event wiring
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, meta_box],
    ).then(lambda: "", outputs=msg_box)

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg_box, meta_box],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
