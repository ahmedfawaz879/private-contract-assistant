"""Gradio demo: pick a contract, ask a question, toggle redaction, and see whether planted data appears."""

from __future__ import annotations

from contract_assistant.leakage import leaks
from contract_assistant.prompts import make_item


def build_app(docs: dict, index: dict, backends: list, top_k: int, embedder=None, default_question: str = ""):
    import gradio as gr

    by_name = {b.name: b for b in backends}

    def answer(doc, question, use_redaction, arm):
        ix = index["redacted" if use_redaction else "raw"][doc]
        chunks = [ix.chunks[i] for i in ix.search(question, top_k, embedder)]
        r = by_name[arm].generate([make_item(question, chunks)])[0]
        lk = leaks(r["text"], docs[doc]["canaries"])
        badge = ("✅ no planted personal data in the answer" if not lk else
                 "⚠️ planted personal data in the answer: " + ", ".join(f"{t} ({lvl})" for t, lvl, _ in lk))
        ex = "\n\n".join(f"**[{i + 1}]** {c[:700]}" for i, c in enumerate(chunks))
        return r["text"], badge + f"  ·  {r['seconds']:.2f}s", ex

    return gr.Interface(
        fn=answer,
        inputs=[gr.Dropdown(list(docs), label="Contract"),
                gr.Textbox(label="Question", value=default_question),
                gr.Checkbox(value=True, label="Redact personal data before indexing"),
                gr.Radio(list(by_name), value=list(by_name)[-1], label="Generator")],
        outputs=[gr.Textbox(label="Answer"), gr.Markdown(), gr.Markdown(label="Retrieved excerpts")],
        title="Private contract assistant (CUAD, planted synthetic personal data)",
    )
