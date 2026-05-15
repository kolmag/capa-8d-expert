"""
app.py — CAPA/8D Expert Knowledge Worker — Gradio UI
Gradio 6.x compatible: type="messages" is default, css in launch(), no tuples.

Usage:
    uv run scripts/app.py
    uv run scripts/app.py --share
"""

from pathlib import Path

# Load .env from project root
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env if _env.exists() else None)
except ImportError:
    pass

import gradio as gr

from capa_8d_expert.answer import (
    DEFAULT_MODEL_STACK,
    MODEL_STACK_OPTIONS,
    AnswerResult,
    _load_bge,
    answer_stream,
)

# ── Constants ──────────────────────────────────────────────────────────────────

TITLE = "CAPA / 8D Expert"

EXAMPLE_QUESTIONS = [
    "What are the most common mistakes teams make in D3 containment?",
    "How do I update the FMEA after closing a CAPA?",
    "What is the difference between ICA and PCA in 8D?",
    "When should I use Ishikawa instead of 5 Whys?",
    "How do I calculate the right PM interval based on cycle count?",
    "What locations must I check when scoping the suspect population?",
    "What does a good VoE pass criterion look like for a dimensional fix?",
    "How does Is/Is Not analysis help identify root causes?",
]

CATEGORY_COLOURS = {
    "methodology": "#4A90D9",
    "example":     "#7B68EE",
    "procedure":   "#2E8B57",
    "reference":   "#D4822A",
    "tool":        "#9B59B6",
    "compliance":  "#C0392B",
    "general":     "#7F8C8D",
}

CSS = """
.gradio-container { max-width: 1400px !important; }
footer { display: none !important; }
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve_default_model_stack_label() -> str:
    for label, stack_id in MODEL_STACK_OPTIONS.items():
        if stack_id == DEFAULT_MODEL_STACK:
            return label
    return next(iter(MODEL_STACK_OPTIONS))


def model_stack_label(stack_id: str) -> str:
    for label, option_id in MODEL_STACK_OPTIONS.items():
        if option_id == stack_id:
            return label
    return stack_id


def category_badge(category: str) -> str:
    colour = CATEGORY_COLOURS.get(category, "#7F8C8D")
    return (
        f'<span style="background:{colour};color:white;font-size:10px;'
        f'font-weight:600;padding:2px 7px;border-radius:10px;margin-right:4px;">'
        f'{category.upper()}</span>'
    )


def format_sources_panel(result: AnswerResult) -> str:
    if not result.ranked_chunks:
        return "<p style='color:#888'>No sources retrieved.</p>"

    reranker_badge = (
        '<span style="background:#2E8B57;color:white;font-size:10px;font-weight:600;'
        'padding:2px 7px;border-radius:10px;margin-left:6px;">BGE</span>'
        if result.reranker_used == "bge" else
        '<span style="background:#7B68EE;color:white;font-size:10px;font-weight:600;'
        'padding:2px 7px;border-radius:10px;margin-left:6px;">LLM</span>'
    )
    parts = [
        f'<div style="font-size:12px;color:#888;margin-bottom:12px;">'
        f'{len(result.ranked_chunks)} chunks · {len(result.sources)} documents '
        f'{reranker_badge}'
        f'<span style="margin-left:8px;">{model_stack_label(result.model_stack)}</span>'
        f'</div>'
    ]

    for chunk in result.ranked_chunks:
        score = chunk.relevance_score
        score_colour = "#2E8B57" if score >= 7 else "#D4822A" if score >= 4 else "#C0392B"
        preview = chunk.original_text[:300].replace("\n", " ").strip()
        if len(chunk.original_text) > 300:
            preview += "…"

        parts.append(f"""
<div style="border:1px solid #e0e0e0;border-radius:8px;padding:12px;margin-bottom:10px;background:#fafafa;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <div>
      {category_badge(chunk.doc_category)}
      <span style="font-size:11px;color:#555;font-weight:500;">
        {chunk.source_file.replace('.md','').replace('_',' ')}
      </span>
    </div>
    <span style="font-size:12px;font-weight:700;color:{score_colour};">{score:.1f}/10</span>
  </div>
  <div style="font-size:12px;font-weight:600;color:#333;margin-bottom:4px;">{chunk.headline}</div>
  <div style="font-size:11px;color:#666;line-height:1.5;">{preview}</div>
</div>""")

    if result.rewritten_queries:
        rewrites = "".join(
            f'<div style="font-size:11px;color:#666;padding:2px 0;">• {q}</div>'
            for q in result.rewritten_queries
        )
        parts.append(f"""
<details style="margin-top:12px;">
  <summary style="font-size:11px;color:#888;cursor:pointer;">
    Query rewrites ({len(result.rewritten_queries)})
  </summary>
  <div style="padding:6px 0 0 8px;">{rewrites}</div>
</details>""")

    return "\n".join(parts)


def format_sources_from_sink(sink: dict) -> str:
    if not sink.get("ranked_chunks"):
        return "<p style='color:#888'>No sources retrieved.</p>"

    result = AnswerResult(
        question="",
        rewritten_queries=sink.get("rewritten_queries", []),
        ranked_chunks=sink.get("ranked_chunks", []),
        answer=sink.get("answer", ""),
        sources=sink.get("sources", []),
        reranker_used=sink.get("reranker_used", "bge"),
        checker_score=sink.get("checker_score", 1.0),
        model_stack=sink.get("model_stack", DEFAULT_MODEL_STACK),
    )
    return format_sources_panel(result)


# ── Chat handler ───────────────────────────────────────────────────────────────

def chat(
    message: str,
    history: list[dict],
    use_rewrite: bool,
    model_stack: str,
):
    """Gradio 6 messages format: history is list[{"role": ..., "content": ...}]"""
    if not message.strip():
        yield history, "", "<p style='color:#888'>Ask a question to see sources.</p>"
        return

    try:
        # Build history for context (exclude the current empty assistant turn)
        # history format from Gradio: list[{"role": ..., "content": ...}]
        # Gradio 6 can return content as list (multimodal) — normalise to str
        def _content_str(m):
            c = m.get("content", "")
            if isinstance(c, list):
                return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
            return str(c) if c else ""
        past_turns = [
            {"role": m["role"], "content": _content_str(m)}
            for m in history
            if _content_str(m).strip()
        ]
        new_history = history + [{"role": "user", "content": message}]
        yield (
            new_history + [{"role": "assistant", "content": "Retrieving and ranking sources..."}],
            "",
            "<p style='color:#888'>Retrieving and ranking sources...</p>",
        )

        sink = {}
        accumulated = ""
        for partial in answer_stream(
            question=message,
            use_rewrite=use_rewrite,
            debug=False,
            reranker_mode='auto',
            history=past_turns if past_turns else None,
            model_stack=model_stack,
            _sink=sink,
        ):
            accumulated = partial
            yield (
                new_history + [{"role": "assistant", "content": accumulated}],
                "",
                format_sources_from_sink(sink),
            )

    except FileNotFoundError as e:
        response_text = (
            f"**Knowledge base not found.**\n\n"
            f"Run: `uv run scripts/ingest.py --reset`\n\nError: {e}"
        )
        sources_html = "<p style='color:#C0392B'>Knowledge base not initialised.</p>"
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text},
        ], "", sources_html
        return

    except Exception as e:
        response_text = f"**Error:** {str(e)}"
        sources_html  = f"<p style='color:#C0392B'>Error: {str(e)}</p>"
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text},
        ], "", sources_html
        return

    yield (
        new_history + [{"role": "assistant", "content": accumulated}],
        "",
        format_sources_from_sink(sink),
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:

        gr.Markdown(f"""
# 🔍 {TITLE}
**Expert knowledge worker for CAPA procedures, 8D methodology, RCA tools, FMEA, and quality standards.**
*Selectable model stacks · BGE Reranker (local) · text-embedding-3-small · Chroma*
        """)

        with gr.Row():

            # ── Left: Chat ──────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="CAPA/8D Expert",
                    height=520,
                    placeholder=(
                        "## Welcome to the CAPA/8D Expert\n\n"
                        "Ask anything about:\n"
                        "- **8D disciplines** D0–D8 and when to use them\n"
                        "- **CAPA procedures** and phase requirements\n"
                        "- **RCA tools** — 5 Whys, Ishikawa, Is/Is Not, FTA\n"
                        "- **FMEA and Control Plans** — how to update after a CAPA\n"
                        "- **Containment** — suspect populations, ICA methods\n"
                        "- **Standards** — ISO 9001, IATF 16949, AS9100"
                    ),
                )

                with gr.Row():
                    msg_input  = gr.Textbox(
                        placeholder="Ask a CAPA or 8D question...",
                        label="", scale=5, lines=1, max_lines=3, autofocus=True,
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1, min_width=80)

                with gr.Row():
                    clear_btn   = gr.Button("Clear", variant="secondary", size="sm")
                    use_rewrite = gr.Checkbox(
                        value=True,
                        label="Query rewriting (better recall, slightly slower)",
                        scale=3,
                    )
                model_stack = gr.Dropdown(
                    choices=list(MODEL_STACK_OPTIONS.keys()),
                    value=resolve_default_model_stack_label(),
                    label="Model stack",
                    interactive=True,
                )

                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=msg_input,
                    label="Example questions",
                )

            # ── Right: Sources ──────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Source Documents")
                sources_panel = gr.HTML(
                    value="<p style='color:#888;font-size:13px;'>Ask a question to see retrieved sources.</p>",
                )

        # State — Gradio 6 messages format (list of dicts)
        history_state = gr.State([])

        # ── Events ──────────────────────────────────────────────────────────
        def on_submit(message, history, use_rw, stack):
            yield from chat(message, history, use_rw, stack)

        submit_inputs  = [msg_input, history_state, use_rewrite, model_stack]
        submit_outputs = [chatbot, msg_input, sources_panel]

        msg_input.submit(
            fn=on_submit, inputs=submit_inputs, outputs=submit_outputs,
        ).then(fn=lambda h: h, inputs=[chatbot], outputs=[history_state])

        submit_btn.click(
            fn=on_submit, inputs=submit_inputs, outputs=submit_outputs,
        ).then(fn=lambda h: h, inputs=[chatbot], outputs=[history_state])

        clear_btn.click(
            fn=lambda: ([], [], "", "<p style='color:#888'>Ask a question to see sources.</p>"),
            outputs=[chatbot, history_state, msg_input, sources_panel],
        )

    return demo


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CAPA/8D Expert — Gradio UI")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port",  type=int, default=7860)
    args = parser.parse_args()

    print(f"\n{'─'*50}")
    print(f"  CAPA/8D Expert — Starting UI")
    print(f"  http://localhost:{args.port}")
    print(f"{'─'*50}\n")

    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
    )
