"""Real-time Turkish chat moderation - Gradio application.

Every message typed into the chat is labeled live (clean / offensive) with a confidence
score by the selected model (default: quantized BERTurk INT8 on ONNX Runtime, CPU only).
Offensive messages are masked: words found by the lexicon matcher are starred out, and if
the model flags a message without any lexicon word, the whole message is hidden.
The side panel logs the model latency of every message against the 25 ms p95 budget and
lets the user switch between all seven models of the ladder.

Run:  python app.py        ->  http://127.0.0.1:7860
"""
import html
import re
import time

import gradio as gr
import numpy as np

from src import registry
from src.evaluate import LATENCY_BUDGET_MS, N_THREADS, set_threads
from src.lexicon import LexiconMatcher

set_threads(N_THREADS)  # same 4-thread CPU budget as in the benchmark
LEXICON = LexiconMatcher()
LABELS = [e.label for e in registry.ENTRIES]
DEFAULT_LABEL = registry.BY_KEY[registry.DEFAULT_MODEL].label

EXAMPLES = [
    "akşam maça kim geliyor, 9da sunucudayım",
    "s.i.k.t.i.r git buradan",
    "4mk bu ne biçim oyun ya",
    "ŞEREFSİZLER yine hile yapıyor",
    "götürdüm takımı sırtımda, gg",
    "senin gibi oyuncuların ta ağzına sıçayım",
    "çok güzel oynadın kanka helal olsun",
]

CSS = """
.chat {height: 460px; overflow-y: auto; padding: 8px; border: 1px solid var(--border-color-primary);
       border-radius: 8px; background: var(--background-fill-secondary);}
.msg {margin: 6px 0; padding: 8px 10px; border-radius: 8px; background: var(--background-fill-primary);
      border-left: 4px solid #2e7d32;}
.msg.off {border-left-color: #c62828;}
.meta {font-size: 12px; opacity: .75; margin-top: 2px;}
.badge {display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 12px; color: #fff;}
.badge.clean {background: #2e7d32;} .badge.off {background: #c62828;}
.hidden-msg {font-style: italic; opacity: .8;}
"""


def mask(text: str) -> tuple[str, list[str]]:
    """Star out every whitespace token that contains a lexicon match."""
    parts, hits = re.split(r"(\s+)", text), []
    for i, p in enumerate(parts):
        if p.strip() and LEXICON.find(p):
            hits.append(p)
            parts[i] = p[0] + "*" * (len(p) - 1)
    return "".join(parts), hits


def classify(text: str, key: str):
    model, threshold = registry.get(key)
    t0 = time.perf_counter()
    p = float(model.predict_proba([text])[0])
    ms = (time.perf_counter() - t0) * 1000
    return p, p >= threshold, ms, threshold


def render_chat(history):
    if not history:
        return "<div class='chat'><i>Type a message to start the chat.</i></div>"
    rows = []
    for h in history:
        if h["offensive"]:
            body = (html.escape(h["masked"]) if h["hits"] else
                    "<span class='hidden-msg'>🚫 message hidden by the moderator</span>")
            badge = f"<span class='badge off'>offensive {h['p']:.2f}</span>"
        else:
            body, badge = html.escape(h["text"]), f"<span class='badge clean'>clean {1 - h['p']:.2f}</span>"
        rows.append(f"<div class='msg{' off' if h['offensive'] else ''}'><b>{html.escape(h['user'])}:</b> "
                    f"{body}<div class='meta'>{badge} · {h['model_short']} · {h['ms']:.1f} ms</div></div>")
    return "<div class='chat'>" + "".join(rows) + "</div>"


def latency_summary(history):
    if not history:
        return f"**Latency budget:** p95 ≤ {LATENCY_BUDGET_MS:.0f} ms per message on {N_THREADS} CPU threads."
    ms = np.array([h["ms"] for h in history])
    p95 = float(np.percentile(ms, 95))
    ok = "✅ within budget" if p95 <= LATENCY_BUDGET_MS else "❌ over budget"
    return (f"**{len(ms)} messages** · mean **{ms.mean():.1f} ms** · p95 **{p95:.1f} ms** · "
            f"budget {LATENCY_BUDGET_MS:.0f} ms → {ok}  \n"
            f"flagged offensive: {sum(h['offensive'] for h in history)}")


def log_rows(history):
    return [[i + 1, h["model_short"], "offensive" if h["offensive"] else "clean",
             round(h["p"], 3), round(h["ms"], 2)] for i, h in enumerate(history)][::-1]


def send(text, user, model_label, history):
    text = (text or "").strip()
    if not text:
        return gr.update(), render_chat(history), latency_summary(history), log_rows(history), history
    entry = registry.BY_LABEL[model_label]
    p, off, ms, _ = classify(text, entry.key)
    masked, hits = mask(text) if off else (text, [])
    history = history + [{"user": user or "player", "text": text, "masked": masked, "hits": hits,
                          "p": p, "offensive": off, "ms": ms, "model_short": entry.key}]
    return "", render_chat(history), latency_summary(history), log_rows(history), history


def compare(text, history):
    text = (text or "").strip() or (history[-1]["text"] if history else "")
    if not text:
        return [], "Type a message first."
    rows = []
    for e in registry.ENTRIES:
        p, off, ms, t = classify(text, e.key)
        r = registry.RESULTS[e.key]
        rows.append([e.label, "offensive" if off else "clean", round(p, 3), t, round(ms, 2),
                     r["macro_f1"], r["size_mb"]])
    return rows, f"Message: “{text}”"


def clear():
    return [], render_chat([]), latency_summary([]), []


def build():
    with gr.Blocks(title="Turkish Chat Moderation") as demo:
        gr.Markdown("# 🛡️ Real-time Turkish Chat Moderation\n"
                    "Every message is classified live as **clean** or **offensive**. Offensive messages are "
                    "masked. Default model: **BERTurk fine-tuned on OffensEval-TR 2020, quantized to INT8, "
                    "running on ONNX Runtime (CPU)**.")
        history = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.HTML(render_chat([]))
                with gr.Row():
                    user = gr.Textbox(value="player1", label="Name", scale=1, max_lines=1)
                    msg = gr.Textbox(label="Message", placeholder="Type a chat message and press Enter",
                                     scale=4, max_lines=2)
                    btn = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(EXAMPLES, inputs=msg, label="Try these messages")
            with gr.Column(scale=2):
                model = gr.Dropdown(LABELS, value=DEFAULT_LABEL, label="Moderation model")
                summary = gr.Markdown(latency_summary([]))
                log = gr.Dataframe(headers=["#", "model", "label", "P(offensive)", "latency ms"],
                                   label="Response-time log (latest first)", interactive=False,
                                   wrap=True, max_height=330)
                clear_btn = gr.Button("Clear chat")
        with gr.Accordion("Compare all seven models on one message", open=False):
            cmp_btn = gr.Button("Run all models on the message box (or the last message)")
            cmp_info = gr.Markdown()
            cmp_table = gr.Dataframe(headers=["model", "label", "P(offensive)", "threshold", "latency ms",
                                              "test macro-F1", "size MB"], interactive=False, wrap=True)

        outs = [msg, chat, summary, log, history]
        msg.submit(send, [msg, user, model, history], outs)
        btn.click(send, [msg, user, model, history], outs)
        clear_btn.click(clear, None, [history, chat, summary, log])
        cmp_btn.click(compare, [msg, history], [cmp_table, cmp_info])
    return demo


if __name__ == "__main__":
    print("loading default model ...")
    registry.get(registry.DEFAULT_MODEL)  # load + warm up before the first chat message
    build().launch(server_name="127.0.0.1", server_port=7860, css=CSS)
