"""Figures and the results table for the README / slides - generated from our own result
files only (results/*.json), so no number is typed by hand.

  docs/results_table.md        results table (R4)
  docs/figures/results.png     macro-F1 | size | p95 latency per model (small multiples)
  docs/figures/ffn_training.png  word2vec + FFN learning curves
  docs/figures/architecture.png  architecture / data flow of the application (R6)

Run:  python -m src.figures
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.evaluate import LATENCY_BUDGET_MS, N_THREADS  # noqa: E402
from src.registry import DEFAULT_MODEL, ENTRIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
R = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))

# reference palette (light mode), validated: blue / orange pass CVD + contrast checks
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

ORDER = ["lexicon_rules", "tfidf_logreg", "w2v_ffn", "berturk_fp32", "berturk_onnx_fp32",
         "berturk_int8_pytorch", "berturk_onnx_int8"]
NAMES = {
    "lexicon_rules": "Lexicon + edit distance",
    "tfidf_logreg": "tf-idf + logistic regression",
    "w2v_ffn": "word2vec + FFN",
    "berturk_fp32": "BERTurk FP32 (PyTorch)",
    "berturk_onnx_fp32": "BERTurk FP32 (ONNX Runtime)",
    "berturk_int8_pytorch": "BERTurk INT8 (PyTorch dynamic)",
    "berturk_onnx_int8": "BERTurk INT8 (ONNX Runtime)",
}
SHORT_PREC = {"lexicon_rules": "rules", "tfidf_logreg": "FP64", "w2v_ffn": "FP32", "berturk_fp32": "FP32",
              "berturk_onnx_fp32": "FP32", "berturk_int8_pytorch": "INT8 weights, dynamic",
              "berturk_onnx_int8": "INT8 weights, dynamic"}
RUNG = {"lexicon_rules": "classical (rules)", "tfidf_logreg": "classical baseline",
        "w2v_ffn": "neural network", "berturk_fp32": "Transformer", "berturk_onnx_fp32": "Transformer",
        "berturk_int8_pytorch": "quantized Transformer", "berturk_onnx_int8": "quantized Transformer"}


def results_table():
    hw = R[DEFAULT_MODEL]["hardware"]
    lines = [f"Official OffensEval-TR 2020 test set (3,528 tweets). Latency: one message at a time, "
             f"mean / p95 over {R[DEFAULT_MODEL]['latency_runs']} runs, {hw}. "
             f"Budget: p95 ≤ {LATENCY_BUDGET_MS:.0f} ms.", "",
             "| Model | Rung | Precision | Macro-F1 | F1 (offensive) | Size (MB) | Latency mean (ms) "
             "| Latency p95 (ms) | Budget |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in ORDER:
        r = R[k]
        name = f"**{NAMES[k]}** (app default)" if k == DEFAULT_MODEL else NAMES[k]
        lines.append(f"| {name} | {RUNG[k]} | {r['precision']} | {r['macro_f1']:.3f} | {r['f1_offensive']:.3f} | "
                     f"{r['size_mb']:.3f} | {r['latency_ms_mean']:.2f} | {r['latency_ms_p95']:.2f} | "
                     f"{'✅' if r['meets_budget'] else '❌'} |")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _hbar(ax, values, title, fmt, xlabel, log=False):
    y = range(len(ORDER))[::-1]
    colors = [BLUE] * len(ORDER)
    ax.barh(list(y), values, color=colors, height=0.6, edgecolor=SURF, linewidth=2)
    for yi, v in zip(y, values):
        ax.text(v * (1.08 if log else 1) + (0 if log else max(values) * 0.015), yi, fmt(v),
                va="center", ha="left", color=INK, fontsize=9, zorder=5,
                bbox={"boxstyle": "square,pad=0.15", "facecolor": SURF, "edgecolor": "none"})
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale("log")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def results_figure():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, gridspec_kw={"wspace": 0.25})
    f1 = [R[k]["macro_f1"] for k in ORDER]
    size = [R[k]["size_mb"] for k in ORDER]
    p95 = [R[k]["latency_ms_p95"] for k in ORDER]
    _hbar(axes[0], f1, "Macro-F1 on the test set (higher is better)", lambda v: f"{v:.3f}", "macro-F1")
    axes[0].set_xlim(0, 1)
    _hbar(axes[1], size, "Model size on disk (log scale)", lambda v: f"{v:g} MB" if v >= 1 else f"{v:.3f} MB",
          "MB", log=True)
    _hbar(axes[2], p95, f"p95 latency per message, {N_THREADS} CPU threads", lambda v: f"{v:.1f} ms", "ms")
    axes[2].axvline(LATENCY_BUDGET_MS, color=ORANGE, linewidth=2, linestyle="--")
    axes[2].text(LATENCY_BUDGET_MS, -0.8, f" budget {LATENCY_BUDGET_MS:.0f} ms", color=INK2,
                 fontsize=9, va="center")
    axes[2].set_xlim(0, max(p95) * 1.25)
    for ax in axes:
        ax.set_ylim(-1.1, len(ORDER) - 0.5)
    axes[0].set_yticks(range(len(ORDER))[::-1], [NAMES[k] for k in ORDER])
    lbl = axes[0].get_yticklabels()[ORDER.index(DEFAULT_MODEL)]
    lbl.set_fontweight("bold")
    lbl.set_color(INK)
    fig.suptitle("Model ladder on OffensEval-TR 2020 (bold = model used by the app)", x=0.01, y=1.05,
                 ha="left", color=INK, fontsize=12)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def ffn_figure():
    info = json.loads((ROOT / "results" / "w2v_ffn_training.json").read_text(encoding="utf-8"))
    h = info["history"]
    ep = [x["epoch"] for x in h]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.3})
    for ax, key, title, color in [(axes[0], "train_loss", "Training loss (weighted BCE)", BLUE),
                                  (axes[1], "val_macro_f1", "Validation macro-F1", BLUE)]:
        ax.plot(ep, [x[key] for x in h], color=color, linewidth=2, marker="o", markersize=4)
        ax.axvline(info["best_epoch"], color=ORANGE, linewidth=2, linestyle="--")
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.grid(color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].text(info["best_epoch"], min(x["val_macro_f1"] for x in h),
                 f" best epoch {info['best_epoch']} (early stopping)", color=INK2, fontsize=9)
    fig.suptitle("word2vec + feedforward network trained with SGD", x=0.01, y=1.06, ha="left", color=INK,
                 fontsize=12)
    fig.savefig(OUT / "ffn_training.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _box(ax, x, y, w, h, title, body="", edge=AXIS, fill="#ffffff", bold_edge=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=2.2 if bold_edge else 1.2, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.07, body, ha="center", va="top", fontsize=7.8, color=INK2, linespacing=1.35)


def _arrow(ax, a, b, color=INK2, text=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, linewidth=1.5))
    if text:  # label below the arrow, in the gap between boxes
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 0.02, text, ha="center", va="top", fontsize=7.5,
                color=MUTED)


def architecture_figure():
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # where it runs
    ax.add_patch(FancyBboxPatch((0.01, 0.13), 0.98, 0.80, boxstyle="round,pad=0.005,rounding_size=0.01",
                                linewidth=1, edgecolor=AXIS, facecolor="#f9f9f7", linestyle="--"))
    ax.text(0.02, 0.915, f"Local machine · Python 3.13 · CPU only, {N_THREADS} threads  "
            "(app.py → src/registry.py)", fontsize=9, color=MUTED, va="top")

    _box(ax, 0.03, 0.62, 0.14, 0.22, "User (browser)", "Gradio chat UI\nhttp://127.0.0.1:7860\n\ntypes a message,\npicks a model")
    _box(ax, 0.21, 0.62, 0.15, 0.22, "Preprocessing",
         "src/normalize.py (regex)\n@USER/URL placeholders\nspaced letters, leetspeak\nTurkish lowercase (I→ı)\nrepeated letters")
    _arrow(ax, (0.17, 0.73), (0.21, 0.73), text="raw text")

    # models
    bx, by, bw, bh = 0.40, 0.20, 0.30, 0.66
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.005,rounding_size=0.012",
                                linewidth=1.2, edgecolor=AXIS, facecolor="#ffffff"))
    ax.text(bx + bw / 2, by + bh - 0.015, "Selected model → P(offensive)", ha="center", va="top",
            fontsize=10, fontweight="bold", color=INK)
    rows = [e for e in ENTRIES]
    step = (bh - 0.07) / len(rows)
    for i, e in enumerate(rows):
        r = R[e.key]
        yy = by + bh - 0.065 - (i + 1) * step + 0.008
        default = e.key == DEFAULT_MODEL
        ax.add_patch(FancyBboxPatch((bx + 0.012, yy), bw - 0.024, step - 0.012,
                                    boxstyle="round,pad=0.003,rounding_size=0.008",
                                    linewidth=2.2 if default else 0.8, edgecolor=BLUE if default else AXIS,
                                    facecolor="#eef4fc" if default else "#ffffff"))
        ax.text(bx + 0.022, yy + (step - 0.012) * 0.62, NAMES[e.key] + ("  ← default" if default else ""),
                fontsize=8.6, fontweight="bold" if default else "normal", color=INK, va="center")
        ax.text(bx + 0.022, yy + (step - 0.012) * 0.25,
                f"{SHORT_PREC[e.key]} · {r['size_mb']:g} MB · threshold {r['threshold']}", fontsize=7.6,
                color=INK2, va="center")
    _arrow(ax, (0.36, 0.73), (0.40, 0.73), text="normalized\ntext")

    _box(ax, 0.74, 0.62, 0.11, 0.22, "Decision",
         "P ≥ threshold\n(chosen on the\nvalidation set)\n→ clean /\n   offensive")
    _arrow(ax, (0.70, 0.73), (0.74, 0.73))
    _box(ax, 0.87, 0.62, 0.11, 0.22, "Post-processing",
         "offensive →\nmask words found by\nthe lexicon matcher\n(weighted edit dist.)\nelse hide message")
    _arrow(ax, (0.85, 0.73), (0.87, 0.73))
    _box(ax, 0.74, 0.22, 0.24, 0.30, "UI output (Gradio)",
         "chat bubble: masked text, label,\nconfidence, model, latency\n\nside panel: per-message latency log,\n"
         f"mean / p95 vs {LATENCY_BUDGET_MS:.0f} ms budget\n\ncompare panel: all 7 models side by side\n\n"
         "→ shown in the user's browser tab")
    _arrow(ax, (0.925, 0.62), (0.925, 0.52))

    # offline / storage
    _box(ax, 0.03, 0.20, 0.33, 0.30, "Model files  (models/, downloaded on first use)",
         "Hugging Face Hub:\nGabi8347/turkish-chat-moderation-models\n\nTraining (offline, scripts in src/):\n"
         "OffensEval-TR 2020 → data.py splits\nlexicon.py · tfidf.py · w2v_ffn.py (CPU)\n"
         "berturk.py (RTX 4050 GPU, fp16)\nquantize.py → INT8 PyTorch / ONNX")
    _arrow(ax, (0.36, 0.35), (0.40, 0.35), text="load\nweights")

    ax.text(0.01, 0.08, "Every box maps to a file in the repository; model names, precisions, sizes and "
            "thresholds are read from src/registry.py and results/results.json when this figure is generated.",
            fontsize=8.5, color=MUTED)
    fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results_table()
    results_figure()
    ffn_figure()
    architecture_figure()
    print(f"figures written to {OUT}")
