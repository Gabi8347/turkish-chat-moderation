"""Shared evaluation + benchmarking for every model in the ladder.

Each model is wrapped as a `predict_proba(list[str]) -> np.ndarray` function that returns
P(offensive) for raw texts (normalization happens inside the model wrapper), so all models
are measured the same way:

  - task metric : macro-F1 on the official test set (classes are imbalanced), plus accuracy,
                  F1 of the offensive class and the confusion matrix
  - size        : bytes of the serialized model files on disk, in MB
  - latency     : one message at a time (as in a live chat), averaged over N_LATENCY = 200
                  test messages after 10 warm-up calls, mean and p95 in ms, CPU only,
                  N_THREADS = 4 CPU threads for every model (a moderation server shares its
                  cores between many chats; 4 threads per request is the deployment budget)

Results are merged into results/results.json (one entry per model).
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.json"
N_LATENCY = 200
N_WARMUP = 10
N_THREADS = 4
LATENCY_BUDGET_MS = 25.0  # p95 per message on N_THREADS CPU threads (see README)


def hardware() -> str:
    try:
        import subprocess
        cpu = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        cpu = ""
    return (f"{cpu or platform.processor()}, CPU only, {N_THREADS} threads, "
            f"{platform.system()} {platform.release()}")


def metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_offensive": round(float(f1_score(y_true, y_pred, pos_label=1)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # [[TN, FP], [FN, TP]]
    }


def best_threshold(y_true, proba) -> float:
    """Pick the decision threshold that maximizes macro-F1 on the VALIDATION set."""
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int), average="macro") for t in grid]
    return round(float(grid[int(np.argmax(scores))]), 2)


def set_threads(n=N_THREADS):
    import torch
    torch.set_num_threads(n)


def latency_ms(predict_proba, texts) -> dict:
    set_threads()
    texts = list(texts)[:N_LATENCY]
    for t in texts[:N_WARMUP]:
        predict_proba([t])
    times = []
    for t in texts:
        t0 = time.perf_counter()
        predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    return {"latency_ms_mean": round(float(np.mean(times)), 3),
            "latency_ms_p95": round(float(np.percentile(times, 95)), 3),
            "latency_runs": len(times), "threads": N_THREADS,
            "meets_budget": bool(np.percentile(times, 95) <= LATENCY_BUDGET_MS)}


def size_mb(paths) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return round(total / 1e6, 3)


def report(name, predict_proba, test_df, threshold, files, extra=None) -> dict:
    """Evaluate one model on the test set, benchmark it and store the row in results.json."""
    set_threads()
    proba = predict_proba(test_df["text"].tolist())
    y_pred = (proba >= threshold).astype(int)
    row = {"model": name, "threshold": threshold, **metrics(test_df["label"].values, y_pred),
           "size_mb": size_mb(files), **latency_ms(predict_proba, test_df["text"]),
           "hardware": hardware(), **(extra or {})}
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[name] = row
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    pred_dir = ROOT / "results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    test_df.assign(proba=proba.round(4), pred=y_pred)[["id", "label", "pred", "proba"]].to_csv(
        pred_dir / f"{name}.csv", index=False)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row
