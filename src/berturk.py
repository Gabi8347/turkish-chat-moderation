"""Transformer rung: BERTurk fine-tuned for offensive-language classification (area C).

Model: dbmdz/bert-base-turkish-cased ("BERTurk", 12-layer BERT-base encoder, 32k WordPiece
vocabulary trained on Turkish text).  The cased model is used because the uncased
tokenizer strips Turkish diacritics ("şerefsiz" -> "ser ##ef ##siz", "çocuğu" -> "cocu ##gu"),
while the cased one keeps them ("şeref ##siz", "çocuğu").
A linear classification head on the [CLS] token is trained together with all encoder layers
(AdamW, linear warm-up + decay, mixed precision on the GPU).  The best epoch and the decision
threshold are chosen on the validation set (macro-F1).  The test set is evaluated on the
CPU in FP32, so it can be compared with the quantized model (src/quantize.py).

Run:  python -m src.berturk            (train on GPU, then evaluate on CPU)
      python -m src.berturk --eval     (only evaluate the saved model)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.normalize import normalize

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "dbmdz/bert-base-turkish-cased"
MODEL_DIR = ROOT / "models" / "berturk-offensive"
INFO_FILE = ROOT / "results" / "berturk_training.json"
MAX_LEN = 128
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BertClassifier:
    """predict_proba wrapper used by the evaluator and by the app."""

    def __init__(self, model, tokenizer, device="cpu", batch_size=64):
        self.model = model.to(device).eval()
        self.tok, self.device, self.batch_size = tokenizer, device, batch_size

    @torch.inference_mode()
    def predict_proba(self, texts):
        texts = [normalize(t) for t in texts]
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    @classmethod
    def load(cls, path=MODEL_DIR, device="cpu"):
        return cls(AutoModelForSequenceClassification.from_pretrained(path),
                   AutoTokenizer.from_pretrained(path), device)


def _loader(tok, texts, labels, batch_size, shuffle, seed=SEED):
    enc = tok([normalize(t) for t in texts], truncation=True, max_length=MAX_LEN)
    items = [{"input_ids": ids, "attention_mask": am, "labels": int(y)}
             for ids, am, y in zip(enc["input_ids"], enc["attention_mask"], labels)]

    def collate(batch):
        padded = tok.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch],
                         return_tensors="pt")
        padded["labels"] = torch.tensor([b["labels"] for b in batch])
        return padded

    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g)


def train(epochs=3, batch_size=32, lr=2e-5, warmup=0.1, weight_decay=0.01):
    from src.data import load
    from src.evaluate import best_threshold

    assert torch.cuda.is_available(), "training expects a CUDA GPU"
    set_seed()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "clean", 1: "offensive"},
        label2id={"clean": 0, "offensive": 1}).to(device)

    tr, va = load("train"), load("val")
    train_dl = _loader(tok, tr.text, tr.label, batch_size, shuffle=True)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    val_clf = BertClassifier(model, tok, device)

    history, best = [], (-1.0, 0, 0.5)
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(batch["labels"])
            n += len(batch["labels"])
            if step % 200 == 0:
                print(f"epoch {ep} step {step}/{len(train_dl)} loss {total / n:.4f} "
                      f"({time.time() - t_start:.0f}s)", flush=True)
        model.eval()
        p = val_clf.predict_proba(va.text.tolist())
        t = best_threshold(va.label, p)
        f1 = f1_score(va.label, (p >= t).astype(int), average="macro")
        history.append({"epoch": ep, "train_loss": round(total / n, 4), "val_macro_f1": round(f1, 4),
                        "threshold": t, "elapsed_s": round(time.time() - t_start)})
        print(f"== epoch {ep}: train loss {total / n:.4f}, val macro-F1 {f1:.4f} (t={t})", flush=True)
        if f1 > best[0]:
            best = (f1, ep, t)
            model.save_pretrained(MODEL_DIR)   # safetensors, FP32 weights
            tok.save_pretrained(MODEL_DIR)
            print(f"   saved best model to {MODEL_DIR}", flush=True)

    info = {"base_model": BASE_MODEL, "max_len": MAX_LEN, "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "warmup": warmup, "weight_decay": weight_decay, "mixed_precision": "fp16 (training only)",
            "gpu": torch.cuda.get_device_name(0), "history": history,
            "best_epoch": best[1], "threshold": best[2], "val_macro_f1": round(best[0], 4)}
    INFO_FILE.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def evaluate():
    from src.data import load
    from src.evaluate import report

    info = json.loads(INFO_FILE.read_text())
    clf = BertClassifier.load(MODEL_DIR, device="cpu")
    files = [f for f in MODEL_DIR.iterdir() if f.is_file()]
    report("berturk_fp32", clf.predict_proba, load("test"), info["threshold"], files,
           extra={"family": "transformer (BERTurk fine-tuned)", "precision": "fp32",
                  "params": {"base_model": BASE_MODEL,
                             "parameters": sum(p.numel() for p in clf.model.parameters()),
                             "best_epoch": info["best_epoch"]},
                  "val_macro_f1": info["val_macro_f1"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="only evaluate the saved model")
    args = ap.parse_args()
    if not args.eval:
        train()
    evaluate()
