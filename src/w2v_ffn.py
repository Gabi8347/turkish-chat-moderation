"""Neural network rung: our own word2vec + a feedforward network (technique area A, + B).

1. word2vec (skip-gram with negative sampling, gensim) trained from scratch on the
   normalized TRAINING tweets only - no pretrained vectors, val/test never seen.
2. Each message -> [mean of its word vectors ; element-wise max of its word vectors]
   (2 x 100 = 200 features; mean = overall topic, max = the strongest single signal,
   e.g. one swear word in a long tweet). Out-of-vocabulary tokens are skipped.
3. Feedforward network 200 -> 128 -> 64 -> 1 with ReLU and dropout, trained with
   mini-batch SGD (momentum 0.9) and backpropagation on a class-weighted binary
   cross-entropy loss (19 % offensive).  Inputs are standardized with training-set mean /
   std.  Early stopping and the decision threshold use the validation set (macro-F1).

Run:  python -m src.w2v_ffn
"""
import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from gensim.models import KeyedVectors, Word2Vec
from sklearn.metrics import f1_score
from torch import nn

from src.normalize import normalize, tokenize

ROOT = Path(__file__).resolve().parents[1]
W2V_FILE = ROOT / "models" / "w2v_tweets.kv"
FFN_FILE = ROOT / "models" / "ffn.pt"
SEED = 42
DIM = 100
def stable_hash(s: str) -> int:
    """gensim seeds each word vector with hash(word); Python's str hash changes every run,
    so use a fixed hash to make word2vec reproducible."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


W2V_PARAMS = dict(vector_size=DIM, window=5, min_count=2, sg=1, negative=10, epochs=15,
                  seed=SEED, workers=1, hashfxn=stable_hash)  # workers=1 + fixed hash -> reproducible


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FFN(nn.Module):
    def __init__(self, d_in=2 * DIM, hidden=(128, 64), dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        # feature standardization (fitted on the training set, saved with the weights)
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sigma", torch.ones(d_in))

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma).squeeze(-1)


def embed(texts, kv):
    """[mean ; max] pooling of word2vec vectors of each normalized message."""
    out = np.zeros((len(texts), 2 * kv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [kv[w] for w in tokenize(t) if w in kv.key_to_index]
        if vecs:
            v = np.stack(vecs)
            out[i, :kv.vector_size] = v.mean(0)
            out[i, kv.vector_size:] = v.max(0)
    return out


class W2VFFNModel:
    def __init__(self, kv, ffn):
        self.kv, self.ffn = kv, ffn.eval()

    @torch.no_grad()
    def predict_proba(self, texts):
        x = torch.from_numpy(embed([normalize(t) for t in texts], self.kv))
        return torch.sigmoid(self.ffn(x)).numpy()

    @classmethod
    def load(cls):
        ffn = FFN()
        ffn.load_state_dict(torch.load(FFN_FILE, map_location="cpu"))
        return cls(KeyedVectors.load(str(W2V_FILE)), ffn)


def train(epochs=80, batch_size=64, lr=0.005, patience=10):
    from src.data import load
    from src.evaluate import best_threshold, report, set_threads

    set_threads()  # fixed CPU thread count -> identical float summation order -> reproducible
    set_seed()
    tr, va = load("train"), load("val")
    tr_norm = [normalize(t) for t in tr.text]
    va_norm = [normalize(t) for t in va.text]

    # 1) word2vec on training tweets only
    w2v = Word2Vec(sentences=[tokenize(t) for t in tr_norm], **W2V_PARAMS)
    kv = w2v.wv
    print(f"word2vec: {len(kv)} words x {kv.vector_size} dims")
    probes = ["şerefsiz", "salak", "aq", "güzel", "maç", "hoca"]
    neighbors = {w: [(n, round(float(s), 3)) for n, s in kv.most_similar(w, topn=6)]
                 for w in probes if w in kv.key_to_index}
    for w, ns in neighbors.items():
        print(f"  {w:10} -> {', '.join(n for n, _ in ns)}")
    oov = np.mean([not any(w in kv.key_to_index for w in tokenize(t)) for t in va_norm])
    print(f"val messages with no in-vocabulary word: {oov:.2%}")

    # 2) features + FFN with SGD
    Xtr, Xva = torch.from_numpy(embed(tr_norm, kv)), torch.from_numpy(embed(va_norm, kv))
    ytr = torch.tensor(tr.label.values, dtype=torch.float32)
    pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
    ffn = FFN()
    ffn.mu.copy_(Xtr.mean(0))
    ffn.sigma.copy_(Xtr.std(0) + 1e-6)
    opt = torch.optim.SGD(ffn.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    g = torch.Generator().manual_seed(SEED)

    best, history, wait = (-1, None, None), [], 0
    for ep in range(1, epochs + 1):
        ffn.train()
        perm = torch.randperm(len(Xtr), generator=g)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(ffn(Xtr[idx]), ytr[idx])
            loss.backward()          # backpropagation
            opt.step()               # SGD update
            total += loss.item() * len(idx)
        ffn.eval()
        with torch.no_grad():
            pva = torch.sigmoid(ffn(Xva)).numpy()
        t_ep = best_threshold(va.label, pva)
        f1 = f1_score(va.label, (pva >= t_ep).astype(int), average="macro")
        history.append({"epoch": ep, "train_loss": round(total / len(Xtr), 4),
                        "val_macro_f1": round(f1, 4), "threshold": t_ep})
        print(f"epoch {ep:2d} train loss {total / len(Xtr):.4f}  val macro-F1 {f1:.4f} (t={t_ep})")
        if f1 > best[0]:
            best, wait = (f1, ep, {k: v.clone() for k, v in ffn.state_dict().items()}), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"early stopping (best epoch {best[1]})")
                break

    ffn.load_state_dict(best[2])
    ffn.eval()
    with torch.no_grad():
        pva = torch.sigmoid(ffn(Xva)).numpy()
    t = best_threshold(va.label, pva)
    val_f1 = f1_score(va.label, (pva >= t).astype(int), average="macro")
    print(f"-> best epoch {best[1]}, threshold {t}, val macro-F1 {val_f1:.4f}")

    W2V_FILE.parent.mkdir(exist_ok=True)
    kv.save(str(W2V_FILE))
    torch.save(ffn.state_dict(), FFN_FILE)
    (ROOT / "results" / "w2v_ffn_training.json").write_text(json.dumps(
        {"word2vec": {**{k: v for k, v in W2V_PARAMS.items() if k != "hashfxn"}, "vocab_size": len(kv)},
         "nearest_neighbors": neighbors, "history": history, "best_epoch": best[1]},
        indent=2, ensure_ascii=False))

    w2v_files = [p for p in W2V_FILE.parent.glob(W2V_FILE.name + "*")]
    report("w2v_ffn", W2VFFNModel(kv, ffn).predict_proba, load("test"), t, w2v_files + [FFN_FILE],
           extra={"family": "neural network (own word2vec + FFN, SGD)", "precision": "fp32",
                  "params": {"ffn_parameters": sum(p.numel() for p in ffn.parameters()),
                             "w2v_vocab": len(kv), "dim": DIM, "best_epoch": best[1]},
                  "val_macro_f1": round(val_f1, 4)})


if __name__ == "__main__":
    train()
