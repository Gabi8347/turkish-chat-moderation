"""Classical learned baseline: tf-idf + logistic regression (technique area B).

Features on normalized text (src/normalize.py):
  - word uni+bigrams tf-idf (regex tokenizer)
  - character 2-5-gram tf-idf inside word boundaries - robust to Turkish suffixes and to
    spelling noise ("şerefsizler" / "serefsizlerr" share most character n-grams)
Logistic regression with class_weight="balanced" (19 % offensive).
The regularization strength C and the decision threshold are chosen on the VALIDATION set
(macro-F1); the test set is used once, for the final numbers.

Run:  python -m src.tfidf
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src.normalize import normalize, tokenize

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "tfidf_logreg.joblib"
SEED = 42


def build_pipeline(C=1.0):
    features = FeatureUnion([
        ("word", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False,
                                 min_df=3, sublinear_tf=True, max_features=300_000)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED)
    return Pipeline([("features", features), ("clf", clf)])


class TfidfModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, texts):
        return self.pipeline.predict_proba([normalize(t) for t in texts])[:, 1]

    @classmethod
    def load(cls, path=MODEL_FILE):
        return cls(joblib.load(path))


def train():
    from src.data import load
    from src.evaluate import best_threshold, report

    tr, va = load("train"), load("val")
    Xtr, Xva = [normalize(t) for t in tr.text], [normalize(t) for t in va.text]

    best = None
    for C in [0.5, 1, 2, 4, 8, 16]:
        pipe = build_pipeline(C).fit(Xtr, tr.label)
        p = pipe.predict_proba(Xva)[:, 1]
        t = best_threshold(va.label, p)
        f1 = f1_score(va.label, (p >= t).astype(int), average="macro")
        print(f"C={C:<5} val macro-F1={f1:.4f} (threshold {t})")
        if best is None or f1 > best[0]:
            best = (f1, C, t, pipe)
    f1, C, t, pipe = best
    print(f"-> best C={C}, threshold={t}, val macro-F1={f1:.4f}")

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_FILE, compress=3)
    n_feat = len(pipe.named_steps["features"].transformer_list[0][1].vocabulary_) + \
        len(pipe.named_steps["features"].transformer_list[1][1].vocabulary_)
    report("tfidf_logreg", TfidfModel(pipe).predict_proba, load("test"), t, [MODEL_FILE],
           extra={"family": "classical (learned)", "precision": "fp64",
                  "params": {"C": C, "n_features": n_feat}, "val_macro_f1": round(f1, 4)})


if __name__ == "__main__":
    train()
