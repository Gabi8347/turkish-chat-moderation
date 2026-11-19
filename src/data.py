"""Download OffensEval-TR 2020 and build fixed train / val / test splits.

Source: Çöltekin (2020), "A Corpus of Turkish Offensive Language on Social Media", LREC.
https://coltekin.github.io/offensive-turkish/

- test  = the official OffensEval-TR 2020 test set (3,528 tweets) - never used for training
          or model selection; every model in the ladder is scored on exactly this file.
- val   = 10 % of the official training set (stratified, seed 42) - for early stopping,
          threshold / hyperparameter choice.
- train = the remaining 90 %.

Labels: 0 = NOT (clean), 1 = OFF (offensive).

Run:  python -m src.data
"""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
BASE = RAW / "offenseval2020-turkish"
SEED = 42
LABELS = {"NOT": 0, "OFF": 1}


def download():
    if BASE.exists():
        return
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(RAW)


def _read_tsv(path, sep="\t"):
    # The files have no quoting: split each line manually (see the corpus README).
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(sep)
        rows = [line.rstrip("\n").split(sep, len(header) - 1) for line in f if line.strip()]
    return pd.DataFrame(rows, columns=header)


def build():
    download()
    train = _read_tsv(BASE / "offenseval-tr-training-v1" / "offenseval-tr-training-v1.tsv")
    train = train.rename(columns={"tweet": "text", "subtask_a": "label"})

    test = _read_tsv(BASE / "offenseval-tr-testset-v1" / "offenseval-tr-testset-v1.tsv")
    with open(BASE / "offenseval-tr-testset-v1" / "offenseval-tr-labela-v1.tsv", encoding="utf-8") as f:
        gold = dict(line.strip().split(",") for line in f if line.strip())
    test = test.rename(columns={"tweet": "text"})
    test["label"] = test["id"].map(gold)
    assert test["label"].notna().all(), "every test tweet must have a gold label"

    for df in (train, test):
        df["label"] = df["label"].map(LABELS)

    tr, va = train_test_split(train, test_size=0.10, stratify=train["label"], random_state=SEED)
    PROC.mkdir(parents=True, exist_ok=True)
    splits = {"train": tr, "val": va, "test": test}
    stats = {}
    for name, df in splits.items():
        df[["id", "text", "label"]].to_json(PROC / f"{name}.jsonl", orient="records",
                                            lines=True, force_ascii=False)
        stats[name] = {"n": len(df), "offensive": int(df["label"].sum()),
                       "offensive_ratio": round(float(df["label"].mean()), 4)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def load(split):
    """Return a DataFrame with columns id, text, label for 'train', 'val' or 'test'."""
    path = PROC / f"{split}.jsonl"
    if not path.exists():
        build()
    return pd.read_json(path, lines=True, dtype={"id": str, "text": str, "label": int})


if __name__ == "__main__":
    build()
