"""Model registry for the application: one entry per rung of the ladder.

Each model is loaded lazily the first time it is selected.  If its files are not in
models/ (fresh clone), they are downloaded from the Hugging Face Hub repository
MODEL_REPO (override with the environment variable MODEL_REPO).
Decision thresholds are the validation-set thresholds stored in results/results.json.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODEL_REPO = os.environ.get("MODEL_REPO", "Gabi8347/turkish-chat-moderation-models")
RESULTS = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
DEFAULT_MODEL = "berturk_onnx_int8"


@dataclass
class Entry:
    key: str            # key in results/results.json
    label: str          # shown in the UI
    files: list         # paths relative to models/ (downloaded if missing)
    loader: Callable    # () -> object with predict_proba(list[str])


def _lexicon():
    from src.lexicon import LexiconMatcher
    return LexiconMatcher()


def _tfidf():
    from src.tfidf import TfidfModel
    return TfidfModel.load()


def _w2v():
    from src.w2v_ffn import W2VFFNModel
    return W2VFFNModel.load()


def _bert_fp32():
    from src.berturk import BertClassifier
    return BertClassifier.load()


def _bert_int8():
    from src.quantize import load_pytorch_int8
    return load_pytorch_int8()


def _onnx(name):
    def load():
        from src.quantize import ONNX_DIR, OnnxClassifier
        return OnnxClassifier(ONNX_DIR / name)
    return load


_TOK = ["tokenizer.json", "tokenizer_config.json"]
ENTRIES = [
    Entry("berturk_onnx_int8", "BERTurk INT8 · ONNX Runtime (quantized, default)",
          [f"berturk-offensive-onnx/{f}" for f in ["model_int8.onnx", *_TOK]], _onnx("model_int8.onnx")),
    Entry("berturk_int8_pytorch", "BERTurk INT8 · PyTorch dynamic",
          [f"berturk-offensive-int8/{f}" for f in ["model_int8.pt", "config.json", *_TOK]], _bert_int8),
    Entry("berturk_onnx_fp32", "BERTurk FP32 · ONNX Runtime",
          [f"berturk-offensive-onnx/{f}" for f in ["model_fp32.onnx", *_TOK]], _onnx("model_fp32.onnx")),
    Entry("berturk_fp32", "BERTurk FP32 · PyTorch",
          [f"berturk-offensive/{f}" for f in ["model.safetensors", "config.json", *_TOK]], _bert_fp32),
    Entry("w2v_ffn", "word2vec + feedforward network", ["w2v_tweets.kv", "ffn.pt"], _w2v),
    Entry("tfidf_logreg", "tf-idf + logistic regression", ["tfidf_logreg.joblib"], _tfidf),
    Entry("lexicon_rules", "Lexicon + weighted edit distance (rules)", [], _lexicon),
]
BY_KEY = {e.key: e for e in ENTRIES}
BY_LABEL = {e.label: e for e in ENTRIES}
_cache = {}


def ensure_files(entry: Entry):
    missing = [f for f in entry.files if not (MODELS / f).exists()]
    if missing:
        from huggingface_hub import hf_hub_download
        for f in missing:
            print(f"downloading {f} from https://huggingface.co/{MODEL_REPO}")
            hf_hub_download(MODEL_REPO, f, local_dir=MODELS)


def get(key: str):
    """Return (model, threshold) for a results.json key, loading it on first use."""
    if key not in _cache:
        entry = BY_KEY[key]
        ensure_files(entry)
        model = entry.loader()
        for _ in range(3):  # warm-up: the first calls pay one-off costs (kernels, caches)
            model.predict_proba(["ısınma mesajı"])
        _cache[key] = model
    return _cache[key], RESULTS[key]["threshold"]
