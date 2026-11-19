"""Efficient-model rung: quantization of the fine-tuned BERTurk (technique area D).

Two INT8 variants of the same FP32 model (models/berturk-offensive), both post-training
*dynamic* quantization: weights of every Linear layer are stored as INT8 (symmetric,
affine map q = round(x / scale) + zero_point); activations are quantized on the fly per
batch, so no calibration data and no retraining are needed.

  1. PyTorch  : torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, qint8)
  2. ONNX     : export the FP32 graph to ONNX, then onnxruntime.quantization.quantize_dynamic
                (QInt8 weights) and run it with ONNX Runtime (graph optimizations: operator
                fusion, constant folding)

The decision threshold is the one chosen on the validation set for the FP32 model, so the
quantized models get no extra tuning.  All variants are measured with the same benchmark
(src/evaluate.py: test set, size on disk, single-message latency on 4 CPU threads).

Run:  python -m src.quantize
"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.berturk import INFO_FILE, MAX_LEN, MODEL_DIR, BertClassifier
from src.evaluate import N_THREADS
from src.normalize import normalize

ROOT = Path(__file__).resolve().parents[1]
INT8_DIR = ROOT / "models" / "berturk-offensive-int8"
ONNX_DIR = ROOT / "models" / "berturk-offensive-onnx"
ONNX_FP32 = ONNX_DIR / "model_fp32.onnx"
ONNX_INT8 = ONNX_DIR / "model_int8.onnx"


# ---------------------------------------------------------------- PyTorch dynamic INT8
def quantize_pytorch():
    from torch.ao.quantization import quantize_dynamic

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    qmodel = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    INT8_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), INT8_DIR / "model_int8.pt")
    model.config.save_pretrained(INT8_DIR)
    AutoTokenizer.from_pretrained(MODEL_DIR).save_pretrained(INT8_DIR)
    return qmodel


def load_pytorch_int8(path=INT8_DIR):
    from torch.ao.quantization import quantize_dynamic

    config = AutoConfig.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_config(config).eval()
    qmodel = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    qmodel.load_state_dict(torch.load(Path(path) / "model_int8.pt", map_location="cpu"))
    return BertClassifier(qmodel, AutoTokenizer.from_pretrained(path), "cpu")


# ---------------------------------------------------------------- ONNX Runtime INT8
def export_onnx():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    enc = tok(["örnek bir mesaj"], return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    axes = {n: {0: "batch", 1: "seq"} for n in names}
    axes["logits"] = {0: "batch"}
    torch.onnx.export(model, tuple(enc[n] for n in names), str(ONNX_FP32), input_names=names,
                      output_names=["logits"], dynamic_axes=axes, opset_version=17, dynamo=False)
    tok.save_pretrained(ONNX_DIR)


def quantize_onnx():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(ONNX_FP32), str(ONNX_INT8), weight_type=QuantType.QInt8)


class OnnxClassifier:
    def __init__(self, path, tokenizer_dir=ONNX_DIR, threads=N_THREADS):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def predict_proba(self, texts, batch_size=64):
        texts = [normalize(t) for t in texts]
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="np")
            logits = self.sess.run(["logits"], {n: enc[n].astype(np.int64) for n in self.inputs})[0]
            e = np.exp(logits - logits.max(-1, keepdims=True))
            out.append((e[:, 1] / e.sum(-1)).astype(np.float32))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------- run everything
def main():
    from src.data import load
    from src.evaluate import report

    info = json.loads(INFO_FILE.read_text())
    t = info["threshold"]
    test = load("test")
    common = {"family": "transformer (BERTurk fine-tuned)", "val_threshold_from": "berturk_fp32"}

    print("== PyTorch dynamic INT8")
    quantize_pytorch()
    report("berturk_int8_pytorch", load_pytorch_int8().predict_proba, test, t,
           [INT8_DIR / "model_int8.pt"],
           extra={**common, "precision": "int8 weights (dynamic), fp32 activations",
                  "method": "torch.ao.quantization.quantize_dynamic, nn.Linear -> qint8"})

    print("== ONNX export (FP32) + ONNX Runtime")
    export_onnx()
    report("berturk_onnx_fp32", OnnxClassifier(ONNX_FP32).predict_proba, test, t, [ONNX_FP32],
           extra={**common, "precision": "fp32", "method": "torch.onnx.export + ONNX Runtime"})

    print("== ONNX Runtime dynamic INT8")
    quantize_onnx()
    report("berturk_onnx_int8", OnnxClassifier(ONNX_INT8).predict_proba, test, t, [ONNX_INT8],
           extra={**common, "precision": "int8 weights (dynamic), fp32 activations",
                  "method": "onnxruntime.quantization.quantize_dynamic, QInt8"})


if __name__ == "__main__":
    main()
