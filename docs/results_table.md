Official OffensEval-TR 2020 test set (3,528 tweets). Latency: one message at a time, mean / p95 over 200 runs, 13th Gen Intel(R) Core(TM) i7-13650HX, CPU only, 4 threads, Windows 11. Budget: p95 ≤ 25 ms.

| Model | Rung | Precision | Macro-F1 | F1 (offensive) | Size (MB) | Latency mean (ms) | Latency p95 (ms) | Budget |
|---|---|---|---|---|---|---|---|---|
| Lexicon + edit distance | classical (rules) | n/a | 0.673 | 0.434 | 0.001 | 4.04 | 9.83 | ✅ |
| tf-idf + logistic regression | classical baseline | fp64 | 0.767 | 0.626 | 2.767 | 1.51 | 1.48 | ✅ |
| word2vec + FFN | neural network | fp32 | 0.691 | 0.507 | 13.082 | 0.13 | 0.20 | ✅ |
| BERTurk FP32 (PyTorch) | Transformer | fp32 | 0.821 | 0.718 | 443.255 | 20.16 | 34.95 | ❌ |
| BERTurk FP32 (ONNX Runtime) | Transformer | fp32 | 0.821 | 0.718 | 442.727 | 16.74 | 27.85 | ❌ |
| BERTurk INT8 (PyTorch dynamic) | quantized Transformer | int8 weights (dynamic), fp32 activations | 0.818 | 0.711 | 186.036 | 14.16 | 20.20 | ✅ |
| **BERTurk INT8 (ONNX Runtime)** (app default) | quantized Transformer | int8 weights (dynamic), fp32 activations | 0.823 | 0.719 | 111.434 | 6.97 | 11.99 | ✅ |
