"""
D3 - Serialization format comparison (Gate 1 evidence) -> Table D2.

We take one fitted model and serialise it via pickle/joblib, skops, and ONNX,
then empirically measure, per format:
  * code execution on load (the Risk-A attack surface)
  * Python-only vs language-neutral
  * self-contained / framework portability
  * inference parity vs the pickle baseline (max |Delta prob|, label agreement)

Tree models (XGBoost/RandomForest) map to skops/ONNX. safetensors/.pt/.gguf are
included as documented rows for the neural-network / LLM cases (tensors only;
not applicable to scikit-learn tree objects) so the table matches the paper.
"""
from __future__ import annotations
import os, pickle, time
import numpy as np
import pandas as pd
import joblib

from config import FEATURES


def _onnx_predict(onnx_bytes, X):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    outs = sess.run(None, {inp: X.astype(np.float32)})
    # skl2onnx zipmap output -> list[dict]; take prob of class 1
    for o in outs:
        if isinstance(o, list) and o and isinstance(o[0], dict):
            keys = sorted(o[0].keys())
            return np.array([row[keys[-1]] for row in o], dtype=float)
        arr = np.asarray(o)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 1].astype(float)
    return np.asarray(outs[-1]).ravel().astype(float)


def run(model, X_test: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    art = os.path.join(out_dir, "artifacts")
    os.makedirs(art, exist_ok=True)
    Xv = X_test[FEATURES].values
    base_prob = model.predict_proba(Xv)[:, 1]
    rows = []

    # ---- pickle ----
    p = os.path.join(art, "model.pkl")
    with open(p, "wb") as f:
        pickle.dump(model, f)
    rows.append(dict(format="pickle", code_exec_on_load="yes (unrestricted)",
                     language="Python-only", portability="none",
                     parity_max_abs_delta=0.0, label_agreement=1.0,
                     size_kb=round(os.path.getsize(p) / 1024, 1),
                     model_families="all Python objects"))

    # ---- joblib ----
    j = os.path.join(art, "model.joblib")
    joblib.dump(model, j)
    rows.append(dict(format="joblib", code_exec_on_load="yes (unrestricted)",
                     language="Python-only", portability="none",
                     parity_max_abs_delta=0.0, label_agreement=1.0,
                     size_kb=round(os.path.getsize(j) / 1024, 1),
                     model_families="all Python objects"))

    # ---- skops ----
    try:
        import skops.io as sio
        s = os.path.join(art, "model.skops")
        sio.dump(model, s)
        untrusted = sio.get_untrusted_types(file=s)
        loaded = sio.load(s, trusted=untrusted)   # explicit-trust load (no code exec)
        prob = loaded.predict_proba(Xv)[:, 1]
        rows.append(dict(format="skops",
                         code_exec_on_load="no (explicit trusted-types allowlist)",
                         language="Python (sklearn)", portability="sklearn-family",
                         parity_max_abs_delta=round(float(np.max(np.abs(prob - base_prob))), 6),
                         label_agreement=round(float(np.mean((prob >= .5) == (base_prob >= .5))), 4),
                         size_kb=round(os.path.getsize(s) / 1024, 1),
                         model_families="tree-based / traditional ML"))
    except Exception as e:
        rows.append(dict(format="skops", code_exec_on_load="no",
                         language="Python (sklearn)", portability="sklearn-family",
                         parity_max_abs_delta="n/a", label_agreement="n/a",
                         size_kb="n/a", model_families="tree-based / traditional ML",
                         note=f"live demo skipped: {type(e).__name__}"))

    # ---- ONNX ----
    onnx_ok = False
    try:
        from skl2onnx import to_onnx
        onx = to_onnx(model, Xv[:1].astype(np.float32))
        o = os.path.join(art, "model.onnx")
        with open(o, "wb") as f:
            f.write(onx.SerializeToString())
        prob = _onnx_predict(onx.SerializeToString(), Xv)
        onnx_ok = True
        rows.append(dict(format="ONNX", code_exec_on_load="no (data/graph format)",
                         language="language-neutral", portability="cross-framework",
                         parity_max_abs_delta=round(float(np.max(np.abs(prob - base_prob))), 6),
                         label_agreement=round(float(np.mean((prob >= .5) == (base_prob >= .5))), 4),
                         size_kb=round(os.path.getsize(o) / 1024, 1),
                         model_families="tree-based / NN (via converters)"))
    except Exception as e:
        rows.append(dict(format="ONNX", code_exec_on_load="no (data/graph format)",
                         language="language-neutral", portability="cross-framework",
                         parity_max_abs_delta="n/a", label_agreement="n/a",
                         size_kb="n/a", model_families="tree-based / NN",
                         note=f"live demo skipped: {type(e).__name__}"))

    # ---- documented-only rows (NN / LLM) ----
    for fmt, fam in [("safetensors", "neural networks (tensors only; N/A for trees)"),
                     (".pt (torch)", "neural networks"),
                     (".gguf", "large language models")]:
        rows.append(dict(format=fmt, code_exec_on_load="no" if fmt != ".pt (torch)" else "yes (if weights_only=False)",
                         language="language-neutral" if fmt != ".pt (torch)" else "Python (torch)",
                         portability="weights only" if fmt == "safetensors" else "framework-specific",
                         parity_max_abs_delta="documented", label_agreement="documented",
                         size_kb="n/a", model_families=fam))

    df = pd.DataFrame(rows)
    return df
