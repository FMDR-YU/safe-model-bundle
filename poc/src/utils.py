"""Shared utilities: hashing, environment capture, metrics, model registry."""
from __future__ import annotations
import hashlib, hmac, json, os, platform, subprocess, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:                       # pragma: no cover
    HAS_XGB = False


# --------------------------------------------------------------------------- #
# Hashing / integrity
# --------------------------------------------------------------------------- #
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_df(df: pd.DataFrame) -> str:
    """Deterministic content hash of a dataframe (column order included)."""
    return sha256_bytes(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
        + ",".join(df.columns).encode()
    )


def hmac_sign(digest: str, key: bytes = b"poc-demo-escrow-key") -> str:
    """Stand-in for an escrow authority's signature over the manifest digest."""
    return hmac.new(key, digest.encode(), hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------- #
# Environment capture (Gate 3)
# --------------------------------------------------------------------------- #
def capture_environment() -> dict:
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        freeze = ""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pip_freeze": freeze,
    }
    env["env_hash"] = sha256_bytes(json.dumps(env, sort_keys=True).encode())
    return env


# --------------------------------------------------------------------------- #
# Model registry  (defended/undefended tree models + formal-DP comparator)
# --------------------------------------------------------------------------- #
def make_model(kind: str, seed: int, epsilon: float = 1.0):
    """
    kind:
      'leaky'  -> high-capacity default model fit to a small cohort (current bad
                  practice; overfits -> leaks). NOT an artificially broken model:
                  it is library defaults applied in a small-sample/rare-outcome regime.
      'safe'   -> the same task done disclosure-aware: a regularised, shallow model
                  that keeps generalisation while not memorising members.
      'dp'      -> formally differentially-private logistic regression (supplementary
                   epsilon-sweep only); `epsilon` selects the privacy budget.
      'portable_rf' -> sklearn RF for the clean ONNX/skops parity demo.
    tree_method="exact" + n_jobs=1: the exact greedy split-finder has a fixed
    floating-point reduction order, so fits are bit-reproducible run-to-run.
    """
    if kind == "leaky":
        if HAS_XGB:
            return XGBClassifier(
                n_estimators=600, max_depth=11, learning_rate=0.3,
                subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0,
                reg_alpha=0.0, gamma=0.0, min_child_weight=1,
                eval_metric="logloss",
                random_state=seed, n_jobs=1, tree_method="exact",
            )
        return GradientBoostingClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.3,
            subsample=1.0, random_state=seed,
        )
    if kind == "safe":
        if HAS_XGB:
            return XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                reg_alpha=0.0, gamma=0.0, min_child_weight=6,
                eval_metric="logloss",
                random_state=seed, n_jobs=1, tree_method="exact",
            )
        return RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=25,
            random_state=seed, n_jobs=1,
        )
    if kind == "dp":
        try:
            from diffprivlib.models import LogisticRegression as DPLogReg
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            # DP is randomised by design (fresh privacy noise per release); we
            # seed it here only so the demo is stable run-to-run.
            return Pipeline([
                ("scale", StandardScaler()),
                ("clf", DPLogReg(epsilon=epsilon, data_norm=6.0, random_state=42)),
            ])
        except Exception:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            return Pipeline([("scale", StandardScaler()),
                             ("clf", LogisticRegression(C=0.05, max_iter=1000))])
    if kind == "portable_rf":
        # sklearn-native model used for the clean ONNX/skops parity demo
        return RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            random_state=seed, n_jobs=1,
        )
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def predict_proba_pos(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p


def per_example_loss(model, X, y) -> np.ndarray:
    p = np.clip(predict_proba_pos(model, X), 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def model_quality(model, X, y) -> dict:
    p = predict_proba_pos(model, X)
    return {
        "auc": float(roc_auc_score(y, p)),
        "acc": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
    }


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_table(df: pd.DataFrame, name: str, out_dir: str):
    csv = os.path.join(out_dir, name + ".csv")
    df.to_csv(csv, index=False)
    md = os.path.join(out_dir, name + ".md")
    with open(md, "w") as f:
        f.write(df.to_markdown(index=False))
    return csv
