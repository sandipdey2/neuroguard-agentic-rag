"""Project 3 lineage: supervised stroke-risk classification with class-imbalance handling.

* Stratified 80/20 split (random_state=42). The 20% test split is never used for fitting or selection.
* Candidates: logistic regression (class_weight="balanced"), random forest (balanced_subsample) and
  gradient boosting (inverse-frequency sample weights), all behind the same preprocessing
  (median imputation of BMI on the training fold, standardisation, one-hot encoding).
* Selection by 5-fold stratified cross-validated ROC-AUC on the training split only.
* Served model: the logistic regression wrapped in CalibratedClassifierCV (sigmoid, 5 folds), because class
  weighting inflates raw probabilities and a clinician-facing number must be calibrated (Brier score and a
  reliability curve are reported). Explanations are exact per-feature contributions to the log-odds of the
  uncalibrated logistic model, relative to the average training patient.
* Flag threshold: the highest probability threshold that keeps out-of-fold recall >= 0.80 on the training split.
Artifacts are cached in data/runtime (git-ignored) and retrained if the data file or sklearn version changes.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import RUNTIME
from .data import BINARY, CATEGORIES, CATEGORICAL, FEATURES, NUMERIC, STROKE_CSV, file_sha256, load_stroke

SEED = 42
TARGET_RECALL = 0.80
MODEL_VERSION = "v1"


def _pre() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(categories=[CATEGORIES[c] for c in CATEGORICAL], drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])


def candidates() -> dict:
    return {
        "logistic_regression": Pipeline([("pre", _pre()), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))]),
        "random_forest": Pipeline([("pre", _pre()), ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                                                   class_weight="balanced_subsample", random_state=SEED, n_jobs=1))]),
        "gradient_boosting": Pipeline([("pre", _pre()), ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                                                                                           random_state=SEED))]),
    }


def _fit(name: str, pipe: Pipeline, X, y):
    if name == "gradient_boosting":
        return pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    return pipe.fit(X, y)


def _metrics(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"roc_auc": round(roc_auc_score(y, p), 3), "pr_auc": round(average_precision_score(y, p), 3),
            "brier": round(brier_score_loss(y, p), 4), "threshold": round(float(thr), 4),
            "recall": round(recall_score(y, pred), 3), "precision": round(precision_score(y, pred, zero_division=0), 3),
            "f1": round(f1_score(y, pred), 3), "specificity": round(tn / (tn + fp), 3),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


@dataclass
class RiskModel:
    served: CalibratedClassifierCV
    explainer: Pipeline
    threshold: float
    report: dict
    baseline_logit_terms: np.ndarray = field(repr=False, default=None)
    feature_names: list = field(default_factory=list)
    test_probs: list = field(default_factory=list, repr=False)

    def predict(self, profile: dict) -> float:
        return float(self.served.predict_proba(pd.DataFrame([profile])[FEATURES])[0, 1])

    def explain(self, profile: dict, top: int = 6) -> list[dict]:
        pre, clf = self.explainer.named_steps["pre"], self.explainer.named_steps["clf"]
        x = pre.transform(pd.DataFrame([profile])[FEATURES])
        x = np.asarray(x.todense() if hasattr(x, "todense") else x).ravel()
        contrib = (x - self.baseline_logit_terms) * clf.coef_.ravel()
        grouped: dict[str, float] = {}
        for f, c in zip(self.feature_names, contrib):
            base = next((cat for cat in CATEGORICAL if f.startswith(cat + "_")), f)
            grouped[base] = grouped.get(base, 0.0) + float(c)
        rows = [{"feature": f, "patient_value": profile.get(f), "log_odds_contribution": round(c, 3),
                 "direction": "raises" if c > 0 else "lowers"} for f, c in grouped.items()]
        return sorted(rows, key=lambda r: -abs(r["log_odds_contribution"]))[:top]

    def band(self, p: float) -> str:
        prev = self.report["data"]["prevalence"]
        if p >= self.threshold:
            return "high"
        return "elevated" if p >= prev else "low"


def train() -> RiskModel:
    df, _ = load_stroke()
    X, y = df[FEATURES], df["stroke"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    comparison = {}
    for name, pipe in candidates().items():
        oof = np.zeros(len(ytr))
        for tr, va in cv.split(Xtr, ytr):
            m = _fit(name, clone(pipe), Xtr.iloc[tr], ytr[tr])
            oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
        fitted = _fit(name, clone(pipe), Xtr, ytr)
        pt = fitted.predict_proba(Xte)[:, 1]
        comparison[name] = {"cv_roc_auc": round(roc_auc_score(ytr, oof), 3), "test_roc_auc": round(roc_auc_score(yte, pt), 3),
                            "test_pr_auc": round(average_precision_score(yte, pt), 3), "test_brier_uncalibrated": round(brier_score_loss(yte, pt), 4)}
    served = CalibratedClassifierCV(candidates()["logistic_regression"], method="sigmoid", cv=5).fit(Xtr, ytr)
    oof_cal = cross_val_predict(CalibratedClassifierCV(candidates()["logistic_regression"], method="sigmoid", cv=5),
                                Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    thr_grid = np.unique(np.round(oof_cal, 4))[::-1]
    thr = next(t for t in thr_grid if recall_score(ytr, (oof_cal >= t).astype(int)) >= TARGET_RECALL)
    pte = served.predict_proba(Xte)[:, 1]
    frac_pos, mean_pred = calibration_curve(yte, pte, n_bins=8, strategy="quantile")
    explainer = candidates()["logistic_regression"].fit(Xtr, ytr)
    pre = explainer.named_steps["pre"]
    Ztr = pre.transform(Xtr)
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    best = max(comparison, key=lambda k: comparison[k]["cv_roc_auc"])
    report = {
        "data": {"rows": len(df), "train": len(ytr), "test": len(yte), "prevalence": round(float(y.mean()), 4),
                 "test_positives": int(yte.sum()), "sha256": file_sha256(STROKE_CSV)},
        "comparison": comparison, "best_by_cv_auc": best,
        "served_model": "logistic_regression + sigmoid calibration (CalibratedClassifierCV, cv=5)",
        "threshold_rule": f"highest threshold with out-of-fold recall >= {TARGET_RECALL} on the training split",
        "test_metrics": _metrics(yte, pte, thr),
        "calibration_curve": {"mean_predicted": [round(float(v), 4) for v in mean_pred], "observed_rate": [round(float(v), 4) for v in frac_pos]},
        "coefficients": {n: round(float(c), 3) for n, c in zip(names, explainer.named_steps["clf"].coef_.ravel())},
        "sklearn": sklearn.__version__,
    }
    return RiskModel(served=served, explainer=explainer, threshold=float(thr), report=report,
                     baseline_logit_terms=np.asarray(Ztr.mean(axis=0)).ravel(), feature_names=names,
                     test_probs=[[int(a), round(float(b), 5)] for a, b in zip(yte, pte)])


_lock = threading.Lock()
_model: RiskModel | None = None


def get_model() -> RiskModel:
    global _model
    with _lock:
        if _model is None:
            RUNTIME.mkdir(parents=True, exist_ok=True)
            path = RUNTIME / f"risk_model_{MODEL_VERSION}_{file_sha256(STROKE_CSV)[:12]}_{sklearn.__version__}.joblib"
            if path.exists():
                try:
                    _model = joblib.load(path)
                except Exception:
                    _model = None
            if _model is None:
                _model = train()
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                joblib.dump(_model, tmp)
                os.replace(tmp, path)  # atomic: concurrent MCP subprocesses never read a half-written file
        return _model
