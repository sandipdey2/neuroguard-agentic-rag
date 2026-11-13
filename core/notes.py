"""Project 4 lineage: text modelling for free-text clinical notes.

Two layers, deliberately separated:
1. ``red_flags`` - deterministic matching of the sudden-onset warning signs published by CDC (BE FAST and
   "Signs and Symptoms of Stroke"). This drives the emergency guardrail; it is a rule, not a model, because a
   missed emergency is the costliest error and rules are auditable.
2. ``NoteClassifier`` - TF-IDF (word 1-2 grams) + class-balanced logistic regression trained on the public
   MTSamples transcriptions (CC0) to estimate whether a note reads like a neurology/neurosurgery note.
   It is an advisory routing signal only.

Why not the Project 4 LSTM: on ~5,000 short documents a sparse linear model trains in seconds on Streamlit
Community Cloud's CPU, needs no torch wheel (hundreds of MB), and its weights are inspectable. The notebook
reports its held-out metrics honestly (MTSamples specialty labels overlap, so F1 is modest).
"""
from __future__ import annotations

import re
import os
import threading

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn

from .config import RUNTIME
from .data import NOTES_CSV, file_sha256, load_notes

NOTE_MODEL_VERSION = "v1"
SUDDEN = r"(sudden(ly)?|abrupt(ly)?|just now|minutes? ago|this morning|started (today|an hour)|out of nowhere|right now)"
RED_FLAGS = {
    "face_droop": r"(face|facial|mouth|smile)\W+(\w+\W+){0,3}(droop|drooping|uneven|numb|weak)|(droop\w*)\W+(\w+\W+){0,3}(face|mouth)",
    "arm_or_leg_weakness": r"(arm|leg|hand|limb|one side|side of (the|my|his|her) body)\W+(\w+\W+){0,4}(weak|weakness|numb|numbness|drift|can't lift|cannot lift|paraly)|(weak|weakness|numb|numbness|paraly\w*)\W+(\w+\W+){0,4}(arm|leg|hand|one side|face)",
    "speech_difficulty": r"(slurred|slurring|garbled|trouble speaking|difficulty speaking|can't speak|cannot speak|can't talk|cannot talk|aphasia|trouble understanding speech|words? (are|is) (jumbled|mixed))",
    "vision_change": r"(sudden\w*\W+(\w+\W+){0,3}(vision|see|seeing|blind)|(lost|loss of|double|blurred) vision|trouble seeing)",
    "balance_loss": r"(sudden\w*\W+(\w+\W+){0,3}(dizz|balance|coordination|walking)|loss of balance|can't walk|cannot walk|lack of coordination)",
    "worst_headache": r"(worst headache|thunderclap|sudden\w*\W+(\w+\W+){0,3}severe headache|severe headache with no (known )?cause)",
    "confusion": r"sudden\w*\W+(\w+\W+){0,3}(confus|disorient)",
}
NEGATION = r"\b(no|denies|denied|without|negative for|not)\W+(\w+\W+){0,3}$"


def red_flags(text: str) -> dict:
    t = (text or "").lower()
    hits = []
    for name, pat in RED_FLAGS.items():
        for m in re.finditer(pat, t):
            prefix = t[max(0, m.start() - 40):m.start()]
            if re.search(NEGATION, prefix):
                continue
            hits.append({"sign": name, "matched": m.group(0)[:60]})
            break
    sudden = bool(re.search(SUDDEN, t))
    # a single BE-FAST sign with sudden onset, or two or more signs, is treated as a possible emergency
    emergency = (len(hits) >= 1 and sudden) or len(hits) >= 2 or any(h["sign"] == "worst_headache" for h in hits)
    return {"signs": hits, "sudden_onset_language": sudden, "possible_emergency": emergency,
            "rule_source": "CDC Signs and Symptoms of Stroke / BE FAST (https://www.cdc.gov/stroke/signs-symptoms/index.html)"}


class NoteClassifier:
    def __init__(self):
        mt = load_notes()
        Xtr, Xte, ytr, yte = train_test_split(mt["text"], mt["label"], test_size=0.2, stratify=mt["label"], random_state=42)
        self.pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")),
                              ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=4.0))]).fit(Xtr, ytr)
        p = self.pipe.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        self.report = {"dataset": "MTSamples (Kaggle tboyle10/medicaltranscriptions, CC0)", "docs": len(mt),
                       "positives": int(mt["label"].sum()), "train": len(ytr), "test": len(yte),
                       "label": "medical_specialty in {Neurology, Neurosurgery}",
                       "test_roc_auc": round(roc_auc_score(yte, p), 3), "test_pr_auc": round(average_precision_score(yte, p), 3),
                       "threshold": 0.5, "test_precision": round(precision_score(yte, pred), 3),
                       "test_recall": round(recall_score(yte, pred), 3), "test_f1": round(f1_score(yte, pred), 3)}
        vocab = np.array(self.pipe.named_steps["tfidf"].get_feature_names_out())
        coef = self.pipe.named_steps["clf"].coef_.ravel()
        self.report["top_neuro_terms"] = vocab[np.argsort(coef)[::-1][:15]].tolist()

    def score(self, text: str) -> float:
        return float(self.pipe.predict_proba([text])[0, 1])


_lock = threading.Lock()
_clf: NoteClassifier | None = None


def get_note_classifier() -> NoteClassifier:
    global _clf
    with _lock:
        if _clf is None:
            RUNTIME.mkdir(parents=True, exist_ok=True)
            path = RUNTIME / f"note_clf_{NOTE_MODEL_VERSION}_{file_sha256(NOTES_CSV)[:12]}_{sklearn.__version__}.joblib"
            if path.exists():
                try:
                    _clf = joblib.load(path)
                except Exception:
                    _clf = None
            if _clf is None:
                _clf = NoteClassifier()
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                joblib.dump(_clf, tmp)
                os.replace(tmp, path)  # atomic: concurrent MCP subprocesses never read a half-written file
        return _clf


def screen_note(text: str) -> dict:
    rf = red_flags(text)
    clf = get_note_classifier()
    return {**rf, "neuro_note_probability": round(clf.score(text), 3) if text.strip() else None,
            "classifier_test_f1": clf.report["test_f1"],
            "classifier_note": "TF-IDF + logistic regression trained on MTSamples; advisory signal, not a diagnosis"}
