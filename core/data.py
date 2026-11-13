"""Project 1 lineage: a small, explicit, reproducible data pipeline for the public Stroke Prediction Dataset.

Steps (each one is logged in ``PIPELINE_LOG`` so the notebook and model card can show them):
load -> schema check -> type coercion ("N/A" BMI to NaN) -> drop the single gender == "Other" row
-> train-split median imputation of BMI (done inside the sklearn pipeline, not here, to avoid leakage).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import pandas as pd

from .config import DATA, RAW

STROKE_CSV = RAW / "healthcare-dataset-stroke-data.csv"
NOTES_CSV = RAW / "mtsamples_subset.csv"
EXPECTED_COLUMNS = ["id", "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
                    "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"]
NUMERIC = ["age", "avg_glucose_level", "bmi"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
CATEGORIES = {
    "gender": ["Female", "Male"],
    "ever_married": ["No", "Yes"],
    "work_type": ["Govt_job", "Never_worked", "Private", "Self-employed", "children"],
    "Residence_type": ["Rural", "Urban"],
    "smoking_status": ["Unknown", "formerly smoked", "never smoked", "smokes"],
}


def file_sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@lru_cache(maxsize=1)
def load_stroke() -> tuple[pd.DataFrame, tuple]:
    log = []
    raw = pd.read_csv(STROKE_CSV)
    log.append(("loaded", f"{len(raw)} rows x {raw.shape[1]} columns"))
    missing = set(EXPECTED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"schema check failed, missing columns: {missing}")
    log.append(("schema check", "12 expected columns present"))
    df = raw.copy()
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    log.append(("bmi coerced", f"{int(df['bmi'].isna().sum())} 'N/A' values -> NaN (imputed later on the training split)"))
    n_other = int((df["gender"] == "Other").sum())
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    log.append(("gender filter", f"dropped {n_other} row with gender 'Other' (too rare to model or explain)"))
    dup = int(df.duplicated(subset=[c for c in df.columns if c != "id"]).sum())
    log.append(("duplicate check", f"{dup} duplicate records (excluding id)"))
    log.append(("target", f"{int(df['stroke'].sum())} stroke cases of {len(df)} ({df['stroke'].mean():.2%})"))
    return df, tuple(log)


def pipeline_log() -> list[tuple[str, str]]:
    return list(load_stroke()[1])


def load_notes() -> pd.DataFrame:
    mt = pd.read_csv(NOTES_CSV)
    mt["text"] = (mt["description"].fillna("") + " " + mt["transcription_head"].fillna("")).str.strip()
    mt["label"] = mt["medical_specialty"].str.strip().isin(["Neurology", "Neurosurgery"]).astype(int)
    return mt[mt["text"].str.len() > 0].reset_index(drop=True)


@lru_cache(maxsize=1)
def load_guidelines() -> list[dict]:
    return json.loads((DATA / "guidelines.json").read_text(encoding="utf-8"))


def validate_profile(p: dict) -> dict:
    """Coerce and range-check a patient profile typed into the UI. Raises ValueError with a readable message."""
    out = {}
    try:
        out["age"] = float(p["age"])
        out["avg_glucose_level"] = float(p["avg_glucose_level"])
        out["bmi"] = None if p.get("bmi") in (None, "", "unknown") else float(p["bmi"])
        out["hypertension"] = int(bool(int(p["hypertension"])))
        out["heart_disease"] = int(bool(int(p["heart_disease"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid profile field: {exc}") from exc
    if not 0 < out["age"] <= 110:
        raise ValueError("age must be between 0 and 110")
    if not 40 <= out["avg_glucose_level"] <= 400:
        raise ValueError("average glucose must be between 40 and 400 mg/dL")
    if out["bmi"] is not None and not 10 <= out["bmi"] <= 80:
        raise ValueError("BMI must be between 10 and 80")
    for col, allowed in CATEGORIES.items():
        v = p.get(col)
        if v not in allowed:
            raise ValueError(f"{col} must be one of {allowed}")
        out[col] = v
    return out
