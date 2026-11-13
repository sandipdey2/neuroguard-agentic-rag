"""Project 2 lineage: hypothesis tests and effect sizes on the public stroke cohort.

* chi-square test of independence for each categorical / binary factor vs stroke
* Welch's t-test for each numeric factor (stroke vs no stroke)
* odds ratios with 95% Woolf (log) confidence intervals for binary exposures
* Wilson score intervals for subgroup stroke rates (used by the cohort_stats MCP tool)
All numbers are computed from the dataset at run time; nothing is hard-coded.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import pandas as pd
from scipy import stats

from .data import load_stroke

AGE_BANDS = [(0, 18), (18, 45), (45, 55), (55, 65), (65, 75), (75, 120)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def odds_ratio(exposed: pd.Series, outcome: pd.Series) -> dict:
    a = int(((exposed == 1) & (outcome == 1)).sum()); b = int(((exposed == 1) & (outcome == 0)).sum())
    c = int(((exposed == 0) & (outcome == 1)).sum()); d = int(((exposed == 0) & (outcome == 0)).sum())
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5  # Haldane correction
    orr = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {"odds_ratio": round(orr, 2), "ci95": [round(math.exp(math.log(orr) - 1.96 * se), 2),
                                                   round(math.exp(math.log(orr) + 1.96 * se), 2)]}


@lru_cache(maxsize=1)
def factor_tests() -> dict:
    df, _ = load_stroke()
    y = df["stroke"]
    out = {"n": len(df), "stroke_cases": int(y.sum()), "categorical": [], "numeric": [], "odds_ratios": []}
    for col in ["hypertension", "heart_disease", "ever_married", "gender", "Residence_type", "work_type", "smoking_status"]:
        tab = pd.crosstab(df[col], y)
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        rates = (df.groupby(col)["stroke"].mean() * 100).round(2).to_dict()
        out["categorical"].append({"factor": col, "chi2": round(float(chi2), 2), "dof": int(dof), "p_value": float(f"{p:.3g}"),
                                   "stroke_rate_pct_by_level": {str(k): v for k, v in rates.items()}})
    for col in ["age", "avg_glucose_level", "bmi"]:
        s1, s0 = df.loc[y == 1, col].dropna(), df.loc[y == 0, col].dropna()
        t, p = stats.ttest_ind(s1, s0, equal_var=False)
        out["numeric"].append({"factor": col, "mean_stroke": round(float(s1.mean()), 2), "mean_no_stroke": round(float(s0.mean()), 2),
                               "welch_t": round(float(t), 2), "p_value": float(f"{p:.3g}"), "n_missing": int(df[col].isna().sum())})
    exposures = {"hypertension": df["hypertension"], "heart_disease": df["heart_disease"],
                 "age_65_plus": (df["age"] >= 65).astype(int), "glucose_ge_200": (df["avg_glucose_level"] >= 200).astype(int),
                 "bmi_ge_30": (df["bmi"] >= 30).astype(int), "ever_smoked": df["smoking_status"].isin(["smokes", "formerly smoked"]).astype(int)}
    for name, ex in exposures.items():
        out["odds_ratios"].append({"exposure": name, **odds_ratio(ex, y), "n_exposed": int(ex.sum())})
    return out


def age_band(age: float) -> tuple[int, int]:
    for lo, hi in AGE_BANDS:
        if lo <= age < hi:
            return lo, hi
    return AGE_BANDS[-1]


def cohort_stats(age: float, hypertension: int | None = None, heart_disease: int | None = None) -> dict:
    """Observed stroke rate among dataset patients in the same age band (optionally same hypertension / heart disease)."""
    df, _ = load_stroke()
    lo, hi = age_band(age)
    sub = df[(df["age"] >= lo) & (df["age"] < hi)]
    rows = [{"group": f"age {lo}-{hi - 1}", **_rate(sub)}]
    if hypertension is not None:
        s2 = sub[sub["hypertension"] == int(hypertension)]
        rows.append({"group": f"age {lo}-{hi - 1}, hypertension={int(hypertension)}", **_rate(s2)})
    if heart_disease is not None:
        s3 = sub[sub["heart_disease"] == int(heart_disease)]
        rows.append({"group": f"age {lo}-{hi - 1}, heart_disease={int(heart_disease)}", **_rate(s3)})
    best = rows[-1] if len(rows) > 1 and rows[-1]["n"] >= 30 else rows[0]
    headline = (f"Among dataset patients in group '{best['group']}', {best['strokes']} of {best['n']} had a recorded stroke "
                f"({best['rate'] * 100:.1f}%, Wilson 95% CI {best['wilson_ci95'][0] * 100:.1f}-{best['wilson_ci95'][1] * 100:.1f}%); "
                f"the whole dataset rate is {_rate(df)['rate'] * 100:.1f}%.") if best["n"] else "no patients in this group"
    return {"headline": headline, "overall": {"group": "all patients", **_rate(df)}, "subgroups": rows,
            "note": "Observed rates in a public educational dataset of 5,109 patients; not population incidence."}


def _rate(d: pd.DataFrame) -> dict:
    n, k = len(d), int(d["stroke"].sum())
    return {"n": n, "strokes": k, "rate": round(k / n, 4) if n else None, "wilson_ci95": list(wilson(k, n))}


def summary_table() -> pd.DataFrame:
    t = factor_tests()
    return pd.DataFrame(t["odds_ratios"])


if __name__ == "__main__":
    import json
    print(json.dumps(factor_tests(), indent=1))
    print(cohort_stats(67, 1, 0))
    _ = np
