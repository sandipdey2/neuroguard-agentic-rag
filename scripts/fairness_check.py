"""Preliminary subgroup audit of the served risk model on the held-out test split:  python scripts/fairness_check.py

Rebuilds the exact 80/20 stratified split used in core/model.py (random_state=42), scores the test patients with
the served calibrated model and reports, per subgroup (sex, age band, residence): n, strokes, ROC-AUC, recall and
precision at the deployed threshold, and flag rate. Small subgroups (few strokes) give unstable numbers - this is
a screening check for the roadmap, not a fairness certification. Writes eval/fairness_preliminary.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from core.data import load_stroke  # noqa: E402
from core.model import FEATURES, SEED, get_model  # noqa: E402

m = get_model()
df, _ = load_stroke()
_, Xte, _, yte = train_test_split(df[FEATURES], df["stroke"].to_numpy(), test_size=0.2, stratify=df["stroke"].to_numpy(), random_state=SEED)
p = m.served.predict_proba(Xte)[:, 1]
assert abs(roc_auc_score(yte, p) - m.report["test_metrics"]["roc_auc"]) < 0.006, "split does not match the served model's test split"
flag = p >= m.threshold
t = Xte.assign(y=yte, p=p, flag=flag, age_band=pd.cut(Xte["age"], [0, 44.99, 64.99, 200], labels=["<45", "45-64", "65+"]))
rows = []
for col in ("gender", "age_band", "Residence_type"):
    for lvl, g in t.groupby(col, observed=True):
        pos = int(g.y.sum())
        tp = int((g.flag & (g.y == 1)).sum())
        rows.append({"attribute": col, "group": str(lvl), "n": len(g), "strokes": pos,
                     "roc_auc": round(roc_auc_score(g.y, g.p), 3) if 0 < pos < len(g) and pos >= 5 else None,
                     "recall_at_threshold": round(tp / pos, 3) if pos else None,
                     "precision_at_threshold": round(tp / int(g.flag.sum()), 3) if g.flag.sum() else None,
                     "flag_rate": round(float(g.flag.mean()), 3)})
out = {"split": "held-out test (n=%d, strokes=%d)" % (len(yte), int(yte.sum())), "threshold": round(m.threshold, 4),
       "note": "preliminary; subgroups with < 5 strokes get no AUC; no race/ethnicity field exists in the dataset", "rows": rows}
(ROOT / "eval" / "fairness_preliminary.json").write_text(json.dumps(out, indent=1))
print(pd.DataFrame(rows).to_string(index=False))
