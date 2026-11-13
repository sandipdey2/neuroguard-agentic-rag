"""Download the two public datasets NeuroGuard uses and verify them.

    python scripts/fetch_data.py

1. Stroke Prediction Dataset (fedesoriano, Kaggle). Kaggle needs a login, so the
   byte-identical copy published in the ray-project/raydp repository is used
   (the same file is also mirrored in PlayingNumbers/ML_Process_Course; both
   have MD5 3f25a4c2f2963d969e1a156c08968577). 5,110 rows, 12 columns.
2. MTSamples medical transcriptions (Kaggle tboyle10/medicaltranscriptions,
   CC0 Public Domain, scraped from mtsamples.com). Only four columns are kept
   (description, medical_specialty, sample_name, first 400 characters of the
   transcription) to keep the repository small.
No rows are generated or altered; the only transformation is column selection.
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
STROKE_URL = ("https://raw.githubusercontent.com/ray-project/raydp/master/tutorials/dataset/"
              "healthcare-dataset-stroke-data.csv")
STROKE_MD5 = "3f25a4c2f2963d969e1a156c08968577"
MT_URL = "https://media.githubusercontent.com/media/Anyesh/data-repo/master/mtsamples.csv"
MT_MD5 = "d8547837775283c95b8a70119da806f8"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "neuroguard-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    blob = _get(STROKE_URL)
    md5 = hashlib.md5(blob).hexdigest()
    assert md5 == STROKE_MD5, f"stroke CSV checksum mismatch: {md5}"
    df = pd.read_csv(io.BytesIO(blob))
    assert df.shape == (5110, 12), df.shape
    (RAW / "healthcare-dataset-stroke-data.csv").write_bytes(blob)
    print("stroke dataset OK", df.shape, md5)

    blob = _get(MT_URL)
    md5 = hashlib.md5(blob).hexdigest()
    assert md5 == MT_MD5, f"MTSamples checksum mismatch: {md5}"
    mt = pd.read_csv(io.BytesIO(blob))
    assert len(mt) == 4999, len(mt)
    out = mt[["description", "medical_specialty", "sample_name"]].copy()
    out["transcription_head"] = mt["transcription"].fillna("").str.slice(0, 400)
    out.to_csv(RAW / "mtsamples_subset.csv", index=False)
    print("MTSamples OK", out.shape, md5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
