# Data sources

NeuroGuard AI uses **public data only**. No rows, notes or guideline sentences were generated or invented.
`scripts/fetch_data.py` re-downloads both datasets and checks their MD5 checksums.

| Data | Where it is used | Source | License / terms |
|---|---|---|---|
| **Stroke Prediction Dataset** (fedesoriano), 5,110 rows x 12 columns, `data/raw/healthcare-dataset-stroke-data.csv` | P1 pipeline, P2 statistics, P3 risk model, `cohort_stats`, eval profiles | Kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset. Kaggle requires a login, so the byte-identical copy at https://raw.githubusercontent.com/ray-project/raydp/master/tutorials/dataset/healthcare-dataset-stroke-data.csv is downloaded (MD5 `3f25a4c2f2963d969e1a156c08968577`, same file in PlayingNumbers/ML_Process_Course) | Kaggle lists "Data files © Original Authors"; the author describes the source as confidential and for **educational use only**. Used here for an educational capstone; excluded from the commercial product zip. |
| **MTSamples medical transcriptions**, 4,999 documents, `data/raw/mtsamples_subset.csv` (columns kept: description, medical_specialty, sample_name, first 400 characters of transcription) | P4 note classifier, eval notes | Kaggle tboyle10/medicaltranscriptions (scraped from https://www.mtsamples.com), mirror https://media.githubusercontent.com/media/Anyesh/data-repo/master/mtsamples.csv (MD5 `d8547837775283c95b8a70119da806f8`) | CC0: Public Domain (Kaggle). MTSamples asks for credit and a link when sharing. |
| **Guideline excerpts**, 30 chunks, `data/guidelines.json` | RAG corpus, MCP resources `guideline://...` | CDC pages: [Risk Factors for Stroke](https://www.cdc.gov/stroke/risk-factors/index.html), [Signs and Symptoms of Stroke](https://www.cdc.gov/stroke/signs-symptoms/index.html), [Preventing Stroke](https://www.cdc.gov/stroke/prevention/index.html), [Stroke Facts](https://www.cdc.gov/stroke/data-research/facts-stats/index.html), [About Stroke](https://www.cdc.gov/stroke/about/index.html), [About High Blood Pressure](https://www.cdc.gov/high-blood-pressure/about/index.html), [Adult BMI Categories](https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html), [Diabetes Testing](https://www.cdc.gov/diabetes/diabetes-testing/index.html); NHLBI [Stroke - Causes and Risk Factors](https://www.nhlbi.nih.gov/health/stroke/causes) | U.S. federal government works (public domain); attribution given per chunk. No WHO or AHA text is included (their terms restrict reuse). |

## How the guideline excerpts were made
Each chunk is copied verbatim from the page named in its `url` field (accessed 2026-11-13). Bullet lists were
joined into sentences and some chunks combine adjacent passages of the same page; no words were changed or
added. Superscript reference numbers were removed. Every chunk records `publisher`, `url`, `accessed`, `license`.

## Evaluation inputs
`eval/cases.json` (built by `scripts/build_eval_cases.py`) uses real profiles from the model's held-out test
split, notes copied from MTSamples descriptions, the CDC warning-sign wording for an emergency probe, and fixed
adversarial strings (prompt injection, tool misuse, dosing, off-topic). One MTSamples note has a fictitious
contact line appended solely to test PHI redaction; this is labelled in the file.

## Citations (APA 7)
- fedesoriano. (2021). *Stroke prediction dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- Boyle, T. (n.d.). *Medical transcriptions* [Data set]. Kaggle. https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
- Centers for Disease Control and Prevention. (n.d.). *Risk factors for stroke*. https://www.cdc.gov/stroke/risk-factors/index.html
- Centers for Disease Control and Prevention. (2026). *Signs and symptoms of stroke*. https://www.cdc.gov/stroke/signs-symptoms/index.html
- National Heart, Lung, and Blood Institute. (2023). *Stroke: Causes and risk factors*. https://www.nhlbi.nih.gov/health/stroke/causes
