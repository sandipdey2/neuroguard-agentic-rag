---
name: cohort-comparison
description: Put a model probability in context with observed stroke rates of similar patients in the public dataset, using cohort_stats and the sandboxed calculator.
agents: risk_analyst
---
# Cohort comparison

1. Call `predict_risk` first and copy `risk_probability`, `risk_percent` and `risk_band` exactly.
2. Call `explain_factors` and list the top factors with their direction ("raises" / "lowers").
   These are contributions to the log-odds of a logistic model relative to the average training patient,
   i.e. associations, not causes.
3. Call `cohort_stats` with the patient's age and the hypertension and heart disease flags. Copy its
   `headline` sentence verbatim as `cohort_comparison`; do not relabel groups.
4. If you compare numbers (for example "x times the cohort rate"), compute it with `calculator`; never do
   arithmetic in your head. Round only at the end.
5. Remind the writer that the dataset is an educational sample, not a population registry.
