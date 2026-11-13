# Plan.md - evidence research plan (read-only knowledge file for agentic search)

This file is part of the read-only `knowledge/` directory that the Evidence Researcher can search in
"agentic search" mode (MCP tools `search_files` and `shell`). It contains no patient data.

## Goal
Collect cited public-domain guideline evidence (CDC / NHLBI excerpts G-01..G-30) for every risk factor the
patient profile shows, plus the stroke warning signs (BE FAST, call 9-1-1).

## Steps
1. List the required topics: each risk factor present (age >= 55, hypertension, heart disease, glucose >= 126 mg/dL,
   BMI >= 30, smoking) plus `emergency` (warning signs).
2. For each topic call `semantic_search` with a short query; use `graph_traverse` when a factor name is known.
3. If a topic returns nothing relevant, reformulate with synonyms (e.g. "high blood pressure" for hypertension,
   "blood sugar" for glucose, "excess weight" for BMI) before giving up.
4. Use `execute_query` only for cohort counts (read-only SQL over table `patients`); copy numbers exactly.
5. `web_search` may refresh a CDC / NIH / WHO page, but cite only G-xx excerpts; web text is context, not a citation.
6. Use `get_memory` for de-identified learned hints from earlier runs.
7. Return short paraphrased findings, one claim per bullet, each with exactly one G-xx citation.

## Do not
- Do not give medication names or doses, and do not state that the patient has or had a stroke.
- Do not cite anything that was not returned by a retrieval tool in this run.
