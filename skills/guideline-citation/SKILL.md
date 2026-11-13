---
name: guideline-citation
description: Search the CDC/NHLBI guideline corpus for each risk factor present and return claims that are copied or closely paraphrased from a retrieved chunk, each with its chunk id.
agents: evidence_researcher, clinical_writer
---
# Guideline citation rules

- Issue one `search_guidelines` query per relevant risk factor (for example "high blood pressure stroke
  risk", "diabetes glucose stroke", "age stroke risk doubles", "smoking stroke"), plus one about warning
  signs (BE FAST / call 9-1-1). Use at most 4 queries.
- Every finding must carry exactly one `citation_id` that appeared in a search result (format `G-07`).
- Write each claim as ONE short sentence (at most 25 words) in your own words that keeps the excerpt's meaning.
  Do not paste whole passages: long verbatim copies are cut off by the provider's content filter.
- Do not merge facts from two chunks into one claim.
- Do not add numbers that are not in the chunk text.
- Prefer chunks whose topic matches a factor the patient actually has.
