---
name: risk-communication
description: Write the clinician-facing decision-support report with calibrated, non-causal risk language, natural frequencies and inline guideline citations.
agents: clinical_writer
---
# Risk communication (clinician audience)

- State the model estimate as a percentage **and** a natural frequency out of 100, e.g. "15.0% (about 15 in
  100 patients with this profile in the dataset)". Use the exact `risk_percent`; the frequency is that number
  rounded to a whole number out of 100 (write "fewer than 1 in 100" below 0.5%). Never use other denominators.
- Say "associated with" for model factors, never "caused by".
- Each bullet under "Guideline evidence" ends with one citation in square brackets, e.g. `[G-03]`, and
  only uses ids the evidence researcher returned.
- Never recommend a drug, a dose, or a test result threshold for treatment; point to "discuss with the
  treating clinician" instead.
- Never state or imply that the patient has had or is having a stroke.
- Restate guideline findings in one short sentence each (at most 30 words); never paste long passages.
- Keep it under 400 words. Use the section headings given in the report prompt exactly.
