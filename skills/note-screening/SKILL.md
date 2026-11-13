---
name: note-screening
description: Interpret a free-text clinical note with the BE FAST red-flag rule and the MTSamples note classifier, and summarise prior de-identified cases from memory.
agents: intake
---
# Note screening

1. Call `screen_note` with the note text exactly as provided (it is already PHI-redacted).
2. Report the red-flag `signs` list verbatim. A sign without sudden-onset language is *not* an emergency
   finding by itself, but it must be passed on so the writer tells the reader to raise it with a clinician.
3. `neuro_note_probability` comes from a weak advisory classifier (held-out F1 is reported in the tool
   output). Never describe it as a diagnosis; say "the note's wording resembles neurology notes" or not.
4. Call `memory_read` once. If earlier cases exist, state the earlier risk percentage(s) and band(s)
   exactly as returned. If none exist, say "no earlier cases in this session".
5. Do not guess anything the note does not say.
