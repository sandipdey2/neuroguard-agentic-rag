# NeuroGuard AI — Multi-Agent Stroke Risk Platform

Startup-grade clinical decision-support system that synthesizes all six prior capstone projects into a deployable multi-agent application.

**GitHub Repository:** https://github.com/sandipdey2/neuroguard-agentic-rag

## What it does
- Accepts a patient profile and runs a LangGraph-style multi-agent workflow
- Risk Agent computes a transparent risk score (integrates Project 3 modeling)
- RAG Agent retrieves public clinical guidelines
- Reviewer and Guardrail Agents enforce consistency and safety
- Full audit trail of every agent step
- Optional real OpenAI summary (key entered in UI, never stored)
- RAGAS-style evaluation metrics displayed
- Ready for Streamlit Cloud

## Dataset
Public Stroke Prediction Dataset (Kaggle / fedesoriano).  
5,110 real patient records, 12 columns. Not used in any prior project.  
Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## How to run locally
```bash
git clone https://github.com/sandipdey2/neuroguard-agentic-rag.git
cd neuroguard-agentic-rag
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
1. Push this folder to a public GitHub repo (already done)
2. Go to share.streamlit.io
3. Connect the repo and set main file to `app.py`
4. Deploy

## Architecture
See `architecture_diagram.pdf` for the full multi-agent diagram (Agent Gateway, Orchestrator, MCP-style tool proxy, Memory, Permission Engine, RAG, Guardrails, Audit Log).

## Synthesis of prior projects
- Project 1: Data cleaning and reproducible workflow
- Project 2: Statistical risk-factor analysis
- Project 3: Supervised classification / risk modeling
- Project 4: Sequence-aware clinical notes (conceptual)
- Project 5: Generative report drafting
- Project 6: Full multi-agent LangGraph + MCP + RAG + evaluation

## Disclaimer
Decision-support tool only. Not a medical device. Not for clinical diagnosis.
