"""Graph RAG: a small knowledge graph built from the 30-excerpt guideline corpus (networkx), plus traversal.

Nodes
  factor:<name>     modifiable / non-modifiable risk factors (hypertension, glucose, bmi, smoking, heart_disease, age,
                    cholesterol, lifestyle, tia, demographics) - the same vocabulary the profile and the RAG topics use
  chunk:G-xx        one guideline excerpt (id, title, publisher, url)
  condition:<name>  outcomes / conditions the excerpts talk about (stroke, tia, heart_attack, atrial_fibrillation,
                    diabetes, high_blood_pressure, obesity, high_cholesterol, emergency)
Edges (built deterministically from the text: keyword patterns, no LLM)
  factor  -[evidenced_by]-> chunk        the excerpt is about / mentions the factor
  chunk   -[mentions]->     condition    the excerpt mentions the condition
  factor  -[increases_risk_of]-> condition   both co-occur in an excerpt whose wording states a risk relationship
Traversal (``traverse``): breadth-first from the patient's factors up to ``max_hops`` (1 = factor -> chunk,
2 = factor -> chunk -> condition -> other chunks about the same condition, e.g. prevention and warning-sign
excerpts for "stroke"). Chunks are ranked by path count and hop distance; every returned chunk carries the path
that reached it so the writer can explain *why* it is relevant (context graph traversal).
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

import networkx as nx

from .data import load_guidelines

FACTOR_PATTERNS = {
    "hypertension": r"blood pressure|hypertension", "glucose": r"diabet|blood sugar|glucose|a1c",
    "bmi": r"\bbmi\b|obes|overweight|weight", "smoking": r"smok|tobacco|cigarette|nicotine",
    "heart_disease": r"heart disease|atrial fibrillation|afib|coronary|heart", "age": r"\bage\b|older|aged|years old",
    "cholesterol": r"cholesterol|ldl", "lifestyle": r"physical activ|diet|alcohol|exercise|healthy",
    "tia": r"\btia\b|transient ischemic|mini-stroke", "demographics": r"\bwomen\b|\bmen\b|race|ethnic|sex|pregnan|black",
}
CONDITION_PATTERNS = {
    "stroke": r"stroke", "tia": r"\btia\b|transient ischemic|mini-stroke", "heart_attack": r"heart attack",
    "atrial_fibrillation": r"atrial fibrillation|afib", "diabetes": r"diabet", "high_blood_pressure": r"blood pressure|hypertension",
    "obesity": r"obes", "high_cholesterol": r"cholesterol", "emergency": r"9-1-1|911|emergency|call",
}
HUB_TOPICS = {"emergency", "tia"}
RISK_WORDS = r"risk|cause|increase|raise|leading|more likely|higher chance|damage"
PROFILE_TO_FACTORS = {"hypertension": "hypertension", "heart_disease": "heart_disease", "glucose": "glucose", "bmi": "bmi",
                      "smoking": "smoking", "age": "age"}


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for f in FACTOR_PATTERNS:
        g.add_node(f"factor:{f}", kind="factor", label=f)
    for c in CONDITION_PATTERNS:
        g.add_node(f"condition:{c}", kind="condition", label=c)
    for d in load_guidelines():
        cid, text = f"chunk:{d['id']}", f"{d['title']}. {d['text']}".lower()
        g.add_node(cid, kind="chunk", label=d["id"], title=d["title"], url=d["url"], topic=d["topic"])
        factors = {f for f, pat in FACTOR_PATTERNS.items() if re.search(pat, text)} | ({d["topic"]} & set(FACTOR_PATTERNS))
        conditions = {c for c, pat in CONDITION_PATTERNS.items() if re.search(pat, text)}
        for f in factors:
            g.add_edge(f"factor:{f}", cid, rel="evidenced_by", weight=1.0 if f == d["topic"] else 0.6)
        for c in conditions:
            g.add_edge(cid, f"condition:{c}", rel="mentions")
        if re.search(RISK_WORDS, text):
            for f in factors:
                for c in conditions & {"stroke", "heart_attack"}:
                    g.add_edge(f"factor:{f}", f"condition:{c}", rel="increases_risk_of")
    return g


TEXT = {d["id"]: d["text"] for d in load_guidelines()}


@lru_cache(maxsize=1)
def get_graph() -> nx.DiGraph:
    return build_graph()


def stats() -> dict:
    g = get_graph()
    kinds = defaultdict(int)
    for _, a in g.nodes(data=True):
        kinds[a["kind"]] += 1
    rels = defaultdict(int)
    for _, _, a in g.edges(data=True):
        rels[a["rel"]] += 1
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "node_kinds": dict(kinds), "edge_relations": dict(rels),
            "built_from": "data/guidelines.json (30 CDC/NHLBI excerpts) with deterministic keyword patterns"}


def traverse(factors: list[str], max_hops: int = 2, k: int = 6) -> dict:
    """Breadth-first traversal from the patient's factor nodes; returns ranked chunks with the path that reached them.

    hop 1: factor -[evidenced_by]-> chunk (weight 1.0 if the chunk's topic is the factor, else 0.6)
    hop 2: factor -[increases_risk_of]-> condition <-[mentions]- chunk (weight 0.3; from the 'stroke' hub only into
            warning-sign / TIA excerpts)
    hop 3: factor -> chunk -> condition (not the stroke hub) <- other chunk (weight 0.1)
    A chunk's score is the sum of its path weights, so excerpts reached by several factors rank higher.
    """
    g = get_graph()
    max_hops = max(1, min(int(max_hops), 3))
    norm = lambda f: f"factor:{str(f).strip().lower().replace(' ', '_')}"  # noqa: E731
    starts = [norm(f) for f in factors if norm(f) in g]
    score: dict[str, float] = defaultdict(float)
    best: dict[str, tuple] = {}

    def add(chunk, path, w, hop):
        score[chunk] += w
        if chunk not in best or hop < best[chunk][0]:
            best[chunk] = (hop, path)

    for s in starts:
        for ch in g.successors(s):
            if g.nodes[ch]["kind"] == "chunk":
                add(ch, [s, ch], g.edges[s, ch]["weight"], 1)
        if max_hops >= 2:
            for cond in g.successors(s):
                if g.nodes[cond]["kind"] == "condition":
                    for ch in g.predecessors(cond):
                        # "stroke" is a hub mentioned by almost every excerpt: from it, only follow edges into
                        # warning-sign / TIA excerpts (what a researcher needs next); other conditions expand fully.
                        if g.nodes[ch]["kind"] == "chunk" and (cond != "condition:stroke" or g.nodes[ch]["topic"] in HUB_TOPICS):
                            add(ch, [s, cond, ch], 0.3, 2)
        if max_hops >= 3:
            for ch in [c for c in g.successors(s) if g.nodes[c]["kind"] == "chunk"]:
                for cond in g.successors(ch):
                    if cond == "condition:stroke":
                        continue
                    for ch2 in g.predecessors(cond):
                        if g.nodes[ch2]["kind"] == "chunk" and ch2 != ch:
                            add(ch2, [s, ch, cond, ch2], 0.1, 3)
    ranked = sorted(score, key=lambda c: (-score[c], best[c][0], g.nodes[c]["label"]))[:max(1, min(int(k), 10))]
    label = lambda n: g.nodes[n]["label"]  # noqa: E731
    return {"start_factors": [label(s) for s in starts], "unknown_factors": [f for f in factors if norm(f) not in g],
            "max_hops": max_hops,
            "chunks": [{"id": label(c), "title": g.nodes[c]["title"], "url": g.nodes[c]["url"], "score": round(score[c], 2), "hop": best[c][0],
                        "path": " -> ".join(label(n) for n in best[c][1]), "text": TEXT[label(c)]} for c in ranked]}
