"""CUAD v1 loading, seeded contract sampling, question building and synthetic-data planting."""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from contract_assistant.config import Config
from contract_assistant.prompts import PROBES
from contract_assistant.synthetic import inject


def load_cuad(cuad_dir: str | Path = "data/cuad", repo: str = "https://github.com/TheAtticusProject/cuad.git") -> dict:
    """CUADv1.json from the Atticus Project repo (shallow-cloned on first use)."""
    cuad_dir = Path(cuad_dir)
    if not (cuad_dir / "data.zip").exists():
        cuad_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", repo, str(cuad_dir)], check=True)
    with zipfile.ZipFile(cuad_dir / "data.zip") as z:
        return json.loads(z.read("CUADv1.json"))


def build_corpus(raw: dict, cfg: Config) -> tuple[dict, list[dict]]:
    """Sample contracts, plant synthetic records, and build questions.

    Returns ``docs`` (title -> dict(text, canaries, gold)) and ``queries`` (list of dict(qid, doc, kind, category,
    question)) where kind is ``answerable``, ``unanswerable`` or ``probe``.
    """
    docs_all = raw["data"]
    rng = np.random.default_rng(cfg.seed)
    pool = [d for d in docs_all if cfg.min_chars <= len(d["paragraphs"][0]["context"]) <= cfg.max_chars]
    pick = rng.choice(len(pool), min(cfg.n_contracts, len(pool)), replace=False)
    used_names: set[str] = set()
    docs, queries = {}, []
    for j in sorted(pick):
        d = pool[j]
        ctx = d["paragraphs"][0]["context"]
        title = d["title"]
        qs = {re.search(r'related to "(.+?)"', q["question"]).group(1): q for q in d["paragraphs"][0]["qas"]}
        gold = {c: [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qs[c]["answers"]]
                for c in cfg.categories}
        all_spans = [s for c in gold for s in gold[c]]
        near = next((gold[c][0] for c in cfg.categories if gold[c]), None)
        text, canaries, new_spans = inject(ctx, all_spans, rng, cfg.records_per_contract, used_names, near_span=near)
        it = iter(new_spans)
        gold_new = {c: [next(it) for _ in gold[c]] for c in cfg.categories}
        for c in cfg.categories:
            for (s, e), (s0, e0) in zip(gold_new[c], gold[c], strict=True):
                assert text[s:e] == ctx[s0:e0], "gold clause moved incorrectly"
        docs[title] = dict(text=text, canaries=canaries, gold=gold_new)
        for c in cfg.categories:
            detail = qs[c]["question"].split("Details:")[-1].strip()
            queries.append(dict(qid=f"{title}__{c}", doc=title, kind="answerable" if gold_new[c] else "unanswerable",
                                category=c, question=detail))
        for k, p in enumerate(PROBES):
            queries.append(dict(qid=f"{title}__probe{k}", doc=title, kind="probe", category="PII probe", question=p))
    return docs, queries


def corpus_table(docs: dict, categories: list[str]) -> pd.DataFrame:
    return pd.DataFrame([dict(contract=t[:60], chars=len(v["text"]), planted_values=len(v["canaries"]),
                              answerable=sum(bool(v["gold"][c]) for c in categories)) for t, v in docs.items()])
