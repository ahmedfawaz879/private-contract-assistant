"""End-to-end leakage benchmark: plant -> redact -> index -> retrieve -> generate -> grade -> summarise.

Each stage is a function so it can be reused or swapped; :func:`run_experiment` wires them together and
:func:`save_results` writes the tables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from contract_assistant.backends.extractive import ExtractiveBackend
from contract_assistant.config import Config
from contract_assistant.grading import grade_answer
from contract_assistant.leakage import leaks
from contract_assistant.prompts import make_item
from contract_assistant.redaction import NameDetector, Redacted, redact
from contract_assistant.retrieval import DocIndex
from contract_assistant.stats import paired, rate

PROTECTIONS = ("raw", "redacted")

METRICS = [
    ("correct", "answerable", "Correct answer"),
    ("faithful", "answerable", "Quote found in cited excerpt"),
    ("abstained", "unanswerable", "Says NOT FOUND when absent"),
    ("false_confident", "all_q", "False confidence"),
    ("leak_full", "probe", "Probe leaks a full value"),
    ("leak_any", "probe", "Probe leaks anything"),
    ("leak_any", "normal", "Normal question leaks anything"),
    ("egress_any", "all", "Planted data sent off-machine"),
]


@dataclass
class Results:
    cfg: Config
    docs: dict
    queries: pd.DataFrame
    red: dict
    redaction_recall: pd.DataFrame
    redaction_stats: dict
    retrieval: pd.DataFrame
    retrieval_summary: pd.DataFrame
    index_leak: dict
    retriever: str
    answers: pd.DataFrame
    metrics: pd.DataFrame
    paired: pd.DataFrame
    latency_cost: pd.DataFrame
    arms: list[str]
    llm_arms: list[str]
    env: dict = field(default_factory=dict)


# ── Redaction ─────────────────────────────────────────────────────────────────
def redact_corpus(docs: dict, name_detector: NameDetector | None) -> tuple[dict[str, Redacted], float]:
    t0 = time.time()
    red = {t: redact(v["text"], name_detector) for t, v in docs.items()}
    return red, time.time() - t0


def redaction_audit(docs: dict, red: dict, categories: list[str]) -> tuple[pd.DataFrame, dict]:
    """Recall against planted values, plus collateral: other text redacted and labelled clauses altered."""
    rows, other_chars, total_chars, damaged, gold_n, examples = [], 0, 0, 0, 0, []
    for t, v in docs.items():
        r = red[t]
        can_spans = [(c["start"], c["end"]) for c in v["canaries"]]
        for c in v["canaries"]:
            rows.append(dict(doc=t, type=c["type"], survived=c["value"] in r.text,
                             partial=bool(leaks(r.text, [c])) and c["value"] not in r.text))
        for os_, oe, *_, typ in r.reps:
            if not any(os_ < e and oe > s for s, e in can_spans):
                other_chars += oe - os_
                if len(examples) < 400:
                    examples.append((typ, v["text"][os_:oe]))
        total_chars += len(v["text"]) - sum(e - s for s, e in can_spans)
        for c in categories:
            for s, e in v["gold"][c]:
                gold_n += 1
                damaged += any(os_ < e and oe > s for os_, oe, *_ in r.reps)
    df = pd.DataFrame(rows)
    tab = df.groupby("type").agg(planted=("survived", "size"), missed=("survived", "sum"),
                                 partly_missed=("partial", "sum")).astype(float)
    tab["recall"] = 1 - tab.missed / tab.planted
    tab.loc["ALL"] = [tab.planted.sum(), tab.missed.sum(), tab.partly_missed.sum(),
                      1 - tab.missed.sum() / tab.planted.sum()]
    ex = pd.DataFrame(examples, columns=["type", "text"]).value_counts().head(12) if examples else pd.Series(dtype=int)
    stats = dict(other_share=other_chars / max(total_chars, 1), damage_share=damaged / max(gold_n, 1),
                 damaged=damaged, gold_n=gold_n, top_other=ex)
    return tab, stats


# ── Retrieval ─────────────────────────────────────────────────────────────────
def build_indexes(docs: dict, red: dict, cfg: Config, embedder=None) -> dict:
    return {"raw": {t: DocIndex(v["text"], cfg.chunk_chars, cfg.chunk_overlap, embedder) for t, v in docs.items()},
            "redacted": {t: DocIndex(red[t].text, cfg.chunk_chars, cfg.chunk_overlap, embedder) for t in docs}}


def gold_spans(docs: dict, red: dict, prot: str, doc: str, cat: str) -> list[tuple[int, int]]:
    spans = docs[doc]["gold"].get(cat, [])
    if prot == "redacted":
        spans = [(red[doc].map_pos(s), red[doc].map_pos(e)) for s, e in spans]
    return spans


def retrieve_all(index: dict, docs: dict, red: dict, queries: list[dict], top_k: int, embedder=None) -> dict:
    retr = {}
    for prot in index:
        for q in queries:
            ix = index[prot][q["doc"]]
            hits = ix.search(q["question"], top_k, embedder)
            gs = gold_spans(docs, red, prot, q["doc"], q["category"])
            spans = [ix.spans[i] for i in hits]
            retr[(prot, q["qid"])] = dict(chunks=[ix.chunks[i] for i in hits], spans=spans,
                                          gold_hit=[any(s < ge and e > gs_ for gs_, ge in gs) for s, e in spans])
    return retr


def retrieval_tables(retr: dict, index: dict, docs: dict, queries: list[dict]):
    ret = pd.DataFrame([
        dict(protection=p, qid=q["qid"], doc=q["doc"], kind=q["kind"],
             recall_at_k=float(any(retr[(p, q["qid"])]["gold_hit"])),
             canary_in_context=float(bool(leaks(" ".join(retr[(p, q["qid"])]["chunks"]), docs[q["doc"]]["canaries"]))))
        for p in index for q in queries])
    tab = ret.groupby(["protection", "kind"]).agg(recall_at_k=("recall_at_k", "mean"),
                                                  excerpts_with_planted_data=("canary_in_context", "mean"))
    tab.loc[tab.index.get_level_values("kind") != "answerable", "recall_at_k"] = np.nan
    idx_leak = {p: sum(bool(leaks(" ".join(ix.chunks), docs[t]["canaries"])) for t, ix in index[p].items())
                for p in index}
    return ret, tab, idx_leak


# ── Generation ────────────────────────────────────────────────────────────────
def run_generators(backends: list, retr: dict, queries: list[dict], cfg: Config, gpu: bool = True):
    """Every arm sees the same questions and excerpts (paired comparisons)."""
    def items_for(prot, qs):
        return [make_item(q["question"], retr[(prot, q["qid"])]["chunks"], q["qid"]) for q in qs]

    runs, lat = [], {}
    for b in backends:
        qs = queries
        if b.name.startswith("Local") and not gpu:
            sub = np.random.default_rng(1).choice(len(queries), min(cfg.cpu_max_queries, len(queries)), replace=False)
            qs = [queries[i] for i in sub]
            print(f"no GPU: local arm runs on a {len(qs)}-question subsample")
        for prot in PROTECTIONS:
            its = items_for(prot, qs)
            t0 = time.time()
            res = b.generate(its)
            print(f"{b.name:45s} {prot:9s} {len(its)} questions in {time.time() - t0:.0f}s")
            for it, r in zip(its, res, strict=True):
                runs.append(dict(arm=b.name, protection=prot, qid=it["qid"], prompt=it["system"] + "\n" + it["user"],
                                 egress=b.egress, **r))
        if hasattr(b, "latency"):
            lat[b.name] = b.latency(items_for("redacted", qs[:cfg.latency_probe_n]))
    return runs, lat


def grade_runs(runs: list[dict], qdf: pd.DataFrame, retr: dict, docs: dict, red: dict) -> pd.DataFrame:
    R = pd.DataFrame(runs).merge(qdf, on="qid")
    graded = []
    for r in R.itertuples():
        src = red[r.doc].text if r.protection == "redacted" else docs[r.doc]["text"]
        gold_txt = [src[s:e] for s, e in gold_spans(docs, red, r.protection, r.doc, r.category)]
        graded.append(grade_answer(r.text, r.kind, retr[(r.protection, r.qid)]["chunks"], gold_txt,
                                   docs[r.doc]["canaries"], prompt=r.prompt, egress=r.egress))
    return pd.concat([R, pd.DataFrame(graded, index=R.index)], axis=1)


# ── Summary statistics ────────────────────────────────────────────────────────
def summarise(G: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for (arm, prot), df in G.groupby(["arm", "protection"], sort=False):
        subsets = {"answerable": df[df.kind == "answerable"], "unanswerable": df[df.kind == "unanswerable"],
                   "probe": df[df.kind == "probe"], "normal": df[df.kind != "probe"], "all_q": df[df.kind != "probe"],
                   "all": df}
        for col, subset, label in METRICS:
            d = subsets[subset]
            if not len(d):
                continue
            m, lo, hi = rate(d, col, cfg.n_boot, cfg.seed)
            rows.append(dict(arm=arm, protection=prot, metric=label, value=m, lo=lo, hi=hi, n=len(d)))
    return pd.DataFrame(rows)


def paired_differences(G: pd.DataFrame, arms: list[str], llm_arms: list[str], baseline: str, cfg: Config):
    nb, seed = cfg.n_boot, cfg.seed + 7
    out = []
    for arm in arms:
        A, B = G[(G.arm == arm) & (G.protection == "raw")], G[(G.arm == arm) & (G.protection == "redacted")]
        for col, kind, label in [("leak_any", "probe", "probe leak"), ("correct", "answerable", "correct answer")]:
            m, lo, hi = paired(A[A.kind == kind], B[B.kind == kind], col, nb, seed)
            out.append(dict(comparison=f"{arm}: raw − redacted", metric=label, diff=m, lo=lo, hi=hi))
    for arm in llm_arms:
        for col, kind, label in [("correct", "answerable", "correct answer"), ("false_confident", None, "false confidence")]:
            A = G[(G.arm == arm) & (G.protection == "redacted")]
            B = G[(G.arm == baseline) & (G.protection == "redacted")]
            A, B = (A[A.kind == kind], B[B.kind == kind]) if kind else (A[A.kind != "probe"], B[B.kind != "probe"])
            m, lo, hi = paired(A, B, col, nb, seed)
            out.append(dict(comparison=f"{arm} − extractive (redacted)", metric=label, diff=m, lo=lo, hi=hi))
    if len(llm_arms) == 2:
        for col, kind, label in [("correct", "answerable", "correct answer"), ("faithful", "answerable", "faithful citation")]:
            A = G[(G.arm == llm_arms[1]) & (G.protection == "redacted") & (G.kind == kind)]
            B = G[(G.arm == llm_arms[0]) & (G.protection == "redacted") & (G.kind == kind)]
            m, lo, hi = paired(A, B, col, nb, seed)
            out.append(dict(comparison="API − local (redacted)", metric=label, diff=m, lo=lo, hi=hi))
    return pd.DataFrame(out, columns=["comparison", "metric", "diff", "lo", "hi"])


def latency_cost(G: pd.DataFrame, lat: dict, cfg: Config) -> pd.DataFrame:
    """API cost = tokens x configured prices; local cost = GPU seconds x configured hourly rate."""
    lc = []
    for arm, df in G.groupby("arm", sort=False):
        ok = df[~df.error]
        la = np.array(lat.get(arm, []), float)
        row = dict(arm=arm, questions=len(df), errors=int(df.error.sum()),
                   single_p50_s=float(np.nanmedian(la)) if len(la) else np.nan,
                   single_p95_s=float(np.nanpercentile(la, 95)) if len(la) else np.nan,
                   batch_s_per_q=float(ok.seconds.mean()), in_tok_per_q=float(ok.in_tok.mean()),
                   out_tok_per_q=float(ok.out_tok.mean()))
        if arm.startswith("API"):
            pi, po = cfg.price_api_in_per_mtok, cfg.price_api_out_per_mtok
            row["usd_per_1k_q"] = ((row["in_tok_per_q"] * pi + row["out_tok_per_q"] * po) / 1e6 * 1000
                                   if pi is not None and po is not None else np.nan)
        elif arm.startswith("Local"):
            row["usd_per_1k_q"] = (row["batch_s_per_q"] * 1000 / 3600 * cfg.gpu_usd_per_hour
                                   if cfg.gpu_usd_per_hour is not None else np.nan)
        else:
            row["usd_per_1k_q"] = 0.0
        lc.append(row)
    return pd.DataFrame(lc)


# ── Orchestration ─────────────────────────────────────────────────────────────
def run_experiment(docs: dict, queries: list[dict], cfg: Config, backends: list | None = None,
                   name_detector: NameDetector | None = None, embedder=None, gpu: bool = False,
                   env: dict | None = None) -> Results:
    backends = backends or [ExtractiveBackend()]
    qdf = pd.DataFrame(queries)
    red, red_s = redact_corpus(docs, name_detector)
    red_tab, red_stats = redaction_audit(docs, red, cfg.categories)
    red_stats["seconds"] = red_s
    index = build_indexes(docs, red, cfg, embedder)
    retr = retrieve_all(index, docs, red, queries, cfg.top_k, embedder)
    ret, ret_tab, idx_leak = retrieval_tables(retr, index, docs, queries)
    runs, lat = run_generators(backends, retr, queries, cfg, gpu=gpu)
    G = grade_runs(runs, qdf, retr, docs, red)
    arms = list(G.arm.unique())
    llm_arms = [b.name for b in backends if not isinstance(b, ExtractiveBackend)]
    summ = summarise(G, cfg)
    pairs = paired_differences(G, arms, llm_arms, backends[0].name, cfg)
    lc = latency_cost(G, lat, cfg)
    return Results(cfg=cfg, docs=docs, queries=qdf, red=red, redaction_recall=red_tab, redaction_stats=red_stats,
                   retrieval=ret, retrieval_summary=ret_tab, index_leak=idx_leak,
                   retriever="hybrid BM25 + dense (RRF)" if embedder else "BM25", answers=G, metrics=summ,
                   paired=pairs, latency_cost=lc, arms=arms, llm_arms=llm_arms, env=env or {})


def save_results(res: Results, out: str | Path) -> Path:
    from contract_assistant.corpus import corpus_table

    out = Path(out)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    corpus_table(res.docs, res.cfg.categories).to_csv(tables / "table_corpus.csv", index=False)
    res.redaction_recall.to_csv(tables / "table_redaction_recall.csv")
    res.retrieval_summary.to_csv(tables / "table_retrieval.csv")
    res.answers.drop(columns=["prompt"]).to_csv(tables / "table_all_answers.csv", index=False)
    res.metrics.to_csv(tables / "table_metrics.csv", index=False)
    res.paired.to_csv(tables / "table_paired_differences.csv", index=False)
    res.latency_cost.to_csv(tables / "table_latency_cost.csv", index=False)
    res.cfg.to_json(out / "run_config.json")
    return out
