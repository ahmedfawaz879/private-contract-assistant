"""Figures and the Markdown evaluation report. Wording follows the intervals."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from contract_assistant.experiment import Results

SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8b8a85"


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRIDC, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC,
        "grid.linewidth": 0.6, "legend.frameon": False, "font.size": 10, "figure.dpi": 110,
    })


def make_figures(res: Results, out: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    _style(plt)
    figs = Path(out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    rt = res.redaction_recall.drop("ALL").sort_values("recall")
    ax.barh(rt.index, rt.recall, color=BLUE, height=0.6)
    for i, (_, r_) in enumerate(rt.iterrows()):
        ax.text(r_.recall, i, f"  {r_.recall:.0%} ({int(r_.planted - r_.missed)}/{int(r_.planted)})", va="center",
                fontsize=9)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Share of planted values removed before indexing")
    ax.grid(axis="y", visible=False)
    ax.set_title("Redaction recall by type of personal data")
    plt.tight_layout()
    paths.append(figs / "fig1_redaction_recall.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    arm_col = {a: c for a, c in zip(res.arms, [GREY, BLUE, ORANGE], strict=False)}
    summ = res.metrics

    def bars(ax, metric, title):
        d = summ[summ.metric == metric]
        labels = []
        for k, r in enumerate(d.itertuples()):
            raw = r.protection == "raw"
            ax.barh(k, r.value, color=arm_col.get(r.arm, GREY), height=0.62, alpha=1 if raw else 0.55,
                    hatch=None if raw else "//", edgecolor=SURFACE)
            ax.plot([r.lo, r.hi], [k, k], color=INK, lw=1.3)
            labels.append(f"{r.arm.split(' (')[0]} · {r.protection}")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(PercentFormatter(1))

    for name, panels in [
        ("fig2_leakage.png", [("Probe leaks anything", "Probe questions that leaked planted data"),
                              ("Normal question leaks anything", "Ordinary questions that leaked planted data"),
                              ("Planted data sent off-machine", "Questions whose prompt sent planted data")]),
        ("fig3_quality.png", [("Correct answer", "Correct answers (answerable questions)"),
                              ("Quote found in cited excerpt", "Quote found in the cited excerpt"),
                              ("False confidence", "False confidence (all non-probe questions)")]),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        for ax, (metric, title) in zip(axes, panels, strict=True):
            bars(ax, metric, title)
        plt.tight_layout()
        paths.append(figs / name)
        fig.savefig(paths[-1], dpi=160)
        plt.close(fig)
    return paths


def _sig(lo: float, hi: float) -> str:
    return "a clear" if (lo > 0 or hi < 0) else "no clear"


def render_report(res: Results) -> str:
    cfg, summ, pairs, lc, red_tab = res.cfg, res.metrics, res.paired, res.latency_cost, res.redaction_recall
    qdf, docs, ret = res.queries, res.docs, res.retrieval

    def S(arm, prot, metric):
        r = summ[(summ.arm == arm) & (summ.protection == prot) & (summ.metric == metric)]
        return r.iloc[0] if len(r) else None

    def fmt(r):
        return "–" if r is None else f"{r.value:.0%} [{r.lo:.0%}, {r.hi:.0%}]"

    def f2(x):
        return "–" if np.isnan(x) else f"{x:.2f}"

    cols = ["Correct answer", "Quote found in cited excerpt", "Says NOT FOUND when absent", "False confidence",
            "Probe leaks anything", "Normal question leaks anything", "Planted data sent off-machine"]
    lines = [f"| {a} | {p} | " + " | ".join(fmt(S(a, p, c)) for c in cols) + " |"
             for a in res.arms for p in ("raw", "redacted")]
    lc_lines = "\n".join(
        f"| {r.arm} | {f2(r.single_p50_s)} / {f2(r.single_p95_s)} | {r.batch_s_per_q:.3f} | "
        f"{r.in_tok_per_q:.0f} / {r.out_tok_per_q:.0f} | "
        f"{'not priced' if np.isnan(r.usd_per_1k_q) else f'${r.usd_per_1k_q:.2f}'} |" for r in lc.itertuples())

    find = []
    for a in res.llm_arms:
        raw_p, red_p = S(a, "raw", "Probe leaks anything"), S(a, "redacted", "Probe leaks anything")
        pr = pairs[(pairs.comparison == f"{a}: raw − redacted") & (pairs.metric == "probe leak")].iloc[0]
        find.append(f"* **{a}.** Told not to reveal personal data, it still leaked planted data on {raw_p.value:.0%} "
                    f"of probe questions without redaction, and {red_p.value:.0%} with redaction "
                    f"({_sig(pr.lo, pr.hi)} difference: {pr['diff']:+.0%} [{pr.lo:+.0%}, {pr.hi:+.0%}]).")
    per_type = red_tab.drop("ALL")
    worst = per_type.recall.idxmin()
    rs = res.redaction_stats
    find.append(f"* **Redaction** removed {red_tab.loc['ALL', 'recall']:.0%} of planted values; weakest type: {worst} "
                f"({per_type.loc[worst, 'recall']:.0%}). It also redacted {rs['other_share']:.2%} of other text and "
                f"altered {rs['damage_share']:.0%} of labelled clauses.")
    api = [a for a in res.arms if a.startswith("API")]
    if api:
        e_raw = S(api[0], "raw", "Planted data sent off-machine")
        e_red = S(api[0], "redacted", "Planted data sent off-machine")
        find.append(f"* **Egress.** Without redaction, {e_raw.value:.0%} of API prompts carried planted personal data "
                    f"to a third party; with redaction {e_red.value:.0%}. Local and extractive arms send nothing.")
    for r in pairs[pairs.comparison.str.contains("extractive") | pairs.comparison.str.startswith("API − local")].itertuples():
        find.append(f"* **{r.comparison}, {r.metric}:** {r.diff:+.0%} [{r.lo:+.0%}, {r.hi:+.0%}] "
                    f"({_sig(r.lo, r.hi)} difference).")

    n_can = sum(len(v["canaries"]) for v in docs.values())
    rec = {p: ret[(ret.protection == p) & (ret.kind == "answerable")].recall_at_k.mean() for p in ("raw", "redacted")}
    env = res.env
    nl = "\n"
    return f"""# Private contract assistant: evaluation report

*Generated {date.today():%Y-%m-%d} · CUAD v1, {len(docs)} contracts, {len(qdf)} questions ({(qdf.kind == 'answerable').sum()} answerable, {(qdf.kind == 'unanswerable').sum()} unanswerable, {(qdf.kind == 'probe').sum()} personal-data probes) · {n_can} planted values · retriever {res.retriever}, top-{cfg.top_k} · seed {cfg.seed}*

## 1. Results

Intervals: 95% bootstrap over contracts.

| Generator | Index | Correct | Faithful citation | Says NOT FOUND when absent | False confidence | Probe leak | Ordinary-question leak | Sent off-machine |
|---|---|---|---|---|---|---|---|---|
{nl.join(lines)}

## 2. Findings

{nl.join(find)}

## 3. Latency and cost

| Generator | Single question p50 / p95 (s) | Batch s per question | Tokens in / out | Cost per 1,000 questions |
|---|---|---|---|---|
{lc_lines}

Prices: API {cfg.price_api_in_per_mtok} / {cfg.price_api_out_per_mtok} USD per million tokens in/out; GPU {cfg.gpu_usd_per_hour} USD per hour (configured, not measured).

## 4. Retrieval

Recall@{cfg.top_k} on answerable questions: raw {rec['raw']:.0%}, redacted {rec['redacted']:.0%}. Contracts whose index still holds planted data: raw {res.index_leak['raw']}/{len(docs)}, redacted {res.index_leak['redacted']}/{len(docs)}.

## 5. Limitations

* **Planted, not natural, personal data.** The records follow three sentence templates; real personal data is messier, so redaction recall here is an upper bound for these formats and says nothing about formats not planted.
* **Names are the hard case.** Names are found by a small general-purpose NER model; a stronger model, a domain-tuned one, or a list of known people would change the name numbers.
* **Correctness is lexical.** A quote counts as correct if at least half its words lie in the labelled clause; paraphrased answers without a quote are not credited.
* **One local model, one API model, default prompts, greedy decoding.** No prompt tuning for either.
* **Probe questions are direct.** Indirect extraction attacks (role-play, multi-turn) are not tested.
* **{len(docs)} contracts.** Intervals are over contracts; small differences will not be resolved.
* **Costs depend on configured prices**; latency depends on this GPU and on the API's load at run time.

## 6. Reproducibility

Python {env.get('python', '?')}, spaCy {env.get('spacy', 'not used')}; local model `{cfg.local_model}`; API model `{cfg.api_model_name}`{'' if api else ' (not run)'}; seed {cfg.seed}; configuration in `run_config.json`.
"""


def write_report(res: Results, out: str | Path) -> Path:
    p = Path(out) / "rag_privacy_eval_report.md"
    p.write_text(render_report(res), encoding="utf-8")
    return p
