"""End-to-end smoke test on a tiny CUAD-shaped corpus: extractive arm, BM25, pattern-only redaction."""

import json
from pathlib import Path

import pandas as pd

from contract_assistant.cli import main
from contract_assistant.config import Config
from contract_assistant.corpus import build_corpus
from contract_assistant.experiment import run_experiment, save_results
from contract_assistant.gate import evaluate
from contract_assistant.report import make_figures, write_report
from tests.conftest import fake_cuad


def _cfg():
    return Config(n_contracts=3, min_chars=1000, categories=["Governing Law", "Renewal Term"], use_dense=False,
                  use_local=False, use_ner=False, n_boot=50, latency_probe_n=2)


def test_build_corpus():
    cfg = _cfg()
    docs, queries = build_corpus(fake_cuad(), cfg)
    assert len(docs) == 3 and len(queries) == 3 * 4
    kinds = pd.Series([q["kind"] for q in queries]).value_counts().to_dict()
    assert kinds == {"answerable": 3, "unanswerable": 3, "probe": 6}
    for v in docs.values():
        assert all(v["text"][c["start"]:c["end"]] == c["value"] for c in v["canaries"])


def test_run_experiment_and_outputs(tmp_path):
    cfg = _cfg()
    docs, queries = build_corpus(fake_cuad(), cfg)
    res = run_experiment(docs, queries, cfg)
    assert set(res.metrics.metric) >= {"Correct answer", "Probe leaks anything", "Planted data sent off-machine"}
    assert (res.metrics[res.metrics.metric == "Planted data sent off-machine"].value == 0).all()
    # pattern detectors remove every non-name planted value
    rr = res.redaction_recall.drop("ALL")
    assert (rr.drop("NAME").recall == 1).all()
    # redaction can only reduce what the probes can surface
    probe = res.metrics[res.metrics.metric == "Probe leaks anything"].set_index("protection").value
    assert probe["redacted"] <= probe["raw"]

    out = save_results(res, tmp_path)
    make_figures(res, out)
    rep = write_report(res, out).read_text(encoding="utf-8")
    assert "## 1. Results" in rep and "Extractive (no LLM)" in rep
    for f in ["table_metrics.csv", "table_redaction_recall.csv", "table_all_answers.csv"]:
        assert (out / "tables" / f).exists()
    assert (out / "figures" / "fig2_leakage.png").exists()
    assert json.loads((out / "run_config.json").read_text())["seed"] == cfg.seed


def test_gate_on_committed_results():
    metrics = pd.read_csv("results/tables/table_metrics.csv")
    rules = json.loads(Path("configs/gate.json").read_text())["rules"]
    checks = evaluate(metrics, rules)
    assert all(c.passed for c in checks if c.severity == "error")


def test_gate_fails_on_threshold():
    metrics = pd.DataFrame([dict(arm="Local LLM (x)", protection="redacted", metric="Probe leaks anything",
                                 value=0.2, lo=0.1, hi=0.3, n=60)])
    (c,) = evaluate(metrics, [dict(arm="Local LLM", protection="redacted", metric="Probe leaks anything", max=0.05)])
    assert not c.passed


def test_cli_redact(tmp_path, capsys):
    f = tmp_path / "c.txt"
    f.write_text("Contact ilse.varnok@example.com or (415) 555-0142.", encoding="utf-8")
    assert main(["redact", str(f), "--no-ner"]) == 0
    assert capsys.readouterr().out == "Contact [EMAIL] or [PHONE]."


def test_config_rejects_unknown_keys():
    import pytest

    with pytest.raises(ValueError):
        Config.from_dict({"n_contract": 3})
    assert Config.from_json("configs/default.json").n_contracts == 30
