"""Evaluation gate: fail (non-zero exit) when a deployed configuration crosses declared thresholds.

Rules (JSON list) look like::

    {"arm": "Local LLM", "protection": "redacted", "metric": "Probe leaks anything", "max": 0.05, "severity": "error"}

``arm`` matches by prefix. ``min``/``max`` are checked against the point estimate, or against the interval bound
when ``"use_bound": true`` (``hi`` for ``max``, ``lo`` for ``min``), which is the conservative choice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Check:
    rule: dict
    value: float | None
    passed: bool
    message: str

    @property
    def severity(self) -> str:
        return self.rule.get("severity", "error")


def evaluate(metrics: pd.DataFrame, rules: list[dict]) -> list[Check]:
    checks = []
    for rule in rules:
        m = metrics[metrics.arm.str.startswith(rule["arm"]) & (metrics.protection == rule["protection"])
                    & (metrics.metric == rule["metric"])]
        label = f"{rule['arm']} | {rule['protection']} | {rule['metric']}"
        if m.empty:
            checks.append(Check(rule, None, False, f"{label}: no such metric in results"))
            continue
        r = m.iloc[0]
        ok, parts = True, []
        if "max" in rule:
            v = r.hi if rule.get("use_bound") else r.value
            ok &= v <= rule["max"]
            parts.append(f"{v:.3f} <= {rule['max']}")
        if "min" in rule:
            v = r.lo if rule.get("use_bound") else r.value
            ok &= v >= rule["min"]
            parts.append(f"{v:.3f} >= {rule['min']}")
        checks.append(Check(rule, float(r.value), bool(ok), f"{label}: {' and '.join(parts)}"))
    return checks


def run_gate(metrics_csv: str | Path, rules_json: str | Path) -> int:
    metrics = pd.read_csv(metrics_csv)
    rules = json.loads(Path(rules_json).read_text(encoding="utf-8"))["rules"]
    checks = evaluate(metrics, rules)
    failed = 0
    for c in checks:
        tag = "PASS" if c.passed else ("FAIL" if c.severity == "error" else "WARN")
        failed += (not c.passed) and c.severity == "error"
        print(f"[{tag}] {c.message}")
    print(f"\n{len(checks)} checks, {failed} blocking failure(s)")
    return 1 if failed else 0
