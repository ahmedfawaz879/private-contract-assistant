"""Run configuration. Every setting a reviewer may want to challenge lives here."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_CATEGORIES = [
    "Governing Law",
    "Anti-Assignment",
    "Cap On Liability",
    "Audit Rights",
    "Termination For Convenience",
    "Renewal Term",
]


@dataclass
class Config:
    seed: int = 42
    cuad_repo: str = "https://github.com/TheAtticusProject/cuad.git"
    # contracts sampled (seeded) within this length range
    n_contracts: int = 30
    min_chars: int = 8_000
    max_chars: int = 120_000
    # mix of mostly-present and often-absent clause types
    categories: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    # synthetic personal records planted per contract (one next to a gold clause)
    records_per_contract: int = 3
    chunk_chars: int = 1200
    chunk_overlap: int = 200
    top_k: int = 4
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_dense: bool = True  # falls back to BM25 only if sentence-transformers is unavailable
    spacy_model: str = "en_core_web_sm"
    use_ner: bool = True  # spaCy person-name detector; pattern detectors always run
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Apache-2.0
    use_local: bool = True
    local_batch: int = 8
    max_new_tokens: int = 160
    cpu_max_queries: int = 30  # without a GPU the local arm runs on a subsample
    latency_probe_n: int = 20  # single-query (unbatched) latency measured on this many queries
    api_provider: str = "anthropic"
    api_model: str = "claude-haiku-5-5"
    openai_model: str = "gpt-4o-mini"
    api_workers: int = 6
    # PRICES: fill in from the provider's price page. Left as None, cost is reported in tokens/seconds only.
    price_api_in_per_mtok: float | None = None
    price_api_out_per_mtok: float | None = None
    gpu_usd_per_hour: float | None = None
    n_boot: int = 1000
    run_demo: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @property
    def api_model_name(self) -> str:
        return self.api_model if self.api_provider == "anthropic" else self.openai_model
