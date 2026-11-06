"""Command-line interface.

    contract-assistant run     --config configs/default.json --out results/
    contract-assistant redact  contract.txt [--no-ner]
    contract-assistant gate    --metrics results/tables/table_metrics.csv --rules configs/gate.json
    contract-assistant demo    --config configs/default.json [--share]
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from contract_assistant.config import Config


def _load_cfg(args) -> Config:
    cfg = Config.from_json(args.config) if args.config else Config()
    for key in ("n_contracts", "seed"):
        if getattr(args, key, None) is not None:
            setattr(cfg, key, getattr(args, key))
    if getattr(args, "no_dense", False):
        cfg.use_dense = False
    if getattr(args, "no_local", False):
        cfg.use_local = False
    if getattr(args, "no_ner", False):
        cfg.use_ner = False
    return cfg


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _setup(cfg: Config, cuad_dir: str, with_api: bool = True):
    """Corpus, name detector, embedder and backends according to the config and what is installed."""
    from contract_assistant.backends import ExtractiveBackend, get_secret
    from contract_assistant.corpus import build_corpus, load_cuad
    from contract_assistant.redaction import SpacyNameDetector
    from contract_assistant.retrieval import load_embedder

    gpu = _has_gpu()
    raw = load_cuad(cuad_dir, cfg.cuad_repo)
    docs, queries = build_corpus(raw, cfg)
    print(f"CUAD v1: {len(raw['data'])} contracts | sampled {len(docs)} | "
          f"{sum(len(v['canaries']) for v in docs.values())} planted values | {len(queries)} questions")

    env = {"python": platform.python_version()}
    ner = None
    if cfg.use_ner:
        try:
            ner = SpacyNameDetector(cfg.spacy_model)
            ner("Warm up.")
            env["spacy"] = ner.version
        except Exception as e:  # noqa: BLE001
            print("name detector off (pattern detectors only):", type(e).__name__, str(e)[:120])
            ner = None
    embedder = load_embedder(cfg.dense_model, "cuda" if gpu else "cpu") if cfg.use_dense else None

    backends = [ExtractiveBackend()]
    if cfg.use_local:
        try:
            from contract_assistant.backends.local_hf import LocalHFBackend

            backends.append(LocalHFBackend(cfg.local_model, cfg.max_new_tokens, cfg.local_batch))
        except Exception as e:  # noqa: BLE001
            print("local LLM arm off:", type(e).__name__, str(e)[:200])
    if with_api:
        keys = {"anthropic": get_secret("ANTHROPIC_API_KEY"), "openai": get_secret("OPENAI_API_KEY")}
        if not keys["anthropic"] and keys["openai"]:
            cfg.api_provider = "openai"
        if keys[cfg.api_provider]:
            from contract_assistant.backends.api import APIBackend

            backends.append(APIBackend(cfg.api_provider, cfg.api_model_name, keys[cfg.api_provider],
                                       cfg.max_new_tokens, cfg.api_workers))
    print(f"GPU {gpu} | arms: {[b.name for b in backends]}")
    return docs, queries, ner, embedder, backends, gpu, env


def cmd_run(args) -> int:
    from contract_assistant.experiment import run_experiment, save_results
    from contract_assistant.report import make_figures, write_report

    cfg = _load_cfg(args)
    docs, queries, ner, embedder, backends, gpu, env = _setup(cfg, args.cuad_dir, with_api=not args.no_api)
    res = run_experiment(docs, queries, cfg, backends, name_detector=ner, embedder=embedder, gpu=gpu, env=env)
    out = save_results(res, args.out)
    make_figures(res, out)
    rep = write_report(res, out)
    print(res.metrics.pivot_table(index="metric", columns=["arm", "protection"], values="value", sort=False).round(3))
    print(f"\nwrote tables, figures and {rep}")
    return 0


def cmd_redact(args) -> int:
    from contract_assistant.redaction import SpacyNameDetector, redact

    text = Path(args.file).read_text(encoding="utf-8") if args.file != "-" else sys.stdin.read()
    ner = None if args.no_ner else SpacyNameDetector(args.spacy_model)
    sys.stdout.write(redact(text, ner).text)
    return 0


def cmd_gate(args) -> int:
    from contract_assistant.gate import run_gate

    return run_gate(args.metrics, args.rules)


def cmd_demo(args) -> int:
    from contract_assistant.demo import build_app
    from contract_assistant.experiment import build_indexes, redact_corpus

    cfg = _load_cfg(args)
    docs, queries, ner, embedder, backends, _, _ = _setup(cfg, args.cuad_dir)
    red, _ = redact_corpus(docs, ner)
    index = build_indexes(docs, red, cfg, embedder)
    app = build_app(docs, index, backends, cfg.top_k, embedder, queries[0]["question"])
    app.launch(share=args.share, server_name=args.host, server_port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="contract-assistant", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", default="configs/default.json")
        sp.add_argument("--cuad-dir", default="data/cuad")
        sp.add_argument("--n-contracts", type=int)
        sp.add_argument("--seed", type=int)
        sp.add_argument("--no-dense", action="store_true", help="BM25 only")
        sp.add_argument("--no-local", action="store_true", help="skip the local LLM arm")
        sp.add_argument("--no-ner", action="store_true", help="pattern detectors only (no spaCy)")

    r = sub.add_parser("run", help="run the full leakage / faithfulness / latency benchmark")
    common(r)
    r.add_argument("--out", default="results")
    r.add_argument("--no-api", action="store_true", help="skip the API arm even if a key is set")
    r.set_defaults(fn=cmd_run)

    rd = sub.add_parser("redact", help="redact personal data from a text file ('-' for stdin)")
    rd.add_argument("file")
    rd.add_argument("--no-ner", action="store_true")
    rd.add_argument("--spacy-model", default="en_core_web_sm")
    rd.set_defaults(fn=cmd_redact)

    g = sub.add_parser("gate", help="fail if metrics cross declared thresholds (for CI)")
    g.add_argument("--metrics", default="results/tables/table_metrics.csv")
    g.add_argument("--rules", default="configs/gate.json")
    g.set_defaults(fn=cmd_gate)

    d = sub.add_parser("demo", help="launch the Gradio demo")
    common(d)
    d.add_argument("--share", action="store_true")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=7860)
    d.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
