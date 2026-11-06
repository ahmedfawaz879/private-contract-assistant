from __future__ import annotations

import re
import time

from contract_assistant.retrieval import toks

STOP = {"the", "of", "a", "to", "and", "or", "in", "this", "is", "be", "any"}


class ExtractiveBackend:
    """No LLM: returns the excerpt sentence sharing the most words with the question.

    Free, instant and cannot invent anything, so it sets the bar an LLM has to beat.
    """

    name = "Extractive (no LLM)"
    egress = False

    def generate(self, items: list[dict]) -> list[dict]:
        out = []
        for it in items:
            t0 = time.perf_counter()
            qt = set(toks(it["question"])) - STOP
            best = (0, None, None)
            for ci_, c in enumerate(it["chunks"]):
                for s in re.split(r"(?<=[.;])\s+", c):
                    sc = len(qt & set(toks(s)))
                    if sc > best[0] and len(s) > 25:
                        best = (sc, s.strip(), ci_ + 1)
            txt = "NOT FOUND" if best[0] < 2 else f'ANSWER: {best[1][:200]}\nQUOTE: "{best[1]}" [{best[2]}]'
            out.append(dict(text=txt, in_tok=0, out_tok=0, seconds=time.perf_counter() - t0))
        return out
