from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class APIBackend:
    """Commercial model over the internet. Everything in the prompt leaves the machine (``egress=True``)."""

    egress = True

    def __init__(self, provider: str, model: str, api_key: str, max_new_tokens: int = 160, workers: int = 6):
        self.provider, self.model = provider, model
        self.max_new_tokens, self.workers = max_new_tokens, workers
        if provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            import openai

            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"unknown provider {provider!r}")
        self.name = f"API LLM ({self.model})"

    def _call(self, it: dict) -> tuple[str, int, int]:
        if self.provider == "anthropic":
            kw = dict(model=self.model, max_tokens=self.max_new_tokens, system=it["system"],
                      messages=[{"role": "user", "content": it["user"]}])
            try:
                r = self.client.messages.create(temperature=0, **kw)
            except Exception as e:  # some models reject temperature
                if "temperature" not in str(e).lower():
                    raise
                r = self.client.messages.create(**kw)
            text = "".join(getattr(b, "text", "") for b in r.content)
            return text, r.usage.input_tokens, r.usage.output_tokens
        msgs = [{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}]
        try:
            r = self.client.chat.completions.create(model=self.model, messages=msgs,
                                                    max_tokens=self.max_new_tokens, temperature=0)
        except Exception:  # noqa: BLE001 - newer models take max_completion_tokens and no temperature
            r = self.client.chat.completions.create(model=self.model, messages=msgs,
                                                    max_completion_tokens=self.max_new_tokens)
        return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens

    def _one(self, it: dict) -> dict:
        err: Exception | None = None
        for attempt in range(4):
            try:
                t0 = time.perf_counter()
                text, i_, o_ = self._call(it)
                return dict(text=text, in_tok=i_, out_tok=o_, seconds=time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001 - retry with backoff, then record the error
                err = e
                time.sleep(2 ** attempt)
        return dict(text=f"ERROR: {type(err).__name__}", in_tok=0, out_tok=0, seconds=np.nan)

    def generate(self, items: list[dict]) -> list[dict]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, items))

    def latency(self, items: list[dict]) -> list[float]:
        return [self._one(it)["seconds"] for it in items]
