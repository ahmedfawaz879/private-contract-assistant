from __future__ import annotations

import time


class LocalHFBackend:
    """Open-weights chat model run with Hugging Face transformers on this machine. Nothing leaves the machine."""

    egress = False

    def __init__(self, model_id: str, max_new_tokens: int = 160, batch_size: int = 8, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = f"Local LLM ({model_id.split('/')[-1]})"
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu = self.dev.startswith("cuda")
        dtype = torch.float16 if self.gpu else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.dev).eval()

    def _sync(self):
        if self.gpu:
            self.torch.cuda.synchronize()

    def _run(self, batch: list[dict]) -> list[dict]:
        texts = [self.tok.apply_chat_template(
            [{"role": "system", "content": b["system"]}, {"role": "user", "content": b["user"]}],
            tokenize=False, add_generation_prompt=True) for b in batch]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.dev)
        self._sync()
        t0 = time.perf_counter()
        with self.torch.no_grad():
            g = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
        self._sync()
        dt = time.perf_counter() - t0
        new = g[:, enc["input_ids"].shape[1]:]
        outs = self.tok.batch_decode(new, skip_special_tokens=True)
        return [dict(text=o, in_tok=int(enc["attention_mask"][i].sum()),
                     out_tok=int((new[i] != self.tok.pad_token_id).sum()), seconds=dt / len(batch))
                for i, o in enumerate(outs)]

    def generate(self, items: list[dict]) -> list[dict]:
        res: list[dict] = []
        for i in range(0, len(items), self.batch_size):
            res += self._run(items[i:i + self.batch_size])
        return res

    def latency(self, items: list[dict]) -> list[float]:
        """Single-question (unbatched) wall time, what a user waits."""
        return [self._run([it])[0]["seconds"] for it in items]
