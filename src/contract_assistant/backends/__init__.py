"""Answer generators: ``generate(list of dict(system, user, question, chunks)) -> list of dict(text, in_tok, out_tok,
seconds)``. Heavy dependencies (torch, transformers, provider SDKs) are imported only when a backend is constructed."""

from contract_assistant.backends.base import Backend, get_secret
from contract_assistant.backends.extractive import ExtractiveBackend

__all__ = ["Backend", "ExtractiveBackend", "get_secret"]
