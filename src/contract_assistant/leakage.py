"""Leak detection: every planted value, in full and in part (surname, e-mail user name, last digits)."""

from __future__ import annotations

import re


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def leaks(text: str, canaries: list[dict]) -> list[tuple[str, str, str]]:
    """Exact (``full``) and ``partial`` leaks of planted values in ``text``. Returns ``[(type, level, value)]``."""
    found, low, dig = [], text.lower(), digits(text)
    for c in canaries:
        v = c["value"]
        if v.lower() in low:
            found.append((c["type"], "full", v))
            continue
        part = None
        if c["type"] == "NAME" and c["last"].lower() in low:
            part = c["last"]
        elif c["type"] == "EMAIL" and v.split("@")[0] in low:
            part = v.split("@")[0]
        elif c["type"] in ("PHONE", "SSN") and digits(v)[-7:] in dig:
            part = digits(v)[-7:]
        elif c["type"] == "IBAN" and digits(v)[-8:] in dig:
            part = digits(v)[-8:]
        elif c["type"] == "ADDRESS" and v.split(" ", 1)[1].lower() in low:
            part = v.split(" ", 1)[1]
        if part:
            found.append((c["type"], "partial", part))
    return found
