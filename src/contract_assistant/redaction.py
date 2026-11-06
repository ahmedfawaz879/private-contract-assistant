"""Redaction: pattern detectors + a pluggable person-name detector, with an offset map.

Spans are replaced by ``[TYPE]``; overlapping detections are merged longest-first. The :class:`Redacted` offset map lets
expert-labelled answers be located again in the redacted text.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

Span = tuple[int, int, str]
NameDetector = Callable[[str], list[Span]]

PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]*\w",
    "IBAN": r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PHONE": r"(?:\+?1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]?\d{3}[ .-]\d{4}\b",
    "ADDRESS": r"\b\d{1,5} [A-Z][a-z]+(?: [A-Z][a-z]+)? "
               r"(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Lane|Ln\.|Drive|Dr\.|Boulevard|Blvd\.)",
}

# spaCy otherwise tags "Licensee" or "Manufacturer" as people and redacts the clauses we need.
ROLE_WORDS = {"licensee", "licensor", "manufacturer", "supplier", "distributor", "company", "party", "parties",
              "subcontractor", "contractor", "consultant", "customer", "client", "vendor", "purchaser", "seller",
              "buyer", "agent", "franchisee", "franchisor", "employee", "executive", "recipient", "affiliate",
              "agreement", "section", "exhibit", "schedule"}


def is_person_name(s: str) -> bool:
    """Keep NER PERSON hits that look like a personal name: 2-4 title-case words, none a defined contract role."""
    w = re.findall(r"[A-Za-z][A-Za-z.'-]*", s)
    return (2 <= len(w) <= 4
            and all(x[0].isupper() and not x.isupper() for x in w)
            and not any(x.lower().strip(".") in ROLE_WORDS for x in w))


class SpacyNameDetector:
    """Person names from a spaCy NER model, filtered by :func:`is_person_name`. Loaded lazily."""

    def __init__(self, model: str = "en_core_web_sm", window: int = 100_000, auto_download: bool = True):
        self.model_name, self.window, self.auto_download = model, window, auto_download
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy

            disable = ["parser", "lemmatizer", "attribute_ruler", "tagger"]
            try:
                self._nlp = spacy.load(self.model_name, disable=disable)
            except OSError:
                if not self.auto_download:
                    raise
                import subprocess
                import sys

                subprocess.run([sys.executable, "-m", "spacy", "download", self.model_name], check=True)
                self._nlp = spacy.load(self.model_name, disable=disable)
            self._nlp.max_length = 2_000_000
        return self._nlp

    @property
    def version(self) -> str:
        import spacy

        return spacy.__version__

    def __call__(self, text: str) -> list[Span]:
        w = self.window
        spans: list[Span] = []
        for i, doc in enumerate(self.nlp.pipe([text[i:i + w] for i in range(0, len(text), w)])):
            spans += [(e.start_char + i * w, e.end_char + i * w, "NAME")
                      for e in doc.ents if e.label_ == "PERSON" and is_person_name(e.text)]
        return spans


def merge_longest_first(spans: list[Span]) -> list[Span]:
    """Sort by start (longest first on ties) and merge overlaps, extending the earlier span."""
    spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[Span] = []
    for s, e, t in spans:
        if merged and s < merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1] = (merged[-1][0], e, merged[-1][2])
            continue
        merged.append((s, e, t))
    return merged


def detect(text: str, name_detector: NameDetector | None = None) -> list[Span]:
    spans = [(m.start(), m.end(), t) for t, p in PATTERNS.items() for m in re.finditer(p, text)]
    if name_detector is not None:
        spans += name_detector(text)
    return merge_longest_first(spans)


@dataclass
class Redacted:
    text: str
    reps: list  # (orig_start, orig_end, new_start, new_end, type)

    def map_pos(self, p: int) -> int:
        """Map an offset in the original text to the redacted text."""
        d = 0
        for os_, oe, ns, ne, _ in self.reps:
            if oe <= p:
                d += (ne - ns) - (oe - os_)
            elif os_ < p:
                return ns + (p - os_ if p - os_ < ne - ns else ne - ns)
            else:
                break
        return p + d


def redact(text: str, name_detector: NameDetector | None = None) -> Redacted:
    out, reps, last, pos = [], [], 0, 0
    for s, e, t in detect(text, name_detector):
        out.append(text[last:s])
        pos += s - last
        tag = f"[{t}]"
        reps.append((s, e, pos, pos + len(tag), t))
        out.append(tag)
        pos += len(tag)
        last = e
    out.append(text[last:])
    return Redacted("".join(out), reps)
