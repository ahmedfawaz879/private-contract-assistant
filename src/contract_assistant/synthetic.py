"""Synthetic personal data, fake by construction, planted at known character offsets.

* social-security numbers start with 9 (never issued);
* phone numbers use the 555-01xx fictional range;
* e-mails use the reserved ``example.*`` domains;
* IBANs carry a valid check digit on a made-up account;
* surnames are generated from syllables so they match nothing real.
"""

from __future__ import annotations

import re

import numpy as np

FIRST = ["Ilse", "Marek", "Odalys", "Teodor", "Saoirse", "Anouk", "Bram", "Liesl", "Caius", "Nerea", "Oskar", "Yara",
         "Emeric", "Junia", "Pavlo", "Rosalind", "Tobiah", "Wren", "Zelda", "Idris", "Ottilie", "Lucan", "Mireille", "Soren"]
SYL = ["var", "nok", "quen", "neva", "bel", "tor", "ash", "mir", "oss", "dra", "vik", "lund", "hal", "zem", "rov", "tal",
       "ibr", "kess"]
STREETS = ["Harlow", "Quillon", "Brackett", "Mirefield", "Calder", "Tamsin", "Wexley", "Orrin", "Pellam", "Sorrel"]
STYPES = ["Street", "Avenue", "Road", "Lane", "Drive"]
CITIES = ["Tulsa, OK 74103", "Dayton, OH 45402", "Fresno, CA 93721", "Boise, ID 83702", "Albany, NY 12207",
          "Mobile, AL 36602"]
ROLES = ["the Distributor", "the Licensee", "the Company", "the Supplier", "the Consultant"]

# (sentence, which values it plants)
TEMPLATES = [
    ("Notices to {role} shall be sent to the attention of {NAME}, {ADDRESS}, {CITY}; telephone {PHONE}; e-mail {EMAIL}.",
     ["NAME", "ADDRESS", "PHONE", "EMAIL"]),
    ("All payments due to {role} hereunder shall be wired to the personal account of {NAME}, IBAN {IBAN}.",
     ["NAME", "IBAN"]),
    ("Executed on behalf of {role} by {NAME} (Social Security No. {SSN}), residing at {ADDRESS}, {CITY}.",
     ["NAME", "SSN", "ADDRESS"]),
]


def iban_de(rng: np.random.Generator) -> str:
    """A German-format IBAN with a valid ISO 13616 check digit on a random (fake) account."""
    bban = f"{rng.integers(10_000_000, 99_999_999)}{rng.integers(1_000_000_000, 9_999_999_999)}"
    check = 98 - int(bban + "131400") % 97  # D=13, E=14, '00' placeholder
    s = f"DE{check:02d}{bban}"
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4))


def iban_valid(iban: str) -> bool:
    s = iban.replace(" ", "")
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def make_person(rng: np.random.Generator, used: set[str]) -> dict:
    """One synthetic person with a unique surname (tracked in ``used``)."""
    while True:
        last = "".join(rng.choice(SYL, 3)).capitalize()
        if last not in used:
            used.add(last)
            break
    first = str(rng.choice(FIRST))
    return dict(
        first=first, last=last, NAME=f"{first} {last}",
        ADDRESS=f"{rng.integers(10, 9899)} {rng.choice(STREETS)} {rng.choice(STYPES)}",
        CITY=str(rng.choice(CITIES)),
        PHONE=f"({rng.integers(201, 989)}) 555-01{rng.integers(0, 100):02d}",
        EMAIL=f"{first.lower()}.{last.lower()}@example.{rng.choice(['com', 'org', 'net'])}",
        SSN=f"{rng.integers(900, 1000)}-{rng.integers(10, 100)}-{rng.integers(1000, 10000)}",
        IBAN=iban_de(rng),
    )


def boundaries(text: str) -> list[int]:
    """Insertion points: after a blank line or a sentence end."""
    return [m.end() for m in re.finditer(r"\n\s*\n|(?<=[a-z0-9\)])\.\s{1,3}(?=[A-Z])", text)]


def inject(text: str, gold_spans: list[tuple[int, int]], rng: np.random.Generator, n_records: int, used: set[str],
           near_span: tuple[int, int] | None = None):
    """Plant ``n_records`` contract-style sentences at boundaries outside any gold span.

    Returns ``(new_text, canaries, shifted_gold_spans)`` where each canary is
    ``dict(type, value, start, end, record, last)`` with offsets into ``new_text``.
    If ``near_span`` is given, the first record goes right after that clause.
    """
    pts = [p for p in boundaries(text) if not any(s < p < e for s, e in gold_spans)]
    if len(pts) < n_records:
        pts = [len(text)] * n_records
    chosen = list(rng.choice(pts, n_records, replace=len(pts) < n_records))
    if near_span is not None:
        after = [p for p in pts if p >= near_span[1]]
        if after:
            chosen[0] = after[0]
    inserts = []
    for r, p in enumerate(chosen):
        person = make_person(rng, used)
        tpl, keys = TEMPLATES[r % len(TEMPLATES)]
        sent = " " + tpl.format(role=rng.choice(ROLES), **person) + " "
        inserts.append((int(p), sent, person, keys, r))
    inserts.sort(key=lambda x: x[0])

    out, canaries, shift, last = [], [], 0, 0
    for p, sent, person, keys, r in inserts:
        out.append(text[last:p])
        base = p + shift
        for k in keys:
            off = sent.index(person[k])
            canaries.append(dict(type=k, value=person[k], start=base + off, end=base + off + len(person[k]),
                                 record=r, last=person["last"]))
        out.append(sent)
        shift += len(sent)
        last = p
    out.append(text[last:])
    new = "".join(out)

    def mv(x: int) -> int:
        return x + sum(len(s) for p, s, *_ in inserts if p <= x)

    new_gold = [(mv(s), mv(s) + (e - s)) for s, e in gold_spans]
    return new, canaries, new_gold
