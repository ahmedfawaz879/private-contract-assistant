import numpy as np
import pytest

from contract_assistant.synthetic import inject

BASE = ("This Agreement is made between Alpha Corp. and Beta LLC. The Term shall be two years.\n\n"
        "This Agreement shall be governed by the laws of the State of New York. Either party may terminate on notice.\n\n"
        "Miscellaneous provisions follow. Nothing herein creates a partnership.")


@pytest.fixture
def base_text():
    return BASE


@pytest.fixture
def gold_span():
    return (BASE.index("This Agreement shall be governed"), BASE.index("New York.") + 9)


@pytest.fixture
def injected(gold_span):
    rng = np.random.default_rng(0)
    new, canaries, new_gold = inject(BASE, [gold_span], rng, 3, set(), near_span=gold_span)
    return new, canaries, new_gold


def fake_cuad(n_docs: int = 3) -> dict:
    """A CUAD-shaped dict: governing law present, renewal term absent."""
    data = []
    for i in range(n_docs):
        filler = " ".join(f"Section {k}. The parties agree to clause number {k} of this document." for k in range(60))
        gov = f"This Agreement shall be governed by the laws of the State of Delaware number {i}."
        ctx = f"MASTER SERVICES AGREEMENT {i}.\n\n{filler}\n\n{gov} {filler}\n\nEnd of agreement."
        start = ctx.index(gov)
        qas = [
            dict(question='Highlight the parts (if any) of this contract related to "Governing Law" that should be '
                          "reviewed by a lawyer. Details: Which state/country's law governs the interpretation of "
                          "the contract?",
                 answers=[dict(text=gov, answer_start=start)]),
            dict(question='Highlight the parts (if any) of this contract related to "Renewal Term" that should be '
                          "reviewed by a lawyer. Details: What is the renewal term after the initial term expires?",
                 answers=[]),
        ]
        data.append(dict(title=f"Contract_{i}", paragraphs=[dict(context=ctx, qas=qas)]))
    return {"data": data}
