import numpy as np
import pandas as pd

from contract_assistant.retrieval import DocIndex, chunk, rrf
from contract_assistant.stats import boot_by_group, ci, paired


def test_fusion_and_chunking():
    assert rrf([[0, 1, 2], [1, 2, 0]])[0] == 1
    assert len(chunk("a " * 3000, 1200, 200)) > 4


def test_chunks_cover_text_with_overlap():
    text = "word " * 2000
    spans = chunk(text, 1200, 200)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s1, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert s2 < e1 and s2 > s1


def test_bm25_finds_relevant_chunk():
    text = ("Payment terms are net thirty days. " * 40 + "This Agreement shall be governed by the laws of Delaware. "
            + "Delivery shall occur at the warehouse. " * 40)
    ix = DocIndex(text, 400, 50)
    top = ix.chunks[ix.search("which law governs this agreement", 1)[0]]
    assert "governed by the laws" in top


def test_bootstrap_constant_and_ci():
    b = boot_by_group(np.ones(30), np.repeat(np.arange(10), 3), 200, 0)
    assert np.allclose(b, 1.0)
    lo, hi = ci(boot_by_group(np.r_[np.zeros(15), np.ones(15)], np.repeat(np.arange(10), 3), 500, 0))
    assert 0 <= lo < 0.5 < hi <= 1


def test_paired_difference():
    a = pd.DataFrame(dict(qid=list("abcd"), doc=list("xxyy"), v=[1, 1, 1, 1]))
    b = pd.DataFrame(dict(qid=list("abcd"), doc=list("xxyy"), v=[0, 1, 0, 1]))
    m, lo, hi = paired(a, b, "v", 200, 0)
    assert m == 0.5 and lo <= m <= hi
