import importlib.util

import pytest

from contract_assistant.redaction import (
    SpacyNameDetector,
    is_person_name,
    merge_longest_first,
    redact,
)


def test_offset_map_relocates_clause(injected, base_text, gold_span):
    new, _, new_gold = injected
    r = redact(new)
    s2, e2 = r.map_pos(new_gold[0][0]), r.map_pos(new_gold[0][1])
    assert r.text[s2:e2] == base_text[gold_span[0]:gold_span[1]]


def test_pattern_detectors_catch_every_non_name_type(injected):
    new, canaries, _ = injected
    r = redact(new)
    for c in canaries:
        if c["type"] != "NAME":
            assert c["value"] not in r.text, f"pattern detector missed {c['type']}: {c['value']}"


@pytest.mark.parametrize("text,tag", [
    ("mail jane.doe@example.org now", "[EMAIL]"),
    ("call (415) 555-0123 today", "[PHONE]"),
    ("SSN 912-34-5678 on file", "[SSN]"),
    ("IBAN DE89 3704 0044 0532 0130 00 here", "[IBAN]"),
    ("lives at 42 Harlow Street, Tulsa", "[ADDRESS]"),
])
def test_each_pattern(text, tag):
    assert tag in redact(text).text


def test_name_filter():
    assert is_person_name("Tom Horner")
    assert not is_person_name("Licensee")
    assert not is_person_name("ZYNGA GAME IRELAND")
    assert not is_person_name("Licensed Party")


def test_custom_name_detector_is_used():
    text = "Signed by Ilse Varnok for the Company."
    start = text.index("Ilse")
    r = redact(text, lambda t: [(start, start + len("Ilse Varnok"), "NAME")])
    assert r.text == "Signed by [NAME] for the Company."


def test_merge_longest_first():
    assert merge_longest_first([(5, 8, "B"), (0, 10, "A")]) == [(0, 10, "A")]
    assert merge_longest_first([(0, 4, "A"), (2, 9, "B")]) == [(0, 9, "A")]
    assert merge_longest_first([(0, 2, "A"), (3, 5, "B")]) == [(0, 2, "A"), (3, 5, "B")]


@pytest.mark.skipif(importlib.util.find_spec("spacy") is None, reason="spaCy not installed")
def test_spacy_detector_finds_names():
    try:
        det = SpacyNameDetector(auto_download=False)
        det.nlp  # noqa: B018
    except OSError:
        pytest.skip("en_core_web_sm not downloaded")
    r = redact("This agreement was executed by John Smith on behalf of the Licensee.", det)
    assert "John Smith" not in r.text and "Licensee" in r.text
