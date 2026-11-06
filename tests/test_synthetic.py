import numpy as np

from contract_assistant.synthetic import boundaries, iban_valid, make_person


def test_planted_values_at_recorded_offsets(injected):
    new, canaries, _ = injected
    assert canaries
    assert all(new[c["start"]:c["end"]] == c["value"] for c in canaries)


def test_gold_clause_intact_and_relocated(injected, base_text, gold_span):
    new, _, new_gold = injected
    assert new[new_gold[0][0]:new_gold[0][1]] == base_text[gold_span[0]:gold_span[1]]


def test_near_answer_record_placed_after_clause(injected):
    _, canaries, new_gold = injected
    assert min(c["start"] for c in canaries if c["record"] == 0) >= new_gold[0][1]


def test_iban_check_digit_valid():
    for seed in range(20):
        assert iban_valid(make_person(np.random.default_rng(seed), set())["IBAN"])


def test_values_are_fake_by_construction():
    used = set()
    rng = np.random.default_rng(3)
    people = [make_person(rng, used) for _ in range(50)]
    assert len({p["last"] for p in people}) == 50, "surnames are unique"
    for p in people:
        assert p["SSN"].startswith("9")
        assert "555-01" in p["PHONE"]
        assert "@example." in p["EMAIL"]


def test_boundaries_never_split_words(base_text):
    for b in boundaries(base_text):
        assert b == len(base_text) or base_text[b - 1].isspace()
