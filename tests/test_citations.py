from docrag.generation.citations import (
    check_citations,
    cited_aliases,
    strip_unknown_markers,
    uncited_sentences,
)


def test_parses_markers_in_order_without_duplicates() -> None:
    assert cited_aliases("A [S2]. B [S1][S2]. C [S10].") == ["S2", "S1", "S10"]


def test_valid_answer() -> None:
    ans = "Hooks are external contracts [S1]. Each pool has at most one hook [S2]."
    r = check_citations(ans, n_sources=3, abstained=False)
    assert r.valid and r.cited == ["S1", "S2"] and r.unknown == []


def test_cites_source_outside_retrieved_set() -> None:
    r = check_citations("Fee is 0.3% [S6].", n_sources=5, abstained=False)
    assert not r.valid
    assert r.unknown == ["S6"]


def test_answer_without_citations_is_invalid() -> None:
    r = check_citations("The fee tier is 0.3 percent for this pool.", 5, abstained=False)
    assert not r.valid
    assert r.reason == "answer has no citations"


def test_uncited_claim_sentence_detected() -> None:
    ans = "Hooks are external contracts [S1]. They can also mint tokens for free."
    r = check_citations(ans, 3, abstained=False)
    assert not r.valid
    assert r.uncited_sentences == ["They can also mint tokens for free."]


def test_abstention_needs_no_citations() -> None:
    r = check_citations("The documentation provided does not cover this.", 5, abstained=True)
    assert r.valid


def test_lead_in_and_code_are_not_claims() -> None:
    ans = (
        "Call it like this:\n\n```solidity\nmanager.initialize(key, price);\n```\n\n"
        "The price is a Q64.96 value [S2]."
    )
    assert uncited_sentences(ans) == []


def test_short_fragments_ignored() -> None:
    assert uncited_sentences("Yes. It is [S1].") == []


def test_strip_unknown_markers() -> None:
    assert strip_unknown_markers("A [S1]. B [S9].", 3) == "A [S1]. B ."
