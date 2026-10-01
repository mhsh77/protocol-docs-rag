from docrag.retrieval.lexical import BM25Index, tokenize


def test_tokenize_keeps_identifiers_and_their_parts() -> None:
    toks = tokenize("Call PoolManager.initialize with sqrtPriceX96")
    assert "poolmanager.initialize" in toks
    assert {"pool", "manager", "initialize"} <= set(toks)
    assert "sqrtpricex96" in toks
    assert {"sqrt", "price"} <= set(toks)
    assert "with" not in toks  # stopword


def test_bm25_ranks_exact_identifier_match_first() -> None:
    idx = BM25Index.build(
        ["a", "b", "c"],
        [
            "Hooks run beforeSwap and afterSwap callbacks.",
            "Liquidity providers earn fees from swaps.",
            "The router batches commands.",
        ],
    )
    hits = idx.search("when is beforeSwap called", k=3)
    assert hits[0][0] == "a"
    assert all(score > 0 for _, score in hits)


def test_bm25_empty_query_returns_nothing() -> None:
    idx = BM25Index.build(["a"], ["text"])
    assert idx.search("the of and", k=5) == []
