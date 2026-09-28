import unicodedata

import pytest

from src import ingest, retriever
from src.retriever import bm25_search, rrf, tokenize_lexical
from tests.conftest import FakeEmbedder, ws_offsets


class FakeTokenizer:
    def __call__(self, a, b=None, **_):
        n = len(a.split()) + (len(b.split()) + 4 if b is not None else 0)
        return {"input_ids": [0] * n}


class FakeReranker:
    """Scores a passage by how many question words it contains."""
    tokenizer = FakeTokenizer()

    def predict(self, pairs, **_):
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]


def test_lexical_tokens_keep_codes_and_parts():
    toks = tokenize_lexical("QCVN 01:2008/BCT, pump P-101A/B.")
    assert "01:2008/bct" in toks and "p-101a/b" in toks
    assert {"qcvn", "2008", "bct", "p", "101a", "b"} <= set(toks)


def test_lexical_nfc_and_lowercase():
    nfd = unicodedata.normalize("NFD", "Kiểm Định")
    assert tokenize_lexical(nfd) == tokenize_lexical("kiểm định") == ["kiểm", "định"]


def test_rrf_manual_values_rank_from_one():
    fused = dict(rrf([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_dedup_ties_and_empty_branch():
    assert [cid for cid, _ in rrf([["b", "a"], ["a", "b"]])] == ["a", "b"]  # equal scores -> id order
    assert rrf([["a", "a", "b"], []]) == [("a", pytest.approx(1 / 61)), ("b", pytest.approx(1 / 63))]
    assert rrf([[], []]) == []
    assert len(rrf([["a", "b", "c"]], top=2)) == 2


def test_bm25_code_query_finds_right_chunk(built):
    cfg, gen, _ = built
    hits = bm25_search(gen, "What about KC-4561A?", 20)
    assert hits and "KC-4561A" in gen.by_id[hits[0][0]]["text"]
    assert all(isinstance(score, float) for _, score in hits)


def test_bm25_unseen_or_punctuation_query_is_empty(built):
    cfg, gen, _ = built
    assert bm25_search(gen, "zzzunseen-9999", 20) == []
    assert bm25_search(gen, "?!... --", 20) == []


def test_bm25_keeps_matched_docs_with_nonpositive_scores(built):
    cfg, gen, _ = built
    # "shall" occurs in most chunks of this tiny corpus, so its IDF is not positive; matches must still be returned.
    hits = bm25_search(gen, "shall", 20)
    assert len(hits) == sum("shall" in toks for toks in gen.token_sets) >= 2


def test_vector_search_small_corpus_returns_what_exists(built):
    cfg, gen, _ = built
    qvec = FakeEmbedder().encode(["hydrostatic test pressure"])[0]
    hits = retriever.vector_search(gen, qvec, 50)
    assert len(hits) == len(gen.chunks)
    assert "hydrostatic" in gen.by_id[hits[0][0]]["text"]
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


@pytest.mark.parametrize("mode", ["vector", "hybrid", "hybrid-rerank"])
def test_three_modes_return_ranked_hydrated_chunks(built, mode):
    cfg, gen, _ = built
    out = retriever.retrieve(gen, "hydrostatic test pressure of vessel V-101A/B", mode, cfg, FakeEmbedder(), FakeReranker())
    ranked = out["ranked"]
    assert 1 <= len(ranked) <= cfg.context_k
    assert [c["rank"] for c in ranked] == list(range(1, len(ranked) + 1))
    assert ranked[0]["file"] == "vessel.pdf" and ranked[0]["page"] == 1
    assert (out["timings_ms"]["rerank"] is not None) == (mode == "hybrid-rerank")
    assert (out["bm25"] != []) == (mode != "vector")


def test_hybrid_with_empty_bm25_falls_back_to_vector_order(built):
    cfg, gen, _ = built
    out = retriever.retrieve(gen, "zzzunseen", "hybrid", cfg, FakeEmbedder())
    assert out["bm25"] == [] and [c for c, _ in out["fused"]] == [c for c, _ in out["vector"]]


def test_reranker_rejects_overlong_pair_instead_of_truncating(built):
    cfg, gen, _ = built
    with pytest.raises(retriever.RetrievalInputError, match="exceeds max_length"):
        retriever.rerank(FakeReranker(), "pressure", gen.chunks[:2], 5, 4, max_length=6)
    with pytest.raises(retriever.RetrievalInputError, match="too long"):
        retriever.rerank(FakeReranker(), "word " * 50, gen.chunks[:2], 5, 4, max_length=60)


def test_rerank_orders_by_score_with_stable_ties(built):
    cfg, gen, _ = built
    ranked = retriever.rerank(FakeReranker(), "hydrostatic design pressure", gen.chunks, 2, 4, 512)
    assert "hydrostatic" in gen.by_id[ranked[0][0]]["text"] and len(ranked) == 2


def test_empty_question_and_unknown_mode(built):
    cfg, gen, _ = built
    with pytest.raises(retriever.RetrievalInputError):
        retriever.retrieve(gen, "   ", "vector", cfg, FakeEmbedder())
    with pytest.raises(retriever.RetrievalInputError):
        retriever.retrieve(gen, "pressure", "magic", cfg, FakeEmbedder())


def test_loaded_generation_is_unaffected_by_pointer_change(built):
    cfg, gen, first = built
    ingest.publish_pointer(cfg.index_root, "0000000000000000")
    out = retriever.retrieve(gen, "crane inspection", "hybrid", cfg, FakeEmbedder())
    assert out["ranked"] and gen.gen_id == first["generation_id"]
