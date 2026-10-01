import json

import pytest

from src import pipeline
from tests.conftest import FakeEmbedder
from tests.test_generator import FakeHTTP, answer_json, count_words, ok
from tests.test_retriever import FakeReranker

import requests


@pytest.fixture
def res():
    return pipeline.Resources(embedder=FakeEmbedder(), count_tokens=count_words, reranker=FakeReranker())


@pytest.mark.parametrize("mode", ["vector", "hybrid", "hybrid-rerank"])
def test_answered_result_contract(built, res, mode):
    cfg, gen, _ = built
    http = ok(answer_json([{"text": "1.5 times design pressure.", "source_ids": ["S1"]}]))
    r = pipeline.answer("hydrostatic test pressure of vessel V-101A/B", mode, cfg, res, gen, http)
    assert r["status"] == "answered" and r["mode"] == mode and r["generation_id"] == gen.gen_id
    assert r["retrieved_chunks"][0]["file"] == "vessel.pdf" and r["retrieved_chunks"][0]["rank"] == 1
    assert all({"chunk_id", "doc_id", "file", "page", "chunk_ordinal", "text", "rank"} <= set(c) for c in r["retrieved_chunks"])
    assert r["context_chunks"][0]["source_id"] == "S1"
    t = r["timings_ms"]
    assert t["wall_end_to_end"] > 0 and t["embed"] is not None and t["ollama"]["total"] == 2000.0
    assert (t["rerank"] is None) == (mode != "hybrid-rerank")
    json.dumps(r)  # QueryResult must be serialisable as-is


def test_error_and_refusal_stay_distinct(built, res):
    cfg, gen, _ = built
    refused = pipeline.answer("crane", "vector", cfg, res, gen, ok(answer_json([], False, "no")))
    broken = pipeline.answer("crane", "vector", cfg, res, gen, ok("garbage"))
    down = pipeline.answer("crane", "vector", cfg, res, gen, FakeHTTP(exc=requests.ConnectionError()))
    assert [x["status"] for x in (refused, broken, down)] == ["refused", "error", "error"]
    assert (broken["error_code"], down["error_code"]) == ("invalid_model_output", "ollama_unreachable")
    assert down["retrieved_chunks"] and down["timings_ms"]["ollama"]["total"] is None  # no fake zeros


def test_input_error_is_reported_not_raised(built, res):
    cfg, gen, _ = built
    r = pipeline.answer("   ", "vector", cfg, res, gen, ok("{}"))
    assert (r["status"], r["error_code"]) == ("error", "input_error") and r["timings_ms"]["wall_end_to_end"] is not None


def test_trimmed_passage_remains_in_retrieved_snapshot(built, res):
    cfg, gen, _ = built
    from src.config import load_config

    from src import generator, retriever

    q = "crane inspection wire ropes pressure"
    ranked = retriever.retrieve(gen, q, "vector", cfg, res.embedder)["ranked"]
    two = count_words(generator.SYSTEM_PROMPT, generator.build_user_message(q, ranked[:2]))
    tight = load_config(env={}, **{**cfg.as_dict(), "num_ctx": two + 60, "num_predict": 50, "margin": 10})
    http = ok(answer_json([{"text": "x", "source_ids": ["S1"]}]))
    r = pipeline.answer(q, "vector", tight, res, gen, http)
    assert len(r["context_chunks"]) < len(r["retrieved_chunks"])
    assert r["context_dropped"] == len(r["retrieved_chunks"]) - len(r["context_chunks"])


def test_render_text_never_shows_raw_output_for_errors(built, res):
    cfg, gen, _ = built
    r = pipeline.answer("crane", "vector", cfg, res, gen, ok("SECRET RAW"))
    assert pipeline.render_text(r) == "[error:invalid_model_output]"


def test_llm_state_handles_unreachable_server(built):
    cfg, _, _ = built

    class Down:
        def get(self, *a, **k):
            raise requests.ConnectionError()

    assert pipeline.llm_state(cfg, Down()) == {"digest": None, "loaded": None, "gpu_fraction": None}
