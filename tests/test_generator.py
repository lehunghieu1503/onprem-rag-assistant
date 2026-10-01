import json

import pytest
import requests

from src import generator
from src.config import load_config
from src.generator import validate_output

CFG = load_config(env={})
CHUNKS = [{"chunk_id": f"sha256:{i}", "file": "a.pdf", "page": i, "source_version": "v1",
           "text": f"Passage {i} says the test pressure is 1.5 times design pressure."} for i in range(1, 6)]


def count_words(system, user):
    return len(system.split()) + len(user.split())


class FakeResponse:
    def __init__(self, status=200, payload=None, text=None):
        self.status_code, self._payload = status, payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeHTTP:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.response


def ok(content, **extra):
    payload = {"message": {"content": content}, "done": True, "done_reason": "stop", "prompt_eval_count": 321,
               "eval_count": 40, "total_duration": 2_000_000_000, "load_duration": 5_000_000,
               "prompt_eval_duration": 500_000_000, "eval_duration": 1_400_000_000}
    payload.update(extra)
    return FakeHTTP(FakeResponse(200, payload))


def answer_json(claims, answerable=True, reason=None):
    return json.dumps({"answerable": answerable, "claims": claims, "refusal_reason": reason})


def test_answered_with_valid_citations_and_request_shape():
    http = ok(answer_json([{"text": "Test pressure is 1.5x design.", "source_ids": ["S1", "S1", "S3"]}], reason=""))
    out = generator.generate(CFG, "What is the test pressure?", CHUNKS, count_words, http)
    assert out["status"] == "answered" and out["error_code"] is None
    assert out["claims"] == [{"text": "Test pressure is 1.5x design.", "source_ids": ["S1", "S3"]}]
    assert [s["source_id"] for s in out["sources"]] == ["S1", "S2", "S3", "S4", "S5"]
    assert out["ollama_ms"] == {"total": 2000.0, "load": 5.0, "prompt_eval": 500.0, "eval": 1400.0}
    body = http.calls[0]["json"]
    assert body["think"] is False and body["stream"] is False and body["truncate"] is False
    assert body["options"]["num_ctx"] == 8192 and body["options"]["num_predict"] == 512
    assert body["format"]["properties"]["claims"]["items"]["properties"]["source_ids"]["items"]["enum"] == ["S1", "S2", "S3", "S4", "S5"]
    assert http.calls[0]["timeout"] == (5.0, 180.0)
    user = body["messages"][1]["content"]
    assert "[S5] file=a.pdf page=5" in user and user.rstrip().endswith("QUESTION: What is the test pressure?")


def test_model_refusal_is_refused_not_error():
    out = generator.generate(CFG, "q", CHUNKS, count_words, ok(answer_json([], False, "Not in sources.")))
    assert (out["status"], out["refusal_reason"], out["error_code"]) == ("refused", "Not in sources.", None)


def test_empty_context_refuses_without_calling_llm():
    http = ok("{}")
    out = generator.generate(CFG, "q", [], count_words, http)
    assert out["status"] == "refused" and out["refusal_kind"] == "no_evidence" and http.calls == []


@pytest.mark.parametrize("content,code", [
    ("not json", "invalid_model_output"),
    (json.dumps(["list"]), "invalid_model_output"),
    (json.dumps({"answerable": "yes", "claims": []}), "invalid_model_output"),
    (answer_json([]), "invalid_model_output"),  # answerable but no claims
    (answer_json([{"text": "  ", "source_ids": ["S1"]}]), "invalid_model_output"),
    (answer_json([{"text": "x", "source_ids": ["S1"]}], answerable=False), "invalid_model_output"),
    (answer_json([{"text": "x", "source_ids": []}]), "invalid_citation"),
    (answer_json([{"text": "x", "source_ids": ["S9"]}]), "invalid_citation"),
    (answer_json([{"text": "x"}]), "invalid_citation"),
])
def test_bad_model_output_is_error_never_refusal(content, code):
    out = generator.generate(CFG, "q", CHUNKS, count_words, ok(content))
    assert out["status"] == "error" and out["error_code"] == code
    assert out["raw_output"] == content and out["claims"] == []


def test_truncated_output_is_error():
    out = generator.generate(CFG, "q", CHUNKS, count_words, ok('{"answerable": true, "cla', done_reason="length"))
    assert (out["status"], out["error_code"]) == ("error", "output_truncated")


@pytest.mark.parametrize("http,code", [
    (FakeHTTP(exc=requests.Timeout()), "timeout"),
    (FakeHTTP(exc=requests.ConnectionError()), "ollama_unreachable"),
    (FakeHTTP(FakeResponse(404, {"error": "model not found"})), "model_missing"),
    (FakeHTTP(FakeResponse(500, None, "boom")), "http_500"),
    (FakeHTTP(FakeResponse(503, None, "busy")), "ollama_overloaded"),
    (FakeHTTP(FakeResponse(400, None, '{"type":"exceed_context_size_error"}')), "input_too_long"),
    (FakeHTTP(FakeResponse(200, None, "<html>")), "invalid_http_response"),
    (FakeHTTP(FakeResponse(200, {"message": {}, "done": True})), "invalid_http_response"),
    (FakeHTTP(FakeResponse(200, {"message": {"content": "{}"}, "done": False})), "invalid_http_response"),
    (FakeHTTP(FakeResponse(200, {"error": "oops"})), "invalid_http_response"),
])
def test_transport_failures_have_distinct_error_codes(http, code):
    out = generator.generate(CFG, "q", CHUNKS, count_words, http)
    assert (out["status"], out["error_code"]) == ("error", code)
    assert len(http.calls) == 1  # no silent retry


def test_context_trimmed_whole_chunks_from_the_end():
    tight = load_config(env={}, num_ctx=1000, num_predict=100, margin=50)
    big = [dict(c, text="word " * 300) for c in CHUNKS]
    kept, tokens = generator.select_context("q", big, count_words, 850)
    assert [c["chunk_id"] for c in kept] == ["sha256:1", "sha256:2"] and tokens <= 850
    http = ok(answer_json([{"text": "x", "source_ids": ["S2"]}]))
    out = generator.generate(tight, "q", big, count_words, http)
    assert out["context_dropped"] == 3 and [c["source_id"] for c in out["context_chunks"]] == ["S1", "S2"]
    assert http.calls[0]["json"]["format"]["properties"]["claims"]["items"]["properties"]["source_ids"]["items"]["enum"] == ["S1", "S2"]


def test_question_too_long_never_calls_llm():
    tight = load_config(env={}, num_ctx=1000, num_predict=100, margin=50)
    http = ok("{}")
    out = generator.generate(tight, "why " * 2000, CHUNKS, count_words, http)
    assert (out["status"], out["error_code"]) == ("error", "input_too_long") and http.calls == []


def test_only_context_ids_are_valid_and_gold_never_enters_prompt():
    assert validate_output(answer_json([{"text": "x", "source_ids": ["S3"]}]), ["S1", "S2"])["error_code"] == "invalid_citation"
    user = generator.build_user_message("q", CHUNKS[:1])
    assert "expected" not in user and "gold" not in user


def test_render_matches_ollama_template_shape():
    text = generator.render_ollama_qwen3("SYS", "USER")
    assert text == ("<|im_start|>system\n\nSYS<|im_end|>\n<|im_start|>user\nUSER /no_think<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n")
