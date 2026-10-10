"""UI logic tests with Streamlit's AppTest; models, index and Ollama are replaced by doubles."""
import pytest
from streamlit.testing.v1 import AppTest

from src import pipeline, retriever
from tests.conftest import FakeEmbedder
from tests.test_generator import answer_json, count_words, ok
from tests.test_retriever import FakeReranker


@pytest.fixture
def app_env(built, monkeypatch):
    cfg, gen, _ = built
    for key, value in cfg.as_dict().items():
        monkeypatch.setenv("RAG_" + key.upper(), str(value))
    loads = {"models": 0}

    def fake_resources(cfg_, with_reranker=True):
        loads["models"] += 1
        return pipeline.Resources(FakeEmbedder(), count_words, FakeReranker())

    monkeypatch.setattr(pipeline, "load_resources", fake_resources)
    monkeypatch.setattr(pipeline, "llm_state", lambda cfg_, http=None: {"digest": "abc", "loaded": True, "gpu_fraction": 1})
    return cfg, gen, loads


def run_app():
    from pathlib import Path

    return AppTest.from_file(str(Path(__file__).parent.parent / "src" / "app.py"), default_timeout=30).run()


def test_missing_index_and_model_show_remediation(app_env, monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_INDEX_ROOT", str(tmp_path / "nowhere"))
    monkeypatch.setattr(pipeline, "llm_state", lambda cfg_, http=None: {"digest": None, "loaded": False, "gpu_fraction": None})
    at = run_app()
    errors = " ".join(e.value for e in at.error)
    assert "python -m src.ingest" in errors and "ollama pull" in errors
    assert not at.chat_input  # no chat until ready


def test_answer_refusal_and_error_render_differently(app_env, monkeypatch):
    cfg, gen, loads = app_env
    outcomes = iter([
        ok(answer_json([{"text": "Inspect every 12 months <b>*bold*</b>.", "source_ids": ["S1"]}])),
        ok(answer_json([], False, "Not in the documents.")),
        ok("garbage SECRET"),
    ])
    real_answer = pipeline.answer
    monkeypatch.setattr(pipeline, "answer", lambda q, mode, cfg_, res, g: real_answer(q, mode, cfg_, res, g, next(outcomes)))

    at = run_app()
    page = lambda: " ".join(m.value for m in at.markdown)
    assert len(at.button) == 4  # suggested questions on the empty screen
    at.chat_input[0].set_value("crane inspection interval").run()
    assert "Inspect every 12 months" in page() and "Nguồn 1" in page()
    assert "&lt;b&gt;" in page() and "<b>*bold*</b>" not in page()  # model text is escaped, never rendered as HTML
    assert len(at.button) == 0  # suggestions disappear once there is history
    at.chat_input[0].set_value("something absent").run()
    assert "Không tìm thấy trong tài liệu" in page() and "Not in the documents." in page()
    at.chat_input[0].set_value("break it").run()
    assert "invalid_model_output" in page() and "SECRET" not in page()
    assert len(at.session_state["history"]) == 3
    assert loads["models"] == 1  # reruns and repeated queries never reload models


def test_suggested_question_is_asked(app_env, monkeypatch):
    cfg, gen, loads = app_env
    real_answer = pipeline.answer
    reply = ok(answer_json([{"text": "02 năm.", "source_ids": ["S1"]}]))
    monkeypatch.setattr(pipeline, "answer", lambda q, mode, cfg_, res, g: real_answer(q, mode, cfg_, res, g, reply))
    at = run_app()
    at.button[0].click().run()
    assert at.session_state["history"][0]["question"].startswith("Thời hạn kiểm định định kỳ")
    assert "02 năm." in " ".join(m.value for m in at.markdown)
