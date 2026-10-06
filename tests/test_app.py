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
    at.chat_input[0].set_value("crane inspection interval").run()
    assert any("Inspect every 12 months" in m.value for m in at.markdown)
    assert any("\\<b\\>" in m.value for m in at.markdown)  # model text is escaped, not rendered as HTML
    at.chat_input[0].set_value("something absent").run()
    assert any("Not in the documents." in w.value for w in at.warning)
    at.chat_input[0].set_value("break it").run()
    assert any("invalid_model_output" in e.value for e in at.error)
    assert "SECRET" not in " ".join(m.value for m in at.markdown)
    assert len(at.session_state["history"]) == 3
    assert loads["models"] == 1  # reruns and repeated queries never reload models
