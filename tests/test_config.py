import pytest

from src.config import ConfigError, load_config, prompt_budget


def test_defaults_are_valid():
    cfg = load_config(env={})
    assert cfg.llm_model == "qwen3:4b-q4_K_M"
    assert prompt_budget(cfg) == 8192 - 512 - 256


def test_env_override_and_types():
    cfg = load_config(env={"RAG_CHUNK_TOKENS": "350", "RAG_TEMPERATURE": "0", "RAG_OLLAMA_URL": "http://ollama:11434"})
    assert (cfg.chunk_tokens, cfg.temperature, cfg.ollama_url) == (350, 0.0, "http://ollama:11434")


@pytest.mark.parametrize("env", [
    {"RAG_EMBED_BATCH": "abc"},
    {"RAG_EMBED_BATCH": "0"},
    {"RAG_PDF_DEADLINE_S": "-1"},
    {"RAG_CHUNK_OVERLAP": "500"},
    {"RAG_NUM_PREDICT": "8000"},
    {"RAG_CONTEXT_K": "50"},
    {"RAG_OLLAMA_URL": "ollama:11434"},
    {"RAG_DEVICE": "tpu"},
    {"RAG_RERANK_DEVICE": "gpu0"},
])
def test_invalid_values_rejected(env):
    with pytest.raises(ConfigError):
        load_config(env=env)


def test_index_hash_ignores_query_time_settings():
    base = load_config(env={})
    assert base.index_config_hash() == load_config(env={}, temperature=0.0, context_k=3).index_config_hash()
    assert base.index_config_hash() != load_config(env={}, chunk_tokens=350).index_config_hash()
