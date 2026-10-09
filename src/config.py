"""Runtime configuration read from environment variables (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

MODES = ("vector", "hybrid", "hybrid-rerank")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    # models
    llm_model: str = "qwen3:4b-q4_K_M"
    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "BAAI/bge-m3"
    embed_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    llm_tokenizer: str = "Qwen/Qwen3-4B"
    llm_tokenizer_revision: str = "1cfa9a7208912126459214e8b04321603b3df60c"
    device: str = "cpu"
    rerank_device: str = "cpu"  # "cuda" runs the reranker in fp16 next to the LLM (needs requirements-gpu.txt)
    torch_threads: int = 8  # tuned on dev: 8 threads rerank faster than all 24 hybrid cores
    # ingestion
    chunk_tokens: int = 500
    chunk_overlap: int = 80
    embed_batch: int = 8
    pdf_deadline_s: int = 120
    min_page_chars: int = 40
    # retrieval
    vector_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60
    candidates: int = 10  # tuned on dev: same Hit@k as 20, half the CPU rerank time
    context_k: int = 5
    rerank_batch: int = 4
    rerank_max_length: int = 1024
    # generation
    num_ctx: int = 8192
    num_predict: int = 512
    margin: int = 256
    temperature: float = 0.1
    seed: int = 7
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 180.0
    keep_alive: str = "30m"
    # paths
    raw_dir: str = "data/raw"
    sources_path: str = "data/sources.jsonl"
    index_root: str = "data/index"

    def index_config(self) -> dict:
        """Only the settings that change index contents; top_k/temperature do not force a re-ingest."""
        return {
            "embed_model": self.embed_model,
            "embed_revision": self.embed_revision,
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap": self.chunk_overlap,
            "min_page_chars": self.min_page_chars,
            "cleanup_version": 1,
            "lexical_version": 1,
            "distance": "cosine",
        }

    def index_config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.index_config(), sort_keys=True).encode()).hexdigest()

    def as_dict(self) -> dict:
        return asdict(self)


_ENV_PREFIX = "RAG_"


def _cast(name: str, raw: str, default):
    try:
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
    except ValueError as e:
        raise ConfigError(f"{_ENV_PREFIX}{name.upper()}={raw!r} is not a valid {type(default).__name__}") from e
    return raw


def load_config(env=None, **overrides) -> Config:
    env = os.environ if env is None else env
    values = {}
    for name, default in asdict(Config()).items():
        raw = env.get(_ENV_PREFIX + name.upper())
        if raw is not None and raw != "":
            values[name] = _cast(name, raw, default)
    values.update(overrides)
    cfg = Config(**values)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    positive = ("chunk_tokens", "embed_batch", "rerank_batch", "pdf_deadline_s", "vector_k", "bm25_k", "rrf_k",
                "candidates", "context_k", "num_ctx", "num_predict", "rerank_max_length", "torch_threads")
    for name in positive:
        if getattr(cfg, name) <= 0:
            raise ConfigError(f"{name} must be > 0, got {getattr(cfg, name)}")
    if not 0 <= cfg.chunk_overlap < cfg.chunk_tokens:
        raise ConfigError("chunk_overlap must be >= 0 and smaller than chunk_tokens")
    if cfg.margin < 0 or cfg.num_predict + cfg.margin >= cfg.num_ctx:
        raise ConfigError("num_predict + margin must be smaller than num_ctx")
    if cfg.context_k > cfg.candidates:
        raise ConfigError("context_k cannot exceed candidates")
    if not 0 <= cfg.temperature <= 2:
        raise ConfigError("temperature must be within [0, 2]")
    if cfg.connect_timeout_s <= 0 or cfg.read_timeout_s <= 0:
        raise ConfigError("timeouts must be > 0")
    if not cfg.ollama_url.startswith(("http://", "https://")):
        raise ConfigError("ollama_url must be an http(s) URL")
    if cfg.device not in ("cpu", "cuda") or cfg.rerank_device not in ("cpu", "cuda"):
        raise ConfigError("device and rerank_device must be cpu or cuda")


def prompt_budget(cfg: Config) -> int:
    """Maximum prompt tokens so that the answer still fits in the context window."""
    return cfg.num_ctx - cfg.num_predict - cfg.margin
