"""Index generation loading and the three retrieval modes: vector, hybrid (RRF), hybrid-rerank."""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

LEXICAL_VERSION = 1
_TOKEN = re.compile(r"\w+(?:[/\-.:]\w+)*")
_SPLIT = re.compile(r"[/\-.:]")


class IndexError_(RuntimeError):
    """Missing, corrupt or incompatible index generation."""


class RetrievalInputError(ValueError):
    pass


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def tokenize_lexical(text: str) -> list:
    """NFC + lowercase; codes such as 'p-101a/b' are kept whole and also split into their parts."""
    tokens = []
    for tok in _TOKEN.findall(normalize(text).lower()):
        tokens.append(tok)
        parts = _SPLIT.split(tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return tokens


# ---------------------------------------------------------------- generation

@dataclass
class Generation:
    gen_id: str
    path: Path
    manifest: dict
    chunks: list
    by_id: dict
    bm25: object
    collection: object
    token_sets: list

    @property
    def vocab(self) -> set:
        return set().union(*self.token_sets)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_chunks(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_bm25(chunks: list):
    from rank_bm25 import BM25Okapi

    if not chunks:
        raise IndexError_("cannot build BM25 over an empty corpus")
    tokenized = [tokenize_lexical(c["text"]) for c in chunks]
    return BM25Okapi(tokenized), [set(t) for t in tokenized]


def open_collection(chroma_dir, create: bool = False):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    if create:
        return client.create_collection("chunks", embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
    return client.get_collection("chunks", embedding_function=None)


def active_generation_id(index_root) -> str:
    pointer = Path(index_root) / "active.json"
    if not pointer.exists():
        raise IndexError_(f"no active index at {pointer}; run: python -m src.ingest")
    return json.loads(pointer.read_text(encoding="utf-8"))["generation_id"]


def load_generation(index_root, gen_id: str | None = None, expected_config_hash: str | None = None) -> Generation:
    gen_id = gen_id or active_generation_id(index_root)
    path = Path(index_root) / "generations" / gen_id
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise IndexError_(f"generation {gen_id} has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("ready"):
        raise IndexError_(f"generation {gen_id} is not marked ready")
    if expected_config_hash and manifest["index_config_hash"] != expected_config_hash:
        raise IndexError_("index was built with a different chunking/embedding config; re-run ingest")
    if file_sha256(path / "chunks.jsonl") != manifest["chunks_sha256"]:
        raise IndexError_(f"generation {gen_id}: chunks.jsonl checksum mismatch")
    chunks = read_chunks(path / "chunks.jsonl")
    collection = open_collection(path / "chroma")
    if collection.count() != len(chunks) or len(chunks) != manifest["chunk_count"]:
        raise IndexError_(f"generation {gen_id}: Chroma/JSONL/manifest counts differ")
    bm25, token_sets = build_bm25(chunks)
    return Generation(gen_id, path, manifest, chunks, {c["chunk_id"]: c for c in chunks}, bm25, collection, token_sets)


# ---------------------------------------------------------------- search

def vector_search(gen: Generation, query_embedding, k: int) -> list:
    n = min(k, len(gen.chunks))
    res = gen.collection.query(query_embeddings=[list(map(float, query_embedding))], n_results=n, include=["distances"])
    pairs = sorted(zip(res["distances"][0], res["ids"][0]))
    return [(cid, 1.0 - dist) for dist, cid in pairs]


def bm25_search(gen: Generation, question: str, k: int) -> list:
    q_tokens = tokenize_lexical(question)
    wanted = set(q_tokens)
    hits = [i for i, toks in enumerate(gen.token_sets) if wanted & toks]
    if not hits:
        return []  # no arbitrary keyword candidates for unseen/punctuation-only queries
    scores = gen.bm25.get_scores(q_tokens)
    matched = [(float(scores[i]), gen.chunks[i]["chunk_id"]) for i in hits]
    matched.sort(key=lambda x: (-x[0], x[1]))  # scores may be <= 0 in tiny corpora; keep matched docs anyway
    return [(cid, score) for score, cid in matched[:k]]


def rrf(ranked_lists: list, k: int = 60, top: int | None = None) -> list:
    """Reciprocal rank fusion over lists of chunk ids; rank starts at 1, ties broken by id."""
    scores: dict = {}
    for ids in ranked_lists:
        seen = set()
        for rank, cid in enumerate(ids, start=1):
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused[:top] if top is not None else fused


def pair_token_count(tokenizer, question: str, passage: str) -> int:
    return len(tokenizer(question, passage, add_special_tokens=True, truncation=False)["input_ids"])


def rerank(reranker, question: str, chunks: list, top: int, batch_size: int, max_length: int) -> list:
    if not chunks:
        return []
    tokenizer = reranker.tokenizer
    q_len = len(tokenizer(question, add_special_tokens=False, truncation=False)["input_ids"])
    if q_len > max_length // 2:
        raise RetrievalInputError(f"question is too long for the reranker ({q_len} tokens)")
    for c in chunks:
        n = pair_token_count(tokenizer, question, c["text"])
        if n > max_length:
            raise RetrievalInputError(
                f"reranker pair of {n} tokens exceeds max_length={max_length} for chunk {c['chunk_id'][:12]}")
    scores = reranker.predict([(question, c["text"]) for c in chunks], batch_size=batch_size, show_progress_bar=False)
    order = sorted(zip((float(s) for s in scores), (c["chunk_id"] for c in chunks)), key=lambda x: (-x[0], x[1]))
    return [(cid, score) for score, cid in order[:top]]


def retrieve(gen: Generation, question: str, mode: str, cfg, embedder, reranker=None) -> dict:
    """Return ranked top chunks plus per-branch candidates and stage timings (ms)."""
    question = normalize(question).strip()
    if not question:
        raise RetrievalInputError("empty question")
    timings, out = {}, {"mode": mode, "vector": [], "bm25": [], "fused": []}

    t = time.perf_counter()
    qvec = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    timings["embed"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    k_dense = cfg.context_k if mode == "vector" else cfg.vector_k
    out["vector"] = vector_search(gen, qvec, k_dense)
    if mode == "vector":
        ranked = out["vector"][: cfg.context_k]
    else:
        out["bm25"] = bm25_search(gen, question, cfg.bm25_k)
        out["fused"] = rrf([[c for c, _ in out["vector"]], [c for c, _ in out["bm25"]]], cfg.rrf_k, cfg.candidates)
        ranked = out["fused"][: cfg.context_k]
    timings["retrieval"] = (time.perf_counter() - t) * 1000

    timings["rerank"] = None
    if mode == "hybrid-rerank":
        if reranker is None:
            raise RetrievalInputError("hybrid-rerank mode needs a reranker")
        t = time.perf_counter()
        ranked = rerank(reranker, question, [gen.by_id[c] for c, _ in out["fused"]], cfg.context_k,
                        cfg.rerank_batch, cfg.rerank_max_length)
        timings["rerank"] = (time.perf_counter() - t) * 1000
    elif mode not in ("vector", "hybrid"):
        raise RetrievalInputError(f"unknown mode {mode!r}")

    out["ranked"] = [dict(gen.by_id[cid], rank=i, score=score) for i, (cid, score) in enumerate(ranked, start=1)]
    out["timings_ms"] = timings
    return out


def load_embedder(cfg):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(cfg.embed_model, revision=cfg.embed_revision, device=cfg.device, local_files_only=True)


def load_reranker(cfg):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(cfg.rerank_model, revision=cfg.rerank_revision, device=cfg.device, local_files_only=True,
                        max_length=cfg.rerank_max_length)
