"""Query pipeline shared by the CLI, Streamlit app and eval runner: retrieve -> generate -> QueryResult."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass

import requests

from . import generator, retriever
from .config import MODES, Config, load_config


@dataclass
class Resources:
    """Loaded models. Independent of index generation and retrieval mode."""
    embedder: object
    count_tokens: object
    reranker: object = None


def load_resources(cfg: Config, with_reranker: bool = True) -> Resources:
    import torch

    torch.set_num_threads(cfg.torch_threads)
    return Resources(
        embedder=retriever.load_embedder(cfg),
        count_tokens=generator.make_token_counter(generator.load_llm_tokenizer(cfg)),
        reranker=retriever.load_reranker(cfg) if with_reranker else None,
    )


def llm_state(cfg: Config, http=requests) -> dict:
    """What Ollama reports for the configured model: digest, loaded or not, CPU/GPU split."""
    state = {"digest": None, "loaded": None, "gpu_fraction": None}
    try:
        tags = http.get(cfg.ollama_url.rstrip("/") + "/api/tags", timeout=cfg.connect_timeout_s).json()
        for m in tags.get("models", []):
            if m.get("name") == cfg.llm_model:
                state["digest"] = m.get("digest")
        ps = http.get(cfg.ollama_url.rstrip("/") + "/api/ps", timeout=cfg.connect_timeout_s).json()
        state["loaded"] = False
        for m in ps.get("models", []):
            if m.get("name") == cfg.llm_model:
                state["loaded"] = True
                if m.get("size"):
                    state["gpu_fraction"] = round(m.get("size_vram", 0) / m["size"], 3)
    except (requests.RequestException, ValueError):
        pass
    return state


def answer(question: str, mode: str, cfg: Config, res: Resources, gen: retriever.Generation, http=requests) -> dict:
    """Run one query against an already-resolved generation. Never raises for expected failures."""
    t0 = time.perf_counter()
    result = {
        "request_id": uuid.uuid4().hex, "question": question, "mode": mode, "generation_id": gen.gen_id,
        "status": "error", "claims": [], "sources": [], "refusal_reason": None, "error_code": None,
        "retrieved_chunks": [], "context_chunks": [], "candidates": {}, "llm_model": cfg.llm_model,
        "timings_ms": {"wall_end_to_end": None, "embed": None, "retrieval": None, "rerank": None,
                       "generation": None, "ollama": {"total": None, "load": None, "prompt_eval": None, "eval": None}},
    }
    try:
        found = retriever.retrieve(gen, question, mode, cfg, res.embedder, res.reranker)
    except retriever.RetrievalInputError as e:
        result.update(error_code="input_error", error_detail=str(e))
    else:
        result["retrieved_chunks"] = found["ranked"]
        result["candidates"] = {k: [cid for cid, _ in found[k]] for k in ("vector", "bm25", "fused")}
        result["timings_ms"].update(found["timings_ms"])
        t = time.perf_counter()
        out = generator.generate(cfg, question, found["ranked"], res.count_tokens, http)
        result["timings_ms"]["generation"] = (time.perf_counter() - t) * 1000
        result["timings_ms"]["ollama"] = out.pop("ollama_ms")
        result.update(out)
    result["timings_ms"]["wall_end_to_end"] = (time.perf_counter() - t0) * 1000
    return result


def render_text(result: dict) -> str:
    if result["status"] == "error":
        return f"[error:{result['error_code']}]"
    if result["status"] == "refused":
        return f"[refused] {result['refusal_reason']}"
    lines = [f"- {c['text']} [{', '.join(c['source_ids'])}]" for c in result["claims"]]
    lines += [f"  {s['source_id']}: {s['file']} p.{s['page']}" for s in result["sources"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ask one question against the active index.")
    p.add_argument("--question", required=True)
    p.add_argument("--mode", choices=MODES, default="hybrid-rerank")
    p.add_argument("--json", action="store_true", help="print the full QueryResult")
    args = p.parse_args(argv)
    cfg = load_config()
    try:
        gen = retriever.load_generation(cfg.index_root, expected_config_hash=cfg.index_config_hash())
    except retriever.IndexError_ as e:
        print(f"index not ready: {e}", file=sys.stderr)
        return 1
    res = load_resources(cfg, with_reranker=args.mode == "hybrid-rerank")
    result = answer(args.question, args.mode, cfg, res, gen)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        print(render_text(result))
        t = result["timings_ms"]
        print(f"({result['status']}, {t['wall_end_to_end']:.0f} ms, generation {gen.gen_id})")
    return 0 if result["status"] != "error" else 2


if __name__ == "__main__":
    sys.exit(main())
