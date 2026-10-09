"""Streamlit chat UI over the shared pipeline. Run from the project root:

    python -m streamlit run src/app.py --server.address=127.0.0.1
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # streamlit runs this file as a script

import streamlit as st  # noqa: E402

from src import pipeline, retriever  # noqa: E402
from src.config import MODES, ConfigError, load_config  # noqa: E402

MODE_LABELS = {"vector": "Vector only", "hybrid": "Hybrid (vector + BM25, RRF)", "hybrid-rerank": "Hybrid + reranker"}


@st.cache_resource(show_spinner="Loading embedding, reranker and tokenizer…")
def get_models(embed_rev: str, rerank_rev: str, tokenizer_rev: str, device: str, rerank_device: str):
    """Keyed by immutable revisions/device only, so a new index generation never reloads models."""
    return pipeline.load_resources(load_config())


@st.cache_resource(max_entries=2, show_spinner="Loading index…")
def get_generation(index_root: str, gen_id: str, config_hash: str):
    return retriever.load_generation(index_root, gen_id, config_hash)


@st.cache_resource
def query_lock() -> threading.Lock:
    """One lock for the whole process: one query at a time across tabs and generations."""
    return threading.Lock()


def readiness(cfg) -> list:
    """Human-readable problems that block answering; empty when ready."""
    problems = []
    try:
        retriever.active_generation_id(cfg.index_root)
    except retriever.IndexError_:
        problems.append("No index found. Run `python -m src.ingest` (see docs/setup.md).")
    state = pipeline.llm_state(cfg)
    if state["loaded"] is None:
        problems.append(f"Ollama is not reachable at {cfg.ollama_url}. Start it and reload this page.")
    elif state["digest"] is None:
        problems.append(f"Model `{cfg.llm_model}` is not installed. Run `ollama pull {cfg.llm_model}`.")
    return problems


def show_result(result: dict) -> None:
    if result["status"] == "error":
        st.error(f"Could not produce a verified answer (error: {result['error_code']}). Nothing is shown as an answer.")
    elif result["status"] == "refused":
        st.warning(result["refusal_reason"])
    else:
        for claim in result["claims"]:
            st.markdown("- " + st_escape(claim["text"]) + "  " + " ".join(f"`{s}`" for s in claim["source_ids"]))
    cited = {s for c in result["claims"] for s in c["source_ids"]}
    for src in result["sources"]:
        mark = "● " if src["source_id"] in cited else ""
        with st.expander(f"{mark}{src['source_id']} · {src['file']} · page {src['page']} · {src['source_version']}"):
            st.text(src["text"])
    t = result["timings_ms"]
    part = lambda v: "–" if v is None else f"{v / 1000:.1f}s"
    st.caption(f"{MODE_LABELS[result['mode']]} · total {part(t['wall_end_to_end'])} · retrieval "
               f"{part((t['embed'] or 0) + (t['retrieval'] or 0))} · rerank {part(t['rerank'])} · "
               f"LLM {part(t['generation'])} · index {result['generation_id']}")


def st_escape(text: str) -> str:
    """Model text is rendered as plain markdown text: neutralise markdown/HTML control characters."""
    for ch in "\\`*_{}[]()#+!|<>$~":
        text = text.replace(ch, "\\" + ch)
    return text


def main() -> None:
    st.set_page_config(page_title="On-prem RAG · Technical & HSE", layout="centered")
    st.title("On-prem RAG assistant")
    st.caption("Answers come only from the local PDF corpus, with page-level sources. Local demo, not HSE advice.")
    try:
        cfg = load_config()
    except ConfigError as e:
        st.error(f"Invalid configuration: {e}")
        return

    with st.sidebar:
        mode = st.radio("Retrieval mode", MODES, index=2, format_func=MODE_LABELS.get)
        state = pipeline.llm_state(cfg)
        placement = "not loaded" if not state["loaded"] else (
            "GPU" if state["gpu_fraction"] == 1 else "CPU" if not state["gpu_fraction"] else f"{state['gpu_fraction']:.0%} GPU")
        st.markdown(f"**LLM** `{cfg.llm_model}` ({placement})  \n**Embedding** `{cfg.device}` · **Reranker** `{cfg.rerank_device}`")

    problems = readiness(cfg)
    if problems:
        for p in problems:
            st.error(p)
        return
    try:
        gen_id = retriever.active_generation_id(cfg.index_root)  # read once per rerun; a request keeps its generation
        gen = get_generation(cfg.index_root, gen_id, cfg.index_config_hash())
        res = get_models(cfg.embed_revision, cfg.rerank_revision, cfg.llm_tokenizer_revision, cfg.device, cfg.rerank_device)
    except Exception as e:  # missing snapshot, stale index, ... -> actionable message, nothing cached as ready
        st.error(f"Not ready: {e}")
        return
    st.sidebar.markdown(f"**Index** `{gen.gen_id}` · {gen.manifest['source_count']} PDFs · {gen.manifest['chunk_count']} chunks")

    history = st.session_state.setdefault("history", [])  # display only; every query is independent
    for item in history:
        st.chat_message("user").write(item["question"])
        with st.chat_message("assistant"):
            show_result(item["result"])

    question = st.chat_input("Ask about the documents (Vietnamese or English)")
    if question:
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            lock = query_lock()
            if not lock.acquire(blocking=False):
                st.info("Another question is being answered. Try again in a moment.")
                return
            try:
                with st.spinner("Searching and answering…"):
                    result = pipeline.answer(question, mode, cfg, res, gen)
            finally:
                lock.release()
            show_result(result)
        history.append({"question": question, "result": result})


main()
