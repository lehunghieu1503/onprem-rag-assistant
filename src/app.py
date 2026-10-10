"""Streamlit chat UI over the shared pipeline. Run from the project root:

    python -m streamlit run src/app.py --server.address=127.0.0.1
"""
from __future__ import annotations

import html
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # streamlit runs this file as a script

import streamlit as st  # noqa: E402

from src import pipeline, retriever  # noqa: E402
from src.config import MODES, ConfigError, load_config  # noqa: E402

MODE_LABELS = {"vector": "Theo nghĩa", "hybrid": "Nghĩa + từ khóa", "hybrid-rerank": "Chính xác nhất"}
MODE_HINTS = {"vector": "Nhanh, hợp câu hỏi diễn đạt tự do", "hybrid": "Bắt được mã hiệu như QCVN, TCVN",
              "hybrid-rerank": "Thêm bước xếp hạng lại · khuyên dùng"}
SUGGESTIONS = [
    ("Số liệu", "Thời hạn kiểm định định kỳ của cần trục tự hành là bao lâu?"),
    ("Mã hiệu", "QCVN 25:2015/BLĐTBXH được ban hành theo Thông tư nào?"),
    ("Tài liệu tiếng Anh", "Theo OSHA, phải lưu giấy phép vào không gian hạn chế đã hủy trong bao lâu?"),
    ("Thiết bị", "Tốc độ di chuyển tối đa của pa lăng điện điều khiển từ sàn là bao nhiêu?"),
]

# Static chrome only. Model and document text is always passed through html.escape before it reaches HTML.
CSS = """
<style>
.stApp { background: #F3F5F8; color: #101B2D; }
.block-container { max-width: 860px; padding-top: 2.5rem; }
[data-testid="stSidebar"] { background: #101B2D; }
[data-testid="stSidebar"] * { color: #E6EBF2; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] .rag-muted { color: #A9B4C4 !important; }
.rag-brand { display: flex; gap: 12px; align-items: center; margin-bottom: 8px; }
.rag-logo { width: 36px; height: 36px; border-radius: 8px; background: #1456B8; display: flex; align-items: center; justify-content: center; font-weight: 600; }
.rag-label { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: #A9B4C4; margin: 18px 0 6px; }
.rag-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.rag-stat { background: #1A2740; border-radius: 8px; padding: 12px; }
.rag-stat b { display: block; font-size: 22px; font-weight: 600; }
.rag-stat span { font-size: 13px; color: #A9B4C4; }
.rag-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
.rag-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.rag-q { margin: 6px 0 14px auto; width: fit-content; max-width: 80%; background: #1456B8; color: #fff; padding: 12px 16px; border-radius: 14px 14px 4px 14px; }
.rag-card { background: #fff; border: 1px solid #DCE2EA; border-radius: 14px; padding: 20px; margin-bottom: 18px; }
.rag-ok { color: #18794E; font-size: 13px; font-weight: 500; margin-bottom: 12px; }
.rag-claim { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; font-size: 16px; margin-bottom: 10px; }
.rag-claim > span:first-child { flex: 1 1 320px; }
.rag-chip { font-family: ui-monospace, Menlo, monospace; font-size: 13px; background: #E8F0FC; color: #1456B8; padding: 3px 10px; border-radius: 6px; text-decoration: none; }
.rag-sub { font-size: 13px; font-weight: 600; color: #46546A; border-top: 1px solid #E6EAF0; padding-top: 14px; margin: 14px 0 10px; }
.rag-src { border: 1px solid #DCE2EA; border-radius: 10px; margin-bottom: 8px; overflow: hidden; }
.rag-src summary { display: flex; gap: 12px; align-items: center; min-height: 48px; padding: 8px 14px; cursor: pointer; list-style: none; font-size: 14px; }
.rag-src summary::-webkit-details-marker { display: none; }
.rag-badge { flex: none; width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-family: ui-monospace, Menlo, monospace; font-size: 13px; background: #E6EAF0; color: #46546A; }
.rag-badge.cited { background: #1456B8; color: #fff; }
.rag-src-title { flex: 1; min-width: 0; overflow-wrap: anywhere; }
.rag-src-title b { font-weight: 500; display: block; }
.rag-src-title span { font-size: 13px; color: #5B6A80; }
.rag-src summary::after { content: "Xem đoạn trích"; font-size: 13px; color: #1456B8; white-space: nowrap; }
.rag-src[open] summary::after { content: "Thu gọn"; }
.rag-excerpt { padding: 12px 14px 14px 54px; background: #F8FAFC; border-top: 1px solid #E6EAF0; font-size: 14px; color: #2B3850; white-space: pre-wrap; }
.rag-meta { font-size: 13px; color: #5B6A80; margin-top: 12px; }
.rag-refused { background: #FFF8EB; border-color: #F0D9A8; color: #6B4000; }
.rag-error { background: #FDF0EE; border-color: #EBC3BC; color: #7A2318; }
.rag-refused b, .rag-error b { display: block; margin-bottom: 4px; }
.rag-hero h1 { font-size: 34px; line-height: 1.2; font-weight: 600; margin: 24px 0 10px; padding: 0; }
.rag-hero p { font-size: 17px; color: #46546A; max-width: 640px; }
.rag-notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; font-size: 14px; color: #46546A; margin-top: 24px; }
.stButton > button { min-height: 72px; width: 100%; text-align: left; justify-content: flex-start; background: #fff; border: 1px solid #DCE2EA; border-radius: 12px; color: #101B2D; }
</style>
"""


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


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def result_html(result: dict) -> str:
    """One card per result. Errors never show model text; refusals and answers are visually distinct."""
    if result["status"] == "error":
        return ('<div class="rag-card rag-error"><b>Chưa trả lời được</b>Hệ thống gặp lỗi '
                f'<span class="rag-mono">{esc(result["error_code"])}</span> nên không hiển thị câu trả lời. Hãy thử lại.</div>')
    if result["status"] == "refused":
        return ('<div class="rag-card rag-refused"><b>Không tìm thấy trong tài liệu</b>'
                f'{esc(result["refusal_reason"])}<br>Trợ lý chỉ trả lời từ các tài liệu đã nạp, không dùng kiến thức bên ngoài.</div>')
    rid = esc(result["request_id"][:8])
    cited = {s for c in result["claims"] for s in c["source_ids"]}
    parts = ['<div class="rag-card"><div class="rag-ok">✓ Trả lời từ tài liệu, có nguồn kèm theo</div>']
    for claim in result["claims"]:
        chips = "".join(f'<a class="rag-chip" href="#{rid}-{esc(s)}">Nguồn {esc(s[1:])}</a>' for s in claim["source_ids"])
        parts.append(f'<div class="rag-claim"><span>{esc(claim["text"])}</span>{chips}</div>')
    parts.append('<div class="rag-sub">Nguồn đã dùng</div>')
    for src in result["sources"]:
        sid = src["source_id"]
        badge = "rag-badge cited" if sid in cited else "rag-badge"
        opened = " open" if sid in cited and sid == min(cited) else ""
        parts.append(
            f'<details class="rag-src" id="{rid}-{esc(sid)}"{opened}><summary><span class="{badge}">{esc(sid[1:])}</span>'
            f'<span class="rag-src-title"><b>{esc(src["source_version"])}</b><span>{esc(src["file"])} · trang {esc(src["page"])}</span></span>'
            f'</summary><div class="rag-excerpt">{esc(src["text"])}</div></details>')
    seconds = result["timings_ms"]["wall_end_to_end"]
    took = "" if seconds is None else f"Trả lời trong {seconds / 1000:.1f} giây · ".replace(".", ",")
    parts.append(f'<div class="rag-meta">{took}Cách tìm: {esc(MODE_LABELS[result["mode"]])}</div></div>')
    return "".join(parts)


def show_turn(question: str, result: dict | None) -> None:
    st.markdown(f'<div class="rag-q">{esc(question)}</div>', unsafe_allow_html=True)
    if result is not None:
        st.markdown(result_html(result), unsafe_allow_html=True)


def sidebar(cfg, gen=None) -> str:
    with st.sidebar:
        st.markdown('<div class="rag-brand"><div class="rag-logo">TL</div><div><b>Trợ lý tài liệu</b><br>'
                    '<span class="rag-muted" style="font-size:13px">Kỹ thuật &amp; HSE · chạy nội bộ</span></div></div>'
                    '<div class="rag-label">Cách tìm kiếm</div>', unsafe_allow_html=True)
        mode = st.radio("Cách tìm kiếm", MODES, index=2, format_func=MODE_LABELS.get,
                        captions=[MODE_HINTS[m] for m in MODES], label_visibility="collapsed")
        if gen is not None:
            langs = [c["lang"] for c in {c["doc_id"]: c for c in gen.chunks}.values()]
            st.markdown(
                '<div class="rag-label">Kho tài liệu</div><div class="rag-stats">'
                f'<div class="rag-stat"><b>{gen.manifest["source_count"]}</b><span>tài liệu PDF</span></div>'
                f'<div class="rag-stat"><b>{gen.manifest["page_count"]}</b><span>trang đã đọc</span></div></div>'
                f'<div class="rag-muted" style="font-size:13px;margin-top:8px">{langs.count("vi")} văn bản tiếng Việt · '
                f'{langs.count("en")} tài liệu tiếng Anh</div>', unsafe_allow_html=True)
        state = pipeline.llm_state(cfg)
        ready = state["digest"] is not None and gen is not None
        placement = "chưa nạp" if not state["loaded"] else ("GPU" if state["gpu_fraction"] == 1 else "CPU/GPU")
        st.markdown(
            f'<div style="margin-top:28px;font-size:13px"><span class="rag-dot" style="background:{"#4ADE80" if ready else "#F59E0B"}"></span>'
            f'{"Sẵn sàng · không cần Internet" if ready else "Chưa sẵn sàng"}</div>'
            f'<div class="rag-mono rag-muted" style="margin-top:6px">{esc(cfg.llm_model)} · {placement}<br>'
            f'reranker {esc(cfg.rerank_device)}{" · index " + esc(gen.gen_id[:8]) if gen is not None else ""}</div>',
            unsafe_allow_html=True)
    return mode


def empty_state() -> str | None:
    st.markdown('<div class="rag-hero"><h1>Hỏi về quy chuẩn, quy trình và hướng dẫn an toàn</h1>'
                '<p>Trợ lý tìm trong kho tài liệu nội bộ và trả lời kèm trang nguồn để bạn đối chiếu. '
                'Nếu tài liệu không có, trợ lý sẽ nói rõ là không tìm thấy.</p></div>', unsafe_allow_html=True)
    st.caption("Thử một câu hỏi")
    picked = None
    cols = st.columns(2)
    for i, (tag, text) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(f"**{tag}** — {text}", key=f"suggest-{i}"):
            picked = text
    st.markdown('<div class="rag-notes"><div>Mỗi ý trong câu trả lời đều dẫn tới trang tài liệu cụ thể.</div>'
                '<div>Dữ liệu và mô hình chạy trên máy này, không gửi ra ngoài.</div>'
                '<div>Văn bản trong kho có thể đã được sửa đổi; hãy kiểm tra hiệu lực.</div></div>', unsafe_allow_html=True)
    return picked


def main() -> None:
    st.set_page_config(page_title="Trợ lý tài liệu Kỹ thuật & HSE", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        cfg = load_config()
    except ConfigError as e:
        st.error(f"Invalid configuration: {e}")
        return

    problems = readiness(cfg)
    if problems:
        sidebar(cfg)
        for p in problems:
            st.error(p)
        return
    try:
        gen_id = retriever.active_generation_id(cfg.index_root)  # read once per rerun; a request keeps its generation
        gen = get_generation(cfg.index_root, gen_id, cfg.index_config_hash())
        res = get_models(cfg.embed_revision, cfg.rerank_revision, cfg.llm_tokenizer_revision, cfg.device, cfg.rerank_device)
    except Exception as e:  # missing snapshot, stale index, ... -> actionable message, nothing cached as ready
        sidebar(cfg)
        st.error(f"Not ready: {e}")
        return
    mode = sidebar(cfg, gen)

    history = st.session_state.setdefault("history", [])  # display only; every query is independent
    for item in history:
        show_turn(item["question"], item["result"])
    suggested = empty_state() if not history else None

    question = st.chat_input("Đặt câu hỏi bằng tiếng Việt hoặc tiếng Anh") or suggested
    if question:
        show_turn(question, None)
        lock = query_lock()
        if not lock.acquire(blocking=False):
            st.info("Đang trả lời một câu hỏi khác. Vui lòng thử lại sau giây lát.")
            return
        try:
            with st.spinner("Đang tìm trong tài liệu và soạn câu trả lời…"):
                result = pipeline.answer(question, mode, cfg, res, gen)
        finally:
            lock.release()
        history.append({"question": question, "result": result})
        st.rerun()
    st.caption("Bản demo nội bộ. Không dùng câu trả lời để ra quyết định an toàn khi chưa đối chiếu tài liệu gốc.")


main()
