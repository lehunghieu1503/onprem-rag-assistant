"""Shared test doubles: a whitespace tokenizer and a deterministic bag-of-words embedder (no model downloads)."""
import hashlib
import json
import math
import re

import pytest

from src.config import load_config

DIM = 64


def ws_offsets(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, **_):
        out = []
        for text in texts:
            v = [0.0] * DIM
            for word in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def make_pdf(path, pages):
    import pymupdf as fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)
    doc.close()


def write_sources(path, raw_dir, files):
    from src.retriever import file_sha256

    with open(path, "w", encoding="utf-8") as f:
        for name, lang in files:
            f.write(json.dumps({"doc_id": "sha256:" + file_sha256(raw_dir / name), "file": name, "url": "local",
                                "lang": lang, "source_version": "v1", "downloaded_at": "2026-10-09",
                                "redistribution": "test fixture"}) + "\n")


@pytest.fixture
def corpus(tmp_path):
    """Two small PDFs + source manifest + config pointing at tmp dirs."""
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "crane.pdf", [
        "Crane inspection. Each crane KC-4561A shall be inspected every 12 months by a competent person.",
        "",
        "Wire ropes shall be removed from service when six broken wires are found in one rope lay.",
    ])
    make_pdf(raw / "vessel.pdf", [
        "Pressure vessel V-101A/B hydrostatic test pressure is 1.5 times the design pressure for 5 minutes.",
    ])
    sources = tmp_path / "sources.jsonl"
    write_sources(sources, raw, [("crane.pdf", "en"), ("vessel.pdf", "en")])
    cfg = load_config(env={}, raw_dir=str(raw), sources_path=str(sources), index_root=str(tmp_path / "index"),
                      chunk_tokens=40, chunk_overlap=5, min_page_chars=10)
    return cfg


@pytest.fixture
def built(corpus):
    from src import ingest, retriever

    summary = ingest.run_ingest(corpus, FakeEmbedder(), ws_offsets)
    return corpus, retriever.load_generation(corpus.index_root), summary
