"""PDF corpus -> cleaned pages -> chunks -> BGE-M3 embeddings -> Chroma + JSONL generation.

Every run builds a complete generation in a temp directory, validates it, then switches the
active pointer atomically. A failed build leaves the previous generation in service.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import retriever
from .chunking import chunk_page, hf_offsets
from .config import Config, load_config

SOURCE_FIELDS = ("doc_id", "file", "url", "lang", "source_version", "downloaded_at", "redistribution")


class IngestError(RuntimeError):
    pass


# ---------------------------------------------------------------- text

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")


def clean_text(text: str) -> str:
    """NFC, merge wrapped lines inside a paragraph, keep paragraph breaks. Never drops digits/units/headers."""
    text = unicodedata.normalize("NFC", text).replace("\r", "")
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        block = _HYPHEN_BREAK.sub(r"\1-\2", block)
        line = _SPACES.sub(" ", " ".join(l.strip() for l in block.split("\n") if l.strip()))
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------- sources / extraction

def read_sources(path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            missing = [k for k in SOURCE_FIELDS if k not in rec]
            if missing:
                raise IngestError(f"{path}:{n} missing fields {missing}")
            records.append(rec)
    return records


def extract_pdf(path, deadline_s: int, runner=subprocess.run) -> dict:
    """Run PyMuPDF in a child process so a hung native parser can be killed. Returns pages or an error reason."""
    cmd = [sys.executable, "-m", "src.pdf_extract", str(path)]
    try:
        proc = runner(cmd, capture_output=True, timeout=deadline_s, cwd=Path(__file__).resolve().parent.parent)
    except subprocess.TimeoutExpired:  # subprocess.run kills and reaps the child before raising
        return {"error": "timeout"}
    if proc.returncode != 0:
        return {"error": "crashed", "detail": f"exit {proc.returncode}"}
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return {"error": "bad_output"}


def collect_pages(cfg: Config, sources: list, raw_dir, extractor=extract_pdf) -> tuple:
    """Return (pages, report). pages: dicts with doc metadata, 1-based PDF page position and cleaned text."""
    pages, report = [], []
    for src in sources:
        path = Path(raw_dir) / src["file"]
        entry = {"file": src["file"], "doc_id": src["doc_id"], "status": "ok", "pages_total": 0, "pages_used": 0}
        report.append(entry)
        if not path.exists():
            entry["status"] = "skipped:missing_file"
            continue
        if "sha256:" + retriever.file_sha256(path) != src["doc_id"]:
            entry["status"] = "skipped:checksum_mismatch"
            continue
        result = extractor(path, cfg.pdf_deadline_s)
        if "error" in result:
            entry["status"] = "skipped:" + result["error"]
            continue
        entry["pages_total"] = len(result["pages"])
        for number, raw in enumerate(result["pages"], start=1):
            text = clean_text(raw)
            if len(text) < cfg.min_page_chars:
                continue
            entry["pages_used"] += 1
            pages.append({"doc_id": src["doc_id"], "file": src["file"], "lang": src["lang"],
                          "source_version": src["source_version"], "page": number, "text": text})
        if entry["pages_used"] == 0:
            entry["status"] = "skipped:no_text_layer"
    return pages, report


def build_chunks(cfg: Config, pages: list, offsets) -> list:
    config_hash = cfg.index_config_hash()
    chunks = []
    for page in pages:
        for c in chunk_page(page["text"], offsets, cfg.chunk_tokens, cfg.chunk_overlap):
            key = f"{page['doc_id']}|{page['page']}|{c.ordinal}|{config_hash}"
            chunks.append({
                "chunk_id": "sha256:" + hashlib.sha256(key.encode()).hexdigest(),
                "doc_id": page["doc_id"], "file": page["file"], "page": page["page"], "lang": page["lang"],
                "source_version": page["source_version"], "chunk_ordinal": c.ordinal,
                "text": c.text, "token_count": c.token_count,
            })
    return chunks


# ---------------------------------------------------------------- generation publish

def _write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def publish_pointer(index_root, gen_id: str) -> None:
    index_root = Path(index_root)
    tmp = index_root / "active.json.tmp"
    tmp.write_text(json.dumps({"generation_id": gen_id}), encoding="utf-8")
    os.replace(tmp, index_root / "active.json")


def run_ingest(cfg: Config, embedder, offsets, extractor=extract_pdf, before_publish=None) -> dict:
    index_root = Path(cfg.index_root)
    (index_root / "generations").mkdir(parents=True, exist_ok=True)
    lock = open(index_root / ".ingest.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise IngestError("another ingest is running (index lock is held)")
    try:
        return _build(cfg, index_root, embedder, offsets, extractor, before_publish)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _build(cfg, index_root, embedder, offsets, extractor, before_publish) -> dict:
    started = time.perf_counter()
    sources = read_sources(cfg.sources_path)
    source_manifest_hash = retriever.file_sha256(cfg.sources_path)
    pages, report = collect_pages(cfg, sources, cfg.raw_dir, extractor)
    chunks = build_chunks(cfg, pages, offsets)
    if not chunks:
        raise IngestError("no usable text in corpus; active index left unchanged")
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise IngestError("duplicate chunk ids")

    body = "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in chunks).encode("utf-8")
    chunks_sha = hashlib.sha256(body).hexdigest()
    gen_id = hashlib.sha256(f"{cfg.index_config_hash()}|{source_manifest_hash}|{chunks_sha}".encode()).hexdigest()[:16]
    final = index_root / "generations" / gen_id
    summary = {"generation_id": gen_id, "chunk_count": len(chunks), "report": report, "reused": False}

    if (final / "manifest.json").exists():
        retriever.load_generation(index_root, gen_id, cfg.index_config_hash())  # still valid?
        summary["reused"] = True
    else:
        tmp = index_root / "generations" / f".tmp-{gen_id}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            (tmp / "chunks.jsonl").write_bytes(body)
            collection = retriever.open_collection(tmp / "chroma", create=True)
            dim = None
            for i in range(0, len(chunks), 256):
                batch = chunks[i:i + 256]
                vectors = embedder.encode([c["text"] for c in batch], batch_size=cfg.embed_batch,
                                          normalize_embeddings=True, show_progress_bar=False)
                dim = len(vectors[0])
                collection.add(ids=[c["chunk_id"] for c in batch], embeddings=[list(map(float, v)) for v in vectors])
            if collection.count() != len(chunks) or set(collection.get(include=[])["ids"]) != {c["chunk_id"] for c in chunks}:
                raise IngestError("Chroma id set differs from chunks.jsonl")
            manifest = {
                "schema_version": 1, "generation_id": gen_id, "ready": True,
                "index_config": cfg.index_config(), "index_config_hash": cfg.index_config_hash(),
                "source_manifest_hash": source_manifest_hash, "chunks_sha256": chunks_sha,
                "embedding_dimension": dim, "distance": "cosine",
                "source_count": sum(r["status"] == "ok" for r in report),
                "page_count": len(pages), "chunk_count": len(chunks), "report": report,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
            del collection
            os.replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        # Probe both branches on the finished generation before it can serve queries.
        gen = retriever.load_generation(index_root, gen_id, cfg.index_config_hash())
        probe = chunks[0]["text"][:200]
        qvec = embedder.encode([probe], normalize_embeddings=True, show_progress_bar=False)[0]
        if not retriever.vector_search(gen, qvec, 1) or not retriever.bm25_search(gen, probe, 1):
            shutil.rmtree(final, ignore_errors=True)
            raise IngestError("query probes failed on new generation")

    if before_publish:
        before_publish(gen_id)
    publish_pointer(index_root, gen_id)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build a new index generation from local PDFs.")
    p.add_argument("--input", help="directory with PDFs (default: config raw_dir)")
    p.add_argument("--sources", help="source manifest JSONL")
    p.add_argument("--index-root")
    args = p.parse_args(argv)
    overrides = {k: v for k, v in (("raw_dir", args.input), ("sources_path", args.sources),
                                   ("index_root", args.index_root)) if v}
    cfg = load_config(**overrides)
    embedder = retriever.load_embedder(cfg)
    try:
        summary = run_ingest(cfg, embedder, hf_offsets(embedder.tokenizer))
    except (IngestError, retriever.IndexError_) as e:
        print(f"ingest failed: {e}", file=sys.stderr)
        return 1
    for r in summary["report"]:
        print(f"  {r['status']:<28} pages {r['pages_used']}/{r['pages_total']}  {r['file']}")
    print(f"generation {summary['generation_id']} chunks={summary['chunk_count']} "
          f"reused={summary['reused']} seconds={summary['seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
