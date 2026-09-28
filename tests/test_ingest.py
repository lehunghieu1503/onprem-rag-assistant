import fcntl
import json
import subprocess
from pathlib import Path

import pytest

from src import ingest, retriever
from src.config import load_config
from tests.conftest import FakeEmbedder, make_pdf, write_sources, ws_offsets


def test_ingest_builds_consistent_generation(built):
    cfg, gen, summary = built
    assert summary["chunk_count"] == len(gen.chunks) == gen.collection.count() > 0
    assert {c["page"] for c in gen.chunks if c["file"] == "crane.pdf"} == {1, 3}  # blank page 2 skipped, pages 1-based
    assert all(c["chunk_id"].startswith("sha256:") and c["doc_id"].startswith("sha256:") for c in gen.chunks)
    assert gen.manifest["distance"] == "cosine" and gen.manifest["embedding_dimension"] == 64
    crane = next(r for r in gen.manifest["report"] if r["file"] == "crane.pdf")
    assert (crane["pages_total"], crane["pages_used"]) == (3, 2)


def test_reingest_same_corpus_is_idempotent(built):
    cfg, gen, first = built
    second = ingest.run_ingest(cfg, FakeEmbedder(), ws_offsets)
    assert second["generation_id"] == first["generation_id"] and second["reused"]
    assert second["chunk_count"] == first["chunk_count"]
    assert len(list((Path(cfg.index_root) / "generations").iterdir())) == 1


def test_removed_source_disappears_from_new_generation(built):
    cfg, gen, first = built
    write_sources(cfg.sources_path, Path(cfg.raw_dir), [("vessel.pdf", "en")])
    second = ingest.run_ingest(cfg, FakeEmbedder(), ws_offsets)
    new = retriever.load_generation(cfg.index_root)
    assert second["generation_id"] != first["generation_id"]
    assert {c["file"] for c in new.chunks} == {"vessel.pdf"}
    retriever.load_generation(cfg.index_root, first["generation_id"])  # old generation still loadable for rollback


def test_config_change_creates_new_generation_and_old_loader_refuses(built):
    cfg, gen, first = built
    other = load_config(env={}, **{**cfg.as_dict(), "chunk_tokens": 15, "chunk_overlap": 3})
    second = ingest.run_ingest(other, FakeEmbedder(), ws_offsets)
    assert second["generation_id"] != first["generation_id"]
    with pytest.raises(retriever.IndexError_):
        retriever.load_generation(cfg.index_root, expected_config_hash=cfg.index_config_hash())


def test_failure_before_publish_keeps_old_pointer(built):
    cfg, gen, first = built
    make_pdf(Path(cfg.raw_dir) / "new.pdf", ["A brand new document about lockout tagout procedures and isolation."])
    write_sources(cfg.sources_path, Path(cfg.raw_dir), [("crane.pdf", "en"), ("vessel.pdf", "en"), ("new.pdf", "en")])

    def boom(_gen_id):
        raise RuntimeError("injected failure before pointer swap")

    with pytest.raises(RuntimeError):
        ingest.run_ingest(cfg, FakeEmbedder(), ws_offsets, before_publish=boom)
    assert retriever.active_generation_id(cfg.index_root) == first["generation_id"]
    ingest.run_ingest(cfg, FakeEmbedder(), ws_offsets)  # lock was released; next run works
    assert retriever.active_generation_id(cfg.index_root) != first["generation_id"]


def test_embedding_failure_leaves_no_partial_generation(corpus):
    class Broken:
        def encode(self, *a, **k):
            raise RuntimeError("model exploded")

    with pytest.raises(RuntimeError):
        ingest.run_ingest(corpus, Broken(), ws_offsets)
    assert not (Path(corpus.index_root) / "active.json").exists()
    assert list((Path(corpus.index_root) / "generations").iterdir()) == []


def test_second_writer_is_rejected(corpus):
    Path(corpus.index_root).mkdir(parents=True)
    with open(Path(corpus.index_root) / ".ingest.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(ingest.IngestError, match="another ingest"):
            ingest.run_ingest(corpus, FakeEmbedder(), ws_offsets)


def test_empty_corpus_does_not_publish(built):
    cfg, gen, first = built
    with pytest.raises(ingest.IngestError, match="no usable text"):
        ingest.run_ingest(cfg, FakeEmbedder(), ws_offsets, extractor=lambda path, deadline: {"error": "timeout"})
    assert retriever.active_generation_id(cfg.index_root) == first["generation_id"]


def test_hung_or_crashed_parser_becomes_typed_skip(corpus):
    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    assert ingest.extract_pdf("x.pdf", 1, runner=hang) == {"error": "timeout"}
    crashed = lambda cmd, **kw: subprocess.CompletedProcess(cmd, -11, b"", b"")
    assert ingest.extract_pdf("x.pdf", 1, runner=crashed)["error"] == "crashed"
    garbage = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"not json", b"")
    assert ingest.extract_pdf("x.pdf", 1, runner=garbage) == {"error": "bad_output"}

    def one_bad(path, deadline):
        return {"error": "timeout"} if path.name == "crane.pdf" else ingest.extract_pdf(path, deadline)

    summary = ingest.run_ingest(corpus, FakeEmbedder(), ws_offsets, extractor=one_bad)
    statuses = {r["file"]: r["status"] for r in summary["report"]}
    assert statuses == {"crane.pdf": "skipped:timeout", "vessel.pdf": "ok"}


def test_real_child_process_extracts_and_reports_bad_files(corpus, tmp_path):
    pages = ingest.extract_pdf(Path(corpus.raw_dir) / "vessel.pdf", 30)["pages"]
    assert "V-101A/B" in pages[0]
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    assert ingest.extract_pdf(bad, 30)["error"] == "open_failed"


def test_checksum_mismatch_and_scan_are_skipped_with_reason(corpus):
    raw = Path(corpus.raw_dir)
    make_pdf(raw / "scan.pdf", ["", ""])
    write_sources(corpus.sources_path, raw, [("crane.pdf", "en"), ("vessel.pdf", "en"), ("scan.pdf", "vi")])
    make_pdf(raw / "vessel.pdf", ["Tampered content that no longer matches the recorded checksum at all."])
    summary = ingest.run_ingest(corpus, FakeEmbedder(), ws_offsets)
    statuses = {r["file"]: r["status"] for r in summary["report"]}
    assert statuses == {"crane.pdf": "ok", "vessel.pdf": "skipped:checksum_mismatch", "scan.pdf": "skipped:no_text_layer"}


def test_corrupt_manifest_or_chunks_detected(built):
    cfg, gen, _ = built
    with open(gen.path / "chunks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": "sha256:x", "text": "sneaky"}) + "\n")
    with pytest.raises(retriever.IndexError_, match="checksum"):
        retriever.load_generation(cfg.index_root)


def test_missing_index_has_actionable_error(tmp_path):
    with pytest.raises(retriever.IndexError_, match="src.ingest"):
        retriever.load_generation(tmp_path)
