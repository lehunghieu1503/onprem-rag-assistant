import json
from types import SimpleNamespace

import pytest

from eval import run_eval as E

DOC = "sha256:doc1"


def chunk(rank, text, page=3, doc=DOC):
    return {"chunk_id": f"c{rank}", "doc_id": doc, "file": "a.pdf", "page": page, "chunk_ordinal": rank, "rank": rank, "text": text}


def question(qid, qtype="fact", evidence="test pressure is 1.5 times", split="test", text=None):
    answerable = qtype != "absent"
    return {"id": qid, "split": split, "question": text or f"Question {qid}?", "lang": "en", "type": qtype, "tags": [],
            "answerable": answerable,
            "expected_sources": [{"doc_id": DOC, "file": "a.pdf", "page": 3, "evidence": evidence}] if answerable else [],
            "expected_answer": "1.5 times" if answerable else None, "required_facts": ["1.5 times"] if answerable else []}


def output(qid, status, retrieved=(), context=None, claims=(), wall=100.0, error_code=None, load=5.0, gen="G1"):
    r = {"question_id": qid, "generation_id": gen, "status": status, "error_code": error_code,
         "claims": list(claims), "refusal_reason": None, "raw_output": "{}",
         "retrieved_chunks": list(retrieved), "context_chunks": list(retrieved if context is None else context),
         "llm_loaded_before": True,
         "timings_ms": {"wall_end_to_end": wall, "ollama": {"total": 1.0, "load": load, "prompt_eval": 1.0, "eval": 1.0}}}
    r["output_hash"] = E.output_hash(r)
    return r


def write_run(tmp_path, questions, outputs, name="run1", gen="G1"):
    run = tmp_path / name
    run.mkdir()
    snap = "".join(json.dumps(q, sort_keys=True) + "\n" for q in questions).encode()
    (run / "questions.snapshot.jsonl").write_bytes(snap)
    (run / "sources.snapshot.jsonl").write_bytes(b"{}\n")
    manifest = {"run_id": name, "split": "test", "mode": "vector", "generation_id": gen, "generation_manifest_sha256": "m",
                "questions_snapshot_sha256": E.sha256_bytes(snap), "sources_snapshot_sha256": E.sha256_bytes(b"{}\n"),
                "llm_digest": "d", "config": {"llm_model": "qwen3:4b-q4_K_M", **{k: 1 for k in E.BASE_KEYS if k != "llm_model"}}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    with open(run / "outputs.jsonl", "w") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")
    return run, manifest


GOOD = chunk(1, "The hydrostatic test pressure is 1.5 times the design pressure.")
CLAIM = {"text": "1.5 times", "source_ids": ["S1"]}


def label(manifest, out, correct, supported="full", pair=True):
    return {"run_id": manifest["run_id"], "question_id": out["question_id"], "output_hash": out["output_hash"],
            "answer_correct": correct,
            "claims": [{"supported": supported, "pairs": {s: pair for s in c["source_ids"]}} for c in out["claims"]]}


def test_percentiles_and_ratio_formatting():
    assert E.percentile([100, 200, 300], 0.5) == 200
    assert E.percentile([300, 100, 200], 0.95) == pytest.approx(290)
    assert E.percentile([42], 0.95) == 42 and E.percentile([], 0.5) is None
    assert E.fmt(E.ratio(0, 0)) == "N/A" and E.fmt(E.ratio(1, 4)) == "1/4 (25.0%)"


def test_evidence_must_match_span_doc_and_page():
    src = [{"doc_id": DOC, "file": "a.pdf", "page": 3, "evidence": "Test  pressure is 1.5 TIMES"}]
    assert E.evidence_hit(src, [chunk(2, "other"), GOOD | {"rank": 4}]) == 4
    assert E.evidence_hit(src, [chunk(1, "same page but nothing relevant")]) is None
    assert E.evidence_hit(src, [GOOD | {"page": 9}]) is None  # right text, wrong page
    assert E.evidence_hit(src, [GOOD | {"doc_id": "sha256:other-version"}]) is None


def test_evidence_split_across_two_chunks_of_same_page_counts():
    src = [{"doc_id": DOC, "file": "a.pdf", "page": 3, "evidence": "test pressure is 1.5 times the design pressure"}]
    a, b = chunk(2, "The hydrostatic test pressure is"), chunk(5, "1.5 times the design pressure for 5 minutes.")
    assert E.evidence_hit(src, [a, b]) == 5
    assert E.evidence_hit(src, [a]) is None
    assert E.evidence_hit(src, [a, b | {"page": 4}]) is None


def test_metrics_hand_computed(tmp_path):
    qs = [question("q1"), question("q2"), question("q3"), question("q4"), question("a1", "absent"), question("a2", "absent")]
    outs = [
        output("q1", "answered", [GOOD], claims=[CLAIM]),                                   # correct
        output("q2", "answered", [chunk(1, "x"), GOOD | {"rank": 2}], claims=[CLAIM, CLAIM]),  # wrong answer
        output("q3", "refused", [GOOD]),                                                   # false refusal
        output("q4", "error", [], error_code="timeout", wall=180000.0),                    # error counts as wrong
        output("a1", "refused", []),                                                       # correct refusal
        output("a2", "error", [], error_code="invalid_model_output"),                      # error is not a refusal
    ]
    run, manifest = write_run(tmp_path, qs, outs)
    _, questions, outputs = E.load_run(run)
    labels = {"q1": label(manifest, outs[0], True), "q2": label(manifest, outs[1], False, "partial", False)}
    m = E.compute_metrics(manifest, questions, outputs, labels)
    assert (m["answer_accuracy"]["num"], m["answer_accuracy"]["den"]) == (1, 4)
    assert (m["hit_at"]["1"]["num"], m["hit_at"]["5"]["num"], m["hit_at"]["5"]["den"]) == (2, 3, 4)
    assert m["mrr_at_5"] == pytest.approx((1 + 0.5 + 1 + 0) / 4)
    assert (m["correct_refusal"]["num"], m["correct_refusal"]["den"]) == (1, 2)
    assert (m["false_refusal"]["num"], m["false_refusal"]["den"]) == (1, 4)
    assert (m["error_rate"]["num"], m["error_rate"]["den"]) == (2, 6)
    assert (m["citation_precision"]["num"], m["citation_precision"]["den"]) == (1, 3)
    assert (m["citation_coverage"]["num"], m["citation_coverage"]["den"]) == (1, 3)
    assert m["latency_ms"]["all_attempts"]["n"] == 6 and m["latency_ms"]["completed_all"]["n"] == 4
    assert m["accuracy_by_type"]["absent"]["num"] == 1
    assert "1/4 (25.0%)" in E.render_report(m, manifest)


def test_retrieved_hit_but_context_miss_is_scored_from_run_alone(tmp_path):
    out = output("q1", "answered", [chunk(1, "x"), GOOD | {"rank": 5}], context=[chunk(1, "x")], claims=[CLAIM])
    run, manifest = write_run(tmp_path, [question("q1")], [out])
    _, qs, outs = E.load_run(run)
    m = E.compute_metrics(manifest, qs, outs, {"q1": label(manifest, out, False)})
    assert m["hit_at"]["5"]["num"] == 1 and m["context_hit"]["num"] == 0


def test_unlabelled_run_reports_na_not_100_percent(tmp_path):
    run, manifest = write_run(tmp_path, [question("q1"), question("a1", "absent")],
                              [output("q1", "answered", [GOOD], claims=[CLAIM]), output("a1", "refused")])
    _, qs, outs = E.load_run(run)
    m = E.compute_metrics(manifest, qs, outs, None)
    assert m["answer_accuracy"]["value"] is None and m["citation_precision"]["value"] is None
    assert m["correct_refusal"]["value"] == 1.0 and m["hit_at"]["1"]["num"] == 1


def test_citation_denominator_zero_is_na(tmp_path):
    run, manifest = write_run(tmp_path, [question("q1")], [output("q1", "refused", [GOOD])])
    _, qs, outs = E.load_run(run)
    m = E.compute_metrics(manifest, qs, outs, {})
    assert m["citation_precision"] == {"num": 0, "den": 0, "value": None}
    assert m["answer_accuracy"]["num"] == 0  # refusal on answerable is wrong, no label needed


def test_integrity_errors(tmp_path):
    out = output("q1", "answered", [GOOD], claims=[CLAIM])
    run, manifest = write_run(tmp_path, [question("q1"), question("q2")], [out], name="missing")
    with pytest.raises(E.EvalError, match="no output record for q2"):
        E.load_run(run)

    run, _ = write_run(tmp_path, [question("q1")], [output("q1", "answered", [GOOD], claims=[CLAIM], gen="G2")], name="mixed")
    with pytest.raises(E.EvalError, match="another index generation"):
        E.load_run(run)

    run, _ = write_run(tmp_path, [question("q1")], [out], name="tampered-gold")
    (run / "questions.snapshot.jsonl").write_text(json.dumps(question("q1", evidence="easier evidence")) + "\n")
    with pytest.raises(E.EvalError, match="questions.snapshot.jsonl"):
        E.load_run(run)

    edited = dict(out, claims=[{"text": "edited after the fact", "source_ids": ["S1"]}])
    run, _ = write_run(tmp_path, [question("q1")], [edited], name="tampered-output")
    with pytest.raises(E.EvalError, match="was modified"):
        E.load_run(run)

    no_text = output("q1", "answered", [chunk(1, "")], claims=[CLAIM])
    run, _ = write_run(tmp_path, [question("q1")], [no_text], name="no-passage")
    with pytest.raises(E.EvalError, match="lacks retrieved passage text"):
        E.load_run(run)


def test_label_validation(tmp_path):
    out = output("q1", "answered", [GOOD], claims=[CLAIM])
    run, manifest = write_run(tmp_path, [question("q1")], [out])
    _, qs, outs = E.load_run(run)

    def check(lab, match):
        path = tmp_path / "labels.jsonl"
        path.write_text(json.dumps(lab) + "\n")
        with pytest.raises(E.EvalError, match=match):
            E.load_labels(path, manifest, outs)

    good = label(manifest, out, True)
    check(dict(good, output_hash="stale"), "stored output hash")
    check(dict(good, run_id="other"), "another run")
    check(dict(good, answer_correct=None), "true/false")
    check(dict(good, claims=[]), "claim count")
    check(dict(good, claims=[{"supported": None, "pairs": {"S1": True}}]), "incomplete")
    with pytest.raises(E.EvalError, match="missing label"):
        E.compute_metrics(manifest, qs, outs, {})


def test_question_validation_rules():
    E.validate_questions([question("q1"), question("a1", "absent")])
    with pytest.raises(E.EvalError, match="duplicates"):
        E.validate_questions([question("q1", text="Same  question?"), question("q2", split="dev", text="same question?")])
    with pytest.raises(E.EvalError, match="duplicate question id"):
        E.validate_questions([question("q1"), question("q1")])
    bad = question("a1", "absent")
    bad["required_facts"] = ["leak"]
    with pytest.raises(E.EvalError, match="empty gold"):
        E.validate_questions([bad])
    bad = question("q1")
    bad["expected_sources"] = [{"doc_id": DOC, "file": "a.pdf", "page": 3}]
    with pytest.raises(E.EvalError, match="incomplete expected_source"):
        E.validate_questions([bad])


def test_score_and_compare_end_to_end(tmp_path):
    runs = []
    for name, retrieved in (("vec", [chunk(1, "x"), GOOD | {"rank": 2}]), ("hyb", [GOOD])):
        out = output("q1", "answered", retrieved, claims=[CLAIM])
        run, manifest = write_run(tmp_path, [question("q1")], [out], name=name)
        (run / "labels.jsonl").write_text(json.dumps(label(manifest, out, True)) + "\n")
        assert E.score(SimpleNamespace(run=str(run), labels=str(run / "labels.jsonl"), report=str(tmp_path / f"{name}.md"))) == 0
        runs.append(str(run))
    first = json.loads((tmp_path / "vec" / "metrics.json").read_text())
    E.score(SimpleNamespace(run=runs[0], labels=runs[0] + "/labels.jsonl", report=None))
    assert json.loads((tmp_path / "vec" / "metrics.json").read_text()) == first  # rescoring is deterministic
    assert E.compare(SimpleNamespace(runs=runs, report=str(tmp_path / "cmp.md"))) == 0
    assert "| Evidence Hit@1 | 0/1 (0.0%) | 1/1 (100.0%) |" in (tmp_path / "cmp.md").read_text()

    other, _ = write_run(tmp_path, [question("q1")], [output("q1", "refused", [GOOD], gen="G9")], name="other", gen="G9")
    E.score(SimpleNamespace(run=str(other), labels=None, report=None))
    with pytest.raises(E.EvalError, match="does not share"):
        E.compare(SimpleNamespace(runs=[runs[0], str(other)], report=str(tmp_path / "bad.md")))


def test_template_lists_only_answered_outputs(tmp_path):
    run, _ = write_run(tmp_path, [question("q1"), question("q2")],
                       [output("q1", "answered", [GOOD], claims=[CLAIM]), output("q2", "refused", [GOOD])])
    E.template(SimpleNamespace(run=str(run)))
    rows = E.read_jsonl(run / "labels.todo.jsonl")
    assert [r["question_id"] for r in rows] == ["q1"] and rows[0]["claims"][0]["pairs"] == {"S1": None}
