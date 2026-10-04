"""Evaluation runner.

  generate  questions + pipeline -> immutable run directory (snapshots, manifest, outputs.jsonl)
  template  run -> labels skeleton for the human grader
  score     run + labels -> metrics.json / metrics.csv / Markdown report (no model or index needed)
  compare   several scored runs on the same base -> side-by-side report
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import resource
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

QUESTION_FIELDS = ("id", "split", "question", "lang", "type", "tags", "answerable",
                   "expected_sources", "expected_answer", "required_facts")
TYPES = ("fact", "code", "cross", "absent")
KS = (1, 3, 5)


class EvalError(RuntimeError):
    pass


# ---------------------------------------------------------------- helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()


def output_hash(record: dict) -> str:
    core = {k: record.get(k) for k in ("status", "claims", "refusal_reason", "error_code", "raw_output")}
    return sha256_bytes(json.dumps(core, ensure_ascii=False, sort_keys=True).encode())[:16]


def validate_questions(questions: list) -> None:
    seen, seen_text = set(), {}
    for q in questions:
        missing = [k for k in QUESTION_FIELDS if k not in q]
        if missing:
            raise EvalError(f"question {q.get('id')}: missing {missing}")
        if q["id"] in seen:
            raise EvalError(f"duplicate question id {q['id']}")
        seen.add(q["id"])
        if q["split"] not in ("dev", "test") or q["type"] not in TYPES:
            raise EvalError(f"question {q['id']}: bad split/type")
        key = norm(q["question"])
        if key in seen_text:
            raise EvalError(f"questions {seen_text[key]} and {q['id']} are duplicates after normalisation")
        seen_text[key] = q["id"]
        if q["answerable"]:
            if not q["expected_sources"] or not q["required_facts"] or not q["expected_answer"]:
                raise EvalError(f"question {q['id']}: answerable needs sources, answer and required_facts")
            for s in q["expected_sources"]:
                if not all(s.get(k) for k in ("doc_id", "file", "page", "evidence")):
                    raise EvalError(f"question {q['id']}: incomplete expected_source")
        elif q["expected_sources"] or q["required_facts"] or q["expected_answer"] is not None:
            raise EvalError(f"question {q['id']}: unanswerable must have empty gold")
        if (q["type"] == "absent") == q["answerable"]:
            raise EvalError(f"question {q['id']}: type 'absent' must match answerable=false")


def evidence_hit(expected_sources: list, chunks: list) -> int | None:
    """Best (lowest) rank at which gold evidence is fully present, or None.

    A span counts when it sits inside one chunk, or is split across two retrieved chunks of the
    same document page (chunk boundary). Same file/page without the span is a miss.
    """
    best = None
    for src in expected_sources:
        ev = norm(src["evidence"])
        same_page = [c for c in chunks if c["doc_id"] == src["doc_id"] and c["page"] == src["page"]]
        texts = [(c["rank"], norm(c["text"])) for c in same_page]
        for rank, text in texts:
            if ev in text:
                best = rank if best is None else min(best, rank)
        words = ev.split(" ")
        for cut in range(1, len(words)):
            head, tail = " ".join(words[:cut]), " ".join(words[cut:])
            ranks_h = [r for r, t in texts if t.endswith(head)]
            ranks_t = [r for r, t in texts if tail in t]
            pairs = [max(a, b) for a in ranks_h for b in ranks_t if a != b]
            if pairs:
                best = min(pairs) if best is None else min(best, min(pairs))
    return best


def percentile(values: list, p: float):
    """Linear interpolation at rank p*(n-1); None for an empty sample."""
    if not values:
        return None
    s = sorted(values)
    pos = p * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def ratio(num: int, den: int) -> dict:
    return {"num": num, "den": den, "value": (num / den) if den else None}


def fmt(r: dict) -> str:
    return "N/A" if r["value"] is None else f"{r['num']}/{r['den']} ({100 * r['value']:.1f}%)"


def _hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu": platform.processor()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        info["gpu"] = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["gpu"] = None
    return info


# ---------------------------------------------------------------- generate

def generate(args) -> int:
    from src import pipeline, retriever
    from src.config import load_config

    out = Path(args.output)
    if out.exists():
        raise EvalError(f"{out} already exists; runs are never overwritten")
    cfg = load_config()
    raw_questions = Path(args.questions).read_bytes()
    questions = read_jsonl(args.questions)
    validate_questions(questions)
    selected = [q for q in questions if q["split"] == args.split][: args.limit]
    if not selected:
        raise EvalError(f"no questions in split {args.split}")

    gen = retriever.load_generation(cfg.index_root, expected_config_hash=cfg.index_config_hash())
    sources_bytes = Path(cfg.sources_path).read_bytes()
    if sha256_bytes(sources_bytes) != gen.manifest["source_manifest_hash"]:
        raise EvalError("data/sources.jsonl changed since the active index was built; re-run ingest")
    res = pipeline.load_resources(cfg, with_reranker=args.mode == "hybrid-rerank")
    llm = pipeline.llm_state(cfg)

    out.mkdir(parents=True)
    snap_q = "".join(json.dumps(q, ensure_ascii=False, sort_keys=True) + "\n" for q in selected).encode()
    (out / "questions.snapshot.jsonl").write_bytes(snap_q)
    (out / "sources.snapshot.jsonl").write_bytes(sources_bytes)
    test_lines = b"".join(l + b"\n" for l in raw_questions.splitlines() if l.strip() and json.loads(l)["split"] == "test")
    manifest = {
        "schema_version": 1, "run_id": out.name, "split": args.split, "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation_id": gen.gen_id, "generation_manifest_sha256": retriever.file_sha256(gen.path / "manifest.json"),
        "config": cfg.as_dict(), "llm_digest": llm["digest"], "hardware": _hardware(),
        "questions_snapshot_sha256": sha256_bytes(snap_q), "sources_snapshot_sha256": sha256_bytes(sources_bytes),
        "questions_file_sha256": sha256_bytes(raw_questions), "test_only_sha256": sha256_bytes(test_lines),
        "question_count": len(selected),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    with open(out / "outputs.jsonl", "a", encoding="utf-8") as f:
        for i, q in enumerate(selected, 1):
            before = pipeline.llm_state(cfg)["loaded"]
            result = pipeline.answer(q["question"], args.mode, cfg, res, gen)  # gold never reaches the pipeline
            result.update(question_id=q["id"], llm_loaded_before=before)
            result["output_hash"] = output_hash(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(selected)}] {q['id']} {result['status']}"
                  f"{':' + result['error_code'] if result['error_code'] else ''} "
                  f"{result['timings_ms']['wall_end_to_end']:.0f} ms", flush=True)
    after = pipeline.llm_state(cfg)
    resources = {"app_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                 "ollama_gpu_fraction": after["gpu_fraction"], "gpu_after_run": _hardware()["gpu"],
                 "sampler_method": "getrusage(ru_maxrss) for app; one nvidia-smi/api/ps sample after the run"}
    (out / "resources.json").write_text(json.dumps(resources, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- template / score

def load_run(run_dir) -> tuple:
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for name, key in (("questions.snapshot.jsonl", "questions_snapshot_sha256"),
                      ("sources.snapshot.jsonl", "sources_snapshot_sha256")):
        if sha256_bytes((run / name).read_bytes()) != manifest[key]:
            raise EvalError(f"integrity error: {name} does not match the run manifest")
    questions = read_jsonl(run / "questions.snapshot.jsonl")
    outputs = {r["question_id"]: r for r in read_jsonl(run / "outputs.jsonl")}
    for q in questions:
        r = outputs.get(q["id"])
        if r is None:
            raise EvalError(f"integrity error: no output record for {q['id']}")
        if r["generation_id"] != manifest["generation_id"]:
            raise EvalError(f"integrity error: {q['id']} was answered from another index generation")
        if r["output_hash"] != output_hash(r):
            raise EvalError(f"integrity error: output of {q['id']} was modified")
        if r["status"] != "error" and r.get("error_code") != "input_error":
            for c in r["retrieved_chunks"]:
                if not c.get("text"):
                    raise EvalError(f"integrity error: {q['id']} lacks retrieved passage text")
    return manifest, questions, outputs


def template(args) -> int:
    manifest, questions, outputs = load_run(args.run)
    path = Path(args.run) / "labels.todo.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            r = outputs[q["id"]]
            if r["status"] != "answered":
                continue  # refusals and errors are scored automatically
            f.write(json.dumps({
                "run_id": manifest["run_id"], "question_id": q["id"], "output_hash": r["output_hash"],
                "answer_correct": None,
                "claims": [{"text": c["text"], "supported": None, "pairs": {s: None for s in c["source_ids"]}}
                           for c in r["claims"]],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {path}: fill answer_correct (true/false), supported (full/partial/none) and pairs (true/false), "
          f"then save as labels.jsonl")
    return 0


def load_labels(path, manifest, outputs) -> dict:
    labels = {}
    for lab in read_jsonl(path):
        qid = lab["question_id"]
        if lab["run_id"] != manifest["run_id"] or qid not in outputs:
            raise EvalError(f"label for {qid} belongs to another run")
        if lab["output_hash"] != outputs[qid]["output_hash"]:
            raise EvalError(f"label for {qid} does not match the stored output hash")
        if not isinstance(lab["answer_correct"], bool):
            raise EvalError(f"label for {qid}: answer_correct must be true/false")
        if lab["claims"] is None:  # answer graded, citations not graded yet
            labels[qid] = lab
            continue
        if len(lab["claims"]) != len(outputs[qid]["claims"]):
            raise EvalError(f"label for {qid}: claim count differs from output")
        for c in lab["claims"]:
            if c["supported"] not in ("full", "partial", "none") or not all(isinstance(v, bool) for v in c["pairs"].values()):
                raise EvalError(f"label for {qid}: incomplete claim labels")
        labels[qid] = lab
    return labels


def compute_metrics(manifest: dict, questions: list, outputs: dict, labels: dict | None) -> dict:
    answerable = [q for q in questions if q["answerable"]]
    absent = [q for q in questions if not q["answerable"]]
    per_q, hits, ctx_hits, rr = [], {k: 0 for k in KS}, 0, 0.0
    correct = pairs_ok = pairs_all = claims_ok = claims_all = 0
    labelled = labels is not None
    for q in questions:
        r = outputs[q["id"]]
        row = {"id": q["id"], "type": q["type"], "status": r["status"], "error_code": r["error_code"],
               "wall_ms": r["timings_ms"]["wall_end_to_end"], "hit_rank": None, "context_hit": None, "correct": None}
        if q["answerable"]:
            rank = evidence_hit(q["expected_sources"], r["retrieved_chunks"])
            row["hit_rank"] = rank
            for k in KS:
                hits[k] += rank is not None and rank <= k
            rr += 1.0 / rank if rank else 0.0
            ctx = [dict(c, rank=i) for i, c in enumerate(r["context_chunks"], 1)]
            row["context_hit"] = evidence_hit(q["expected_sources"], ctx) is not None
            ctx_hits += row["context_hit"]
            if r["status"] != "answered":
                row["correct"] = False  # refusal or error on an answerable question is wrong
            elif labelled:
                if q["id"] not in labels:
                    raise EvalError(f"missing label for answered question {q['id']}")
                row["correct"] = labels[q["id"]]["answer_correct"]
            correct += bool(row["correct"])
        else:
            row["correct"] = r["status"] == "refused"
        if labelled and r["status"] == "answered":
            if q["id"] not in labels:
                raise EvalError(f"missing label for answered question {q['id']}")
            for c in labels[q["id"]]["claims"] or []:
                claims_all += 1
                claims_ok += c["supported"] == "full"
                pairs_all += len(c["pairs"])
                pairs_ok += sum(c["pairs"].values())
        per_q.append(row)

    status = lambda qs, s: sum(outputs[q["id"]]["status"] == s for q in qs)
    walls = lambda rows: [r["wall_ms"] for r in rows if r["wall_ms"] is not None]
    done = [r for r in per_q if r["status"] in ("answered", "refused")]
    warm = [r for r in done if (outputs[r["id"]]["timings_ms"]["ollama"]["load"] or 0) < 1000
            and outputs[r["id"]].get("llm_loaded_before")]
    lat = lambda rows: {"n": len(rows), "mean": (sum(walls(rows)) / len(walls(rows))) if walls(rows) else None,
                        "p50": percentile(walls(rows), 0.5), "p95": percentile(walls(rows), 0.95)}
    by_type = {}
    for t in TYPES:
        rows = [r for r in per_q if r["type"] == t]
        if rows:
            ok = sum(bool(r["correct"]) for r in rows) if (labelled or t == "absent") else 0
            by_type[t] = ratio(ok, len(rows)) if (labelled or t == "absent") else ratio(0, 0)
    n = len(answerable)
    return {
        "run_id": manifest["run_id"], "mode": manifest["mode"], "split": manifest["split"],
        "generation_id": manifest["generation_id"], "labelled": labelled,
        "questions": len(questions), "answerable": n, "absent": len(absent),
        "hit_at": {str(k): ratio(hits[k], n) for k in KS},
        "mrr_at_5": (rr / n) if n else None,
        "context_hit": ratio(ctx_hits, n),
        "answer_accuracy": ratio(correct, n) if labelled else ratio(0, 0),
        "citation_precision": ratio(pairs_ok, pairs_all) if labelled else ratio(0, 0),
        "citation_coverage": ratio(claims_ok, claims_all) if labelled else ratio(0, 0),
        "correct_refusal": ratio(status(absent, "refused"), len(absent)),
        "false_refusal": ratio(status(answerable, "refused"), n),
        "error_rate": ratio(status(questions, "error"), len(questions)),
        "accuracy_by_type": by_type,
        "latency_ms": {"completed_warm": lat(warm), "completed_all": lat(done), "all_attempts": lat(per_q)},
        "per_question": per_q,
    }


def render_report(m: dict, manifest: dict) -> str:
    L = m["latency_ms"]
    ms = lambda v: "N/A" if v is None else f"{v / 1000:.2f}s"
    lines = [
        f"# Eval run `{m['run_id']}`", "",
        f"- split **{m['split']}**, mode **{m['mode']}**, {m['questions']} questions "
        f"({m['answerable']} answerable, {m['absent']} absent)",
        f"- index generation `{m['generation_id']}`, LLM `{manifest['config']['llm_model']}` digest `{(manifest.get('llm_digest') or '')[:12]}`",
        f"- human labels: {'yes' if m['labelled'] else 'NO - answer/citation metrics unavailable'}", "",
        "| Metric | Value |", "| --- | --- |",
    ]
    lines += [f"| Evidence Hit@{k} | {fmt(m['hit_at'][str(k)])} |" for k in KS]
    lines += [
        f"| MRR@5 | {'N/A' if m['mrr_at_5'] is None else format(m['mrr_at_5'], '.3f')} |",
        f"| Evidence in final context | {fmt(m['context_hit'])} |",
        f"| Answer accuracy (answerable) | {fmt(m['answer_accuracy'])} |",
        f"| Citation precision (claim/source pairs) | {fmt(m['citation_precision'])} |",
        f"| Citation coverage (fully supported claims) | {fmt(m['citation_coverage'])} |",
        f"| Correct refusal (absent) | {fmt(m['correct_refusal'])} |",
        f"| False refusal (answerable) | {fmt(m['false_refusal'])} |",
        f"| Error rate (all) | {fmt(m['error_rate'])} |", "",
        "| Latency sample | n | mean | p50 | p95 |", "| --- | --- | --- | --- | --- |",
    ]
    for name, key in (("completed, warm LLM", "completed_warm"), ("completed, all", "completed_all"),
                      ("all attempts incl. errors", "all_attempts")):
        lines.append(f"| {name} | {L[key]['n']} | {ms(L[key]['mean'])} | {ms(L[key]['p50'])} | {ms(L[key]['p95'])} |")
    lines += ["", "| Type | Correct |", "| --- | --- |"]
    lines += [f"| {t} | {fmt(r)} |" for t, r in m["accuracy_by_type"].items()]
    lines += ["", "| Question | Type | Status | Hit rank | In context | Correct |", "| --- | --- | --- | --- | --- | --- |"]
    for r in m["per_question"]:
        status = r["status"] + (f":{r['error_code']}" if r["error_code"] else "")
        lines.append(f"| {r['id']} | {r['type']} | {status} | {r['hit_rank'] or '-'} | "
                     f"{'-' if r['context_hit'] is None else r['context_hit']} | {'-' if r['correct'] is None else r['correct']} |")
    lines += ["", "Small sample: one answerable question moves accuracy by "
              f"{100 / m['answerable']:.2f} points; no statistical significance is claimed." if m["answerable"] else ""]
    return "\n".join(lines) + "\n"


def score(args) -> int:
    manifest, questions, outputs = load_run(args.run)
    labels = load_labels(args.labels, manifest, outputs) if args.labels else None
    m = compute_metrics(manifest, questions, outputs, labels)
    run = Path(args.run)
    (run / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(run / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "num", "den", "value"])
        for name in ("answer_accuracy", "citation_precision", "citation_coverage", "context_hit",
                     "correct_refusal", "false_refusal", "error_rate"):
            w.writerow([name, m[name]["num"], m[name]["den"], m[name]["value"]])
        for k in KS:
            r = m["hit_at"][str(k)]
            w.writerow([f"hit_at_{k}", r["num"], r["den"], r["value"]])
    report = render_report(m, manifest)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
    print(report.split("\n| Question")[0])
    return 0


# ---------------------------------------------------------------- compare

BASE_KEYS = ("llm_model", "embed_model", "embed_revision", "rerank_revision", "chunk_tokens", "chunk_overlap",
             "num_ctx", "num_predict", "temperature", "seed", "context_k")


def compare(args) -> int:
    runs = []
    for run_dir in args.runs:
        manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
        runs.append((manifest, metrics))
    base = runs[0][0]
    for manifest, _ in runs[1:]:
        same = (manifest["generation_id"] == base["generation_id"]
                and manifest["generation_manifest_sha256"] == base["generation_manifest_sha256"]
                and manifest["questions_snapshot_sha256"] == base["questions_snapshot_sha256"]
                and manifest.get("llm_digest") == base.get("llm_digest")
                and all(manifest["config"][k] == base["config"][k] for k in BASE_KEYS))
        if not same:
            raise EvalError(f"run {manifest['run_id']} does not share index/questions/LLM/config with {base['run_id']}")
    head = "| Metric | " + " | ".join(m["mode"] for _, m in runs) + " |"
    lines = [f"# Comparison on split `{base['split']}` ({runs[0][1]['questions']} questions)", "",
             f"Same index generation `{base['generation_id']}`, questions, LLM and budgets for every column.", "",
             head, "| --- |" + " --- |" * len(runs)]
    rows = [("Evidence Hit@1", lambda m: fmt(m["hit_at"]["1"])), ("Evidence Hit@5", lambda m: fmt(m["hit_at"]["5"])),
            ("MRR@5", lambda m: "N/A" if m["mrr_at_5"] is None else f"{m['mrr_at_5']:.3f}"),
            ("Answer accuracy", lambda m: fmt(m["answer_accuracy"])),
            ("Citation precision", lambda m: fmt(m["citation_precision"])),
            ("Citation coverage", lambda m: fmt(m["citation_coverage"])),
            ("Correct refusal", lambda m: fmt(m["correct_refusal"])),
            ("False refusal", lambda m: fmt(m["false_refusal"])),
            ("Error rate", lambda m: fmt(m["error_rate"])),
            ("Warm latency p50", lambda m: "N/A" if m["latency_ms"]["completed_warm"]["p50"] is None
             else f"{m['latency_ms']['completed_warm']['p50'] / 1000:.2f}s"),
            ("Warm latency p95", lambda m: "N/A" if m["latency_ms"]["completed_warm"]["p95"] is None
             else f"{m['latency_ms']['completed_warm']['p95'] / 1000:.2f}s")]
    for name, get in rows:
        lines.append(f"| {name} | " + " | ".join(get(m) for _, m in runs) + " |")
    lines += ["", "## Per question (hit rank / correct)", "",
              "| Question | Type | " + " | ".join(m["mode"] for _, m in runs) + " |", "| --- | --- |" + " --- |" * len(runs)]
    per = [{r["id"]: r for r in m["per_question"]} for _, m in runs]
    for r in runs[0][1]["per_question"]:
        cells = [f"{p[r['id']]['hit_rank'] or '-'} / {'-' if p[r['id']]['correct'] is None else p[r['id']]['correct']}" for p in per]
        lines.append(f"| {r['id']} | {r['type']} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(text, encoding="utf-8")
    print(text.split("\n## Per question")[0])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--questions", default="eval/questions.jsonl")
    g.add_argument("--split", choices=("dev", "test"), required=True)
    g.add_argument("--mode", choices=("vector", "hybrid", "hybrid-rerank"), required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--limit", type=int, default=None)
    g.set_defaults(func=generate)
    t = sub.add_parser("template")
    t.add_argument("--run", required=True)
    t.set_defaults(func=template)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--labels", help="omit for automatic metrics only (retrieval, refusal, errors, latency)")
    s.add_argument("--report")
    s.set_defaults(func=score)
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--report", required=True)
    c.set_defaults(func=compare)
    args = p.parse_args(argv)
    try:
        return args.func(args)
    except EvalError as e:
        print(f"eval error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
