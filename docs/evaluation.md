# Evaluation

## Question set

`eval/questions.jsonl`: 60 questions, **dev 20** (tuning) and **test 40** (reported once).

| Type | dev | test | Purpose |
| --- | --- | --- | --- |
| fact | 10 | 20 | number / unit / condition lookups |
| code | 4 | 8 | questions that hinge on a code or identifier (QCVN, TCVN, 29 CFR, ANSI) |
| cross | 3 | 6 | Vietnamese question, evidence in an English document |
| absent | 3 | 6 | not in the corpus; the right behaviour is to refuse |

Each answerable question stores the source document checksum, PDF page, a verbatim evidence span, the expected
answer and required facts. Chunk IDs are not part of the gold, so chunking can change without touching it.

## Rubric (fixed before any run)

- **Evidence Hit@k / MRR@5** (automatic): the evidence span, after NFC/lowercase/whitespace normalisation, is
  inside a retrieved chunk of the same document and page at rank ≤ k. A span cut by a chunk boundary counts when
  both halves are retrieved from that page. Same page without the span is a miss. Denominator: answerable questions.
- **Evidence in final context**: same test on the chunks actually sent to the LLM.
- **Answer accuracy** (human): correct only if all required facts are present with the right number, unit and
  condition, and nothing false is added. A refusal or an error on an answerable question is wrong.
- **Citation precision** (human): supporting claim/source pairs ÷ all emitted pairs.
  **Citation coverage**: claims fully supported by their cited sources ÷ all claims.
- **Correct refusal**: refused ÷ absent questions. **False refusal**: refused ÷ answerable questions.
  An error is never a refusal.
- **Latency**: wall-clock from question to validated result; mean, p50, p95 (linear interpolation).
  Reported for completed warm queries, all completed queries, and all attempts including errors.
- A zero denominator is printed as N/A.

## Procedure

```bash
python -m eval.run_eval generate --split dev --mode vector --output eval/results/dev-vector
python -m eval.run_eval template --run eval/results/dev-vector        # writes labels.todo.jsonl
# grade: fill it in, save as labels.jsonl
python -m eval.run_eval score --run eval/results/dev-vector --labels eval/results/dev-vector/labels.jsonl --report docs/reports/dev-vector.md
python -m eval.run_eval compare --runs eval/results/test-vector eval/results/test-hybrid eval/results/test-hybrid-rerank --report docs/reports/test-comparison.md
```

A run directory is never overwritten. It snapshots the questions and source manifest, pins one index generation
for the whole run, and stores the full text of the retrieved top 5 and of the context sent to the LLM, so
`score` needs neither models nor an index. `score` refuses a run whose snapshots, output hashes or generation
do not match its manifest; `compare` refuses runs that differ in index, questions, LLM or budgets.

Tuning uses dev only. If a test result leads to a config change, that test set is spent.

## Limits

34 answerable test questions: one question is 2.94 points of accuracy, and 6 absent questions make one refusal
16.67 points. Differences of one or two questions between modes are not evidence of a real difference.
Questions and gold were written by the same party that built the system.
