# Eval run `dev-hybrid-rerank`

- split **dev**, mode **hybrid-rerank**, 20 questions (17 answerable, 3 absent)
- index generation `ac3e0f44511b6a39`, LLM `qwen3:4b-q4_K_M` digest `2bfd38a7daaf`
- human labels: NO - answer/citation metrics unavailable

| Metric | Value |
| --- | --- |
| Evidence Hit@1 | 16/17 (94.1%) |
| Evidence Hit@3 | 17/17 (100.0%) |
| Evidence Hit@5 | 17/17 (100.0%) |
| MRR@5 | 0.971 |
| Evidence in final context | 17/17 (100.0%) |
| Answer accuracy (answerable) | N/A |
| Citation precision (claim/source pairs) | N/A |
| Citation coverage (fully supported claims) | N/A |
| Correct refusal (absent) | 3/3 (100.0%) |
| False refusal (answerable) | 0/17 (0.0%) |
| Error rate (all) | 0/20 (0.0%) |

| Latency sample | n | mean | p50 | p95 |
| --- | --- | --- | --- | --- |
| completed, warm LLM | 20 | 9.37s | 9.34s | 11.59s |
| completed, all | 20 | 9.37s | 9.34s | 11.59s |
| all attempts incl. errors | 20 | 9.37s | 9.34s | 11.59s |

| Type | Correct |
| --- | --- |
| fact | N/A |
| code | N/A |
| cross | N/A |
| absent | 3/3 (100.0%) |

| Question | Type | Status | Hit rank | In context | Correct |
| --- | --- | --- | --- | --- | --- |
| d01 | fact | answered | 1 | True | - |
| d02 | fact | answered | 1 | True | - |
| d03 | fact | answered | 1 | True | - |
| d04 | fact | answered | 1 | True | - |
| d05 | fact | answered | 1 | True | - |
| d06 | fact | answered | 1 | True | - |
| d07 | fact | answered | 1 | True | - |
| d08 | fact | answered | 1 | True | - |
| d09 | fact | answered | 1 | True | - |
| d10 | fact | answered | 1 | True | - |
| d11 | code | answered | 1 | True | - |
| d12 | code | answered | 1 | True | - |
| d13 | code | answered | 1 | True | - |
| d14 | code | answered | 2 | True | - |
| d15 | cross | answered | 1 | True | - |
| d16 | cross | answered | 1 | True | - |
| d17 | cross | answered | 1 | True | - |
| d18 | absent | refused | - | - | True |
| d19 | absent | refused | - | - | True |
| d20 | absent | refused | - | - | True |

Small sample: one answerable question moves accuracy by 5.88 points; no statistical significance is claimed.
