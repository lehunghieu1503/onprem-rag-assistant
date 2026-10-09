# Eval run `test-hybrid-rerank-gpu`

- split **test**, mode **hybrid-rerank**, 40 questions (34 answerable, 6 absent)
- index generation `ac3e0f44511b6a39`, LLM `qwen3:4b-q4_K_M` digest `2bfd38a7daaf`
- human labels: NO - answer/citation metrics unavailable

| Metric | Value |
| --- | --- |
| Evidence Hit@1 | 27/34 (79.4%) |
| Evidence Hit@3 | 31/34 (91.2%) |
| Evidence Hit@5 | 31/34 (91.2%) |
| MRR@5 | 0.848 |
| Evidence in final context | 31/34 (91.2%) |
| Answer accuracy (answerable) | N/A |
| Citation precision (claim/source pairs) | N/A |
| Citation coverage (fully supported claims) | N/A |
| Correct refusal (absent) | 5/6 (83.3%) |
| False refusal (answerable) | 1/34 (2.9%) |
| Error rate (all) | 0/40 (0.0%) |

| Latency sample | n | mean | p50 | p95 |
| --- | --- | --- | --- | --- |
| completed, warm LLM | 40 | 3.17s | 3.03s | 4.21s |
| completed, all | 40 | 3.17s | 3.03s | 4.21s |
| all attempts incl. errors | 40 | 3.17s | 3.03s | 4.21s |

| Type | Correct |
| --- | --- |
| fact | N/A |
| code | N/A |
| cross | N/A |
| absent | 5/6 (83.3%) |

| Question | Type | Status | Hit rank | In context | Correct |
| --- | --- | --- | --- | --- | --- |
| t01 | fact | answered | 1 | True | - |
| t02 | fact | answered | 1 | True | - |
| t03 | fact | answered | 1 | True | - |
| t04 | fact | answered | 1 | True | - |
| t05 | fact | answered | 1 | True | - |
| t06 | fact | answered | 1 | True | - |
| t07 | fact | answered | 1 | True | - |
| t08 | fact | answered | 1 | True | - |
| t09 | fact | answered | 1 | True | - |
| t10 | fact | answered | 1 | True | - |
| t11 | fact | answered | 1 | True | - |
| t12 | fact | answered | 1 | True | - |
| t13 | fact | answered | 1 | True | - |
| t14 | fact | answered | 1 | True | - |
| t15 | fact | answered | 1 | True | - |
| t16 | fact | answered | 1 | True | - |
| t17 | fact | answered | 1 | True | - |
| t18 | fact | answered | 1 | True | - |
| t19 | fact | answered | 1 | True | - |
| t20 | fact | answered | 1 | True | - |
| t21 | code | answered | 1 | True | - |
| t22 | code | answered | 1 | True | - |
| t23 | code | answered | 2 | True | - |
| t24 | code | answered | 1 | True | - |
| t25 | code | answered | 2 | True | - |
| t26 | code | answered | 3 | True | - |
| t27 | code | answered | 1 | True | - |
| t28 | code | answered | 1 | True | - |
| t29 | cross | answered | 1 | True | - |
| t30 | cross | answered | - | False | - |
| t31 | cross | answered | 1 | True | - |
| t32 | cross | answered | - | False | - |
| t33 | cross | refused | - | False | False |
| t34 | cross | answered | 2 | True | - |
| t35 | absent | refused | - | - | True |
| t36 | absent | refused | - | - | True |
| t37 | absent | refused | - | - | True |
| t38 | absent | answered | - | - | False |
| t39 | absent | refused | - | - | True |
| t40 | absent | refused | - | - | True |

Small sample: one answerable question moves accuracy by 2.94 points; no statistical significance is claimed.
