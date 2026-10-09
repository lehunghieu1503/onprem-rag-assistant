# Eval run `test-hybrid-rerank`

- split **test**, mode **hybrid-rerank**, 40 questions (34 answerable, 6 absent)
- index generation `ac3e0f44511b6a39`, LLM `qwen3:4b-q4_K_M` digest `2bfd38a7daaf`
- human labels: yes

| Metric | Value |
| --- | --- |
| Evidence Hit@1 | 27/34 (79.4%) |
| Evidence Hit@3 | 31/34 (91.2%) |
| Evidence Hit@5 | 31/34 (91.2%) |
| MRR@5 | 0.848 |
| Evidence in final context | 31/34 (91.2%) |
| Answer accuracy (answerable) | 31/34 (91.2%) |
| Citation precision (claim/source pairs) | N/A |
| Citation coverage (fully supported claims) | N/A |
| Correct refusal (absent) | 5/6 (83.3%) |
| False refusal (answerable) | 1/34 (2.9%) |
| Error rate (all) | 0/40 (0.0%) |

| Latency sample | n | mean | p50 | p95 |
| --- | --- | --- | --- | --- |
| completed, warm LLM | 40 | 9.83s | 9.72s | 11.39s |
| completed, all | 40 | 9.83s | 9.72s | 11.39s |
| all attempts incl. errors | 40 | 9.83s | 9.72s | 11.39s |

| Type | Correct |
| --- | --- |
| fact | 18/20 (90.0%) |
| code | 8/8 (100.0%) |
| cross | 5/6 (83.3%) |
| absent | 5/6 (83.3%) |

| Question | Type | Status | Hit rank | In context | Correct |
| --- | --- | --- | --- | --- | --- |
| t01 | fact | answered | 1 | True | True |
| t02 | fact | answered | 1 | True | True |
| t03 | fact | answered | 1 | True | True |
| t04 | fact | answered | 1 | True | True |
| t05 | fact | answered | 1 | True | True |
| t06 | fact | answered | 1 | True | True |
| t07 | fact | answered | 1 | True | True |
| t08 | fact | answered | 1 | True | True |
| t09 | fact | answered | 1 | True | True |
| t10 | fact | answered | 1 | True | True |
| t11 | fact | answered | 1 | True | True |
| t12 | fact | answered | 1 | True | True |
| t13 | fact | answered | 1 | True | True |
| t14 | fact | answered | 1 | True | False |
| t15 | fact | answered | 1 | True | True |
| t16 | fact | answered | 1 | True | True |
| t17 | fact | answered | 1 | True | False |
| t18 | fact | answered | 1 | True | True |
| t19 | fact | answered | 1 | True | True |
| t20 | fact | answered | 1 | True | True |
| t21 | code | answered | 1 | True | True |
| t22 | code | answered | 1 | True | True |
| t23 | code | answered | 2 | True | True |
| t24 | code | answered | 1 | True | True |
| t25 | code | answered | 2 | True | True |
| t26 | code | answered | 3 | True | True |
| t27 | code | answered | 1 | True | True |
| t28 | code | answered | 1 | True | True |
| t29 | cross | answered | 1 | True | True |
| t30 | cross | answered | - | False | True |
| t31 | cross | answered | 1 | True | True |
| t32 | cross | answered | - | False | True |
| t33 | cross | refused | - | False | False |
| t34 | cross | answered | 2 | True | True |
| t35 | absent | refused | - | - | True |
| t36 | absent | refused | - | - | True |
| t37 | absent | refused | - | - | True |
| t38 | absent | answered | - | - | False |
| t39 | absent | refused | - | - | True |
| t40 | absent | refused | - | - | True |

Small sample: one answerable question moves accuracy by 2.94 points; no statistical significance is claimed.
