# Eval run `test-vector`

- split **test**, mode **vector**, 40 questions (34 answerable, 6 absent)
- index generation `ac3e0f44511b6a39`, LLM `qwen3:4b-q4_K_M` digest `2bfd38a7daaf`
- human labels: yes

| Metric | Value |
| --- | --- |
| Evidence Hit@1 | 14/34 (41.2%) |
| Evidence Hit@3 | 27/34 (79.4%) |
| Evidence Hit@5 | 28/34 (82.4%) |
| MRR@5 | 0.591 |
| Evidence in final context | 28/34 (82.4%) |
| Answer accuracy (answerable) | 31/34 (91.2%) |
| Citation precision (claim/source pairs) | N/A |
| Citation coverage (fully supported claims) | N/A |
| Correct refusal (absent) | 5/6 (83.3%) |
| False refusal (answerable) | 1/34 (2.9%) |
| Error rate (all) | 0/40 (0.0%) |

| Latency sample | n | mean | p50 | p95 |
| --- | --- | --- | --- | --- |
| completed, warm LLM | 40 | 2.88s | 2.79s | 3.82s |
| completed, all | 40 | 2.88s | 2.79s | 3.82s |
| all attempts incl. errors | 40 | 2.88s | 2.79s | 3.82s |

| Type | Correct |
| --- | --- |
| fact | 19/20 (95.0%) |
| code | 6/8 (75.0%) |
| cross | 6/6 (100.0%) |
| absent | 5/6 (83.3%) |

| Question | Type | Status | Hit rank | In context | Correct |
| --- | --- | --- | --- | --- | --- |
| t01 | fact | answered | - | False | False |
| t02 | fact | answered | 2 | True | True |
| t03 | fact | answered | 1 | True | True |
| t04 | fact | answered | 2 | True | True |
| t05 | fact | answered | 1 | True | True |
| t06 | fact | answered | 1 | True | True |
| t07 | fact | answered | 1 | True | True |
| t08 | fact | answered | 3 | True | True |
| t09 | fact | answered | 1 | True | True |
| t10 | fact | answered | 4 | True | True |
| t11 | fact | answered | 1 | True | True |
| t12 | fact | answered | 1 | True | True |
| t13 | fact | answered | 1 | True | True |
| t14 | fact | answered | 1 | True | True |
| t15 | fact | answered | 1 | True | True |
| t16 | fact | answered | 2 | True | True |
| t17 | fact | answered | 1 | True | True |
| t18 | fact | answered | 2 | True | True |
| t19 | fact | answered | 3 | True | True |
| t20 | fact | answered | 1 | True | True |
| t21 | code | refused | - | False | False |
| t22 | code | answered | 2 | True | True |
| t23 | code | answered | 2 | True | True |
| t24 | code | answered | 1 | True | True |
| t25 | code | answered | 1 | True | True |
| t26 | code | answered | - | False | True |
| t27 | code | answered | 2 | True | True |
| t28 | code | answered | - | False | False |
| t29 | cross | answered | 3 | True | True |
| t30 | cross | answered | - | False | True |
| t31 | cross | answered | 2 | True | True |
| t32 | cross | answered | - | False | True |
| t33 | cross | answered | 2 | True | True |
| t34 | cross | answered | 3 | True | True |
| t35 | absent | refused | - | - | True |
| t36 | absent | refused | - | - | True |
| t37 | absent | refused | - | - | True |
| t38 | absent | answered | - | - | False |
| t39 | absent | refused | - | - | True |
| t40 | absent | refused | - | - | True |

Small sample: one answerable question moves accuracy by 2.94 points; no statistical significance is claimed.
