# Eval run `test-hybrid`

- split **test**, mode **hybrid**, 40 questions (34 answerable, 6 absent)
- index generation `ac3e0f44511b6a39`, LLM `qwen3:4b-q4_K_M` digest `2bfd38a7daaf`
- human labels: yes

| Metric | Value |
| --- | --- |
| Evidence Hit@1 | 18/34 (52.9%) |
| Evidence Hit@3 | 26/34 (76.5%) |
| Evidence Hit@5 | 29/34 (85.3%) |
| MRR@5 | 0.666 |
| Evidence in final context | 29/34 (85.3%) |
| Answer accuracy (answerable) | 32/34 (94.1%) |
| Citation precision (claim/source pairs) | N/A |
| Citation coverage (fully supported claims) | N/A |
| Correct refusal (absent) | 5/6 (83.3%) |
| False refusal (answerable) | 0/34 (0.0%) |
| Error rate (all) | 0/40 (0.0%) |

| Latency sample | n | mean | p50 | p95 |
| --- | --- | --- | --- | --- |
| completed, warm LLM | 40 | 2.94s | 2.79s | 4.16s |
| completed, all | 40 | 2.94s | 2.79s | 4.16s |
| all attempts incl. errors | 40 | 2.94s | 2.79s | 4.16s |

| Type | Correct |
| --- | --- |
| fact | 20/20 (100.0%) |
| code | 8/8 (100.0%) |
| cross | 4/6 (66.7%) |
| absent | 5/6 (83.3%) |

| Question | Type | Status | Hit rank | In context | Correct |
| --- | --- | --- | --- | --- | --- |
| t01 | fact | answered | 2 | True | True |
| t02 | fact | answered | 2 | True | True |
| t03 | fact | answered | 1 | True | True |
| t04 | fact | answered | 1 | True | True |
| t05 | fact | answered | 1 | True | True |
| t06 | fact | answered | 1 | True | True |
| t07 | fact | answered | 1 | True | True |
| t08 | fact | answered | 1 | True | True |
| t09 | fact | answered | 1 | True | True |
| t10 | fact | answered | 2 | True | True |
| t11 | fact | answered | 1 | True | True |
| t12 | fact | answered | 1 | True | True |
| t13 | fact | answered | 1 | True | True |
| t14 | fact | answered | 1 | True | True |
| t15 | fact | answered | 1 | True | True |
| t16 | fact | answered | 1 | True | True |
| t17 | fact | answered | 1 | True | True |
| t18 | fact | answered | 1 | True | True |
| t19 | fact | answered | 1 | True | True |
| t20 | fact | answered | 2 | True | True |
| t21 | code | answered | 4 | True | True |
| t22 | code | answered | 2 | True | True |
| t23 | code | answered | 2 | True | True |
| t24 | code | answered | 1 | True | True |
| t25 | code | answered | 2 | True | True |
| t26 | code | answered | - | False | True |
| t27 | code | answered | 1 | True | True |
| t28 | code | answered | 2 | True | True |
| t29 | cross | answered | - | False | False |
| t30 | cross | answered | - | False | True |
| t31 | cross | answered | 5 | True | True |
| t32 | cross | answered | - | False | True |
| t33 | cross | answered | 5 | True | True |
| t34 | cross | answered | - | False | False |
| t35 | absent | refused | - | - | True |
| t36 | absent | refused | - | - | True |
| t37 | absent | refused | - | - | True |
| t38 | absent | answered | - | - | False |
| t39 | absent | refused | - | - | True |
| t40 | absent | refused | - | - | True |

Small sample: one answerable question moves accuracy by 2.94 points; no statistical significance is claimed.
