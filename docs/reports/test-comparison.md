# Comparison on split `test` (40 questions)

Same index generation `ac3e0f44511b6a39`, questions, LLM and budgets for every column.

| Metric | vector | hybrid | hybrid-rerank |
| --- | --- | --- | --- |
| Evidence Hit@1 | 14/34 (41.2%) | 18/34 (52.9%) | 27/34 (79.4%) |
| Evidence Hit@5 | 28/34 (82.4%) | 29/34 (85.3%) | 31/34 (91.2%) |
| MRR@5 | 0.591 | 0.666 | 0.848 |
| Answer accuracy | 31/34 (91.2%) | 32/34 (94.1%) | 31/34 (91.2%) |
| Citation precision | N/A | N/A | N/A |
| Citation coverage | N/A | N/A | N/A |
| Correct refusal | 5/6 (83.3%) | 5/6 (83.3%) | 5/6 (83.3%) |
| False refusal | 1/34 (2.9%) | 0/34 (0.0%) | 1/34 (2.9%) |
| Error rate | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) |
| Warm latency p50 | 2.79s | 2.79s | 9.72s |
| Warm latency p95 | 3.82s | 4.16s | 11.39s |

## Per question (hit rank / correct)

| Question | Type | vector | hybrid | hybrid-rerank |
| --- | --- | --- | --- | --- |
| t01 | fact | - / False | 2 / True | 1 / True |
| t02 | fact | 2 / True | 2 / True | 1 / True |
| t03 | fact | 1 / True | 1 / True | 1 / True |
| t04 | fact | 2 / True | 1 / True | 1 / True |
| t05 | fact | 1 / True | 1 / True | 1 / True |
| t06 | fact | 1 / True | 1 / True | 1 / True |
| t07 | fact | 1 / True | 1 / True | 1 / True |
| t08 | fact | 3 / True | 1 / True | 1 / True |
| t09 | fact | 1 / True | 1 / True | 1 / True |
| t10 | fact | 4 / True | 2 / True | 1 / True |
| t11 | fact | 1 / True | 1 / True | 1 / True |
| t12 | fact | 1 / True | 1 / True | 1 / True |
| t13 | fact | 1 / True | 1 / True | 1 / True |
| t14 | fact | 1 / True | 1 / True | 1 / False |
| t15 | fact | 1 / True | 1 / True | 1 / True |
| t16 | fact | 2 / True | 1 / True | 1 / True |
| t17 | fact | 1 / True | 1 / True | 1 / False |
| t18 | fact | 2 / True | 1 / True | 1 / True |
| t19 | fact | 3 / True | 1 / True | 1 / True |
| t20 | fact | 1 / True | 2 / True | 1 / True |
| t21 | code | - / False | 4 / True | 1 / True |
| t22 | code | 2 / True | 2 / True | 1 / True |
| t23 | code | 2 / True | 2 / True | 2 / True |
| t24 | code | 1 / True | 1 / True | 1 / True |
| t25 | code | 1 / True | 2 / True | 2 / True |
| t26 | code | - / True | - / True | 3 / True |
| t27 | code | 2 / True | 1 / True | 1 / True |
| t28 | code | - / False | 2 / True | 1 / True |
| t29 | cross | 3 / True | - / False | 1 / True |
| t30 | cross | - / True | - / True | - / True |
| t31 | cross | 2 / True | 5 / True | 1 / True |
| t32 | cross | - / True | - / True | - / True |
| t33 | cross | 2 / True | 5 / True | - / False |
| t34 | cross | 3 / True | - / False | 2 / True |
| t35 | absent | - / True | - / True | - / True |
| t36 | absent | - / True | - / True | - / True |
| t37 | absent | - / True | - / True | - / True |
| t38 | absent | - / False | - / False | - / False |
| t39 | absent | - / True | - / True | - / True |
| t40 | absent | - / True | - / True | - / True |
