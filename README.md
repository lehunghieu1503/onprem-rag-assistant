# On-Premise RAG Assistant cho tài liệu Kỹ thuật & HSE

Trợ lý hỏi đáp chạy hoàn toàn trên máy cục bộ: đọc PDF tiếng Việt/tiếng Anh, tìm đoạn liên quan và trả lời
kèm nguồn đến từng trang. Không gọi API bên ngoài; sau bước bootstrap, ingest và truy vấn không cần Internet.

> Dự án portfolio cho một người dùng. Không phải công cụ tư vấn an toàn và chưa triển khai cho doanh nghiệp.

## Kiến trúc

```
INGEST   PDF ─▶ PyMuPDF (tiến trình con, có deadline) ─▶ làm sạch NFC ─▶ chunk theo trang (500 token, overlap 80)
             ─▶ BGE-M3 ─▶ Chroma (cosine)  +  chunks.jsonl ─▶ BM25
             ─▶ generation mới, kiểm tra xong mới đổi con trỏ active

QUERY    câu hỏi ─▶ vector top 20 ┐
                                  ├─▶ RRF (k=60) ─▶ top 10 ─▶ bge-reranker-v2-m3 ─▶ top 5
                    BM25 top 20  ─┘
             ─▶ Qwen3 4B qua Ollama, output JSON theo schema: từng khẳng định + ID nguồn
             ─▶ kiểm tra JSON/trích dẫn ─▶ answered | refused | error
```

| Thành phần | Lựa chọn |
| --- | --- |
| LLM | `qwen3:4b-q4_K_M` qua Ollama, GPU |
| Embedding / reranker | `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, CPU |
| Lưu trữ | ChromaDB (vector) + JSONL (văn bản gốc) + `rank_bm25` |
| Giao diện / đóng gói | Streamlit, Docker Compose |

Ba chế độ truy xuất dùng chung một pipeline: `vector`, `hybrid`, `hybrid-rerank`.

Một số quyết định thiết kế:

- **Lỗi không bao giờ được hiển thị như câu trả lời.** JSON sai, trích dẫn không hợp lệ, timeout, output bị cắt
  đều là `error`, tách khỏi `refused` (không đủ bằng chứng).
- **ID nguồn bị ràng buộc bằng schema.** Model chỉ có thể trích các nguồn có trong ngữ cảnh của chính câu hỏi đó.
- **Prompt quá dài bị server từ chối, không bị cắt ngầm** (`truncate: false`); số token đếm cục bộ khớp tuyệt đối
  với `prompt_eval_count` của Ollama trong các phép thử.
- **Index theo generation.** Ingest lỗi hoặc bị gián đoạn không ảnh hưởng index đang phục vụ.

## Kết quả đánh giá

Corpus: **19 PDF, 571 trang (558 trang có text dùng được), 812 chunk** — 12 tài liệu tiếng Anh (OSHA, HSE UK) và
7 văn bản tiếng Việt (QCVN, Nghị định, Thông tư, Công báo).
Bộ test: **40 câu độc lập** (34 có đáp án, 6 ngoài tài liệu), cấu hình chốt trên 20 câu dev trước khi chạy test.

| Chỉ số (test) | vector | hybrid | hybrid + rerank |
| --- | --- | --- | --- |
| Evidence Hit@1 | 14/34 (41,2%) | 18/34 (52,9%) | **27/34 (79,4%)** |
| Evidence Hit@5 | 28/34 (82,4%) | 29/34 (85,3%) | **31/34 (91,2%)** |
| MRR@5 | 0,591 | 0,666 | **0,848** |
| Answer accuracy¹ | 31/34 (91,2%) | 32/34 (94,1%) | 31/34 (91,2%) |
| Từ chối đúng (câu ngoài tài liệu) | 5/6 | 5/6 | 5/6 |
| Tỷ lệ lỗi hệ thống | 0/40 | 0/40 | 0/40 |
| Độ trễ warm, p50 / p95 | 2,8s / 3,8s | 2,8s / 4,2s | 9,7s / 11,4s |

¹ Chấm tạm thời bằng cách đối chiếu câu trả lời với đáp án; chưa được người chấm độc lập rà lại. Citation
precision/coverage chưa chấm.

Cách đọc kết quả:

- Hybrid và reranker cải thiện rõ **thứ hạng** của đoạn bằng chứng (Hit@1 từ 41% lên 79%).
- **Độ chính xác câu trả lời không tăng** trên bộ test này: cả ba chế độ sai 2–3 câu, khác câu nhau. Với 34 câu,
  một câu là 2,94 điểm %; chênh lệch 1 câu không có ý nghĩa thống kê.
- Reranker chạy CPU là phần chậm nhất (khoảng 7 giây cho 10 ứng viên).

Máy đo: Intel Core Ultra 9 275HX, 32 GB RAM, RTX 5060 Laptop 8 GB; LLM 100% trên GPU, RAM đỉnh của app 4,4 GB.
Báo cáo từng lượt chạy: [`docs/reports/`](docs/reports/). Phương pháp và rubric: [`docs/evaluation.md`](docs/evaluation.md).

## Chạy thử

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python --torch-backend cpu -r requirements-dev.txt
ollama pull qwen3:4b-q4_K_M
env PYTHONPATH=. .venv/bin/python scripts/bootstrap-models.py

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m src.ingest
.venv/bin/python -m src.pipeline --question "Thời hạn kiểm định định kỳ của cần trục tự hành là bao lâu?"
.venv/bin/python -m streamlit run src/app.py --server.address=127.0.0.1
```

PDF không nằm trong repo; tải theo URL và checksum trong [`data/sources.jsonl`](data/sources.jsonl).
Hướng dẫn đầy đủ, gồm Docker Compose offline: [`docs/setup.md`](docs/setup.md).

## Tài liệu

- [`docs/setup.md`](docs/setup.md) — cài đặt, Compose, cấu hình, mã lỗi
- [`docs/sources.md`](docs/sources.md) — nguồn tài liệu và điều kiện sử dụng
- [`docs/evaluation.md`](docs/evaluation.md) — bộ câu hỏi, rubric, quy trình
- [`docs/limitations.md`](docs/limitations.md) — giới hạn hiện tại

## Hướng mở rộng

OCR cho tài liệu scan · phân quyền theo người dùng · nhớ ngữ cảnh hội thoại · trích xuất bảng · reranker trên GPU ·
theo dõi hiệu lực văn bản.
