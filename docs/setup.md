# Setup

Two phases: an **online bootstrap** (downloads everything once) and an **offline runtime** (never downloads).

## Native (development, evaluation)

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python --torch-backend cpu -r requirements-dev.txt   # CPU torch, no CUDA wheels
ollama pull qwen3:4b-q4_K_M
env PYTHONPATH=. .venv/bin/python scripts/bootstrap-models.py     # BGE-M3, reranker, Qwen tokenizer only

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                    # from here on nothing is downloaded
.venv/bin/python -m src.ingest                                    # data/raw/*.pdf + data/sources.jsonl -> data/index
.venv/bin/python -m src.pipeline --question "Thời hạn kiểm định định kỳ của cần trục tự hành là bao lâu?"
.venv/bin/python -m streamlit run src/app.py --server.address=127.0.0.1
.venv/bin/python -m pytest -m "not integration"
```

PDFs are not committed. `data/sources.jsonl` lists URL and SHA-256 for each file; download them into `data/raw/`
under the listed file names. Ingest skips any file whose checksum differs.

If your shell exports a foreign `PYTHONPATH` (for example ROS), run pytest as
`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest`.

### Reranker on GPU (optional, native only)

```bash
uv pip install --python .venv/bin/python --torch-backend auto -r requirements-gpu.txt   # CUDA build of torch
export RAG_RERANK_DEVICE=cuda
```

Runs the reranker in fp16 beside the LLM. On the 8 GB test laptop this cut reranking from about 6.8 s to 0.2 s per
query with identical top-5 results, but left little free VRAM. The Docker image stays CPU-only.

## Docker Compose

```bash
export COMPOSE_FILE=docker-compose.yml:docker-compose.bootstrap.yml    # online bootstrap
docker compose build app
docker compose pull ollama ui
docker compose up -d --wait --wait-timeout 180 ollama
docker compose exec ollama ollama pull qwen3:4b-q4_K_M
docker compose run --rm --no-deps app python scripts/bootstrap-models.py
docker compose run --rm --no-deps app python -m src.ingest
docker compose down                         # keeps volumes; never use `down -v`

unset COMPOSE_FILE                          # offline runtime, UI at http://127.0.0.1:8501
docker compose up -d --no-build --pull never --wait
# LLM on GPU instead:
# COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml docker compose up -d --no-build --pull never --wait
```

Runtime network layout: `app` and `ollama` are attached only to an `internal: true` network and have no route
to the Internet. Docker does not publish ports for containers on internal-only networks, so a third container,
`ui` (socat), forwards `127.0.0.1:8501` to `app:8501`. It holds no data or models.

Stop the host Ollama model (`ollama stop qwen3:4b-q4_K_M`) before using the GPU override: both would share 8 GB VRAM.

## Configuration

Every setting is an environment variable `RAG_<NAME>`; names and defaults are the fields of `Config` in
[`src/config.py`](../src/config.py). Changing chunking or embedding settings requires a new ingest; changing
`RAG_TEMPERATURE`, `RAG_CONTEXT_K` and other query-time settings does not.

## Status and error codes

A query ends as `answered`, `refused` (model found no evidence, or retrieval returned nothing) or `error`.
Errors are never shown as answers: `ollama_unreachable`, `model_missing`, `timeout`, `input_too_long`,
`output_truncated`, `invalid_model_output`, `invalid_citation`, `input_error`.
