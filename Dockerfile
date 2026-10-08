FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 HF_HOME=/cache/hf \
    HF_HUB_DISABLE_TELEMETRY=1 ANONYMIZED_TELEMETRY=False

COPY requirements.txt .
RUN uv pip install --system --no-cache --torch-backend cpu -r requirements.txt

COPY src ./src
COPY eval ./eval
COPY scripts ./scripts
COPY .streamlit ./.streamlit

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
