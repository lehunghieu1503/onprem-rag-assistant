"""Online, one-off download of model assets. Runtime never downloads.

Full snapshots for the embedding and reranker models; tokenizer/config files only for Qwen
(the LLM weights are served by Ollama, not Hugging Face).

Run from the project root:  env PYTHONPATH=. python scripts/bootstrap-models.py
"""
from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from src.config import load_config

TOKENIZER_ONLY = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json", "added_tokens.json", "config.json", "chat_template.jinja"]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".onnx", ".pt")
SKIP_FOR_FULL = ["onnx/*", "*.onnx", "*.onnx_data", "imgs/*", "*.jpg", "*.png", "*.DS_Store"]


def plan(cfg) -> list:
    return [
        {"repo": cfg.embed_model, "revision": cfg.embed_revision, "ignore_patterns": SKIP_FOR_FULL},
        {"repo": cfg.rerank_model, "revision": cfg.rerank_revision, "ignore_patterns": SKIP_FOR_FULL},
        {"repo": cfg.llm_tokenizer, "revision": cfg.llm_tokenizer_revision, "allow_patterns": TOKENIZER_ONLY},
    ]


def main(download=snapshot_download) -> int:
    cfg = load_config()
    for item in plan(cfg):
        patterns = item.get("allow_patterns")
        if patterns and any(p.endswith(WEIGHT_SUFFIXES) for p in patterns):
            raise SystemExit(f"refusing to download weights for {item['repo']}")
        path = download(item["repo"], revision=item["revision"], allow_patterns=patterns,
                        ignore_patterns=item.get("ignore_patterns"))
        print(f"{item['repo']}@{item['revision']} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
