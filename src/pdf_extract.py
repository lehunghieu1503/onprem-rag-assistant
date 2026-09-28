"""Child-process PDF text extraction. Kept free of Torch/model imports on purpose.

Usage: python -m src.pdf_extract FILE  -> JSON on stdout: {"pages": [text, ...]} or {"error": reason}
"""
from __future__ import annotations

import json
import sys


def extract(path: str) -> dict:
    import pymupdf as fitz

    with open(path, "rb") as f:
        if b"%PDF" not in f.read(1024):
            return {"error": "open_failed", "detail": "not a PDF file"}
    try:
        doc = fitz.open(path)
    except Exception as e:  # corrupt / not a PDF
        return {"error": "open_failed", "detail": str(e)[:200]}
    try:
        if doc.needs_pass:
            return {"error": "encrypted"}
        return {"pages": [page.get_text("text") for page in doc]}
    except Exception as e:
        return {"error": "parse_failed", "detail": str(e)[:200]}
    finally:
        doc.close()


if __name__ == "__main__":
    json.dump(extract(sys.argv[1]), sys.stdout)
