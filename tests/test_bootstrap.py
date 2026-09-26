import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("bootstrap_models", Path(__file__).parent.parent / "scripts" / "bootstrap-models.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_qwen_download_is_tokenizer_only_and_revisions_are_pinned():
    from src.config import load_config

    items = bootstrap.plan(load_config(env={}))
    qwen = items[-1]
    assert qwen["repo"] == "Qwen/Qwen3-4B"
    assert not any(p.endswith(bootstrap.WEIGHT_SUFFIXES) for p in qwen["allow_patterns"])
    assert all(len(i["revision"]) == 40 for i in items)  # immutable commit, never a branch name


def test_main_downloads_each_item_once(monkeypatch):
    calls = []
    monkeypatch.setattr("os.environ", {})
    assert bootstrap.main(download=lambda repo, **kw: calls.append((repo, kw)) or "/cache/x") == 0
    assert [c[0] for c in calls] == ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", "Qwen/Qwen3-4B"]
    assert calls[0][1]["allow_patterns"] is None and "tokenizer.json" in calls[2][1]["allow_patterns"]
