"""Checks against the running offline Compose stack (`docker compose up -d --wait`)."""
import json
import subprocess

import pytest
import requests

pytestmark = pytest.mark.integration


def sh(*args, timeout=600):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def in_app(code):
    return sh("docker", "compose", "exec", "-T", "app", "python", "-c", code)


@pytest.fixture(scope="module", autouse=True)
def stack_running():
    out = sh("docker", "compose", "ps", "--format", "json").stdout
    names = {json.loads(line)["Service"] for line in out.splitlines() if line.strip()}
    if not {"app", "ollama", "ui"} <= names:
        pytest.skip("compose stack is not running")


def test_ui_reachable_on_localhost_only():
    assert requests.get("http://127.0.0.1:8501/_stcore/health", timeout=10).text == "ok"
    ports = sh("docker", "compose", "port", "ui", "8501").stdout.strip()
    assert ports.startswith("127.0.0.1:")


def test_app_and_ollama_have_no_internet_and_only_the_internal_network():
    for service in ("app", "ollama"):
        cid = sh("docker", "compose", "ps", "-q", service).stdout.strip()
        nets = json.loads(sh("docker", "inspect", "-f", "{{json .NetworkSettings.Networks}}", cid).stdout)
        assert list(nets) == ["onprem-rag_runtime"]
    probe = ("import urllib.request,sys\n"
             "try:\n    urllib.request.urlopen('https://huggingface.co', timeout=6); print('ONLINE')\n"
             "except Exception as e:\n    print('BLOCKED', type(e).__name__)")
    assert "BLOCKED" in in_app(probe).stdout


def test_query_through_container_pipeline_three_modes():
    code = ("import json\nfrom src import pipeline, retriever\nfrom src.config import load_config\n"
            "cfg=load_config(); gen=retriever.load_generation(cfg.index_root); res=pipeline.load_resources(cfg)\n"
            "out={m: pipeline.answer('Chứng chỉ kiểm định viên có thời hạn bao lâu?', m, cfg, res, gen) for m in ('vector','hybrid','hybrid-rerank')}\n"
            "print(json.dumps({m:[r['status'], r['generation_id'], ' '.join(c['text'] for c in r['claims'])] for m,r in out.items()}, ensure_ascii=False))")
    result = json.loads(in_app(code).stdout.strip().splitlines()[-1])
    for mode, (status, _gen, text) in result.items():
        assert status == "answered" and "05 năm" in text, (mode, status, text)


def test_restart_keeps_model_and_index_without_downloads():
    before = sh("docker", "compose", "exec", "-T", "app", "cat", "/app/data/index/active.json").stdout
    digest = sh("docker", "compose", "exec", "-T", "ollama", "ollama", "list").stdout
    assert sh("docker", "compose", "restart").returncode == 0
    assert sh("docker", "compose", "up", "-d", "--no-build", "--pull", "never", "--wait").returncode == 0
    assert sh("docker", "compose", "exec", "-T", "app", "cat", "/app/data/index/active.json").stdout == before
    assert sh("docker", "compose", "exec", "-T", "ollama", "ollama", "list").stdout.split()[5] == digest.split()[5]
