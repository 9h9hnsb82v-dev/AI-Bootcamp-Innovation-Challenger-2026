import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.environ.setdefault("RESOURCE_WARMUP", "0")  # 테스트에서는 백그라운드 모델 예열을 끈다


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("RESOURCE_DB", str(tmp_path / "test.db"))
    yield


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_chat_answer_cache():
    """챗봇 답 저장(2026-11-20)이 테스트 사이에 남아 다른 테스트 결과를 바꾸지 않도록 매번 비운다."""
    import agent.ollama_client as oc
    oc._ANSWERS.clear(); oc._INFLIGHT["n"] = 0
    yield
    oc._ANSWERS.clear(); oc._INFLIGHT["n"] = 0
