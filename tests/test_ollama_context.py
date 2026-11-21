"""챗봇 컨텍스트 관리(2026-11-18): 보낼 기록 줄이기·넘침 표시·오류 처리. Ollama 없이 돈다."""
import json

import pytest
import requests

import agent.ollama_client as oc


def _long_history(n_turns: int, tool_chars: int = 2500) -> list[dict]:
    h = []
    for i in range(n_turns):
        h += [{"role": "user", "content": f"질문 {i} " + "보관량과 수거 일정 " * 20},
              {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_storage_projection", "arguments": {}}}]},
              {"role": "tool", "content": json.dumps({"status": "OK", "summary_ko": f"요약 {i}", "rows": "x" * tool_chars},
                                                     ensure_ascii=False), "tool_name": "get_storage_projection"},
              {"role": "assistant", "content": f"답변 {i} " + "보관량은 충분합니다. " * 15},
              {"role": "assistant", "content": f"**근거(Python 도구 계산 결과)**\n- 요약 {i}"}]
    return h


@pytest.mark.parametrize("limit", [4096, 8192])
def test_compact_fits_budget_and_keeps_current_question(limit):
    hist = _long_history(30) + [{"role": "user", "content": "지금 질문"}]
    sent, info = oc.compact(hist, limit)
    assert sent[-1]["content"] == "지금 질문"
    assert info["est_tokens"] <= limit - oc.REPLY_RESERVE and not info["over_budget"]
    assert info["dropped_turns"] > 0
    # 이전 질문의 도구 메시지·빈 도구 호출 메시지는 보내지 않는다(근거 요약은 남김)
    assert not any(m["role"] == "tool" or m.get("tool_calls") for m in sent[:-1])
    assert any("근거" in m["content"] for m in sent[:-1])
    # 가장 최근 질문부터 남긴다
    users = [m["content"] for m in sent[:-1] if m["role"] == "user"]
    assert users and users[-1].startswith("질문 29") and len(users) == 30 - info["dropped_turns"]


def test_compact_short_history_untouched():
    hist = [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요"}, {"role": "user", "content": "범퍼 1000개?"}]
    sent, info = oc.compact(hist, 8192)
    assert sent == hist and info["dropped_turns"] == 0 and info["shrunk_tool_results"] == 0


def test_compact_shrinks_huge_tool_results_in_current_turn():
    cur = [{"role": "user", "content": "출처 다 보여줘"}]
    for _ in range(4):
        cur += [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_provenance", "arguments": {}}}]},
                {"role": "tool", "content": json.dumps({"status": "OK", "summary_ko": "출처 요약", "rows": "가" * 3000},
                                                       ensure_ascii=False), "tool_name": "get_provenance"}]
    sent, info = oc.compact(cur, 4096)
    assert info["shrunk_tool_results"] >= 1 and not info["over_budget"]
    assert any("출처 요약" in m["content"] and len(m["content"]) <= 600 for m in sent if m["role"] == "tool")
    assert len(cur[2]["content"]) > 3000  # 원본은 바꾸지 않는다


class _Resp:
    def __init__(self, lines):
        self.lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(json.dumps(x).encode() for x in self.lines)


def test_stream_sends_num_ctx_and_returns_full_history(monkeypatch):
    seen = {}

    def fake_post(url, json=None, **kw):
        seen.update(json)
        return _Resp([{"message": {"content": "답"}}, {"done": True, "prompt_eval_count": 8150}])

    monkeypatch.setattr(oc, "available", lambda: True)
    monkeypatch.setattr(oc.requests, "post", fake_post)
    monkeypatch.setattr(oc, "NUM_CTX", 8192)
    hist = _long_history(30) + [{"role": "user", "content": "지금 질문"}]
    done = [v for e, v in oc.chat_stream(hist) if e == "done"][0]
    assert seen["options"]["num_ctx"] == 8192
    assert len(seen["messages"]) < len(hist)                       # 모델에는 줄여서
    assert done["messages"][:len(hist)] == hist                    # 화면 기록은 전부 유지
    assert done["messages"][-1]["content"] == "답"
    assert done["context"]["truncated"] is True                    # 실제 읽은 토큰이 한도에 닿음


def test_num_ctx_zero_uses_ollama_default(monkeypatch):
    monkeypatch.setattr(oc, "NUM_CTX", 0)
    assert "num_ctx" not in oc._options() and oc.ctx_limit() == oc.OLLAMA_DEFAULT_CTX


@pytest.mark.parametrize("exc", [requests.Timeout("t"), requests.ConnectionError("c")])
def test_stream_errors_become_message(monkeypatch, exc):
    monkeypatch.setattr(oc, "available", lambda: True)
    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: (_ for _ in ()).throw(exc))
    hist = [{"role": "user", "content": "안녕"}]
    done = [v for e, v in oc.chat_stream(hist) if e == "done"][0]
    assert done["status"] == "OLLAMA_ERROR" and done["messages"] == hist and "다시 답변" in done["message"]


def test_stream_error_chunk_from_ollama(monkeypatch):
    monkeypatch.setattr(oc, "available", lambda: True)
    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: _Resp([{"error": "model requires more system memory"}]))
    done = [v for e, v in oc.chat_stream([{"role": "user", "content": "안녕"}]) if e == "done"][0]
    assert done["status"] == "OLLAMA_ERROR"
