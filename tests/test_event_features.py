"""대회 체험 대비 기능(2026-11-20): 발표자 비밀번호 · 챗봇 답 저장 · 대기 안내."""

from streamlit.testing.v1 import AppTest

import agent.ollama_client as oc
from test_rag_guard import _fake_ollama  # tests 폴더는 패키지가 아니라 같은 폴더에서 가져옴

GATE_SANDBOX_APP = """
import streamlit as st
from ui.access_gate import require_password
require_password()
from ui.sandbox import enable
enable()
from services.production_plan import list_plans
st.write(f"PLANS={len(list_plans())}")
"""


def _plans(at):
    return next((m.value for m in at.markdown if m.value.startswith("PLANS=")), None)


def _login(at, pw):
    at.text_input[0].input(pw); at.button[0].click(); at.run()


def test_presenter_uses_shared_data_visitor_gets_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("RESOURCE_DB", str(tmp_path / "resource.db"))
    from services.production_plan import create_production_plan
    create_production_plan("FENDER", "2026-11-30", 5000, user_confirmed=True)
    monkeypatch.setenv("RESOURCE_SANDBOX", "1")
    monkeypatch.setenv("RESOURCE_APP_PASSWORD", "visit-1234")
    monkeypatch.setenv("RESOURCE_PRESENTER_PASSWORD", "stage-5678")
    p = AppTest.from_string(GATE_SANDBOX_APP).run(); _login(p, "stage-5678")
    v = AppTest.from_string(GATE_SANDBOX_APP).run(); _login(v, "visit-1234")
    assert _plans(p) == "PLANS=1"                       # 발표자: 팀 공용 시연 데이터
    assert _plans(v) == "PLANS=0"                       # 체험: 빈 데이터
    assert not any("체험 모드" in s.value for s in p.sidebar.info)   # 발표자: 사이드바 안내 없음
    assert any(s.value == "**체험 모드**" for s in v.sidebar.info)   # 체험: '체험 모드' 글씨만
    w = AppTest.from_string(GATE_SANDBOX_APP).run(); _login(w, "wrong")
    assert _plans(w) is None


# ── 챗봇 답 저장 ─────────────────────────────────────────────────────────────
def _reset_cache():
    oc._ANSWERS.clear(); oc._INFLIGHT["n"] = 0


def test_static_answer_is_cached(monkeypatch):
    _reset_cache()
    tc = [{"function": {"name": "search_docs", "arguments": {"query": "인계서 입력 기한"}}}]
    calls = _fake_ollama(monkeypatch, [("", tc), ("인계서는 10일 이내에 입력합니다.", None)])
    q = [{"role": "user", "content": "폐기물 인계서는 언제까지 입력해야 해?"}]
    first = oc.chat(q)
    n = len(calls["bodies"])
    second = oc.chat([{"role": "user", "content": "폐기물 인계서는  언제까지 입력해야 해?"}])  # 공백만 다름
    assert len(calls["bodies"]) == n                      # 모델을 다시 부르지 않음
    assert second["trace"]["cached"] and second["messages"][-1]["content"] == first["messages"][-1]["content"]


def test_user_data_answer_not_cached(monkeypatch):
    _reset_cache()
    tc = [{"function": {"name": "list_production_plans", "arguments": {}}}]
    calls = _fake_ollama(monkeypatch, [("", tc), ("등록된 계획은 없습니다.", None)])
    q = [{"role": "user", "content": "내 생산계획 보여줘"}]
    oc.chat(q); n = len(calls["bodies"])
    again = oc.chat(q)
    assert len(calls["bodies"]) > n and not again["trace"].get("cached")  # 사람마다 다른 데이터 → 저장 안 함


def test_followup_question_not_cached(monkeypatch):
    _reset_cache()
    _fake_ollama(monkeypatch, [("답", None)])
    hist = [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요"}, {"role": "user", "content": "또 안녕"}]
    oc.chat(hist)
    assert oc._ANSWERS == {}


def test_wait_notice_when_slots_busy(monkeypatch):
    _reset_cache()
    _fake_ollama(monkeypatch, [("답", None)])
    monkeypatch.setenv("RESOURCE_OLLAMA_SLOTS", "2")
    oc._INFLIGHT["n"] = 2                                  # 이미 2개 처리 중
    evs = list(oc.chat_stream([{"role": "user", "content": "대기 시험 질문"}]))
    assert any(e == "thinking" and "다른 사람의 질문 2개" in v for e, v in evs)
    assert oc._INFLIGHT["n"] == 2                          # 끝나면 내 몫은 빠짐
    oc._INFLIGHT["n"] = 0
