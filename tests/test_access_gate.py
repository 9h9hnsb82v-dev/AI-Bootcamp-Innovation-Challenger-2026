"""접속 비밀번호 화면(ui/access_gate.py): 비밀번호가 없으면 통과, 있으면 맞을 때만 통과."""
from streamlit.testing.v1 import AppTest

GATE_APP = """
from ui.access_gate import require_password
import streamlit as st
require_password()
st.write("INSIDE")
"""


def _texts(at):
    return " ".join(m.value for m in at.markdown) + " " + " ".join(str(e.value) for e in at.error)


def test_no_password_configured_passes(monkeypatch):
    monkeypatch.delenv("RESOURCE_APP_PASSWORD", raising=False)
    at = AppTest.from_string(GATE_APP).run()
    assert "INSIDE" in _texts(at)


def test_wrong_then_right_password(monkeypatch):
    monkeypatch.setenv("RESOURCE_APP_PASSWORD", "demo-pass-123")
    at = AppTest.from_string(GATE_APP).run()
    assert "INSIDE" not in _texts(at)
    at.text_input[0].input("wrong"); at.button[0].click(); at.run()
    assert "INSIDE" not in _texts(at) and "맞지 않습니다" in _texts(at)
    at.text_input[0].input("demo-pass-123"); at.button[0].click(); at.run()
    assert "INSIDE" in _texts(at)
