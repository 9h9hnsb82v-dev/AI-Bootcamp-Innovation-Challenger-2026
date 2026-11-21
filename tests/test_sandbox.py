"""체험 모드(ui/sandbox.py): 세션마다 빈 DB(생산계획 0건) — 한 세션의 변경이 원본·다른 세션에 보이지 않는다."""
import sqlite3

from streamlit.testing.v1 import AppTest

APP = """
import streamlit as st
from ui.sandbox import enable
enable()
from services import db
from services.production_plan import create_production_plan, list_plans
if st.button("add"):
    create_production_plan("DOOR", "2026-12-01", 777, user_confirmed=True)
st.write(f"PLANS={len(list_plans())}")
st.write(f"DB={db.db_path().name}|{db.db_path().parent.name}")
"""


def _plans(at):
    return next(m.value for m in at.markdown if m.value.startswith("PLANS="))


def _seed(tmp_path, monkeypatch):
    seed = tmp_path / "resource.db"
    monkeypatch.setenv("RESOURCE_DB", str(seed))
    from services.production_plan import create_production_plan
    create_production_plan("FENDER", "2026-11-30", 5000, user_confirmed=True)
    return seed


def test_sessions_are_isolated(tmp_path, monkeypatch):
    seed = _seed(tmp_path, monkeypatch)
    monkeypatch.setenv("RESOURCE_SANDBOX", "1")
    a = AppTest.from_string(APP).run()
    b = AppTest.from_string(APP).run()
    assert _plans(a) == _plans(b) == "PLANS=0"                # 원본에 1건이 있어도 0건에서 시작
    [x for x in a.button if x.label == "add"][0].click(); a.run()  # 세션 A만 계획 추가
    assert _plans(a) == "PLANS=1"
    b.run()
    assert _plans(b) == "PLANS=0"                      # 세션 B는 그대로
    with sqlite3.connect(seed) as con:                  # 원본도 그대로
        assert con.execute("SELECT COUNT(*) FROM production_plan").fetchone()[0] == 1
    assert (tmp_path / "sandbox").exists()


def test_reset_button_restores_seed(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    monkeypatch.setenv("RESOURCE_SANDBOX", "1")
    a = AppTest.from_string(APP).run()
    [b for b in a.button if b.label == "add"][0].click(); a.run()
    assert _plans(a) == "PLANS=1"
    [b for b in a.sidebar.button if b.key == "_sandbox_reset"][0].click(); a.run()
    assert _plans(a) == "PLANS=0"


def test_off_by_default_uses_shared_db(tmp_path, monkeypatch):
    seed = _seed(tmp_path, monkeypatch)
    monkeypatch.delenv("RESOURCE_SANDBOX", raising=False)
    a = AppTest.from_string(APP).run()
    [b for b in a.button if b.label == "add"][0].click(); a.run()
    with sqlite3.connect(seed) as con:
        assert con.execute("SELECT COUNT(*) FROM production_plan").fetchone()[0] == 2
