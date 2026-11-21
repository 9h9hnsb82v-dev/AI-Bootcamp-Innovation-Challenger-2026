"""공장 배치도: 기본 배치, 사람별 저장, 값 검사, 화면 렌더."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import factory_layout as fl
from services.config import ServiceError

ROOT = Path(__file__).resolve().parents[1]


def test_default_and_per_owner_save_reset():
    base = fl.get_layout("alice")
    assert base["saved"] is False and base["width_m"] == 80 and len(base["items"]) >= 9
    moved = {**base, "items": [{**base["items"][0], "x": 40, "rot": 45}] + base["items"][1:]}
    moved["items"].append({"id": "new", "type": "custom", "label": "파쇄기", "x": 10, "y": 10, "w": 4, "h": 4, "rot": 0})
    fl.save_layout("alice", moved)
    a, b = fl.get_layout("alice"), fl.get_layout("bob")
    assert a["saved"] and a["items"][0]["x"] == 40 and a["items"][0]["rot"] == 45 and a["items"][-1]["label"] == "파쇄기"
    assert b["saved"] is False and b["items"][0]["x"] == base["items"][0]["x"]  # 사람마다 따로
    fl.reset_layout("alice")
    assert fl.get_layout("alice")["saved"] is False


def test_validation_rejects_out_of_bounds_and_tiny():
    base = fl.default_layout()
    with pytest.raises(ServiceError):
        fl.save_layout("x", {**base, "items": [{"id": "a", "type": "line", "label": "a", "x": 500, "y": 0, "w": 5, "h": 5, "rot": 0}]})
    with pytest.raises(ServiceError):
        fl.save_layout("x", {**base, "items": [{"id": "a", "type": "line", "label": "a", "x": 1, "y": 1, "w": 0.1, "h": 5, "rot": 0}]})
    with pytest.raises(ServiceError):
        fl.save_layout(" ", base)


def test_page_renders_with_live_storage():
    at = AppTest.from_file(str(ROOT / "ui/pages/9_factory_layout.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("보관량" in str(m.label) for m in at.metric)


def test_room_projection_matches_total_and_storage_page():
    from datetime import date
    from services import production_plan as pp
    from services.storage_projection import get_storage_projection
    from services.storage_rooms import room_projection, schedule_collection, list_collections, cancel_collection
    pp.create_production_plan("DOOR", date.today().isoformat(), 1000)
    pp.create_production_plan("CN7_LH", date.today().isoformat(), 5000)
    rp, tot = room_projection(days=7), get_storage_projection(days=7)
    for a, b in zip(rp["days"], tot["projection"]):
        assert abs(a["total_kg"] - b["end_kg"]) < 0.5
    sch = schedule_collection(date.today().isoformat(), 100.0)
    assert any(c["source_id"] == sch["source_id"] for c in list_collections())
    cancel_collection(sch["source_id"])
    assert not list_collections()
    at = AppTest.from_file(str(ROOT / "ui/pages/6_storage.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    at.radio(key="room_view").set_value("3D 입체").run()
    assert not at.exception, at.exception


def test_evidence_and_chatbot_pages_render():
    for pg in ("11_evidence.py", "8_chatbot.py"):
        at = AppTest.from_file(str(ROOT / "ui/pages" / pg), default_timeout=60).run()
        assert not at.exception, (pg, at.exception)
