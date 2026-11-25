"""공장 배치도: 기본 배치, 공장 배치 하나 저장, 보관함별 용량·채우기, 값 검사, 화면 렌더."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import db
from services import factory_layout as fl
from services.config import ServiceError, facility

ROOT = Path(__file__).resolve().parents[1]


def _with_bins(base, *caps):
    """기본 배치 + 보관함 추가(용량 caps[1:]). 첫 보관함 용량 = caps[0]."""
    items = [dict(i) for i in base["items"]]
    first = next(i for i in items if i["type"] == "scrap_bin")
    first["capacity_kg"] = caps[0]
    for n, c in enumerate(caps[1:], 2):
        items.append({"id": f"bin{n}", "type": "scrap_bin", "label": f"보관함 {n}", "x": 40, "y": 40, "w": 6, "h": 4,
                      "rot": 0, "capacity_kg": c})
    return {**base, "items": items}


def test_default_save_reset_single_factory_layout():
    fl.reset_layout()
    base = fl.get_layout()
    assert base["saved"] is False and base["width_m"] == 80 and len(base["items"]) >= 9
    assert [b["capacity_kg"] for b in fl.bins(base)] == [1000.0]
    moved = {**base, "items": [{**base["items"][0], "x": 40, "rot": 45}] + base["items"][1:]}
    moved["items"].append({"id": "new", "type": "custom", "label": "파쇄기", "x": 10, "y": 10, "w": 4, "h": 4, "rot": 0})
    fl.save_layout(moved)
    a = fl.get_layout()
    assert a["saved"] and a["items"][0]["x"] == 40 and a["items"][0]["rot"] == 45 and a["items"][-1]["label"] == "파쇄기"
    fl.reset_layout()
    assert fl.get_layout()["saved"] is False


def test_old_per_owner_layout_is_migrated():
    fl.reset_layout()
    old = fl.default_layout()
    for i in old["items"]:
        i.pop("capacity_kg", None)  # 용량 기능 이전 저장본
    old["items"][0]["x"] = 33
    with db.connect() as con:
        con.execute("INSERT INTO factory_layout (owner, layout_json, updated_at) VALUES (?,?,?)",
                    ("operator01", json.dumps(old, ensure_ascii=False), db.now()))
    got = fl.get_layout()
    assert got["saved"] and got["items"][0]["x"] == 33
    assert [b["capacity_kg"] for b in fl.bins(got)] == [1000.0]
    fl.reset_layout()


def test_split_bins_fills_front_first():
    assert fl.split_bins(1300, [1000, 1000, 1000]) == [1000, 300, 0]
    assert fl.split_bins(500, [1000, 800]) == [500, 0]
    assert fl.split_bins(2000, [1000, 800]) == [1000, 1000]  # 넘치는 200은 마지막 보관함(초과)
    assert fl.split_bins(-5, [1000]) == [0]
    assert fl.split_bins(100, []) == []


def test_bin_capacity_sum_drives_app_capacity():
    from services.storage_projection import get_storage_projection
    from services.storage_rooms import room_projection
    fl.reset_layout()
    assert facility()["max_capacity_kg"] == 1000.0
    fl.save_layout(_with_bins(fl.default_layout(), 1000, 800))
    try:
        assert facility()["max_capacity_kg"] == 1800.0
        s = get_storage_projection(days=7)
        assert s["max_capacity_kg"] == 1800.0 and abs(s["alert_kg"] - 1440.0) < 1e-6
        rp = room_projection(days=7)
        assert abs(sum(r["capacity_kg"] for r in rp["rooms"]) - 1800.0) < 0.5
        live = fl.live_status(days=7)
        assert [b["capacity_kg"] for b in live["bins"]] == [1000.0, 800.0]
        assert abs(sum(b["kg"] for b in live["bins"]) - live["scrap_kg"]) < 0.5
        k1, k2 = (b["kg"] for b in live["bins"])
        assert k2 == 0 or k1 == 1000.0  # 앞 보관함이 가득 차야 다음 보관함에 들어간다
    finally:
        fl.reset_layout()


def test_validation_rejects_out_of_bounds_tiny_and_bad_capacity():
    base = fl.default_layout()
    with pytest.raises(ServiceError):
        fl.save_layout({**base, "items": [{"id": "a", "type": "line", "label": "a", "x": 500, "y": 0, "w": 5, "h": 5, "rot": 0}]})
    with pytest.raises(ServiceError):
        fl.save_layout({**base, "items": [{"id": "a", "type": "line", "label": "a", "x": 1, "y": 1, "w": 0.1, "h": 5, "rot": 0}]})
    with pytest.raises(ServiceError):
        fl.save_layout(_with_bins(base, 1000, 0.5))
    with pytest.raises(ServiceError):
        fl.save_layout(_with_bins(base, "abc"))


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
