"""위탁 처리업체 추천: 폐합성수지류 허가 업체만, 공장에서 가까운 순, 화면 렌더."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import waste_processors as wp

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not wp.CSV.exists(), reason="처리업체 CSV 없음")


def test_recommend_sorted_and_filtered():
    r = wp.recommend(n=10)
    d = [it["직선거리_km"] for it in r["items"]]
    assert len(d) == 10 and d == sorted(d) and d[0] < 5  # 성서공단 인근
    assert all(it["업종명"] in wp.RECYCLE_TYPES for it in r["items"])
    assert r["factory"]["lat"] == 35.8557 and r["truck"]["height_m"] == 3.4


def test_page_renders():
    at = AppTest.from_file(str(ROOT / "ui/pages/10_processors.py"), default_timeout=60).run()
    assert not at.exception, at.exception


def test_tmap_parse_and_no_key_fallback(monkeypatch):
    from services import truck_route as tr
    resp = {"type": "FeatureCollection", "properties": {"totalDistance": "11200", "totalTime": "1140"},
            "features": [{"geometry": {"type": "Point", "coordinates": [128.48, 35.85]}, "properties": {"pointType": "S"}},
                         {"geometry": {"type": "LineString", "coordinates": [[128.48, 35.85], [128.40, 35.90]]},
                          "properties": {"distance": 11200, "time": 1140}}]}
    out = tr.parse(resp)
    assert out["distance_m"] == 11200 and out["time_s"] == 1140 and len(out["coords"]) == 2
    monkeypatch.delenv("TMAP_APP_KEY", raising=False)
    monkeypatch.setattr(tr, "app_key", lambda: None)
    assert tr.route((35.8557, 128.4889), (35.86, 128.47)) is None  # 키 없으면 None → 직선거리로 동작


def test_route_uses_cache_without_network(monkeypatch):
    from services import truck_route as tr
    from services import db
    import json
    a, b = (35.8557, 128.4889), (35.86, 128.47)
    with db.connect() as con:
        con.execute(tr.SCHEMA)
        con.execute("INSERT INTO route_cache VALUES (?,?,?,?,?,?)", (tr._key(a, b), 5000, 600, json.dumps([[1, 2], [3, 4]]), "t", "now"))
    monkeypatch.setattr(tr, "app_key", lambda: "dummy")
    r = tr.route(a, b)
    assert r["cached"] and r["distance_km"] == 5 and r["minutes"] == 10


def test_factory_location_set_get_reset_and_bounds():
    """공장 위치: 저장·조회·기본값 복귀, 국내 범위 밖 거부, 추천은 바뀐 위치 기준으로 다시 계산."""
    from services import factory_location as fl
    from services.config import ServiceError
    assert fl.get()["source"] == "DEFAULT" and fl.get()["lat"] == 35.8557
    with pytest.raises(ServiceError):
        fl.set_location(48.85, 2.35, "파리")  # 국외
    v = fl.set_location(37.5665, 126.9780, "서울 시청", "서울 중구 세종대로 110", source="TMAP_SEARCH")
    assert fl.get() == v and v["provenance"] == "USER_SET_FACTORY_LOCATION"
    r = wp.recommend(n=5)
    assert r["factory"]["name"] == "서울 시청" and r["items"][0]["직선거리_km"] < 30  # 서울 근처 업체가 먼저
    assert "USER_SET_FACTORY_LOCATION" in r["provenance"]
    assert fl.reset()["source"] == "DEFAULT" and fl.get()["source"] == "DEFAULT"


def test_page_manual_location_flow(monkeypatch):
    """화면: '위도·경도 직접 입력' → 설정 → 지표·추천이 새 위치로, '기본 위치로 되돌리기' 동작. TMAP 호출 없음."""
    from services import factory_location as fl
    monkeypatch.setattr(fl, "reverse_geocode", lambda lat, lon: "테스트 주소")
    at = AppTest.from_file(str(ROOT / "ui/pages/10_processors.py"), default_timeout=60).run()
    at.segmented_control(key="loc_how").set_value("위도·경도 직접 입력").run()
    at.text_input(key="man_name").set_value("테스트 공장")
    at.number_input(key="man_lat").set_value(37.5665)
    at.number_input(key="man_lon").set_value(126.9780)
    at.button(key="man_set").click().run()
    assert not at.exception, at.exception
    assert fl.get()["name"] == "테스트 공장" and fl.get()["source"] == "MANUAL"
    assert any(m.value == "테스트 공장" for m in at.metric)
    at.button(key="loc_reset").click().run()
    assert not at.exception and fl.get()["source"] == "DEFAULT"
