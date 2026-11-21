"""1단계(차량 전체 사진 → 부품 위치) 불변식: 안내 전용, DB·kg 불변, 계획 품목 검증, 미검출 시 차단하지 않음."""
import pytest

from services import db, part_locator
from services import production_plan as pp
from services.config import ServiceError
from services.inspection_registry import unit_buckets


def _fake(found: dict):
    def f(_b):
        return {"status": "OK", "model_version": "fake", "detections": [], "overlay": None,
                "parts": {p: {"present": found.get(p, False), "max_score": 0.9 if found.get(p) else 0.0, "area_frac": 0.1}
                          for p in ("DOOR", "BUMPER", "FENDER")}}
    return f


def _snapshot():
    with db.connect() as con:
        return (con.execute("SELECT COUNT(*) FROM inspection").fetchone()[0],
                con.execute("SELECT COUNT(*) FROM stock_ledger").fetchone()[0])


def test_rejects_non_image_plan():
    p = pp.create_production_plan("CN7_LH", "2026-11-12", 5000)
    with pytest.raises(ServiceError) as e:
        part_locator.locate_plan_part(p["plan_id"], b"x")
    assert e.value.code == "PRODUCT_MISMATCH"


def test_found_guides_and_changes_nothing(monkeypatch):
    p = pp.create_production_plan("FENDER", "2026-11-12", 100)
    before, b0 = _snapshot(), unit_buckets(p["plan_id"])
    monkeypatch.setattr(part_locator, "_locate", _fake({"FENDER": True, "DOOR": True}))
    r = part_locator.locate_plan_part(p["plan_id"], b"img")
    assert r["plan_part_found"] is True and r["kg_changed"] is False
    assert "PARTSEG_GUIDE_ONLY" in r["provenance"] and "휀더" in r["guidance"]
    assert _snapshot() == before and unit_buckets(p["plan_id"]) == b0


def test_missing_part_warns_not_blocks(monkeypatch):
    p = pp.create_production_plan("BUMPER", "2026-11-12", 100)
    monkeypatch.setattr(part_locator, "_locate", _fake({"DOOR": True}))
    r = part_locator.locate_plan_part(p["plan_id"], b"img")
    assert r["plan_part_found"] is False
    assert "찾지 못했습니다" in r["guidance"] and "도어" in r["guidance"] and "차단하지 않음" in r["guidance"]


def test_model_not_ready_falls_back(monkeypatch):
    p = pp.create_production_plan("DOOR", "2026-11-12", 100)
    monkeypatch.setattr(part_locator, "_locate", lambda b: {"status": "MODEL_NOT_READY", "message": "없음"})
    r = part_locator.locate_plan_part(p["plan_id"], b"img")
    assert r["status"] == "MODEL_NOT_READY" and "바로 검사" in r["guidance"] and _snapshot() == (0, 0)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CAR = ROOT / "data/demo/photos/car/car_1.jpg"
CLOSE = ROOT / "data/demo/photos/closeup/fender_defect.jpg"


@pytest.mark.skipif(not ((ROOT / "models/vision/partseg/maskrcnn_partseg.pt").exists() and CAR.exists() and CLOSE.exists()),
                    reason="부품 분할 모델 또는 시연 사진 없음")
def test_two_stage_flow_real_models():
    """① 전체 사진으로 휀더 위치 확인(기록 없음) → ② 근접 사진 판정(기록 1건, kg 불변) → 작업자 확정 시에만 ledger."""
    from services.inspection_registry import inspect_part_image, record_operator_decision
    p = pp.create_production_plan("FENDER", "2026-11-12", 100)
    r1 = part_locator.locate_plan_part(p["plan_id"], CAR.read_bytes())
    assert r1["status"] == "OK" and r1["plan_part_found"] and _snapshot() == (0, 0)
    r2 = inspect_part_image(p["plan_id"], "U1", CLOSE.read_bytes())
    assert r2["ai_result"]["status"] == "OK" and _snapshot() == (1, 0)
    record_operator_decision(p["plan_id"], "U1", "CONFIRMED_SCRAP", "op1")
    assert _snapshot() == (1, 1)


def test_empty_image_rejected():
    p = pp.create_production_plan("DOOR", "2026-11-12", 100)
    with pytest.raises(ServiceError):
        part_locator.locate_plan_part(p["plan_id"], b"")
