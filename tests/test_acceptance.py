"""MASTER_PROMPT §13 필수 자동화 수용 테스트."""
from datetime import date, timedelta

import pytest

from services import inspection_registry as ir
from services import production_plan as pp
from services import supply_to_plan_draft as sd
from services.config import ServiceError, defect_rate, get_product, products
from services.storage_projection import project_storage
from services.waste_forecast import expected_scrap, predict_preproduction_waste
from services.waste_forecast_update import compute_update


# 0. 승인 확정값 불변
def test_product_master_values_unchanged():
    exp = {"CN7_LH": (1.30, None), "CN7_RH": (1.35, None), "RG3_LH": (1.45, None), "RG3_RH": (1.50, None),
           "DOOR": (5.00, 0.015), "BUMPER": (3.50, 0.020), "FENDER": (2.80, 0.025)}
    ps = products()
    assert set(ps) == set(exp)
    for k, (w, r) in exp.items():
        assert ps[k]["unit_weight_kg"] == w and ps[k]["planned_defect_rate"] == r


# 1. CN7_LH 5,000 × KAMP 학습구간 불량률 × 1.30
def test_cn7_lh_uses_kamp_train_baseline():
    r = expected_scrap("CN7_LH", 5000)
    rate = defect_rate("CN7_LH")
    assert r["provenance"][0] == "KAMP_TRAIN_BASELINE"
    assert "XGBOOST_SENSOR" not in r["provenance"]
    assert r["expected_scrap_kg"] == pytest.approx(5000 * rate["rate"] * 1.30)
    assert rate["sample"]["train_n"] > 0 and r["expected_scrap_kg_ci95"][0] <= r["expected_scrap_kg"]


# 2. DOOR/BUMPER/FENDER 각 1,000개 → 75/70/70, 합 215kg
def test_aihub_demo_products():
    kg = {p: expected_scrap(p, 1000)["expected_scrap_kg"] for p in ("DOOR", "BUMPER", "FENDER")}
    assert kg == pytest.approx({"DOOR": 75.0, "BUMPER": 70.0, "FENDER": 70.0})
    assert sum(kg.values()) == pytest.approx(215.0)


def _bumper_scenario():
    plan = pp.create_production_plan("BUMPER", "2026-11-12", 1000)
    for i in range(400):
        uid = f"U{i:04d}"
        ir.register_inspection(plan["plan_id"], uid, f"img-{uid}".encode(), None)
        ir.record_operator_decision(plan["plan_id"], uid, "CONFIRMED_SCRAP" if i < 12 else "CONFIRMED_GOOD", "op1")
    return plan


# 3. BUMPER 최초 70kg → 400 검사(12 폐기·388 양품), 600 미생산 → 84kg; 중복 이미지 집계 0
def test_bumper_update_84kg_and_no_double_count():
    plan = _bumper_scenario()
    assert predict_preproduction_waste(plan["plan_id"])["expected_scrap_kg"] == pytest.approx(70.0)
    u = compute_update(plan["plan_id"])
    assert u["updated_expected_total_kg"] == pytest.approx(84.0)
    assert u["buckets"] == {"CONFIRMED_SCRAP": 12, "CONFIRMED_GOOD": 388, "REWORK_PENDING": 0,
                            "PRODUCED_UNINSPECTED": 0, "UNPRODUCED": 600}
    # 같은 폐기 개체의 추가 사진 2장 + 같은 사진 재업로드
    ir.register_inspection(plan["plan_id"], "U0000", b"img-U0000-angle2", None)
    ir.register_inspection(plan["plan_id"], "U0000", b"img-U0000-angle3", None)
    again = ir.register_inspection(plan["plan_id"], "U0000", b"img-U0000", None)
    assert again["duplicate"] is True
    assert compute_update(plan["plan_id"])["updated_expected_total_kg"] == pytest.approx(84.0)
    with pytest.raises(ServiceError) as e:  # 같은 사진을 다른 개체로
        ir.register_inspection(plan["plan_id"], "U0001", b"img-U0000", None)
    assert e.value.code == "DUPLICATE_IMAGE"


# 4. 도어 이미지는 CN7/RG3 또는 범퍼 계획 kg를 갱신할 수 없다
def test_door_image_cannot_update_other_plans():
    cn7 = pp.create_production_plan("CN7_LH", "2026-11-12", 5000)
    bumper = pp.create_production_plan("BUMPER", "2026-11-12", 1000)
    door_ai = {"status": "OK", "predicted_part": "DOOR", "part_confidence": 0.97, "defect_prob": 0.9}
    for plan in (cn7, bumper):
        before = compute_update(plan["plan_id"])["updated_expected_total_kg"]
        with pytest.raises(ServiceError) as e:
            ir.register_inspection(plan["plan_id"], "U1", b"door-photo", door_ai)
        assert e.value.code == "PRODUCT_MISMATCH"
        assert compute_update(plan["plan_id"])["updated_expected_total_kg"] == pytest.approx(before)


def test_low_confidence_part_mismatch_flagged_without_kg_change():
    bumper = pp.create_production_plan("BUMPER", "2026-11-12", 1000)
    ai = {"status": "OK", "predicted_part": "DOOR", "part_confidence": 0.77, "defect_prob": 1.0}
    rec = ir.register_inspection(bumper["plan_id"], "U1", b"door-ish", ai)
    assert rec["review_status"] == "PART_MISMATCH_SUSPECTED"
    assert compute_update(bumper["plan_id"])["updated_expected_total_kg"] == pytest.approx(70.0)


# 5. AI_SUSPECTED_DEFECT 만으로 확인 폐기중량 불변
def test_ai_suspected_defect_does_not_add_confirmed_kg():
    plan = pp.create_production_plan("FENDER", "2026-11-12", 100)
    ai = {"status": "OK", "predicted_part": "FENDER", "part_confidence": 0.99, "defect_prob": 0.95,
          "threshold": 0.5, "model_version": "t"}
    rec = ir.register_inspection(plan["plan_id"], "U1", b"fender", ai)
    assert rec["review_status"] == "AI_SUSPECTED_DEFECT"
    u = compute_update(plan["plan_id"])
    assert u["confirmed_scrap_kg"] == 0
    assert u["updated_expected_total_kg"] == pytest.approx(100 * 0.025 * 2.8)


# 6. 모델/데이터 미확보 시 MODEL_NOT_READY / DATA_NOT_AVAILABLE
def test_model_not_ready_status(monkeypatch, tmp_path):
    import ml.vision.infer as vi
    monkeypatch.setattr(vi, "WEIGHTS", tmp_path / "none.pt")
    monkeypatch.setattr(vi, "YOLO_WEIGHTS", tmp_path / "none_yolo.pt")  # 어느 백엔드든 모델 없으면 MODEL_NOT_READY
    monkeypatch.setattr(vi, "ENS_EFF_WEIGHTS", tmp_path / "none_eff_full.pt")
    monkeypatch.setattr(vi, "ENS_YOLO_WEIGHTS", tmp_path / "none_yolo_full.pt")
    monkeypatch.setattr(vi, "ENSD_DIR", tmp_path / "none_ensemble_sd_dino")  # EfficientNet+DINO 조합(2026-11-18)
    assert vi.predict(b"x")["status"] == "MODEL_NOT_READY"
    plan = pp.create_production_plan("DOOR", "2026-11-12", 10)
    rec = ir.register_inspection(plan["plan_id"], "U1", b"door", None)
    assert rec["review_status"] == "MODEL_NOT_READY"
    from ml.supply import data as D
    from ml.supply.audit import audit_supply_dataset
    assert audit_supply_dataset(tmp_path / "missing.zip")["status"] == "DATA_NOT_AVAILABLE"
    with pytest.raises(FileNotFoundError):
        D.load_raw(tmp_path / "missing.zip")


# 7. KAMP ID40 CSV: 34,617×20, 117종, 타깃 확인
def test_id40_shape():
    from ml.supply import data as D
    df = D.load_raw()
    assert df.drop(columns=["DateTime_raw", "Date"]).shape == (34617, 20)
    assert df[D.KEY].nunique() == 117
    assert D.TARGET in df.columns and D.TARGET_SEMANTICS.startswith("PLANNED")


# 8. ID40 결과는 개수이며 kg와 자동 합산 불가, 미매핑 상태에서 계획 자동 생성 없음
def test_order_forecast_is_units_and_not_in_kg_ledger():
    from services.storage_projection import get_storage_projection
    from services.supply_forecast import forecast_order_quantity
    r = forecast_order_quantity("Product_c0", "2022-03-12", 1)
    assert "units" in r["unit"] and "kg" not in r.get("expected_scrap_kg", "")
    assert pp.list_plans() == []
    proj = get_storage_projection("2026-11-10", 7)
    assert proj["plans"] == [] and proj["stock_after_7_days_kg"] == pytest.approx(300.0)
    with pytest.raises(ServiceError) as e:
        pp.create_production_plan("Product_c0", "2026-11-12", 100)
    assert e.value.code == "UNKNOWN_PRODUCT"


# 9. Product_Number → BUMPER 자동 변환 차단, 경고 확인 + 초안 + 재확정 후에만 계획
def test_demo_bridge_requires_explicit_steps():
    from services.supply_forecast import forecast_order_quantity
    f = forecast_order_quantity("Product_c0", "2022-03-12", 1)
    with pytest.raises(ServiceError) as e:
        sd.create_demo_plan_draft_from_order_forecast(f["forecast_id"], "BUMPER", 400, "2026-11-15")
    assert e.value.code == "MAPPING_WARNING_NOT_ACKNOWLEDGED"
    with pytest.raises(ServiceError):
        pp.create_production_plan("BUMPER", "2026-11-15", 400, _source_forecast_id=f["forecast_id"])
    d = sd.create_demo_plan_draft_from_order_forecast(f["forecast_id"], "BUMPER", 400, "2026-11-15",
                                                      acknowledge_mapping_warning=True)
    assert d["approval_status"] == "DRAFT" and pp.list_plans() == []
    with pytest.raises(ServiceError):
        sd.confirm_demo_plan_draft(d["draft_id"])
    out = sd.confirm_demo_plan_draft(d["draft_id"], user_confirmed=True, acknowledge_mapping_warning=True)
    plans = pp.list_plans()
    assert len(plans) == 1 and plans[0]["source"] == "DEMO_SUPPLY_BRIDGE"
    assert plans[0]["source_forecast_id"] == f["forecast_id"] and plans[0]["planned_units"] == 400
    assert "DEMO_SYNTHETIC_BRIDGE" in predict_preproduction_waste(out["plan"]["plan_id"])["provenance"]


# 10. 지원 밖 as-of / horizon → 수치 날조 없이 오류
def test_unsupported_as_of_errors():
    from services.supply_forecast import forecast_order_quantity
    for args in [("Product_c0", "2026-11-10", 1), ("Product_c0", "2022-03-12", 7), ("CN7_LH", "2022-03-12", 1)]:
        with pytest.raises(ServiceError) as e:
            forecast_order_quantity(*args)
        assert e.value.code in {"UNSUPPORTED_AS_OF", "UNSUPPORTED_HORIZON", "UNKNOWN_SUPPLY_PRODUCT"}


# 11. 시계열 분할 누출 점검 + 타깃 명칭
def test_id40_no_temporal_leakage():
    import json
    import pandas as pd
    from ml.supply.features import build_training_table, original_snapshots
    from services.supply_forecast import MODEL_DIR, TARGET_DESC
    m = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    sp = {k: pd.to_datetime(v) for k, v in m["split"].items()}
    assert sp["train"].max() < sp["val"].min() and sp["val"].max() < sp["test"].min()
    snap = original_snapshots()
    for h in (1, 4):
        t = build_training_table(snap, h)
        assert (t.as_of == t.Date - pd.Timedelta(days=h)).all()
        # 피처 값이 as-of 이후 스냅샷에서 오지 않았는지: plan_j는 D-j(>= h) 스냅샷에서만
        row = t.dropna(subset=[f"plan_{h}"]).iloc[0]
        src = snap[(snap.Product_Number == row.Product_Number) & (snap.Date == row.Date - pd.Timedelta(days=h))]
        assert src[f"T+{h}일 예정 수주량"].iloc[0] == row[f"plan_{h}"]
        assert (t.last_T_age.dropna() >= 0).all()
    assert m["data_policy"].startswith("ORIGINAL")
    assert "확정 주문 실적 아님" in TARGET_DESC


# 12. 보관 300 + 신규 80 - 수거 50 = 330
def test_storage_balance():
    d = date(2026, 11, 10)
    out = project_storage(300, {d: 80}, {d: 50}, d, 1)
    assert out[0]["end_kg"] == pytest.approx(330.0)


def test_storage_no_double_count_confirmed_scrap():
    from services.storage_projection import get_storage_projection
    plan = _bumper_scenario()
    today = date.today().isoformat()
    proj = get_storage_projection(today, 7)
    # 확정 폐기 42kg(ledger, 오늘 입고) + 남은 예상 42kg = 84kg, 기준 300kg
    assert proj["stock_after_7_days_kg"] == pytest.approx(300 + 84.0)


# 13. Ollama 연결 불가여도 Python 계산 동작
def test_ollama_down_fallback(monkeypatch):
    import agent.ollama_client as oc
    from agent.tools import dispatch
    monkeypatch.setattr(oc, "BASE_URL", "http://127.0.0.1:9")
    assert oc.chat([{"role": "user", "content": "hi"}])["status"] == "OLLAMA_UNAVAILABLE"
    r = dispatch("predict_preproduction_waste", {"product_id": "DOOR", "planned_units": 1000})
    assert r["status"] == "OK" and r["result"]["expected_scrap_kg"] == pytest.approx(75.0)
    p = dispatch("create_production_plan", {"product_id": "DOOR", "plan_date": "2026-11-12", "planned_units": 10})
    assert p["status"] == "PENDING_USER_CONFIRMATION" and pp.list_plans() == []
