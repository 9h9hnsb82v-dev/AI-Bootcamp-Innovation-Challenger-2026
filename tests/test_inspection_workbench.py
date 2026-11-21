"""이미지 검사 작업대(C안) 화면 동작: 부품 → 계획 자동 연결, 다음 개체 번호, 확정 버튼 후 자동 이동, 차량 위치 탭."""
from datetime import date, timedelta
from pathlib import Path

from streamlit.testing.v1 import AppTest

from services import production_plan as pp
from services.inspection_registry import list_inspections

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "ui/pages/4_inspection.py")


def _app():
    return AppTest.from_file(PAGE, default_timeout=120).run()


def test_part_picks_todays_plan_over_other_dates():
    today = date.today()
    pp.create_production_plan("FENDER", (today - timedelta(days=3)).isoformat(), 50)
    todays = pp.create_production_plan("FENDER", today.isoformat(), 500)
    pp.create_production_plan("DOOR", today.isoformat(), 100)
    at = _app()
    assert not at.exception, at.exception
    at.segmented_control(key="insp_part").set_value("FENDER").run()
    assert not at.exception, at.exception
    assert any(todays["plan_id"] in str(c.value) for c in at.caption)


def test_decision_button_saves_and_moves_to_next_unit():
    plan = pp.create_production_plan("DOOR", date.today().isoformat(), 100)
    pp.set_produced_units(plan["plan_id"], 10)
    at = _app()
    key = f"insp_unit_{plan['plan_id']}"
    assert at.text_input(key=key).value == "U0001"
    at.button[[b.label for b in at.button].index("폐기 확정")].click().run()
    assert not at.exception, at.exception
    saved = [r for r in list_inspections(plan["plan_id"]) if r["unit_id"] == "U0001"]
    assert saved and saved[0]["scrap_decision"] == "CONFIRMED_SCRAP"
    assert at.text_input(key=key).value == "U0002"  # 다음 개체로 자동 이동


def test_vehicle_locate_tab_offers_camera():
    pp.create_production_plan("BUMPER", date.today().isoformat(), 10)
    at = _app()
    at.segmented_control(key="insp_task").set_value("차량 위치 찾기 (선택)").run()
    assert not at.exception, at.exception
    assert at.segmented_control(key="car_mode").options == ["카메라 촬영", "파일 업로드"]


def test_gap_shot_toggle_only_for_fender_and_renders():
    pp.create_production_plan("DOOR", date.today().isoformat(), 10)
    pp.create_production_plan("FENDER", date.today().isoformat(), 10)
    at = _app()
    at.segmented_control(key="insp_part").set_value("DOOR").run()
    assert not any(s.key == "shot_kind" for s in at.segmented_control)
    at.segmented_control(key="insp_part").set_value("FENDER").run()
    at.segmented_control(key="shot_kind").set_value("틈(단차)").run()
    assert not at.exception, at.exception
    assert any("가이드 박스" in str(i.value) for i in at.info)


def test_gap_shot_rejected_for_non_fender_plan():
    import pytest
    from services.config import ServiceError
    from services.inspection_registry import inspect_part_image, preview_part_image
    door = pp.create_production_plan("DOOR", date.today().isoformat(), 10)
    for fn in (lambda: inspect_part_image(door["plan_id"], "U1", b"x", shot="gap"),
               lambda: preview_part_image(door["plan_id"], b"x", shot="gap")):
        with pytest.raises(ServiceError) as e:
            fn()
        assert e.value.code == "INVALID_INPUT"


def test_gap_shot_rule_on_fender_photo():
    import pytest
    from ml.vision.infer import gap_ready, predict, status
    photo = ROOT / "data/demo/photos/closeup/fender_defect.jpg"
    if not (gap_ready() and status("ensemble")["status"] == "OK" and photo.exists()):
        pytest.skip("단차·조합 모델 또는 사진 없음")
    r = predict(photo.read_bytes(), "FENDER", with_cam=False, backend_name="ensemble", shot="gap")
    names = [m["name"] for m in r["members"]]
    assert r["shot"] == "gap" and len(names) == 4 and "단차 모델(가이드 영역)" in names
    g, s = r["members"][2], r["members"][3]
    auto = g["verdict"] == s["verdict"] and abs(s["defect_prob"] - s["threshold"]) >= 0.10
    assert r["verdict"] == (s["verdict"] if auto else "AI_DISAGREE")
    surface = predict(photo.read_bytes(), "FENDER", with_cam=False, backend_name="ensemble")
    assert len(surface["members"]) == 3  # 표면 촬영은 기존 규칙 그대로


def test_aihub_sample_mode_inspects_and_compares_truth():
    import pytest
    from ml.vision.infer import status
    if not (ROOT / "data/demo/aihub_samples/index.csv").exists() or status()["status"] != "OK":
        pytest.skip("샘플 세트 또는 모델 없음")
    plan = pp.create_production_plan("DOOR", date.today().isoformat(), 10)
    at = _app()
    at.segmented_control(key="photo_mode").set_value("AI-Hub 샘플").run()
    assert not at.exception, at.exception
    at.segmented_control(key="sample_truth_filter").set_value("불량").run()
    at.button[[b.label for b in at.button].index("이 샘플로 검사")].click().run()
    assert not at.exception, at.exception
    assert len(list_inspections(plan["plan_id"])) == 1
    assert any("AI-Hub 정답" in str(i.value) for i in at.info)
