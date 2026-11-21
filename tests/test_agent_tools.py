"""에이전트 도구 정제·요약·확인 대기 계약."""
import pytest

from agent.tools import dispatch, sanitize
from services import production_plan as pp


def test_sanitize_drops_fake_plan_ids_and_casts_numbers():
    a = sanitize({"plan_id": "CN7_LH_5000", "product_id": "CN7_LH", "planned_units": "5000", "horizon": 9, "x": "null"})
    assert a == {"product_id": "CN7_LH", "planned_units": 5000}


def test_waste_tool_with_junk_plan_id_still_computes():
    r = dispatch("predict_preproduction_waste", {"plan_id": "None", "product_id": "BUMPER", "planned_units": "1000"})
    assert r["status"] == "OK" and r["result"]["expected_scrap_kg"] == pytest.approx(70.0)
    assert "70.00 kg" in r["summary_ko"]


def test_mutating_tools_never_execute_from_llm_path():
    for name, args in [("create_production_plan", {"product_id": "DOOR", "plan_date": "2026-11-20", "planned_units": 5}),
                       ("confirm_demo_plan_draft", {"draft_id": "DRAFT_x"})]:
        r = dispatch(name, args)
        assert r["status"] == "PENDING_USER_CONFIRMATION" and r["executed"] is False
    assert pp.list_plans() == []


def test_backtest_summary_is_product_specific():
    r = dispatch("get_supply_backtest", {"product_number": "Product_c0", "horizon": 0})
    assert r["status"] == "OK" and "metrics" not in r["result"]
    assert "Product_c0" not in r["summary_ko"] or "해당 제품" in r["summary_ko"]
    assert "T+1" in r["summary_ko"] and "T+4" in r["summary_ko"]


def test_demo_scenario_reproduces_key_numbers(tmp_path, monkeypatch):
    import scripts.demo_scenario as ds
    monkeypatch.setattr(ds, "DEMO_DB", tmp_path / "demo.db")
    log = "\n".join(ds.main()["log"])
    # CN7_LH 예상 kg = 5,000개 x KAMP 학습구간 Jeffreys 불량률 x 단중 1.30kg (분할 변경 시 값이 바뀌므로 모델 파일에서 계산)
    import json
    from pathlib import Path
    rate = json.loads((Path(ds.ROOT) / "models/kamp/baseline_rates.json").read_text(encoding="utf-8"))[
        "products"]["CN7_LH"]["estimated_defect_rate"]
    assert "갱신 84.00 kg" in log and f"예상 {5000 * rate * 1.30:.2f} kg" in log and "DEMO_SYNTHETIC_BRIDGE" in log
