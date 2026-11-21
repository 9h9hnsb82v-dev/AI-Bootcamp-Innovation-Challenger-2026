"""검사 후 예상 폐기물 kg 갱신 (해당 생산계획만 재계산).

updated_expected_total_kg = confirmed_scrap_kg
    + (produced_uninspected_units + unproduced_units) * remaining_defect_rate * unit_weight_kg * scrap_fraction
REWORK_PENDING 수량은 승인 정책이 없으므로 kg에 넣지 않고 별도 표시한다.
"""
from __future__ import annotations

from services import db
from services.config import SCRAP_FRACTION, defect_rate, get_product
from services.inspection_registry import unit_buckets
from services.production_plan import get_plan
from services.waste_forecast import expected_scrap


def compute_update(plan_id: str) -> dict:
    plan = get_plan(plan_id)
    p = get_product(plan["product_id"])
    w = float(p["unit_weight_kg"])
    r = defect_rate(plan["product_id"])
    b = unit_buckets(plan_id)["buckets"]
    initial = expected_scrap(plan["product_id"], plan["planned_units"])["expected_scrap_kg"]
    confirmed_kg = b["CONFIRMED_SCRAP"] * w * SCRAP_FRACTION
    remaining_units = b["PRODUCED_UNINSPECTED"] + b["UNPRODUCED"]
    remaining_kg = remaining_units * r["rate"] * w * SCRAP_FRACTION
    return {"plan_id": plan_id, "product_id": plan["product_id"], "plan_date": plan["plan_date"],
            "initial_expected_kg": initial, "confirmed_scrap_kg": confirmed_kg,
            "remaining_expected_kg": remaining_kg, "updated_expected_total_kg": confirmed_kg + remaining_kg,
            "remaining_defect_rate": r["rate"], "buckets": b,
            "rework_pending_units_not_in_kg": b["REWORK_PENDING"], "unit": "kg",
            "provenance": r["provenance"] + ["DEMO_SYNTHETIC_WEIGHT"] +
                          (["DEMO_OPERATOR_CONFIRMATION"] if b["CONFIRMED_SCRAP"] or b["CONFIRMED_GOOD"] else [])}


def update_waste_forecast(plan_id: str) -> dict:
    res = compute_update(plan_id)
    with db.connect() as con:
        con.execute("INSERT INTO forecast_snapshot VALUES (?,?,?,?,?,?,?,?)",
                    (db.new_id("SNAP"), plan_id, res["initial_expected_kg"], res["confirmed_scrap_kg"],
                     res["remaining_expected_kg"], res["updated_expected_total_kg"], ",".join(res["provenance"]),
                     db.now()))
    return res


def snapshots(plan_id: str) -> list[dict]:
    with db.connect() as con:
        return db.rows(con, "SELECT * FROM forecast_snapshot WHERE plan_id=? ORDER BY timestamp", (plan_id,))
