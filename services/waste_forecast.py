"""생산 전 폐플라스틱 예상 발생량(kg).

expected_defective_units = planned_units * estimated_defect_rate   (반올림 금지)
expected_scrap_kg        = expected_defective_units * unit_weight_kg * scrap_fraction
"""
from __future__ import annotations

from services.config import SCRAP_FRACTION, defect_rate, get_product


def expected_scrap(product_id: str, planned_units: float) -> dict:
    p = get_product(product_id)
    r = defect_rate(product_id)
    w = float(p["unit_weight_kg"])
    units = float(planned_units) * r["rate"]
    out = {
        "product_id": product_id, "planned_units": planned_units, "estimated_defect_rate": r["rate"],
        "expected_defective_units": units, "unit_weight_kg": w, "scrap_fraction": SCRAP_FRACTION,
        "expected_scrap_kg": units * w * SCRAP_FRACTION, "unit": "kg",
        "provenance": r["provenance"] + ["DEMO_SYNTHETIC_WEIGHT"],
        "assumptions": ["가상 단중(실측 아님)", "불량 완제품 전량 폐기 가정(scrap_fraction=1.0)",
                        "runner/sprue/purge/rework 미포함"],
        "warnings": r["warnings"],
    }
    if r.get("ci95"):
        out["expected_scrap_kg_ci95"] = [planned_units * c * w * SCRAP_FRACTION for c in r["ci95"]]
        out["rate_detail"] = {k: r[k] for k in ("raw_rate", "sample", "method", "ci95")}
    return out


def predict_preproduction_waste(plan_id: str | None = None, product_id: str | None = None,
                                planned_units: float | None = None) -> dict:
    if plan_id:
        from services.production_plan import get_plan
        plan = get_plan(plan_id)
        res = expected_scrap(plan["product_id"], plan["planned_units"])
        res.update(plan_id=plan_id, plan_date=plan["plan_date"], plan_source=plan["source"])
        if plan["source"] == "DEMO_SUPPLY_BRIDGE":
            res["provenance"].append("DEMO_SYNTHETIC_BRIDGE")
        return res
    return expected_scrap(product_id, planned_units)
