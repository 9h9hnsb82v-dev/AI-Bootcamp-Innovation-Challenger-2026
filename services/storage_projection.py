"""가상 제조창고 폐플라스틱 보관량(kg) 예측. S(t+1) = S(t) + G(t) - C(t).

상호배타 구성
- 현재 보관량: facility_demo.current_stock_kg(승인된 시연 기준값) + ledger 실현 이벤트(오늘까지)
- G(t): 확정 생산계획의 *남은 예상* kg(remaining_expected_kg)를 계획일에 배치 + 미래일자 확정 폐기 이벤트
  (이미 확정된 폐기 kg는 ledger로 들어가 있으므로 예상치에서 빠져 중복되지 않음)
- C(t): 예정된 수거(COLLECTION_OUT)
KAMP ID40 수주예측(개수)은 이 계산에 들어오지 않는다.
"""
from __future__ import annotations

from datetime import date, timedelta

from services import db
from services.config import facility


def project_storage(initial_kg: float, generation: dict[date, float], collections: dict[date, float],
                    start: date, days: int) -> list[dict]:
    out, s = [], float(initial_kg)
    for i in range(days):
        d = start + timedelta(days=i)
        g, c = float(generation.get(d, 0.0)), float(collections.get(d, 0.0))
        end = max(s + g - c, 0.0)
        out.append({"date": d.isoformat(), "start_kg": s, "generation_kg": g, "collection_kg": c, "end_kg": end})
        s = end
    return out


def _inputs(start: date, days: int):
    from services.production_plan import list_plans
    from services.waste_forecast_update import compute_update
    fac = facility()
    end = start + timedelta(days=days - 1)
    with db.connect() as con:
        led = db.rows(con, "SELECT * FROM stock_ledger")
    current = float(fac["current_stock_kg"])
    gen: dict[date, float] = {}
    col: dict[date, float] = {}
    for e in led:
        d = date.fromisoformat(e["date"])
        sign = -1 if e["event_type"] == "COLLECTION_OUT" else 1
        if d < start:
            current += sign * e["kg"]
        elif d <= end:
            (col if sign < 0 else gen)[d] = (col if sign < 0 else gen).get(d, 0.0) + e["kg"]
    plan_detail = []
    for p in list_plans():
        u = compute_update(p["plan_id"])
        d = max(date.fromisoformat(p["plan_date"]), start)  # 지난 계획의 남은 예상분은 시작일에 반영
        if d <= end and u["remaining_expected_kg"] > 0:
            gen[d] = gen.get(d, 0.0) + u["remaining_expected_kg"]
        plan_detail.append({"plan_id": p["plan_id"], "product_id": p["product_id"], "plan_date": p["plan_date"],
                            "remaining_expected_kg": u["remaining_expected_kg"],
                            "confirmed_scrap_kg": u["confirmed_scrap_kg"]})
    return fac, current, gen, col, plan_detail


def get_storage_projection(start_date: str | None = None, days: int = 7) -> dict:
    start = date.fromisoformat(start_date) if start_date else date.today()
    fac, current, gen, col, plans = _inputs(start, days)
    proj = project_storage(current, gen, col, start, days)
    cap = float(fac["max_capacity_kg"])
    alert = cap * float(fac["alert_ratio"])
    return {"start_date": start.isoformat(), "current_stock_kg": current, "max_capacity_kg": cap,
            "alert_kg": alert, "projection": proj,
            "stock_after_3_days_kg": proj[min(2, days - 1)]["end_kg"],
            "stock_after_7_days_kg": proj[min(6, days - 1)]["end_kg"],
            "first_alert_date": next((r["date"] for r in proj if r["end_kg"] >= alert), None),
            "first_over_capacity_date": next((r["date"] for r in proj if r["end_kg"] > cap), None),
            "plans": plans, "unit": "kg", "provenance": ["DEMO_ASSUMED", "DEMO_SYNTHETIC_WEIGHT"]}
