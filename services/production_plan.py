"""확정 생산계획(7종 제조품목 전용).

KAMP ID40 수주예측은 여기로 직접 들어올 수 없다. DEMO_SUPPLY_BRIDGE 출처 계획은
services.supply_to_plan_draft.confirm_demo_plan_draft() 만 생성할 수 있다.
"""
from __future__ import annotations

from datetime import date

from services import db
from services.config import ServiceError, get_product

_BRIDGE_TOKEN = object()  # supply_to_plan_draft 전용 내부 토큰


def create_production_plan(product_id: str, plan_date: str | date, planned_units: int,
                           *, user_confirmed: bool = True, _bridge=None, _source_forecast_id=None) -> dict:
    get_product(product_id)  # 7종 이외(예: Product_ac) 차단
    if not user_confirmed:
        raise ServiceError("CONFIRMATION_REQUIRED", "생산계획은 사용자 확인 후에만 등록됩니다.")
    if int(planned_units) != planned_units or planned_units <= 0:
        raise ServiceError("INVALID_INPUT", "planned_units는 양의 정수여야 합니다.")
    source = "DEMO_SUPPLY_BRIDGE" if _bridge is _BRIDGE_TOKEN else "DIRECT_INPUT"
    if _source_forecast_id and source != "DEMO_SUPPLY_BRIDGE":
        raise ServiceError("BRIDGE_NOT_ALLOWED", "수주예측 기반 계획은 초안 확정 절차로만 생성됩니다.")
    plan = {"plan_id": db.new_id("PLAN"), "product_id": product_id,
            "plan_date": str(date.fromisoformat(str(plan_date))), "planned_units": int(planned_units),
            "produced_units": 0, "source": source, "source_forecast_id": _source_forecast_id,
            "approved_at": db.now(), "is_demo": 1}
    with db.connect() as con:
        con.execute("INSERT INTO production_plan VALUES (:plan_id,:product_id,:plan_date,:planned_units,"
                    ":produced_units,:source,:source_forecast_id,:approved_at,:is_demo)", plan)
    return plan


def get_plan(plan_id: str) -> dict:
    with db.connect() as con:
        r = db.rows(con, "SELECT * FROM production_plan WHERE plan_id=?", (plan_id,))
    if not r:
        raise ServiceError("NOT_FOUND", f"생산계획 {plan_id} 없음")
    return r[0]


def list_plans() -> list[dict]:
    with db.connect() as con:
        return db.rows(con, "SELECT * FROM production_plan ORDER BY plan_date, approved_at")


def set_produced_units(plan_id: str, produced_units: int) -> dict:
    """작업자 입력: 현재까지 생산된(고유) 수량. 검사 수량 이상, 계획 수량 이하."""
    plan = get_plan(plan_id)
    with db.connect() as con:
        inspected = con.execute("SELECT COUNT(DISTINCT unit_id) FROM inspection WHERE plan_id=? AND cancelled_at IS NULL",
                                (plan_id,)).fetchone()[0]
        if not (inspected <= produced_units <= plan["planned_units"]):
            raise ServiceError("INVALID_INPUT", f"생산수량은 검사된 고유 개체 수({inspected}) 이상, "
                                                f"계획수량({plan['planned_units']}) 이하여야 합니다.")
        con.execute("UPDATE production_plan SET produced_units=? WHERE plan_id=?", (int(produced_units), plan_id))
    return get_plan(plan_id)


def delete_plan(plan_id: str) -> None:
    """시연 데이터 정리용(UI에서 확인 후 호출)."""
    with db.connect() as con:
        con.execute("DELETE FROM inspection WHERE plan_id=?", (plan_id,))
        con.execute("DELETE FROM forecast_snapshot WHERE plan_id=?", (plan_id,))
        con.execute("DELETE FROM stock_ledger WHERE plan_id=?", (plan_id,))
        con.execute("DELETE FROM production_plan WHERE plan_id=?", (plan_id,))
