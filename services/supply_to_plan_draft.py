"""KAMP ID40 수주예측 → 가상 제조품목 '시뮬레이션 계획 초안' (명시적 사용자 작업만 허용).

자동 매핑·자동 계획 생성 기능은 없다. 사용자가
 (1) 출처 불일치 경고를 확인하고  (2) 7종 중 대상 품목·생산일·수량을 직접 고른 뒤
 (3) 초안을 다시 확정해야만 production_plan(source=DEMO_SUPPLY_BRIDGE)이 생긴다.
주문량 = 생산계획량으로 자동 가정하지 않는다(edited_planned_units 필수).
"""
from __future__ import annotations

import json
from datetime import date

from services import db
from services.config import ServiceError, get_product
from services.production_plan import _BRIDGE_TOKEN, create_production_plan

MAPPING_WARNING = ("서로 다른 기업의 데이터이며 실제 제품 매핑이 없습니다. KAMP ID40 비식별 제품의 예정 수주량을 "
                   "가상 제조품목의 생산계획으로 옮기는 것은 시연용 가상 시나리오(DEMO_SYNTHETIC_BRIDGE)입니다.")
ASSUMPTION_FLAGS = ["DEMO_SYNTHETIC_BRIDGE", "CROSS_COMPANY_NO_REAL_MAPPING",
                    "ORDER_TO_PRODUCTION_NOT_ASSUMED_EQUAL", "USER_SELECTED_TARGET_PRODUCT", "USER_EDITED_QUANTITY"]


def create_demo_plan_draft_from_order_forecast(supply_forecast_id: str, target_product_id: str,
                                               edited_planned_units: int, planned_date: str,
                                               acknowledge_mapping_warning: bool = False) -> dict:
    if not acknowledge_mapping_warning:
        raise ServiceError("MAPPING_WARNING_NOT_ACKNOWLEDGED", MAPPING_WARNING)
    if not target_product_id:
        raise ServiceError("TARGET_PRODUCT_REQUIRED", "대상 가상 제조품목을 사용자가 직접 선택해야 합니다.")
    get_product(target_product_id)
    if edited_planned_units is None or int(edited_planned_units) != edited_planned_units or edited_planned_units <= 0:
        raise ServiceError("QUANTITY_REVIEW_REQUIRED", "사용자가 검토한 생산 수량(양의 정수)을 입력해야 합니다.")
    planned_date = str(date.fromisoformat(str(planned_date)))
    with db.connect() as con:
        f = db.rows(con, "SELECT * FROM supply_forecast WHERE forecast_id=?", (supply_forecast_id,))
        if not f:
            raise ServiceError("NOT_FOUND", f"수주예측 {supply_forecast_id} 없음")
        f = f[0]
        draft = {"draft_id": db.new_id("DRAFT"), "supply_forecast_id": supply_forecast_id,
                 "source_product_number": f["product_number"], "model_version": f["model_version"],
                 "target_manufacturing_product_id": target_product_id, "planned_date": planned_date,
                 "quantity_proposed": f["predicted_planned_order_units"],
                 "quantity_approved": int(edited_planned_units), "approval_status": "DRAFT", "is_demo": 1,
                 "assumption_flags": json.dumps(ASSUMPTION_FLAGS), "warning_acknowledged_at": db.now(),
                 "confirmed_at": None, "plan_id": None}
        con.execute("INSERT INTO supply_plan_draft VALUES (:draft_id,:supply_forecast_id,:source_product_number,"
                    ":model_version,:target_manufacturing_product_id,:planned_date,:quantity_proposed,"
                    ":quantity_approved,:approval_status,:is_demo,:assumption_flags,:warning_acknowledged_at,"
                    ":confirmed_at,:plan_id)", draft)
    return {**draft, "assumption_flags": ASSUMPTION_FLAGS, "warning": MAPPING_WARNING,
            "next_step": "confirm_demo_plan_draft(draft_id, user_confirmed=True, acknowledge_mapping_warning=True)"}


def confirm_demo_plan_draft(draft_id: str, user_confirmed: bool = False,
                            acknowledge_mapping_warning: bool = False) -> dict:
    if not (user_confirmed and acknowledge_mapping_warning):
        raise ServiceError("CONFIRMATION_REQUIRED", MAPPING_WARNING + " 확정하려면 경고 재확인과 사용자 확인이 모두 필요합니다.")
    with db.connect() as con:
        d = db.rows(con, "SELECT * FROM supply_plan_draft WHERE draft_id=?", (draft_id,))
    if not d:
        raise ServiceError("NOT_FOUND", f"초안 {draft_id} 없음")
    d = d[0]
    if d["approval_status"] != "DRAFT":
        raise ServiceError("ALREADY_PROCESSED", f"초안 상태 {d['approval_status']}")
    plan = create_production_plan(d["target_manufacturing_product_id"], d["planned_date"], d["quantity_approved"],
                                  user_confirmed=True, _bridge=_BRIDGE_TOKEN,
                                  _source_forecast_id=d["supply_forecast_id"])
    with db.connect() as con:
        con.execute("UPDATE supply_plan_draft SET approval_status='CONFIRMED', confirmed_at=?, plan_id=? WHERE draft_id=?",
                    (db.now(), plan["plan_id"], draft_id))
    return {"draft_id": draft_id, "plan": plan, "provenance": ["DEMO_SYNTHETIC_BRIDGE"], "warning": MAPPING_WARNING}


def list_drafts() -> list[dict]:
    with db.connect() as con:
        return db.rows(con, "SELECT * FROM supply_plan_draft ORDER BY warning_acknowledged_at DESC")
