"""Ollama Tool Calling용 도구 정의와 디스패처. 모든 숫자 계산은 Python 서비스가 수행한다.

상태를 바꾸는 도구(MUTATING)는 LLM 호출만으로 실행되지 않는다. dispatch()는
PENDING_USER_CONFIRMATION을 돌려주고, UI가 사용자의 버튼 클릭 후 execute_confirmed()로 실행한다.
KECO/지역별 인계량/처리방법 도구는 존재하지 않는다(범위 제외).
"""
from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pandas as pd

from services.config import ServiceError


def _s(name, desc, props=None, req=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": req or []}}}


STR, INT, NUM, BOOL = ({"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"})
PRODUCT = {"type": "string", "enum": ["CN7_LH", "CN7_RH", "RG3_LH", "RG3_RH", "DOOR", "BUMPER", "FENDER"]}

TOOLS = [
    _s("create_production_plan", "7종 가상 제조품목 생산계획 등록(사용자 확인 필요). KAMP ID40 제품번호 사용 불가.",
       {"product_id": PRODUCT, "plan_date": STR, "planned_units": INT}, ["product_id", "plan_date", "planned_units"]),
    _s("list_production_plans", "등록된 확정 생산계획 목록"),
    _s("predict_preproduction_waste", "품목·수량(product_id, planned_units)으로 생산 전 예상 폐플라스틱 kg 계산. "
       "등록된 계획 ID(PLAN_로 시작)가 있을 때만 plan_id 사용",
       {"plan_id": STR, "product_id": PRODUCT, "planned_units": NUM}),
    _s("record_operator_decision", "작업자 최종 판정 기록(사용자 확인 필요)",
       {"plan_id": STR, "unit_id": STR, "decision": {"type": "string", "enum": ["CONFIRMED_SCRAP", "CONFIRMED_GOOD", "REWORK_PENDING"]},
        "operator_id": STR}, ["plan_id", "unit_id", "decision", "operator_id"]),
    _s("inspect_part_image", "UI에 업로드된 이미지로 도어/범퍼/휀더 결함 의심 검사(폐기 확정 아님)",
       {"plan_id": STR, "unit_id": STR, "attachment_name": STR}, ["plan_id", "unit_id", "attachment_name"]),
    _s("update_waste_forecast", "검사·판정 결과로 해당 생산계획 kg만 재계산", {"plan_id": STR}, ["plan_id"]),
    _s("get_storage_projection", "3일·7일 보관량(kg) 예측", {"start_date": STR, "days": INT}),
    _s("recommend_collection", "경보일·용량초과일·수거 권장일 추천", {"start_date": STR, "days": INT}),
    _s("get_provenance", "결과 출처 태그 설명", {"tag": STR}),
    _s("forecast_order_quantity", "KAMP ID40 비식별 제품의 T일 예정 수주량(개수) 예측. 2022 백테스트 기간만 지원.",
       {"product_number": STR, "forecast_as_of": STR, "horizon_days": INT},
       ["product_number", "forecast_as_of", "horizon_days"]),
    _s("get_supply_backtest", "KAMP ID40 제품의 백테스트 결과·예측 성능(2022 홀드아웃 예측 vs 실제 예정 수주량). "
       "'백테스트', '성능', '정확도' 질문에 사용",
       {"product_number": STR, "horizon": INT}, ["product_number"]),
    _s("create_demo_plan_draft_from_order_forecast",
       "수주예측을 사용자가 직접 고른 가상 제조품목의 시뮬레이션 계획 '초안'으로 복사(경고 확인+사용자 확인 필요)",
       {"supply_forecast_id": STR, "target_product_id": PRODUCT, "edited_planned_units": INT, "planned_date": STR},
       ["supply_forecast_id", "target_product_id", "edited_planned_units", "planned_date"]),
    _s("confirm_demo_plan_draft", "시뮬레이션 계획 초안을 확정 생산계획으로 반영(사용자 확인 필요)",
       {"draft_id": STR}, ["draft_id"]),
    _s("search_docs", "문서 검색: 폐기물관리법·시행령·시행규칙·자원재활용법·순환경제법 조문, 폐기물 분류번호, 인계서 제도, "
       "데이터 출처, 이미지 판정 모델 비교, 재활용자원 가격(원/kg)·폐자원에너지 통계, 앱 사용법. 법·제도·가격·모델 성능·사용법 질문에 사용",
       {"query": STR}, ["query"]),
    _s("get_recycling_price", "재활용자원 가격(원/kg) 조회: PP 압축 · PE 펠렛 · 폐골판지 등 26개 품목, 전국·8개 권역, "
       "2025-01~2026-10 월별(period=2026-10) 또는 연평균(period=2025), 범위는 start·end. 가격 숫자 질문에 사용",
       {"item": STR, "period": STR, "region": STR, "start": STR, "end": STR}, ["item"]),
    _s("find_waste_processors", "전국 폐기물 처리업체를 지역(주소 일부)·폐기물(분류번호 또는 이름)·업종으로 찾기. 공장에서 가까운 순",
       {"region": STR, "waste": STR, "business": STR, "limit": INT}),
]
MUTATING = {"create_production_plan", "record_operator_decision", "create_demo_plan_draft_from_order_forecast",
            "confirm_demo_plan_draft"}


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return o.head(50).to_dict("records")
    if isinstance(o, (date, datetime, pd.Timestamp)):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if k != "cam"}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _run(name: str, a: dict, attachments: dict | None = None):
    from services import (collection_recommendation as cr, inspection_registry as ir, production_plan as pp,
                          provenance as pv, storage_projection as sp, supply_forecast as sf,
                          supply_to_plan_draft as sd, waste_forecast as wf, waste_forecast_update as wu)
    if name == "create_production_plan":
        return pp.create_production_plan(a["product_id"], a["plan_date"], int(a["planned_units"]), user_confirmed=True)
    if name == "list_production_plans":
        return pp.list_plans()
    if name == "predict_preproduction_waste":
        return wf.predict_preproduction_waste(a.get("plan_id"), a.get("product_id"), a.get("planned_units"))
    if name == "record_operator_decision":
        return ir.record_operator_decision(a["plan_id"], a["unit_id"], a["decision"], a["operator_id"])
    if name == "inspect_part_image":
        img = (attachments or {}).get(a["attachment_name"])
        if img is None:
            raise ServiceError("DATA_NOT_AVAILABLE", "UI에서 이미지를 먼저 업로드해야 합니다.")
        return ir.inspect_part_image(a["plan_id"], a["unit_id"], img)
    if name == "update_waste_forecast":
        return wu.update_waste_forecast(a["plan_id"])
    if name == "get_storage_projection":
        return sp.get_storage_projection(a.get("start_date"), int(a.get("days", 7)))
    if name == "recommend_collection":
        return cr.recommend_collection(a.get("start_date"), int(a.get("days", 7)))
    if name == "get_provenance":
        return pv.get_provenance(a.get("tag"))
    if name == "forecast_order_quantity":
        return sf.forecast_order_quantity(a["product_number"], a["forecast_as_of"], int(a["horizon_days"]))
    if name == "get_supply_backtest":
        r = sf.get_supply_backtest(a["product_number"], horizon=a.get("horizon"))
        r["rows"] = r["rows"][["Date", "horizon", "actual_planned_order_units", "predicted", "selected_model"]]
        r.pop("metrics", None)  # 전체 제품 지표를 제품 결과로 오인하지 않도록 LLM에는 제품 행만 전달
        return r
    if name == "search_docs":
        from services.doc_search import search
        return search(a.get("query", ""))
    if name == "get_recycling_price":
        from services.recycling_price import get_price
        return get_price(a.get("item", ""), a.get("period"), a.get("region"), a.get("start"), a.get("end"))
    if name == "find_waste_processors":
        from services.waste_processors import search_processors
        return search_processors(a.get("region"), a.get("waste"), a.get("business"), int(a.get("limit", 10)))
    if name == "create_demo_plan_draft_from_order_forecast":
        return sd.create_demo_plan_draft_from_order_forecast(a["supply_forecast_id"], a["target_product_id"],
                                                             int(a["edited_planned_units"]), a["planned_date"],
                                                             acknowledge_mapping_warning=True)
    if name == "confirm_demo_plan_draft":
        return sd.confirm_demo_plan_draft(a["draft_id"], user_confirmed=True, acknowledge_mapping_warning=True)
    raise ServiceError("UNKNOWN_TOOL", name)


_NULLS = {"", "null", "none", "None", "NULL", "undefined", "PENDING_USER_CONFIRMATION"}
_INTS = {"planned_units", "horizon_days", "horizon", "days", "edited_planned_units", "limit"}


def sanitize(args: dict) -> dict:
    """소형 LLM이 채우는 빈값/가짜 ID를 제거하고 숫자 문자열을 변환한다."""
    out = {}
    for k, v in (args or {}).items():
        if v is None or (isinstance(v, str) and v.strip() in _NULLS):
            continue
        if k == "plan_id" and not str(v).startswith("PLAN_"):
            continue  # 존재하지 않는 형식의 계획 ID는 무시(품목·수량 경로로 계산)
        if k in _INTS:
            try:
                v = int(float(v))
            except (TypeError, ValueError):
                pass
            if k == "horizon" and v not in (1, 2, 3, 4):
                continue  # 백테스트 필터: 범위 밖이면 전체 horizon
        out[k] = v
    return out


def summarize(name: str, r) -> str:
    """도구 결과의 결정적 한국어 요약(LLM 답변 검증용 근거)."""
    try:
        if name == "predict_preproduction_waste":
            return (f"{r['product_id']} {r['planned_units']:,}개 × 불량률 {r['estimated_defect_rate']:.4%} × "
                    f"{r['unit_weight_kg']}kg = 예상 폐기물 {r['expected_scrap_kg']:.2f} kg ({', '.join(r['provenance'])})")
        if name == "forecast_order_quantity":
            act = r["actual_planned_order_units"]
            return (f"{r['product_number']} {r['target_date']} 예정 수주량 예측 {r['predicted_planned_order_units']:.1f}개"
                    f" (기준 {r['forecast_as_of']}, T+{r['horizon']}, 모델 {r['model_used']}"
                    + (f", 실제 {act:.0f}개" if act is not None else "") + ") — 개수이며 kg 아님, 확정 주문 아님")
        if name == "get_storage_projection":
            return (f"{r['start_date']} 기준 보관량 {r['current_stock_kg']:.1f}kg → 3일 후 {r['stock_after_3_days_kg']:.1f}kg, "
                    f"7일 후 {r['stock_after_7_days_kg']:.1f}kg (경보 {r['alert_kg']:.0f}kg, 용량 {r['max_capacity_kg']:.0f}kg)")
        if name == "recommend_collection":
            if r["status"] == "NO_COLLECTION_NEEDED":
                return f"{r['start_date']} 기준 {r['message']} → 수거 요청 불필요"
            return (f"경보일 {r['alert_date']}, 권장 수거일 {r['recommended_collection_date']}, "
                    f"요청 마감 {r['request_by_date']} ({r['status']})")
        if name == "update_waste_forecast":
            return f"{r['plan_id']} 갱신 예상 {r['updated_expected_total_kg']:.2f}kg (최초 {r['initial_expected_kg']:.2f}kg)"
        if name == "get_supply_backtest":
            rows = r["rows"]
            if rows.empty:
                return "해당 조건의 2022 홀드아웃 시험 예측이 없습니다."
            parts = []
            for h, g in rows.groupby("horizon"):
                y, p = g["actual_planned_order_units"], g["predicted"]
                parts.append(f"T+{h}({g.selected_model.iloc[0]}) MAE {abs(p - y).mean():.1f}개, "
                             f"WAPE {abs(p - y).sum() / max(y.abs().sum(), 1e-9):.3f}")
            return (f"{len(rows)}개 시험 예측(2022 홀드아웃) 해당 제품 기준: " + "; ".join(parts)
                    + " — 전체 제품 기준 지표는 수주량 예측 화면 참조")
        if name == "get_provenance":
            return "; ".join(f"{k}: {v}" for k, v in r.items())
        if name == "search_docs":
            return f"문서 {len(r['hits'])}조각({r['mode']}): " + "; ".join(f"{h['doc']} > {h['section'][:40]}" for h in r["hits"])
        if name == "get_recycling_price":
            pts = ", ".join(f"{x['기간']} {x['권역']} {'미공표' if x['원/kg'] is None else format(x['원/kg'], ',.1f')}" for x in r["rows"][:12])
            return f"{r['item']} ({r['unit']}, {r['stage']}): {pts}" + (" …" if len(r["rows"]) > 12 else "")
        if name == "find_waste_processors":
            names = ", ".join(f"{i['시설명']}({i['직선거리_km']}km)" for i in r["items"][:5])
            return f"조건 {r['conditions']} 해당 {r['matched']}곳 중 가까운 {r['shown']}곳: {names}"
    except (KeyError, TypeError, ValueError):
        pass
    return ""


def dispatch(name: str, args: dict, attachments: dict | None = None) -> dict:
    """LLM 경로. 상태 변경 도구는 실행하지 않고 사용자 확인 대기 상태로 반환."""
    args = sanitize(args)
    if name in MUTATING:
        from services.supply_to_plan_draft import MAPPING_WARNING
        warn = MAPPING_WARNING if "draft" in name else None
        return {"status": "PENDING_USER_CONFIRMATION", "executed": False, "tool": name, "args": args, "warning": warn,
                "message": "아직 실행되지 않았습니다(등록/저장 안 됨). 사용자가 화면의 '확인 후 실행' 버튼을 눌러야 실행됩니다."}
    return execute_confirmed(name, args, attachments)


def execute_confirmed(name: str, args: dict, attachments: dict | None = None) -> dict:
    args = sanitize(args)
    try:
        raw = _run(name, args, attachments)
        return {"status": "OK", "summary_ko": summarize(name, raw), "result": _jsonable(raw)}
    except ServiceError as e:
        return {"status": "ERROR", "code": e.code, "message": e.message}
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "ERROR", "code": "INVALID_INPUT", "message": repr(e)}


def to_json(o) -> str:
    return json.dumps(_jsonable(o), ensure_ascii=False, default=str)
