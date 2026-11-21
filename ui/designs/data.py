"""시안 대시보드가 함께 쓰는 실제 데이터 묶음 (DB·서비스 값 그대로, 화면용 가공만)."""
from __future__ import annotations

import json
import re
from datetime import date

import streamlit as st

from services.collection_recommendation import recommend_collection
from services.config import ROOT, ServiceError
from services.inspection_registry import IMAGE_PRODUCTS, pending_units
from services.production_plan import list_plans
from services.storage_rooms import list_collections, schedule_collection
from services.waste_forecast_update import compute_update

PAGES = {  # st.switch_page / st.page_link 경로 (진입점 ui/app.py 기준)
    "dash": "pages/1_dashboard.py", "plan": "pages/2_production_plan.py", "pred": "pages/3_waste_prediction.py",
    "insp": "pages/4_inspection.py", "update": "pages/5_forecast_update.py", "storage": "pages/6_storage.py",
    "layout": "pages/9_factory_layout.py", "proc": "pages/10_processors.py", "evidence": "pages/11_evidence.py",
    "supply": "pages/7_supply_forecast.py", "chat": "pages/8_chatbot.py",
}
WD = "월화수목금토일"
SOURCE_KO = {"DIRECT_INPUT": "직접 입력", "DEMO_SUPPLY_BRIDGE": "수주 예측 초안"}


def md(d: str) -> str:
    x = date.fromisoformat(d)
    return f"{d[5:]}({WD[x.weekday()]})"


def summary() -> dict:
    ss = st.session_state
    as_of = ss.get("as_of") or date.today()
    days = int(ss.get("horizon") or 14)
    plans = list_plans()
    rows, errors = [], []
    for p in plans:
        try:
            u = compute_update(p["plan_id"])
        except ServiceError as e:
            errors.append(f"{p['plan_id']}: {e.message}")
            continue
        rows.append({"계획": p["plan_id"], "품목": p["product_id"], "계획일": p["plan_date"], "계획수량": p["planned_units"],
                     "출처": SOURCE_KO.get(p["source"], p["source"]), "최초 예상 kg": u["initial_expected_kg"],
                     "확인 폐기 kg": u["confirmed_scrap_kg"], "갱신 예상 kg": u["updated_expected_total_kg"],
                     "근거": ", ".join(u["provenance"])})
    by_prod: dict[str, list[float]] = {}
    for r in rows:
        a = by_prod.setdefault(r["품목"], [0.0, 0.0])
        a[0] += r["최초 예상 kg"]
        a[1] += r["갱신 예상 kg"]
    pend = {"REWORK": 0, "CHECK": 0}
    for p in plans:
        if p["product_id"] in IMAGE_PRODUCTS:
            q = pending_units(p["plan_id"])
            pend["REWORK"] += len(q["REWORK"])
            pend["CHECK"] += len(q["CHECK"])
    try:
        rec = recommend_collection(as_of.isoformat(), days)
    except ServiceError as e:
        rec, errors = None, errors + [f"보관 예측: {e.message}"]
    proj = rec["projection_without_collection"] if rec else []
    upcoming = [c for c in list_collections() if c["date"] >= as_of.isoformat()]
    return {
        "as_of": as_of, "days": days, "plans": plans, "rows": rows, "errors": errors,
        "by_product": dict(sorted(by_prod.items(), key=lambda kv: -kv[1][1])),
        "initial_kg": sum(r["최초 예상 kg"] for r in rows), "updated_kg": sum(r["갱신 예상 kg"] for r in rows),
        "scrap_kg": sum(r["확인 폐기 kg"] for r in rows), "pending": pend,
        "rec": rec, "proj": proj, "upcoming": upcoming,
        "now_kg": proj[0]["start_kg"] if proj else 0.0,
        "gen_kg": sum(d["generation_kg"] for d in proj),
        "peak_kg": max((d["end_kg"] for d in proj), default=0.0),
        "acc": accuracy(),
    }


@st.cache_data(ttl=600)
def accuracy() -> dict | None:
    """세션 비겹침 재평가(정직한 수치). 파일이 없으면 None — 화면은 숫자를 지어내지 않는다.
    2026-11-19: 지금 서비스 중인 조합(serving.json = ensemble_dino → EfficientNet+DINOv2)의 시험 성적을 우선 쓴다.
    reports/vision/sd_eval.json 은 이전 조합(EfficientNet+YOLO11m) 재평가라 그 조합일 때만 쓴다."""
    try:
        sv = json.loads((ROOT / "models/vision/serving.json").read_text(encoding="utf-8")).get("backend")
        if sv == "ensemble_dino":
            em = json.loads((ROOT / "models/vision/ensemble_sd_dino/meta.json").read_text(encoding="utf-8"))
            ev, days = em["evaluation"], re.search(r"촬영일 (\d+)일", em["evaluation"]["set"])
            return {"days": int(days.group(1)) if days else None, "model": " + ".join(em.get("member_names", [])),
                    "parts": {k: {"acc": v["auto_accuracy"], "auto": v["auto_rate"]} for k, v in ev["by_part"].items()},
                    "gap": {"acc": ev["gap"]["자동판정_정확도"], "auto": ev["gap"]["자동판정_비율"]} if ev.get("gap") else None}
    except (OSError, ValueError, KeyError):
        pass
    p = ROOT / "reports/vision/sd_eval.json"
    if not p.exists():
        return None
    sd = json.loads(p.read_text(encoding="utf-8"))
    s = sd["surface"]
    return {"days": sd["split"]["days"]["test"],
            "parts": {k: {"acc": s[k]["auto_acc"], "auto": s[k]["auto_rate"]} for k in ("DOOR", "BUMPER", "FENDER") if k in s}}


def open_booking() -> None:
    st.session_state["_book_open"] = True


def _close_booking() -> None:
    st.session_state.pop("_book_open", None)


def booking(rec: dict | None) -> None:
    """'수거 예약하기' 버튼(on_click=open_booking)을 누른 뒤면 확인 창을 띄운다. 페이지 끝에서 한 번 부른다."""
    if rec and st.session_state.get("_book_open"):
        booking_dialog(rec)


@st.dialog("수거 예약", on_dismiss=_close_booking)
def booking_dialog(rec: dict) -> None:
    """수거 예약 확인 창. 확인을 누르면 보관 화면의 '권장 수거 일정 예약'과 같은 장부 이벤트를 넣는다."""
    st.write(f"**{md(rec['recommended_collection_date'])}**에 **{rec['expected_collection_kg']:,.1f} kg** 수거를 예약합니다.")
    st.caption(f"요청 마감일 {rec['request_by_date']} · 리드타임 {rec['lead_time_days']}일 · 가상 시연 장부에 들어가며, "
               "'보관량·수거 추천 → 수거 예약 목록'에서 취소할 수 있습니다.")
    c = st.columns(2)
    if c[0].button("예약 확정", type="primary", width="stretch", icon=":material/local_shipping:"):
        try:
            schedule_collection(rec["recommended_collection_date"], rec["expected_collection_kg"])
        except ServiceError as e:
            st.error(f"**{e.code}** — {e.message}")
            return
        _close_booking()
        st.session_state["_booked"] = rec["recommended_collection_date"]
        st.rerun()
    if c[1].button("취소", width="stretch"):
        _close_booking()
        st.rerun()


def booked_toast() -> None:
    d = st.session_state.pop("_booked", None)
    if d:
        st.toast(f"{md(d)} 수거를 예약했습니다.", icon=":material/check_circle:")
