from datetime import date, timedelta

import pandas as pd
import streamlit as st

from services.config import products
from services.production_plan import create_production_plan, delete_plan, list_plans, set_produced_units
from services.waste_forecast import expected_scrap
from ui.common import kg, product_label, run, tags

st.title("생산계획")
st.caption("7종 가상 제조품목만 등록할 수 있습니다. KAMP ID40 제품번호는 여기서 사용할 수 없습니다.")

with st.form("plan"):
    c1, c2, c3 = st.columns(3)
    pid = c1.selectbox("품목", list(products()), format_func=product_label)
    d = c2.date_input("생산 예정일", date.today() + timedelta(days=1))
    n = c3.number_input("계획 수량(개)", min_value=1, value=1000, step=100)
    preview = st.form_submit_button("예상 kg 미리보기")
    if preview:
        r = run(expected_scrap, pid, n)
        if r:
            st.info(f"예상 불량 {r['expected_defective_units']:.2f}개 × {r['unit_weight_kg']}kg → **{kg(r['expected_scrap_kg'])}**")
            tags(r["provenance"])
    confirm = st.checkbox("위 계획(품목·수량·날짜)을 확인했습니다")
    if st.form_submit_button("생산계획 확정 등록", type="primary"):
        if not confirm:
            st.warning("확인 체크 후 등록할 수 있습니다.")
        else:
            p = run(create_production_plan, pid, d, int(n), user_confirmed=True)
            if p:
                st.success(f"등록됨: {p['plan_id']}")

SOURCE_KO = {"DIRECT_INPUT": "직접 입력", "DEMO_SUPPLY_BRIDGE": "수주량 예측 초안 반영"}


def _plan_table(plans: list[dict]) -> pd.DataFrame:
    """DB 열 이름 대신 사용자가 읽는 말로 (2026-11-21 사용자 요청). 빈 값은 공백."""
    rows = []
    for p in plans:
        exp = expected_scrap(p["product_id"], p["planned_units"])
        rows.append({"품목": product_label(p["product_id"]), "생산 예정일": str(p["plan_date"]),
                     "계획 수량(개)": p["planned_units"], "생산 완료(개)": p["produced_units"],
                     "진행률": p["produced_units"] / p["planned_units"] if p["planned_units"] else 0.0,
                     "예상 폐기물(kg)": exp["expected_scrap_kg"],
                     "등록 방법": SOURCE_KO.get(p["source"], p["source"] or ""),
                     "연결된 수주량 예측": p.get("source_forecast_id") or "",
                     "등록 시각": str(p.get("approved_at") or "").replace("T", " ")[:16],
                     "계획 번호": p["plan_id"]})
    return pd.DataFrame(rows)


def _plan_name(p: dict) -> str:
    return f"{product_label(p['product_id'])} · {p['plan_date']} · {p['planned_units']:,}개 ({p['plan_id']})"


plans = list_plans()
st.subheader("확정 생산계획")
if plans:
    st.dataframe(_plan_table(plans), hide_index=True, width="stretch",
                 column_config={"계획 수량(개)": st.column_config.NumberColumn(format="%d"),
                                "생산 완료(개)": st.column_config.NumberColumn(format="%d"),
                                "진행률": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                "예상 폐기물(kg)": st.column_config.NumberColumn(format="%.1f")})
    with st.expander("생산 진행 수량 입력 / 계획 삭제"):
        by_id = {p["plan_id"]: p for p in plans}
        sel = st.selectbox("계획", list(by_id), format_func=lambda i: _plan_name(by_id[i]))
        p = next(x for x in plans if x["plan_id"] == sel)
        produced = st.number_input("현재까지 생산된 고유 개체 수", 0, p["planned_units"], p["produced_units"])
        c1, c2 = st.columns(2)
        if c1.button("생산수량 저장"):
            if run(set_produced_units, sel, int(produced)):
                st.rerun()
        if c2.button("이 계획 삭제(시연 데이터)", type="secondary"):
            delete_plan(sel)
            st.rerun()
else:
    st.info("등록된 계획 없음")
