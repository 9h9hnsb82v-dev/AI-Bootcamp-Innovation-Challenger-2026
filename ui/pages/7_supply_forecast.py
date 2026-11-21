from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from services.config import products
from services.supply_forecast import (forecast_order_quantity, get_supply_backtest, list_supply_products,
                                      model_status, product_history)
from services.supply_to_plan_draft import (MAPPING_WARNING, confirm_demo_plan_draft,
                                           create_demo_plan_draft_from_order_forecast, list_drafts)
from ui.common import product_label, run, tags

st.title("미래 수주량 예측 (KAMP ID 40)")
st.info("**예측 대상: `T일 예정 수주량`** — 기업 생산계획 시스템의 예정(계획) 수주량이며 **확정 주문 실적이 아닙니다**. "
        "단위는 **개(units)**이고 폐기물 kg가 아닙니다. 비식별 `Product_Number`는 7종 제조품목과 실제 매핑이 없습니다.")
ms = model_status()
if ms["status"] != "TRAINED":
    st.error(f"{ms['status']} — {ms['message']}")
    st.stop()
with st.expander("데이터·평가 방식 (감사 결과)"):
    st.markdown(
        "- 원본 34,617행×20열, 117개 제품. 하루 여러 로그 중 **마지막 로그**를 일 스냅샷으로 사용\n"
        "- **2022-03-24~05-09는 01-26~03-13을 57일 뒤로 재생한 복제본**, 연휴 등 일부 날짜는 전날 복사본 → "
        "**원본(ORIGINAL) 스냅샷 31개 목표일만** 학습·평가(사용자 결정)\n"
        f"- 시간순 분할: train {len(ms['split']['train'])}일 / val {len(ms['split']['val'])}일 / test {len(ms['split']['test'])}일 "
        f"({ms['split']['test'][0]}~{ms['split']['test'][-1]}); horizon h 모델은 평가 시작일−h 이후 정답을 학습에 쓰지 않음\n"
        "- 피처는 as-of(목표일−h) 이전 스냅샷의 T+j 예정/작년/기업예상 값과 과거 타깃만 사용. 목표일 당일 값·기상 미사용\n"
        "- 모델 선택은 검증 WAPE 기준(기준선 포함). 예측은 0 이상으로 클리핑 후 평가")
    rows = []
    bt = get_supply_backtest()
    for h, mm in bt["metrics"].items():
        for name, v in mm.items():
            rows.append({"horizon": f"T+{h}", "모델": name + (" ✅" if bt["selected"][int(h)] == name else ""),
                         "MAE": v["MAE"], "RMSE": v["RMSE"], "WAPE": v["WAPE"], "sMAPE": v["sMAPE"], "n": v["n"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("✅ = 검증 구간에서 선택되어 서비스에 쓰이는 모델. 시험 표본은 4일 × 약 108개 제품입니다.")

def date_chart(df: pd.DataFrame, cols: list[str]) -> None:
    """하루 단위 데이터를 날짜 축(MM-DD)으로 그린다. st.line_chart 는 며칠짜리 구간에서 1시간 눈금을 만들어(2026-11-19 확인) 직접 지정."""
    d = df[["Date"] + cols].copy()
    d["Date"] = pd.to_datetime(d["Date"])
    long = d.melt("Date", var_name="구분", value_name="수량")
    short = d["Date"].dt.normalize().nunique() <= 14
    if short:  # 며칠짜리: 하루 한 눈금(문자 축이라 시간 눈금이 생기지 않음)
        long["날짜"] = long["Date"].dt.strftime("%m-%d")
        x = alt.X("날짜:O", sort=sorted(long["날짜"].unique()), axis=alt.Axis(labelAngle=0, title="날짜"))
    else:
        x = alt.X("Date:T", axis=alt.Axis(format="%m-%d", labelAngle=0, title="날짜", tickCount={"interval": "week", "step": 1}))
    ch = alt.Chart(long).mark_line(point=short).encode(
        x=x, y=alt.Y("수량:Q", title="예정 수주량 (개)"),
        color=alt.Color("구분:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("Date:T", title="날짜", format="%Y-%m-%d"), "구분:N", alt.Tooltip("수량:Q", format=",.0f")])
    st.altair_chart(ch.properties(height=260), width="stretch")


prods = list_supply_products()
pn = st.selectbox("Product_Number (비식별)", [p["product_number"] for p in prods],
                  index=[p["product_number"] for p in prods].index("Product_c0"))
hist = product_history(pn)
st.subheader("과거 예정 수주량 (ORIGINAL 스냅샷)")
date_chart(hist, ["T일 예정 수주량"])

st.subheader("2022 홀드아웃 백테스트: 예측 vs 실제 예정 수주량")
b = get_supply_backtest(pn)
rows = b["rows"]
if len(rows):
    for h in sorted(rows.horizon.unique()):
        r = rows[rows.horizon == h].rename(columns={"actual_planned_order_units": "실제 예정 수주량", "predicted": "예측"})
        st.caption(f"T+{h} · 모델 {b['selected'][int(h)]}")
        date_chart(r, ["실제 예정 수주량", "예측"])
    tags(b["provenance"])

st.subheader("시점 지정 예측")
c1, c2 = st.columns(2)
as_of = c1.date_input("예측 기준일(as-of)", date(2022, 3, 12), min_value=date(2022, 1, 26), max_value=date(2026, 12, 31))
h = c2.selectbox("horizon", [1, 2, 3, 4], format_func=lambda x: f"T+{x}일")
if st.button("예측"):
    f = run(forecast_order_quantity, pn, as_of.isoformat(), h)
    if f:
        st.session_state["last_forecast"] = f
f = st.session_state.get("last_forecast")
if f and f["product_number"] == pn:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{f['target_date']} 예측 예정 수주량", f"{f['predicted_planned_order_units']:.1f} 개")
    c2.metric("실제 예정 수주량", "-" if f["actual_planned_order_units"] is None else f"{f['actual_planned_order_units']:.0f} 개")
    c3.metric("80% 구간(검증 잔차)", f"{f['interval_80_from_val_residuals'][0]:.0f} ~ {f['interval_80_from_val_residuals'][1]:.0f}")
    for w in f["warnings"]:
        st.caption("⚠️ " + w)
    tags(f["provenance"])

    st.divider()
    st.subheader("선택: 가상 제조품목 시뮬레이션 계획 초안으로 복사")
    st.warning(MAPPING_WARNING)
    with st.form("bridge"):
        ack = st.checkbox("위 경고를 읽었으며, 실제 매핑이 아닌 가상 시나리오임을 이해합니다.")
        c1, c2, c3 = st.columns(3)
        tgt = c1.selectbox("대상 가상 제조품목(직접 선택)", [""] + list(products()),
                           format_func=lambda x: "— 선택 —" if not x else product_label(x))
        qty = c2.number_input("생산 수량(직접 검토·수정, 주문량=생산량 자동 가정 없음)", 1, 1_000_000,
                              max(int(round(f["predicted_planned_order_units"])), 1))
        pdate = c3.date_input("생산 예정일", date.today() + timedelta(days=3))
        if st.form_submit_button("수주예측 기반 시뮬레이션 초안 만들기"):
            d = run(create_demo_plan_draft_from_order_forecast, f["forecast_id"], tgt, int(qty), pdate.isoformat(),
                    acknowledge_mapping_warning=ack)
            if d:
                st.success(f"초안 생성: {d['draft_id']} (아직 생산계획 아님)")

drafts = [d for d in list_drafts() if d["approval_status"] == "DRAFT"]
if drafts:
    st.subheader("확정 대기 초안")
    st.dataframe(pd.DataFrame(drafts)[["draft_id", "source_product_number", "target_manufacturing_product_id",
                                       "planned_date", "quantity_proposed", "quantity_approved", "model_version"]],
                 hide_index=True, width="stretch")
    with st.form("confirm"):
        did = st.selectbox("초안", [d["draft_id"] for d in drafts])
        ack2 = st.checkbox("서로 다른 기업 데이터의 가상 매핑임을 다시 확인합니다.")
        ok = st.checkbox("이 초안을 확정 생산계획으로 반영합니다.")
        if st.form_submit_button("초안 확정", type="primary"):
            r = run(confirm_demo_plan_draft, did, user_confirmed=ok, acknowledge_mapping_warning=ack2)
            if r:
                st.success(f"확정 생산계획 {r['plan']['plan_id']} 생성 (source=DEMO_SUPPLY_BRIDGE). 이제 kg 예측이 계산됩니다.")
                tags(r["provenance"])
