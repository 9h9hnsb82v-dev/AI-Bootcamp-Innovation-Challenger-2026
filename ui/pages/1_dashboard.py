import pandas as pd
import streamlit as st

from services.production_plan import list_plans
from services.storage_projection import get_storage_projection
from services.waste_forecast_update import compute_update
from ui.common import kg, run, tags

st.title("대시보드")
st.caption("7종 가상 제조품목의 확정 생산계획 기준 폐플라스틱 예상 발생량(kg). KAMP ID40 수주량(개수)은 포함하지 않습니다.")

plans = list_plans()
if not plans:
    st.info("등록된 확정 생산계획이 없습니다. **생산계획** 화면에서 등록하세요.")
rows = []
for p in plans:
    u = run(compute_update, p["plan_id"])
    if u:
        rows.append({"계획": p["plan_id"], "품목": p["product_id"], "계획일": p["plan_date"], "계획수량": p["planned_units"],
                     "출처": p["source"], "최초 예상 kg": u["initial_expected_kg"], "확인 폐기 kg": u["confirmed_scrap_kg"],
                     "갱신 예상 kg": u["updated_expected_total_kg"], "근거": ",".join(u["provenance"])})
df = pd.DataFrame(rows)
c1, c2, c3, c4 = st.columns(4)
c1.metric("확정 계획 수", len(plans))
c2.metric("최초 예상 합계", kg(df["최초 예상 kg"].sum()) if len(df) else "0 kg")
c3.metric("갱신 예상 합계", kg(df["갱신 예상 kg"].sum()) if len(df) else "0 kg")
proj = run(get_storage_projection, None, 7)
if proj:
    c4.metric("7일 후 보관량", kg(proj["stock_after_7_days_kg"]), help=f"용량 {proj['max_capacity_kg']:.0f}kg, 경보 {proj['alert_kg']:.0f}kg")
if len(df):
    st.subheader("품목별 예상 폐기물")
    st.bar_chart(df.groupby("품목")[["최초 예상 kg", "갱신 예상 kg"]].sum())
    st.dataframe(df, hide_index=True, width="stretch")
if proj:
    st.subheader("창고 상태 (가상 정책)")
    if proj["first_over_capacity_date"]:
        st.error(f"용량 초과 예상일: {proj['first_over_capacity_date']}")
    elif proj["first_alert_date"]:
        st.warning(f"경보 기준 도달 예상일: {proj['first_alert_date']}")
    else:
        st.success("7일 내 경보 기준 도달 예상 없음")
    tags(proj["provenance"])
