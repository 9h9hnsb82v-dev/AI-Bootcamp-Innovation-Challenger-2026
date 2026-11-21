import pandas as pd
import streamlit as st

from services.production_plan import list_plans
from services.waste_forecast_update import compute_update, snapshots, update_waste_forecast
from ui.common import kg, run, tags

st.title("검사 후 예측 갱신")
st.latex(r"\text{갱신 kg} = \text{확인 폐기 kg} + (\text{생산·미검사} + \text{미생산}) \times \text{불량률} \times \text{단중}")
st.caption("재작업 대기(REWORK_PENDING) 수량은 승인 정책이 없어 kg에 넣지 않고 별도 표시합니다. 갱신은 해당 계획에만 적용됩니다.")

plans = list_plans()
if not plans:
    st.info("확정 생산계획이 없습니다.")
    st.stop()
plan_id = st.selectbox("생산계획", [p["plan_id"] for p in plans],
                       format_func=lambda x: next(f"{x} · {p['product_id']} {p['planned_units']}개" for p in plans if p["plan_id"] == x))
u = run(compute_update, plan_id)
if u:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최초 예상", kg(u["initial_expected_kg"]))
    c2.metric("확인 폐기", kg(u["confirmed_scrap_kg"]))
    c3.metric("남은 예상", kg(u["remaining_expected_kg"]))
    c4.metric("갱신 예상 합계", kg(u["updated_expected_total_kg"]),
              delta=f"{u['updated_expected_total_kg'] - u['initial_expected_kg']:+.2f} kg")
    st.dataframe(pd.DataFrame([u["buckets"]]), hide_index=True)
    if u["rework_pending_units_not_in_kg"]:
        st.warning(f"재작업 대기 {u['rework_pending_units_not_in_kg']}개 — kg 미반영")
    tags(u["provenance"])
    if st.button("갱신 스냅샷 저장", type="primary"):
        update_waste_forecast(plan_id)
        st.rerun()
s = snapshots(plan_id)
if s:
    df = pd.DataFrame(s)
    st.line_chart(df.set_index("timestamp")[["initial_expected_kg", "updated_expected_total_kg", "confirmed_scrap_kg"]])
    st.dataframe(df, hide_index=True, width="stretch")
