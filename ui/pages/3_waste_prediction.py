import json

import pandas as pd
import streamlit as st

from services.config import KAMP_BASELINE_FILE, ROOT, defect_rate, products
from services.production_plan import list_plans
from services.waste_forecast import expected_scrap, predict_preproduction_waste
from ui.common import kg, run, tags

st.title("폐플라스틱 사전 예측")
st.latex(r"\text{kg} = \text{계획수량} \times \text{불량률} \times \text{가상 단중} \times \text{scrap\_fraction}(1.0)")

st.subheader("품목별 불량률 근거")
rows = []
for pid, p in products().items():
    r = run(defect_rate, pid)
    if r:
        rows.append({"품목": pid, "가상 단중(kg)": p["unit_weight_kg"], "불량률": r["rate"],
                     "95% 구간": "" if not r["ci95"] else f"{r['ci95'][0]:.4%} ~ {r['ci95'][1]:.4%}",
                     "학습표본(불량/전체)": "" if not r["sample"] else f"{r['sample']['train_defects']}/{r['sample']['train_n']}",
                     "출처": r["provenance"][0], "경고": ",".join(r["warnings"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={"불량률": st.column_config.NumberColumn(format="%.4f")})
st.caption("KAMP 품목: `_id` 중복 제거, 날짜 단위 시간순 앞 70% 구간, Jeffreys 사후평균((k+0.5)/(n+1)). "
           "AI-Hub 품목: 사용자 승인 시연값(이미지 학습셋 불량 비율과 무관).")
meta_p = ROOT / "models/kamp/xgb_sensor_meta.json"
if meta_p.exists():
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    with st.expander(f"보조 모델 XGBOOST_SENSOR 상태: {meta['status']}"):
        st.write("샷 단위 공정 센서 불량확률 모델(생산 중/후 센서값 필요). 홀드아웃 성능이 낮아 kg 계산에는 사용하지 않습니다.")
        st.json(meta["metrics"])

st.subheader("시나리오 비교")
c1, c2 = st.columns(2)
units = c1.number_input("공통 계획 수량", 1, 1_000_000, 1000, step=100)
sel = c2.multiselect("품목", list(products()), default=list(products()))
res = [r for r in (run(expected_scrap, p, units) for p in sel) if r]
if res:
    df = pd.DataFrame([{"품목": r["product_id"], "예상 불량수": r["expected_defective_units"], "예상 kg": r["expected_scrap_kg"],
                        "kg 95% 하한": r.get("expected_scrap_kg_ci95", [None, None])[0],
                        "kg 95% 상한": r.get("expected_scrap_kg_ci95", [None, None])[1]} for r in res])
    st.bar_chart(df.set_index("품목")["예상 kg"])
    st.dataframe(df, hide_index=True, width="stretch")
    st.metric("합계", kg(df["예상 kg"].sum()))

st.subheader("확정 계획별 예측")
for p in list_plans():
    r = run(predict_preproduction_waste, p["plan_id"])
    if r:
        st.write(f"**{p['plan_id']}** {p['product_id']} {p['planned_units']}개 ({p['plan_date']}) → {kg(r['expected_scrap_kg'])}")
        tags(r["provenance"])
