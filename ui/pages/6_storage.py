"""보관량 예측 · 수거 추천 · 방별 적재 모습 (2026-11-14 팀원 앱 방식 반영: 카드 + 경고선 그래프 + 방별 2D/3D + 날짜 슬라이더)."""
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from services.collection_recommendation import recommend_collection
from services.config import facility
from services.storage_rooms import cancel_collection, list_collections, room_projection, schedule_collection
from ui.common import run, tags
from ui.storage_view import plan_svg, rooms_3d

ss = st.session_state
as_of = ss.get("as_of") or date.today()
days = int(ss.get("horizon") or 14)
fac = facility()
cap, alert = float(fac["max_capacity_kg"]), float(fac["max_capacity_kg"]) * float(fac["alert_ratio"])

st.title("가상 공통 폐플라스틱 보관창고 적재량 예측")
r = run(recommend_collection, as_of.isoformat(), days)
if not r:
    st.stop()
proj = pd.DataFrame(r["projection_without_collection"])
at = lambda i: proj.end_kg.iloc[min(i, len(proj) - 1)]  # noqa: E731
c = st.columns(4)
for col, (lab, i) in zip(c[:3], [("현재", 0), ("3일 후", 3), ("7일 후", 7)]):
    d = (as_of + timedelta(days=i)).isoformat()
    col.metric(f"{lab} ({d})", f"{at(i):,.1f} kg", delta=f"적재율 {at(i) / cap:.0%}", delta_color="off")
c[3].metric("용량 초과 위험일", r["over_capacity_date"] or "없음",
            delta=f"경고선 {r['alert_date'][5:] if r['alert_date'] else '없음'}", delta_color="off")

df = proj.assign(date=pd.to_datetime(proj.date))
base = alt.Chart(df).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d", labelAngle=0, tickCount=days)))
line = base.mark_line(color="#3b5bdb", strokeWidth=2.5).encode(y=alt.Y("end_kg:Q", title="예상 적재량 (kg)"),
                                                              tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("end_kg:Q", format=",.1f", title="kg")])
pts = base.mark_point(color="#3b5bdb", filled=True, size=45).encode(y="end_kg:Q")
rules = pd.DataFrame({"y": [cap, alert], "label": ["최대 용량", "경고선"]})
rule = alt.Chart(rules).mark_rule(strokeDash=[5, 4], color="#adb5bd").encode(y="y:Q")
txt = alt.Chart(rules).mark_text(align="left", dx=4, dy=-8, color="#adb5bd").encode(x=alt.value(0), y="y:Q", text="label:N")
st.altair_chart((line + pts + rule + txt).properties(height=380), width="stretch")
st.caption(f"현재 보관량 = 시연 기준 보관량 {fac['current_stock_kg']:.0f}kg + 작업자 폐기 확정 − 수거(장부), 이후는 확정 생산계획의 남은 예상 발생량을 "
           "계획일에 더한 값입니다(수거 예약 반영 전). 중량(kg) 기준이며 부피는 추정하지 않습니다. 창고 설정은 가상 운영값입니다.")

# ── 수거 요청 권장 ──
st.subheader("수거 요청 권장")
state = {"NO_COLLECTION_NEEDED": ("🟢 수거 불필요", st.success), "RECOMMENDED": ("🟠 수거 요청 권장", st.warning),
         "URGENT": ("🔴 즉시 요청", st.error)}[r["status"]]
st.markdown(f"#### {state[0]}")
if r["status"] != "NO_COLLECTION_NEEDED":
    k = st.columns(3)
    k[0].metric("수거 요청 마감일", r["request_by_date"])
    k[1].metric("수거 예정일", r["recommended_collection_date"])
    k[2].metric("예상 수거량", f"{r['expected_collection_kg']:,.1f} kg")
    state[1](f"경고선 도달 {r['alert_date']} · 용량 초과 {r['over_capacity_date'] or '-'} → {r['message']} (리드타임 {r['lead_time_days']}일)")
    if st.button("권장 수거 일정 예약(SCHEDULED)", help="장부에 수거(COLLECTION_OUT) 예정 이벤트를 넣어 예측에 반영합니다(가상 시연)."):
        run(schedule_collection, r["recommended_collection_date"], r["expected_collection_kg"])
        st.rerun()
    ss["expected_collection_kg"] = r["expected_collection_kg"]
else:
    st.success(r["message"])
    ss["expected_collection_kg"] = None
with st.expander("수거 예약 목록"):
    cols = list_collections()
    if not cols:
        st.caption("예약된 수거가 없습니다.")
    for i, e in enumerate(cols):
        a, b = st.columns([4, 1])
        a.write(f"{e['date']} · {e['kg']:,.1f} kg · `{e['source_id']}`")
        if str(e["source_id"]).startswith("COLLECT:") and b.button("취소", key=f"cc{i}"):
            cancel_collection(e["source_id"])
            st.rerun()

# ── 방별 적재 모습 ──
st.subheader("방별 적재 모습")
rp = run(room_projection, as_of.isoformat(), days)
if rp:
    v1, v2 = st.columns([1, 3])
    view = v1.radio("보기", ["2D 평면도", "3D 입체"], horizontal=True, key="room_view")
    dates = [d["date"] for d in rp["days"]]
    WD = "월화수목금토일"

    def dlabel(d: str) -> str:
        k = (date.fromisoformat(d) - as_of).days
        return f"{d[5:]}({WD[date.fromisoformat(d).weekday()]}) {'현재' if k == 0 else f'+{k}일'}"

    peak_day = max(rp["days"], key=lambda x: x["total_kg"])["date"]
    if ss.get("room_day") not in dates:
        ss["room_day"] = dates[0]
    with v2:
        q = st.columns(4)
        for col, (lab, tgt) in zip(q, [("현재", dates[0]), ("+3일", dates[min(3, len(dates) - 1)]),
                                       ("+7일", dates[min(7, len(dates) - 1)]), ("최대 적재일", peak_day)]):
            col.button(lab, key=f"jump_{lab}", width="stretch", on_click=lambda t=tgt: ss.update(room_day=t),
                       help=dlabel(tgt))
        pick = st.select_slider("날짜", dates, key="room_day", format_func=dlabel)
    day = next(d for d in rp["days"] if d["date"] == pick)
    mc = st.columns(len(rp["rooms"]))
    for col, rm in zip(mc, rp["rooms"]):
        s = day["rooms"][rm["id"]]
        col.metric(rm["name"], f"{s['kg']:,.0f} / {rm['capacity_kg']:,.0f} kg", delta=f"{s['ratio']:.0%} · {s['state']}", delta_color="off")
    label = dlabel(pick)
    if view == "2D 평면도":
        import base64
        svg = plan_svg(rp["rooms"], day, f"2D 평면도 · {label}", rp["bag_kg"], rp["stack"])
        st.markdown(f'<img alt="방별 적재 2D 평면도 {label}" style="width:100%;max-width:900px" '
                    f'src="data:image/svg+xml;base64,{base64.b64encode(svg.encode("utf-8")).decode()}"/>', unsafe_allow_html=True)
    else:
        rooms_3d(key="rooms3d", rooms=rp["rooms"], day=day, bag_kg=rp["bag_kg"], stack=rp["stack"])
        st.caption("3D: 마우스로 끌어 회전, 휠로 확대·축소. 브라우저가 three.js를 인터넷에서 불러옵니다.")
    st.caption("방 구성: " + ", ".join(f"{rm['name']} ({'·'.join(rm['products'])})" for rm in rp["rooms"])
               + " — 가정값. 품목 정보가 없는 기준 보관량·수거는 방 용량 비율로 배분. 방별 경고선 = 방 용량의 80%. "
               "그림은 포대 1개 = 25kg, 최대 2단으로 무게를 칸으로 환산한 것이며 실제 부피·배치가 아닙니다.")
    peak = []
    for rm in rp["rooms"]:
        best = max(rp["days"], key=lambda x: x["rooms"][rm["id"]]["kg"])
        s_ = best["rooms"][rm["id"]]
        peak.append({"방": rm["name"], "기간 중 최대 적재": f"{s_['kg']:,.0f} / {rm['capacity_kg']:,.0f} kg",
                     "적재율": f"{s_['ratio']:.0%}", "상태": s_["state"], "날짜": dlabel(best["date"])})
    st.caption(f"기간 중 방별 최대 적재 ({dates[0]} ~ {dates[-1]})")
    st.dataframe(pd.DataFrame(peak), hide_index=True)
    with st.expander("일자별 표"):
        st.dataframe(pd.DataFrame([{"날짜": d["date"], "합계 kg": d["total_kg"],
                                    **{rm["name"]: d["rooms"][rm["id"]]["kg"] for rm in rp["rooms"]}} for d in rp["days"]]),
                     hide_index=True)
tags(r["provenance"] + ["DEMO_SYNTHETIC_WEIGHT"])
