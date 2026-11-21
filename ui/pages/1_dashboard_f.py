"""대시보드 — 테이스트안 (design-taste-frontend). 결론 한 문장 + 실제 보관 추이 + 흐름도 + 큰 숫자 하나.
숫자는 모두 DB·서비스 실제 값이고, 문장도 그 값에서 만든다."""
import html
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from services.config import facility
from ui.common import tags
from ui.designs.data import PAGES, booked_toast, booking, md, open_booking, summary

booked_toast()
S = summary()
fac = facility()
cap, alert = float(fac["max_capacity_kg"]), float(fac["max_capacity_kg"]) * float(fac["alert_ratio"])
rec, as_of = S["rec"], S["as_of"]
need = bool(rec) and rec["status"] != "NO_COLLECTION_NEEDED"
for e in S["errors"]:
    st.error(e)


def when(d: str) -> str:
    k = (date.fromisoformat(d) - as_of).days
    return "오늘" if k == 0 else ("내일" if k == 1 else f"{md(d)}에")


# ── 결론 한 문장 ──
if rec and rec["over_capacity_date"]:
    head = f"폐플라스틱 보관함이<br>{when(rec['over_capacity_date'])} 용량을 넘습니다"
elif rec and rec["alert_date"]:
    head = f"폐플라스틱 보관함이<br>{when(rec['alert_date'])} 경보선에 닿습니다"
else:
    head = "폐플라스틱 보관함은<br>기간 내내 여유가 있습니다"
fact = (f"지금 {S['now_kg']:,.0f} kg에 기간 내 생산계획분 {S['gen_kg']:,.0f} kg이 더해져 "
        f"<strong>최대 {S['peak_kg']:,.0f} kg</strong>이 됩니다. 보관 용량은 {cap:,.0f} kg입니다.")

hero, frame = st.columns([1, 1.15], gap="large", vertical_alignment="center")
with hero:
    st.html(f'<div class="f-enter"><h1 class="f-h1">{head}</h1><p class="f-fact">{fact}</p></div>')
    with st.container(horizontal=True, key="f_cta"):
        st.button("수거 예약하기", key="f_book", disabled=not need, on_click=open_booking,
                  help=None if need else "기간 안에 경보 기준에 닿지 않아 수거가 필요 없습니다.")
        if st.button("보관 추이 보기", key="f_trend"):
            st.switch_page(PAGES["storage"])
    if S["upcoming"]:
        c = S["upcoming"][0]
        st.caption(f"예약된 수거: {md(c['date'])} · {c['kg']:,.1f} kg")

with frame.container(key="f_frame"):
    st.html(f'<div class="f-bar"><i></i><i></i><i></i><span>보관량 · {S["days"]}일</span></div>')
    if S["proj"]:
        # 계단 그래프: 기간 시작 시점의 보관량(start) → 각 날짜가 끝난 뒤 보관량(end). 오늘 생산분이 더해지는 순간이 보이게.
        pr = S["proj"]
        df = pd.DataFrame([{"date": pr[0]["date"], "end_kg": pr[0]["start_kg"]}]
                          + [{"date": (pd.Timestamp(d["date"]) + pd.Timedelta(days=1)).date().isoformat(), "end_kg": d["end_kg"]} for d in pr])
        df = df.assign(date=lambda d: pd.to_datetime(d.date))
        x = alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d", labelAngle=0, grid=False, tickCount=4))
        y = alt.Y("end_kg:Q", title=None, scale=alt.Scale(domainMin=0), axis=alt.Axis(format=",.0f", tickCount=4))
        area = alt.Chart(df).mark_area(interpolate="step-after", color="#a1ffcb", opacity=0.6, line={"color": "#232323", "strokeWidth": 2}).encode(
            x=x, y=y, tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d", title="날짜"), alt.Tooltip("end_kg:Q", format=",.1f", title="kg")])
        rules = pd.DataFrame({"y": [cap, alert], "label": [f"용량 {cap:,.0f}", f"경보 {alert:,.0f}"], "dash": ["5,4", "2,4"]})
        rule = alt.Chart(rules).mark_rule(color="#5b5b5b").encode(y="y:Q", strokeDash=alt.StrokeDash("dash:N", legend=None, scale=None))
        lab = alt.Chart(rules).mark_text(align="left", dx=4, dy=-7, color="#5b5b5b", fontSize=11).encode(
            x=alt.value(0), y="y:Q", text="label:N")
        st.altair_chart((area + rule + lab).properties(height=280).configure_view(stroke=None), width="stretch")
    else:
        st.caption("보관 예측을 불러오지 못했습니다.")

# ── 생산계획에서 수거까지 ──
with st.container(key="f_flow"):
    st.html('<h2 class="f-h2">생산계획에서 수거까지</h2>')
    a, l1, b, l2, lanes = st.columns([2, 0.6, 2.2, 0.6, 3.6], vertical_alignment="center", gap="small")
    a.html(f'<div class="f-node"><span>생산계획</span><b>{len(S["plans"])}건</b></div>')
    l1.html('<div class="f-wire"><span class="f-pulse"></span></div>')
    b.html(f'<div class="f-node dark"><span>폐플라스틱 사전 예측</span><b>{S["initial_kg"]:,.0f} kg</b></div>')
    l2.html('<div class="f-wire"><span class="f-pulse d2"></span></div>')
    over = S["peak_kg"] >= alert
    n_pend = S["pending"]["REWORK"] + S["pending"]["CHECK"]
    with lanes:
        for k, pg, lab, val in [
            ("insp", "insp", "이미지 검사 · 작업자 확정분" + (f" (대기 {n_pend}건)" if n_pend else ""), f"{S['scrap_kg']:,.0f} kg"),
            ("storage" + ("_hot" if over else ""), "storage", "보관 · " + ("용량 초과" if S["peak_kg"] >= cap else ("경보" if over else "여유")),
             f"{S['peak_kg']:,.0f} / {cap:,.0f} kg"),
            ("proc", "proc", "수거 · 위탁 (5톤 카고)", f"예약 {len(S['upcoming'])}건" if S["upcoming"] else "예약 없음"),
        ]:
            with st.container(key=f"f_lane_{k}"):
                st.page_link(PAGES[pg], label=f"{lab}　**{val}**", width="stretch")

# ── 품목별 · 큰 숫자 ──
left, right = st.columns([3, 2], gap="large")
with left.container(key="f_card_prod", height="stretch"):
    st.html('<h2 class="f-h3">품목별 예상 폐기물</h2>')
    if S["by_product"]:
        mx = max(v[1] for v in S["by_product"].values()) or 1
        st.html("".join(f'<div class="f-row"><span>{html.escape(k)}</span><div class="f-track"><i style="width:{v[1] / mx * 100:.1f}%"></i></div>'
                        f'<span class="f-mono">{v[1]:,.1f}</span></div>' for k, v in S["by_product"].items()))
    else:
        st.caption("등록된 확정 생산계획이 없습니다.")
    st.page_link(PAGES["plan"], label="생산계획 열기 →")
with right.container(key="f_card_acc", height="stretch"):
    acc = S["acc"]
    if acc and "FENDER" in acc["parts"]:
        fe = acc["parts"]["FENDER"]
        share = 1 - fe["auto"]
        frac = "다섯 장 중 한 장" if 0.15 <= share < 0.25 else f"{share:.0%}"
        others = ", ".join(f"{ko} {acc['parts'][k]['acc']:.1%}" for k, ko in (("DOOR", "문"), ("BUMPER", "범퍼")) if k in acc["parts"])
        st.html(f'<h2 class="f-h3">휀더는 {frac}을 작업자가 봅니다</h2><div class="f-big">{fe["acc"]:.1%}</div>'
                f'<p class="f-note">두 모델이 같은 답을 낸 휀더 사진의 정확도입니다. 처음 보는 촬영일 {acc["days"]}일 사진으로 쟀습니다. {others}.</p>')
    else:
        st.html('<h2 class="f-h3">AI 자동 판정 정확도</h2><p class="f-note">정직한 재평가 결과가 없어 표시하지 않습니다.</p>')
    st.page_link(PAGES["insp"], label="처리 대기함 열기 →")

with st.expander("확정 생산계획 표", icon=":material/table:"):
    if S["rows"]:
        st.dataframe(pd.DataFrame(S["rows"]), hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(k, format="%.1f kg")
                                    for k in ("최초 예상 kg", "확인 폐기 kg", "갱신 예상 kg")})
    else:
        st.caption("없음")
if rec:
    tags(rec["provenance"])
booking(rec if need else None)
