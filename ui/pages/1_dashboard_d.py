"""대시보드 — 추천안 (A 기본 + C 5단계 카드). 숫자는 모두 DB·서비스 실제 값."""
import html

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
rec = S["rec"]
need = bool(rec) and rec["status"] != "NO_COLLECTION_NEEDED"
for e in S["errors"]:
    st.error(e)

# ── 머리말 ──
h = st.columns([4, 1], vertical_alignment="bottom")
with h[0]:
    st.title("대시보드")
    st.caption("확정 생산계획 기준 폐플라스틱 예상 발생량 (kg) · KAMP ID40 수주량은 포함하지 않음")
h[1].button("수거 예약하기", type="primary", icon=":material/local_shipping:", width="stretch", disabled=not need,
            help=None if need else "기간 안에 경보 기준에 닿지 않아 수거가 필요 없습니다.", key="d_book_top",
            on_click=open_booking)

# ── 5단계 카드 ──
pct = S["peak_kg"] / cap if cap else 0
n_pend = S["pending"]["REWORK"] + S["pending"]["CHECK"]
n_prod = len({r["품목"] for r in S["rows"]})
steps = [
    ("plan", "1 · 생산계획", f"{len(S['plans'])}", "건 확정", f"{n_prod}개 품목", False),
    ("pred", "2 · 사전 예측", f"{S['initial_kg']:,.0f}", "kg", "계획 기준 예상 발생량", False),
    ("insp", "3 · 이미지 검사", f"{S['scrap_kg']:,.0f}", "kg 반영",
     f"확인 대기 {n_pend}건" if n_pend else "작업자 확정 후에만 변경", False),
    ("storage", "4 · 보관" + (" — 조치 필요" if pct >= 1 else (" — 경보" if S["peak_kg"] >= alert else "")),
     f"{pct:.0%}", "", f"기간 최대 {S['peak_kg']:,.0f} / {cap:,.0f} kg", S["peak_kg"] >= alert),
    ("proc", "5 · 수거 · 위탁", f"예약 {len(S['upcoming'])}건" if S["upcoming"] else "예약 없음", "",
     f"다음 {md(S['upcoming'][0]['date'])}" if S["upcoming"] else "5톤 카고 · 처리업체 지도", False),
]
cols = st.columns(5)
for col, (pg, lab, val, unit, sub, hot) in zip(cols, steps):
    with col.container(border=True, key=f"d_step_{pg}" + ("_hot" if hot else ""), height="stretch"):
        st.html(f'<div class="d-step"><span class="d-step-lab">{html.escape(lab)}</span>'
                f'<span class="d-step-val">{html.escape(val)}<small>{(" " + html.escape(unit)) if unit else ""}</small></span>'
                f'<span class="d-step-sub">{html.escape(sub)}</span></div>')
        st.page_link(PAGES[pg], label="열기", icon=":material/arrow_forward:", icon_position="right")

# ── 품목별 · 지금 할 일 · 정확도 ──
left, right = st.columns([3, 2], gap="medium")
with left.container(border=True, height="stretch"):
    st.subheader("품목별 예상 폐기물 (kg)")
    if S["by_product"]:
        bp = pd.DataFrame([{"품목": k, "최초 예상": v[0], "갱신 예상": v[1]} for k, v in S["by_product"].items()])
        order = bp["품목"].tolist()
        dark = st.context.theme.type == "dark"
        ink, bar_c = ("#E2E8F0", "#94A3B8") if dark else ("#0F172A", "#334155")
        base = alt.Chart(bp).encode(y=alt.Y("품목:N", sort=order, title=None, axis=alt.Axis(labelColor=ink, labelFontSize=13)))
        bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
            x=alt.X("갱신 예상:Q", title=None, axis=alt.Axis(grid=False, labels=False, ticks=False, domain=False)),
            color=alt.value(bar_c),
            tooltip=["품목", alt.Tooltip("최초 예상:Q", format=",.1f"), alt.Tooltip("갱신 예상:Q", format=",.1f")])
        txt = base.mark_text(align="left", dx=6, color=ink, fontSize=13).encode(
            x="갱신 예상:Q", text=alt.Text("갱신 예상:Q", format=",.1f"))
        st.altair_chart((bars + txt).properties(height=40 * len(bp) + 20), width="stretch")
    else:
        st.info("등록된 확정 생산계획이 없습니다.")
        st.page_link(PAGES["plan"], label="생산계획 등록하기", icon=":material/add:")

with right:
    with st.container(border=True, key="d_todo"):
        st.subheader("지금 할 일")
        any_task = False
        if need:
            any_task = True
            when = rec["over_capacity_date"] or rec["alert_date"]
            what = "창고 용량 초과 예상" if rec["over_capacity_date"] else "경보 기준 도달 예상"
            with st.container(key="d_task_hot", horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{md(when)} {what}**  \n경보 {alert:,.0f} kg · {md(rec['request_by_date'])}까지 수거를 요청하세요")
                st.button("예약", type="primary", key="d_book_task", on_click=open_booking)
        if n_pend:
            any_task = True
            with st.container(key="d_task", horizontal=True, vertical_alignment="center"):
                st.markdown(f"**처리 대기함 {n_pend}건**  \n재작업 보류 {S['pending']['REWORK']} · 작업자 확인 필요 {S['pending']['CHECK']}")
                if st.button("열기", key="d_open_queue"):
                    st.switch_page(PAGES["insp"])
        for c in S["upcoming"][:2]:
            any_task = True
            st.success(f"{md(c['date'])} 수거 {c['kg']:,.1f} kg 예약됨", icon=":material/event_available:")
        if not S["plans"]:
            any_task = True
            st.info("확정 생산계획을 먼저 등록하세요.")
        if not any_task:
            st.success("지금 처리할 일이 없습니다.", icon=":material/check_circle:")
    with st.container(border=True):
        st.subheader("AI 자동 판정 정확도")
        acc = S["acc"]
        if acc:
            st.caption(f"처음 보는 촬영일 {acc['days']}일 기준 · 두 모델이 엇갈리면 작업자 확인")
            ko = {"DOOR": "문", "BUMPER": "범퍼", "FENDER": "휀더"}
            st.html("".join(
                f'<div class="d-acc"><span>{ko[k]}</span><div class="d-acc-bar"><i style="width:{v["acc"]*100:.1f}%"'
                f'{" class=warn" if v["acc"] < 0.9 else ""}></i></div><span>{v["acc"]:.1%}</span></div>'
                for k, v in acc["parts"].items()))
            st.page_link(PAGES["evidence"], label="근거 · 성능 보기", icon=":material/arrow_forward:", icon_position="right")
        else:
            st.caption("정직한 재평가 결과(reports/vision/sd_eval.json)가 없어 표시하지 않습니다.")

# ── 확정 생산계획 표 ──
with st.container(border=True):
    st.subheader("확정 생산계획")
    if S["rows"]:
        st.dataframe(pd.DataFrame(S["rows"]), hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(k, format="%.1f kg")
                                    for k in ("최초 예상 kg", "확인 폐기 kg", "갱신 예상 kg")})
    else:
        st.caption("없음")
if rec:
    tags(rec["provenance"])
booking(rec if need else None)
