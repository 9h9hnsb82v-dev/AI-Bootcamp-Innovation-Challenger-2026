"""공장 배치도 (2D 편집 · 3D 보기) — 시연용 기본 배치(80m x 50m)를 베이스로, 공장 배치 하나를 저장(2026-11-25).
폐플라스틱 보관함마다 용량을 두고, 용량 합계가 앱 전체 보관 용량이 된다."""
import pandas as pd
import streamlit as st

from services.config import ServiceError
from services.factory_layout import get_layout, live_status, reset_layout, save_layout, type_table
from ui.common import tags
from ui.factory_layout_view import factory_layout_view

ss = st.session_state
st.title("공장 배치도")
st.caption("실제 공장 크기를 모르므로 **시연용 기본 배치(80m × 50m, 가상)**를 베이스로 둡니다. 실제와 다르면 2D 편집에서 "
           "구역을 **끌어서 옮기고**, 선택한 구역의 오른쪽 아래 흰 손잡이로 **크기**를, 위쪽 편집 줄에서 **회전·이름·종류**를 바꾸고, "
           "구역을 **추가·삭제**한 뒤 **저장**하세요. 폐플라스틱 보관함은 선택하면 **용량(kg)**을 바꿀 수 있고, "
           "보관함 용량의 합계가 앱 전체 보관 용량(보관량·수거 추천·대시보드)이 됩니다.")

rev_key = "layout_rev"
ss.setdefault(rev_key, 0)
layout = get_layout()


def on_save() -> None:
    try:
        save_layout(ss["factory_layout"].save)
        ss["layout_notice"] = "공장 배치를 저장했습니다. 보관함 용량 합계가 앱 전체 보관 용량에 반영됩니다."
        ss[rev_key] += 1
    except ServiceError as e:
        ss["layout_notice"] = f"저장 실패 — {e.code}: {e.message}"


def on_reset() -> None:
    reset_layout()
    ss["layout_notice"] = "기본 배치로 되돌렸습니다."
    ss[rev_key] += 1


from datetime import date as _d
as_of = (ss.get("as_of") or _d.today()).isoformat()
dates = live_status(as_of, days=int(ss.get("horizon") or 14)).get("dates") or [as_of]
day = st.select_slider("보관함 표시 날짜 (예측 보관량)", dates, value=dates[0], key="layout_day",
                       format_func=lambda x: f"{x[5:]}({'월화수목금토일'[_d.fromisoformat(x).weekday()]}) "
                                             f"{'현재' if x == dates[0] else '+' + str(dates.index(x)) + '일'}")
live = live_status(as_of, None if day == dates[0] else day, int(ss.get("horizon") or 14))
factory_layout_view(key="factory_layout", layout=layout, rev=ss[rev_key], live=live, types=type_table(),
                    notice=ss.pop("layout_notice", None)
                    or (f"저장된 공장 배치 · {layout.get('updated_at', '')}" if layout.get("saved") else "기본 배치(아직 저장 안 함)"),
                    on_save=on_save, on_reset=on_reset)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**구역 종류**")
    st.dataframe(pd.DataFrame(type_table()).rename(columns={"type": "코드", "name": "이름", "height_m": "3D 높이(m)",
                                                           "color": "색"}), hide_index=True)
with c2:
    st.markdown(f"**폐플라스틱 보관량 · {day} (3D 보관함 높이·색)**")
    if live.get("scrap_kg") is not None:
        st.metric("보관량", f"{live['scrap_kg']:,.0f} kg", help="시연 기준 현재 보관량 + 작업자 폐기 확정분(보관량·수거 추천 화면과 같은 값)")
        st.progress(min(live["fill"], 1.0), text=f"용량 {live['capacity_kg']:,.0f} kg 중 {live['fill']:.0%} · 경고 기준 {live['alert_kg']:,.0f} kg")
        if live.get("bins"):
            st.dataframe(pd.DataFrame([{"순서": n, "보관함": b["label"], "보관량(kg)": round(b["kg"]),
                                        "용량(kg)": round(b["capacity_kg"]), "채움": f"{b['fill']:.0%}"}
                                       for n, b in enumerate(live["bins"], 1)]), hide_index=True)
            st.caption("앞 보관함이 가득(100%) 차면 다음 보관함(배치에 추가한 순서)으로 넘깁니다. 저장한 배치 기준입니다.")
    else:
        st.info("보관량을 계산하지 못했습니다.")
    st.caption("3D 보기는 브라우저가 인터넷에서 three.js를 불러옵니다. 색: 초록 < 50% ≤ 주황 < 경고 기준(80%) ≤ 빨강")
tags(["DEMO_ASSUMED"])
