"""시안 D/F 공통 틀: 메뉴(st.navigation) + 사이드바(예측 기간·창고 설정·시연 데이터) + 시안 CSS.

기존 화면(classic)은 ui/app.py 의 원래 코드가 그대로 그린다. 여기서는 대시보드만 시안 페이지로 바꾸고,
나머지 화면은 같은 페이지 파일을 쓰되 테마(색·글꼴)와 메뉴 구성만 시안을 따른다.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import streamlit as st

from ui.designs import inject_css


PAGE_DEFS = [  # 키, 파일, 제목, 아이콘, 테이스트안(위쪽 메뉴)용 짧은 제목
    ("plan", "2_production_plan.py", "생산계획", "calendar_month", None),
    ("pred", "3_waste_prediction.py", "폐플라스틱 사전 예측", "scale", None),
    ("insp", "4_inspection.py", "이미지 검사", "search", None),
    ("update", "5_forecast_update.py", "검사 후 예측 갱신", "autorenew", None),
    ("storage", "6_storage.py", "보관량 · 수거 추천", "local_shipping", "보관 · 수거"),
    ("layout", "9_factory_layout.py", "공장 배치도", "factory", None),
    ("proc", "10_processors.py", "위탁 처리업체 지도", "location_on", "처리업체"),
    ("supply", "7_supply_forecast.py", "수주량 예측 (ID40)", "trending_up", None),
    ("chat", "8_chatbot.py", "AI 챗봇", "chat", "챗봇"),
    ("evidence", "11_evidence.py", "근거 · 성능", "description", None),
]


def _pages(P: Path, design: str) -> dict:
    pg = {"dash": st.Page(P / f"1_dashboard_{design}.py", title="대시보드", icon=":material/dashboard:",
                          default=True, url_path=f"dashboard_{design}")}  # 기존 1_dashboard.py 와 URL 이름이 겹치지 않게
    for k, f, t, ic, short in PAGE_DEFS:
        pg[k] = st.Page(P / f, title=(short or t) if design == "f" else t, icon=f":material/{ic}:")
    if design == "d":  # 추천안: 왼쪽 메뉴를 4묶음으로
        return {"계획 · 예측": [pg["dash"], pg["plan"], pg["pred"]], "검사": [pg["insp"], pg["update"]],
                "보관 · 처리": [pg["storage"], pg["layout"], pg["proc"]], "지원": [pg["supply"], pg["chat"], pg["evidence"]]}
    # 테이스트안: 위쪽 가는 메뉴. 자주 쓰는 6개는 바로, 나머지는 '더 보기'
    return {"": [pg["dash"], pg["plan"], pg["insp"], pg["storage"], pg["proc"], pg["chat"]],
            "더 보기": [pg["pred"], pg["update"], pg["layout"], pg["supply"], pg["evidence"]]}


def _period() -> None:
    """예측 기간 (기존 화면과 같은 키·같은 규칙: 최대 30일, 끝일 미선택 시 +13일)."""
    t = date.today()
    rng = st.date_input("예측 기간", (t, t + timedelta(days=13)), min_value=t - timedelta(days=365),
                        max_value=t + timedelta(days=400), key="period", format="YYYY-MM-DD",
                        help="보관량 그래프·수거 추천·방별 적재·공장 배치도가 이 기간을 씁니다. 최대 30일.")
    s = rng[0] if isinstance(rng, (tuple, list)) and rng else t
    e = rng[1] if isinstance(rng, (tuple, list)) and len(rng) > 1 else s + timedelta(days=13)
    n = max(1, min((e - s).days + 1, 30))
    st.session_state["as_of"], st.session_state["horizon"] = s, n
    st.caption(f"{n}일 · 최대 30일" + (" — 30일까지만 계산합니다" if (e - s).days + 1 > 30 else ""))


def _demo_admin() -> None:
    with st.expander("시연 데이터 관리", icon=":material/database:"):
        from services import db as _db
        bdir = _db.db_path().parent / "backups"
        backups = sorted(bdir.glob("*.db"), reverse=True)[:5] if bdir.exists() else []
        st.caption(f"DB `{_db.db_path().name}` · 최근 백업: " + (", ".join(b.name for b in backups) if backups else "없음"))
        st.caption("공용 DB 초기화는 '이미지 검사 → 관리 도구'에서 확인 창을 거쳐 실행합니다(직전 상태 자동 백업).")


def run(root: Path, design: str) -> None:
    inject_css(design)
    here = Path(__file__)
    logo = here.with_name(f"{design}_logo.png")  # f: Loop:in 로고(2026-11-22 이름 변경), 없으면 기존 SVG
    st.logo(str(logo if logo.exists() else here.with_name(f"{design}_logo.svg")), size="large")
    nav = st.navigation(_pages(root / "ui/pages", design), position="sidebar" if design == "d" else "top")
    with st.sidebar:
        _period()
        from services.config import facility
        st.caption(f"창고: 용량 {facility()['max_capacity_kg']:,.0f} kg(공장 배치도 보관함 합계) · 기준 보관량 300 kg · 리드타임 2일 — 단중·시연 불량률·창고 정책은 "
                   "사용자 승인 가상값입니다(실측·법령 기준 아님).")
        _demo_admin()
        if design == "d":
            st.caption("다크 모드: 오른쪽 위 ⋮ → Settings → Theme")
    nav.run()
