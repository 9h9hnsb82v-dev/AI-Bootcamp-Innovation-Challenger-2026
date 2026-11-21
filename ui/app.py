"""Re:Source Predict AI — Streamlit 진입점.  실행: .venv/Scripts/streamlit run ui/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Re:Source Predict AI", page_icon="♻️", layout="wide")

from ui.access_gate import require_password  # noqa: E402

require_password()  # 공개 주소용 접속 비밀번호(secrets.toml 의 app_password 가 있을 때만, 2026-11-20)

from ui.sandbox import enable as _sandbox  # noqa: E402

_sandbox()  # 체험 모드(RESOURCE_SANDBOX=1): 세션마다 시연 DB 사본 — 다른 사람 화면에 영향 없음

from ui.warmup import start as _warmup  # noqa: E402

_warmup()  # 서버 프로세스당 한 번: 모델·지도·문서 색인을 백그라운드로 미리 불러 둔다(탭 첫 진입 대기 줄이기)

# 시안 선택(2026-11-16): RESOURCE_DESIGN=d|f 이면 시안 틀로 그리고 끝낸다. 비어 있거나 classic 이면 아래 기존 코드 그대로.
from ui.designs import current as _design  # noqa: E402

if _design() != "classic":
    from ui.designs.shell import run as _run_design
    _run_design(ROOT, _design())
    st.stop()

P = ROOT / "ui/pages"
pages = [
    st.Page(P / "1_dashboard.py", title="대시보드", icon="📊", default=True),
    st.Page(P / "2_production_plan.py", title="생산계획", icon="🗓️"),
    st.Page(P / "3_waste_prediction.py", title="폐플라스틱 사전 예측", icon="⚖️"),
    st.Page(P / "4_inspection.py", title="이미지 검사", icon="🔍"),
    st.Page(P / "5_forecast_update.py", title="검사 후 예측 갱신", icon="🔄"),
    st.Page(P / "6_storage.py", title="보관량·수거 추천", icon="🚛"),
    st.Page(P / "9_factory_layout.py", title="공장 배치도", icon="🏭"),
    st.Page(P / "10_processors.py", title="위탁 처리업체 지도", icon="🗺️"),
    st.Page(P / "11_evidence.py", title="근거·성능", icon="📑"),
    st.Page(P / "7_supply_forecast.py", title="미래 수주량 예측 (KAMP ID 40)", icon="📦"),
    st.Page(P / "8_chatbot.py", title="AI 챗봇", icon="💬"),
]
with st.sidebar:
    from datetime import date as _date
    st.markdown("**Re:Source Predict AI** v3.0")
    st.caption("생산계획 기반 폐플라스틱(kg) 사전예측 → 이미지 검사 갱신 → 보관·수거 최적화")
    # 앱 전체 공통 기준(사용자 요청 2026-11-14, 팀원 앱 반영): 보관량·수거·배치도 화면이 이 값을 쓴다
    from datetime import timedelta as _td
    _t = _date.today()
    _rng = st.date_input("예측 기간 (시작일 ~ 끝일)", (_t, _t + _td(days=13)), min_value=_t - _td(days=365),
                         max_value=_t + _td(days=400), key="period", format="YYYY-MM-DD",
                         help="보관량 그래프·수거 추천·방별 적재·공장 배치도가 이 기간을 씁니다. 최대 30일.")
    _s = _rng[0] if isinstance(_rng, (tuple, list)) and _rng else _t
    _e = _rng[1] if isinstance(_rng, (tuple, list)) and len(_rng) > 1 else _s + _td(days=13)  # 끝일을 아직 안 골랐으면 +13일
    _days = max(1, min((_e - _s).days + 1, 30))
    if (_e - _s).days + 1 > 30:
        st.caption("⚠️ 최대 30일까지만 계산합니다.")
    st.session_state["as_of"], st.session_state["horizon"] = _s, _days  # 각 화면이 읽는 공통 값
    st.caption(f"{_s} ~ {_s + _td(days=_days - 1)} · {_days}일")
    st.warning("보관창고 설정: 승인된 가상값 · 용량 1,000kg · 기준 보관량 300kg · 리드타임 2일", icon=None)
    st.caption("모든 단중·시연 불량률·창고 정책은 사용자 승인 가상값입니다. 실측·법령 기준이 아닙니다.")
    with st.expander("시연 데이터 관리"):
        from services import db as _db
        st.caption(f"DB: `{_db.db_path().name}` · 백업 폴더 `{_db.db_path().parent.name}/backups`")
        bdir = _db.db_path().parent / "backups"
        backups = sorted(bdir.glob("*.db"), reverse=True)[:5] if bdir.exists() else []
        st.caption("최근 백업: " + (", ".join(b.name for b in backups) if backups else "없음"))
        st.caption("공용 DB 초기화는 '이미지 검사 → 관리 도구'에서 확인 창을 거쳐 실행합니다(직전 상태 자동 백업).")
st.navigation(pages).run()
