"""공개 주소로 열 때의 접속 비밀번호 화면 (2026-11-20, 대회 시연용 팀 VM 공개 주소).

비밀번호는 코드에 두지 않는다. `.streamlit/secrets.toml` 또는 환경변수에 있을 때만 켜진다.
- `app_password`(RESOURCE_APP_PASSWORD): 체험용 — 체험 모드(ui/sandbox.py)가 켜져 있으면 사람마다 빈 데이터
- `presenter_password`(RESOURCE_PRESENTER_PASSWORD): 발표자용 — 체험 모드를 건너뛰고 팀 공용 데이터(시연 계획)로 들어간다
둘 다 없으면(로컬 개발) 아무 것도 하지 않는다. 브라우저 세션마다 한 번 입력하면 된다.
"""
from __future__ import annotations

import hmac
import os
import time

import streamlit as st

MAX_TRIES = 5          # 세션당 연속 실패 허용 횟수
LOCK_SECONDS = 60      # 초과 시 잠금 시간
PRESENTER_KEY = "_presenter"

_BROKEN = object()  # secrets.toml 이 있는데 읽을 수 없음 → 막는다(fail closed)


def _secret(name: str, env: str):
    v = os.environ.get(env)
    if v:
        return v.strip()
    try:
        v = st.secrets.get(name)
        return v.strip() if isinstance(v, str) and v.strip() else None
    except FileNotFoundError:  # secrets.toml 자체가 없음(로컬 개발) → 비밀번호 없음
        return None
    except Exception:  # noqa: BLE001 - 파일은 있는데 형식 오류: 비밀번호 설정이 깨졌을 수 있으니 열어 주지 않는다
        return _BROKEN


def _same(a: str, b: str | None) -> bool:
    return bool(b) and hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def is_presenter() -> bool:
    return bool(st.session_state.get(PRESENTER_KEY))


def require_password() -> None:
    visitor = _secret("app_password", "RESOURCE_APP_PASSWORD")
    presenter = _secret("presenter_password", "RESOURCE_PRESENTER_PASSWORD")
    if visitor is _BROKEN or presenter is _BROKEN:
        st.error("접속 설정을 읽을 수 없습니다. 관리자에게 문의하세요.")
        st.stop()
    ss = st.session_state
    if not (visitor or presenter) or ss.get("_access_ok"):
        return
    st.markdown("## Re:Source Predict AI")
    st.caption("대회 시연용 주소입니다. 안내받은 접속 비밀번호를 입력하세요.")
    locked_until = ss.get("_access_locked_until", 0.0)
    if time.time() < locked_until:
        st.error(f"비밀번호를 여러 번 틀렸습니다. {int(locked_until - time.time()) + 1}초 뒤에 다시 시도하세요.")
        st.stop()
    with st.form("access_gate", clear_on_submit=True):
        pw = st.text_input("접속 비밀번호", type="password")
        ok = st.form_submit_button("들어가기", type="primary")
    if ok:
        pw = pw.strip()
        if _same(pw, presenter):
            ss["_access_ok"] = True
            ss[PRESENTER_KEY] = True
            ss.pop("_access_fail", None)
            st.rerun()
        if _same(pw, visitor):
            ss["_access_ok"] = True
            ss.pop("_access_fail", None)
            st.rerun()
        ss["_access_fail"] = ss.get("_access_fail", 0) + 1
        if ss["_access_fail"] >= MAX_TRIES:
            ss["_access_locked_until"] = time.time() + LOCK_SECONDS
            ss["_access_fail"] = 0
        st.error("비밀번호가 맞지 않습니다.")
    st.stop()
