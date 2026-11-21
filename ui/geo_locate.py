"""브라우저 위치 가져오기 버튼 (Streamlit Custom Components v2, 추가 패키지 없음, 2026-11-17).

navigator.geolocation 으로 한 번만 위치를 읽어 'pos' 트리거({lat, lon, acc})로 보낸다. 실패하면 'fail'({code, msg}).
보안 컨텍스트(localhost·https)에서만 동작한다. PC는 GPS가 없어 Wi-Fi·IP 기반이라 오차가 클 수 있다.
"""
from __future__ import annotations

from collections.abc import Callable

import streamlit as st

_HTML = """
<div class="geo">
  <button id="go" type="button">내 위치 가져오기</button>
  <span id="msg" class="msg" role="status"></span>
</div>
"""
_CSS = """
.geo { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; font-family: var(--st-font, sans-serif); color: var(--st-text-color); }
button { min-height: 2.5rem; padding: 0 1rem; border-radius: 0.5rem; border: 1px solid var(--st-border-color, #cbd5e1);
         background: var(--st-background-color); color: var(--st-text-color); font-size: 0.95rem; cursor: pointer; }
button:disabled { opacity: 0.5; cursor: wait; }
button:focus-visible { outline: 2px solid var(--st-primary-color); outline-offset: 2px; }
.msg { font-size: 0.85rem; opacity: 0.85; }
"""
_JS = """
export default function (component) {
  const { parentElement, setTriggerValue } = component
  const go = parentElement.querySelector("#go")
  const msg = parentElement.querySelector("#msg")
  if (!go || !msg) return
  const WHY = { 1: "위치 권한이 거부됐습니다. 주소창 왼쪽 자물쇠 → 위치 → 허용 후 다시 누르세요. Windows 설정 → 개인 정보 → 위치도 켜져 있어야 합니다.",
                2: "위치를 알 수 없습니다(Windows 위치 서비스가 꺼져 있거나 Wi-Fi 정보 없음).",
                3: "시간 초과 — 다시 누르세요." }
  go.onclick = () => {
    if (!window.isSecureContext || !navigator.geolocation) {
      msg.textContent = "이 접속 주소에서는 위치를 쓸 수 없습니다(localhost 또는 https 필요)."
      setTriggerValue("fail", { code: 0, msg: msg.textContent })
      return
    }
    go.disabled = true
    msg.textContent = "위치 확인 중…"
    navigator.geolocation.getCurrentPosition(
      (p) => {
        go.disabled = false
        msg.textContent = "위치를 받았습니다."
        setTriggerValue("pos", { lat: p.coords.latitude, lon: p.coords.longitude, acc: p.coords.accuracy, t: Date.now() })
      },
      (e) => {
        go.disabled = false
        msg.textContent = WHY[e.code] || ("위치 오류: " + e.message)
        setTriggerValue("fail", { code: e.code, msg: msg.textContent, t: Date.now() })
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 })
  }
}
"""

_GEO = st.components.v2.component("resource_geo_locate", html=_HTML, css=_CSS, js=_JS)


def geo_button(*, key: str, on_pos: Callable[[], None], on_fail: Callable[[], None]):
    """누르면 위치를 한 번 읽는다. 콜백 안에서 st.session_state[key].pos / .fail 로 값을 읽는다."""
    return _GEO(key=key, data={}, on_pos_change=on_pos, on_fail_change=on_fail)
