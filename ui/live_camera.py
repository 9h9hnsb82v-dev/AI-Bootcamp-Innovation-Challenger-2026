"""카메라 실시간 미리보기 컴포넌트 (Streamlit Custom Components v2, 추가 패키지 없음).

- 브라우저가 카메라 영상을 그대로 보여 준다(서버로 영상 스트림을 보내지 않음).
- 약 interval_ms마다 긴 변 512px JPEG 한 장을 'frame' 트리거로 보낸다. 직전 프레임 결과(data.live.seq)가
  돌아오기 전에는 다음 프레임을 보내지 않는다(서버 과부하 방지, 4초 지나면 재시도).
- '이 장면 등록'은 긴 변 1280px JPEG 한 장을 'register' 트리거로 보낸다. 기록은 이 경우에만 생긴다.
- 카메라는 보안 컨텍스트(https 또는 localhost)에서만 열린다.
"""
from __future__ import annotations

import base64
from collections.abc import Callable

import streamlit as st

_HTML = """
<div class="wrap">
  <div class="stage">
    <video id="v" playsinline muted></video>
    <div id="badge" class="badge idle">카메라 꺼짐</div>
    <div id="guide" class="guide" hidden><span>틈이 박스의 2/3를 채우게</span></div>
  </div>
  <canvas id="c" hidden></canvas>
  <div class="bar">
    <label class="devlabel" for="dev">카메라</label>
    <select id="dev" aria-label="사용할 카메라"><option value="">카메라를 켜면 목록이 나타납니다</option></select>
    <button id="toggle" type="button">카메라 켜기</button>
    <button id="reg" type="button" class="primary" disabled>이 장면 등록</button>
    <span id="status" class="status" role="status"></span>
  </div>
</div>
"""

_CSS = """
.wrap { display: flex; flex-direction: column; gap: 0.75rem; font-family: var(--st-font, sans-serif); color: var(--st-text-color); }
.stage { position: relative; width: 100%; max-width: 720px; aspect-ratio: 4 / 3; background: var(--st-secondary-background-color); border-radius: 0.5rem; overflow: hidden; }
video { width: 100%; height: 100%; object-fit: contain; display: block; }
.badge { position: absolute; left: 0.75rem; top: 0.75rem; padding: 0.4rem 0.75rem; border-radius: 0.4rem; font-size: 0.95rem; line-height: 1.35; font-weight: 600; white-space: pre-line; background: rgba(15, 23, 42, 0.82); color: #ffffff; max-width: calc(100% - 1.5rem); }
.badge.defect { background: rgba(185, 28, 28, 0.92); }
.badge.pass { background: rgba(21, 128, 61, 0.92); }
.badge.mismatch { background: rgba(180, 83, 9, 0.94); }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
button { min-height: 2.75rem; padding: 0 1rem; border-radius: 0.5rem; border: 1px solid var(--st-border-color, #cbd5e1); background: var(--st-background-color); color: var(--st-text-color); font-size: 0.95rem; cursor: pointer; }
button.primary { background: var(--st-primary-color); border-color: var(--st-primary-color); color: #ffffff; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.devlabel { font-size: 0.9rem; }
select { min-height: 2.75rem; max-width: 22rem; padding: 0 0.5rem; border-radius: 0.5rem; border: 1px solid var(--st-border-color, #cbd5e1); background: var(--st-background-color); color: var(--st-text-color); font-size: 0.9rem; }
.status { font-size: 0.85rem; opacity: 0.85; }
.guide { position: absolute; border: 3px dashed #ffd166; border-radius: 10px; box-sizing: border-box; pointer-events: none; }
.guide span { position: absolute; left: 50%; top: 100%; transform: translateX(-50%); margin-top: 6px; white-space: nowrap; color: #ffd166; font-size: 0.85rem; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.8); }
"""

_JS = """
const S = new WeakMap()

export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  let s = S.get(parentElement)
  if (!s) {
    s = { stream: null, timer: null, pending: false, seq: 0, lastSeen: -1, sentAt: 0, data: {} }
    S.set(parentElement, s)
  }
  s.data = data || {}
  const video = parentElement.querySelector("#v")
  const canvas = parentElement.querySelector("#c")
  const toggle = parentElement.querySelector("#toggle")
  const reg = parentElement.querySelector("#reg")
  const status = parentElement.querySelector("#status")
  const badge = parentElement.querySelector("#badge")
  if (!video || !canvas || !toggle || !reg || !status || !badge) return

  const snap = !!s.data.snap  // 사진 한 장 모드: 실시간 판정 없이 '촬영' 버튼만 (카메라 선택 기능은 같음)
  badge.hidden = snap
  reg.textContent = s.data.button || "이 장면 등록"
  const live = s.data.live
  if (live && live.seq !== s.lastSeen) {
    s.lastSeen = live.seq
    s.pending = false
  }
  if (!s.stream) {
    badge.className = "badge idle"
    badge.textContent = "카메라 꺼짐"
  } else if (live && live.text) {
    badge.className = "badge " + (live.tone || "idle")
    badge.textContent = live.text
  } else {
    badge.className = "badge idle"
    badge.textContent = "판정 대기 중…"
  }
  if (s.data.notice) status.textContent = s.data.notice

  // 틈(단차) 촬영 가이드: 실제 영상이 보이는 영역 기준으로 짧은 변 x guide 크기의 정사각형을 가운데 표시
  const guide = parentElement.querySelector("#guide")
  const placeGuide = () => {
    if (!guide) return
    const frac = Number(s.data.guide) || 0
    if (!frac) { guide.hidden = true; return }
    const stage = video.parentElement
    const W = stage.clientWidth, H = stage.clientHeight
    const vw = video.videoWidth || 4, vh = video.videoHeight || 3
    const k = Math.min(W / vw, H / vh)
    const side = frac * Math.min(vw * k, vh * k)
    guide.style.width = side + "px"
    guide.style.height = side + "px"
    guide.style.left = (W - side) / 2 + "px"
    guide.style.top = (H - side) / 2 + "px"
    guide.hidden = false
  }
  placeGuide()
  video.onloadedmetadata = placeGuide
  s.placeGuide = placeGuide
  if (!s.resizeBound) {  // 다시 그릴 때마다 리스너가 쌓이지 않게 한 번만 연결
    s.resizeBound = true
    window.addEventListener("resize", () => s.placeGuide && s.placeGuide())
  }

  const grab = (longSide, quality) => {
    const w = video.videoWidth, h = video.videoHeight
    if (!w || !h) return null
    const k = Math.min(1, longSide / Math.max(w, h))
    canvas.width = Math.round(w * k)
    canvas.height = Math.round(h * k)
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height)
    return canvas.toDataURL("image/jpeg", quality)
  }

  const stop = () => {
    if (s.timer) clearInterval(s.timer)
    s.timer = null
    if (s.stream) s.stream.getTracks().forEach((t) => t.stop())
    s.stream = null
    s.pending = false
    video.srcObject = null
    toggle.textContent = "카메라 켜기"
    reg.disabled = true
    badge.className = "badge idle"
    badge.textContent = "카메라 꺼짐"
  }

  const tick = () => {
    if (!s.stream) return
    const now = Date.now()
    if (s.pending && now - s.sentAt < 4000) return
    const image = grab(512, 0.75)
    if (!image) return
    s.seq += 1
    s.pending = true
    s.sentAt = now
    setTriggerValue("frame", { seq: s.seq, image })
  }

  const dev = parentElement.querySelector("#dev")
  const SAVED = "resource_live_camera_device"
  const AVOID = /virtual|\bir\b|ir-|infrared|obs|snap camera/i  // 가상·적외선 카메라는 기본 선택에서 제외
  const loadSaved = () => { try { return localStorage.getItem(SAVED) || "" } catch (e) { return "" } }
  const save = (id) => { try { localStorage.setItem(SAVED, id) } catch (e) { /* 저장 불가여도 동작 */ } }

  const listCameras = async () => {
    const all = await navigator.mediaDevices.enumerateDevices()
    return all.filter((d) => d.kind === "videoinput")
  }

  const fillSelect = (cams, chosenId) => {
    dev.textContent = ""
    cams.forEach((c, i) => {
      const o = document.createElement("option")
      o.value = c.deviceId
      o.textContent = (c.label || ("카메라 " + (i + 1))) + (AVOID.test(c.label) ? " (가상·적외선)" : "")
      if (c.deviceId === chosenId) o.selected = true
      dev.appendChild(o)
    })
  }

  const open = async (deviceId) => {
    if (s.stream) s.stream.getTracks().forEach((t) => t.stop())
    const video_c = { width: { ideal: 1280 }, height: { ideal: 960 } }
    if (deviceId) video_c.deviceId = { exact: deviceId }
    s.stream = await navigator.mediaDevices.getUserMedia({ video: video_c, audio: false })
    video.srcObject = s.stream
    await video.play()
    const track = s.stream.getVideoTracks()[0]
    const settings = track && track.getSettings ? track.getSettings() : {}
    return { id: settings.deviceId || deviceId || "", label: track ? track.label : "" }
  }

  const start = async (wantId) => {
    // 1) 권한을 받아야 장치 이름이 보인다 → 2) 목록에서 실제 카메라를 고른다 → 3) 필요하면 그 장치로 다시 연다
    let cur = await open(wantId || loadSaved() || "").catch(() => open(""))
    const cams = await listCameras()
    let pick = wantId || loadSaved()
    if (!cams.some((c) => c.deviceId === pick)) {
      const real = cams.find((c) => !AVOID.test(c.label))
      pick = real ? real.deviceId : cur.id
    }
    if (pick && pick !== cur.id) cur = await open(pick)
    fillSelect(cams, cur.id)
    save(cur.id)
    toggle.textContent = "카메라 끄기"
    reg.disabled = false
    s.pending = false
    status.textContent = "사용 중: " + (cur.label || "카메라") + (snap ? " · '" + reg.textContent + "'을 누르세요" : " · 실시간 판정 중 (기록 안 됨)")
    if (!snap && !s.timer) s.timer = setInterval(tick, Math.max(300, Number(s.data.interval_ms) || 800))
  }

  toggle.onclick = async () => {
    if (s.stream) { stop(); status.textContent = "카메라를 껐습니다."; return }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      status.textContent = "이 브라우저 또는 접속 주소(http 원격)에서는 카메라를 쓸 수 없습니다. localhost 또는 https로 접속하세요."
      return
    }
    try {
      await start("")
    } catch (e) {
      stop()
      const name = e && e.name ? e.name : String(e)
      status.textContent = "카메라를 열 수 없습니다: " + name +
        (name === "NotReadableError" ? " (다른 프로그램이 카메라를 쓰는 중이거나 카메라 끄기 키가 켜져 있음)" : "")
    }
  }

  dev.onchange = async () => {
    if (!dev.value) return
    save(dev.value)
    if (!s.stream) return
    try {
      await start(dev.value)
    } catch (e) {
      status.textContent = "이 카메라를 열 수 없습니다: " + (e && e.name ? e.name : e)
    }
  }

  reg.onclick = () => {
    if (!s.stream) return
    const image = grab(1280, 0.92)
    if (!image) { status.textContent = "영상이 아직 준비되지 않았습니다."; return }
    status.textContent = snap ? "촬영했습니다. 아래에서 확인하세요." : "등록 중…"
    setTriggerValue("register", { image })
  }

  return () => stop()
}
"""

_LIVE_CAMERA = st.components.v2.component("resource_live_camera", html=_HTML, css=_CSS, js=_JS)


def decode_image(payload: dict | None) -> bytes | None:
    """컴포넌트가 보낸 data URL(JPEG)을 바이트로 바꾼다."""
    if not payload or not isinstance(payload.get("image"), str) or "," not in payload["image"]:
        return None
    head, b64 = payload["image"].split(",", 1)
    if not head.startswith("data:image/jpeg"):
        return None
    return base64.b64decode(b64)


def live_camera(*, key: str, live: dict | None, notice: str | None = None, interval_ms: int = 800,
                on_frame: Callable[[], None], on_register: Callable[[], None], guide: float | None = None):
    """실시간 미리보기를 표시한다. 트리거 값은 콜백 안에서 st.session_state[key].frame / .register 로 읽는다.
    guide: 0~1이면 영상 가운데에 (짧은 변 x guide) 정사각형 가이드 박스를 그린다(휀더 틈 촬영)."""
    return _LIVE_CAMERA(key=key, data={"live": live, "notice": notice, "interval_ms": interval_ms, "guide": guide},
                        on_frame_change=on_frame, on_register_change=on_register)


def camera_snapshot(*, key: str, button: str = "촬영", guide: float | None = None) -> bytes | None:
    """사진 한 장 찍기 (st.camera_input 대신, 2026-11-17).

    st.camera_input 은 브라우저 기본 카메라만 써서, 기본값이 적외선(Windows Hello)·가상 카메라인 PC에서는
    영상이 안 나온다(빨간 X). 이 컴포넌트는 실시간 미리보기와 같은 방식으로 실제 카메라를 골라 열고,
    목록에서 다른 카메라로 바꿀 수 있다. 찍은 사진(JPEG 바이트)은 st.session_state[f"{key}__img"]에 둔다."""
    store = f"{key}__img"

    def on_shot():
        b = decode_image(st.session_state[key].register)
        if b is not None:
            st.session_state[store] = b

    _LIVE_CAMERA(key=key, data={"snap": True, "button": button, "guide": guide, "live": None, "notice": None},
                 on_frame_change=lambda: None, on_register_change=on_shot)
    return st.session_state.get(store)


def clear_snapshot(key: str) -> None:
    st.session_state.pop(f"{key}__img", None)
