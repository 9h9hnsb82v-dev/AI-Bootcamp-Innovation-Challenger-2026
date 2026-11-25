"""공장 배치도 컴포넌트 (Streamlit Custom Components v2, 추가 Python 패키지 없음).

- 2D 편집: SVG 평면도. 구역을 끌어서 옮기기, 오른쪽 아래 손잡이로 크기 조절, 회전(15° 단위/직접 입력), 구역 추가·삭제, 이름·종류 변경.
  0.5m 격자에 맞춘다. 편집 내용은 '저장'을 눌러야 DB에 남는다(공장 배치 하나).
- 3D 보기: three.js(브라우저가 cdn.jsdelivr.net에서 불러옴). 마우스 끌기 = 회전, 휠 = 확대/축소, 오른쪽 버튼 끌기 = 이동.
  폐플라스틱 보관함 높이·색 = 보관함별 보관량 / 보관함 용량. 쌓인 양은 배치 순서대로 앞 보관함을 100%까지 채운 뒤 다음으로(binFill).
트리거: save(배치 JSON), reset(true)
"""
from __future__ import annotations

from collections.abc import Callable

import streamlit as st

_HTML = """
<div class="fl">
  <div class="bar">
    <div class="seg" role="group" aria-label="보기 방식">
      <button id="m2d" type="button" class="on">2D 편집</button>
      <button id="m3d" type="button">3D 보기</button>
    </div>
    <label class="lab">구역 추가 <select id="addType" aria-label="추가할 구역 종류"></select></label>
    <button id="add" type="button">추가</button>
    <span class="sp"></span>
    <button id="reset" type="button">기본 배치로</button>
    <button id="save" type="button" class="primary">저장</button>
  </div>
  <div class="edit" id="edit" hidden>
    <label class="lab">이름 <input id="eLabel" maxlength="30"></label>
    <label class="lab">종류 <select id="eType"></select></label>
    <label class="lab">가로(m) <input id="eW" type="number" step="0.5" min="0.5"></label>
    <label class="lab">세로(m) <input id="eH" type="number" step="0.5" min="0.5"></label>
    <label class="lab">회전(°) <input id="eRot" type="number" step="15"></label>
    <label class="lab" id="eCapWrap">용량(kg) <input id="eCap" type="number" step="100" min="1" max="100000"></label>
    <button id="rl" type="button" aria-label="왼쪽으로 15도 회전">⟲ 15°</button>
    <button id="rr" type="button" aria-label="오른쪽으로 15도 회전">⟳ 15°</button>
    <button id="del" type="button" class="danger">삭제</button>
  </div>
  <div class="stage" id="stage">
    <svg id="svg" xmlns="http://www.w3.org/2000/svg"></svg>
    <div id="three" class="three" hidden></div>
  </div>
  <div class="status" id="status" role="status"></div>
</div>
"""

_CSS = """
.fl { display: flex; flex-direction: column; gap: 0.6rem; font-family: var(--st-font, sans-serif); color: var(--st-text-color); }
.bar, .edit { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
.edit { padding: 0.5rem 0.6rem; border: 1px solid var(--st-border-color, #3a3f4b); border-radius: 0.5rem; }
.edit[hidden], .three[hidden], .lab[hidden] { display: none; }
.sp { flex: 1; }
.lab { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.85rem; }
button, select, input { min-height: 2.5rem; border-radius: 0.45rem; border: 1px solid var(--st-border-color, #cbd5e1);
  background: var(--st-background-color); color: var(--st-text-color); font-size: 0.9rem; padding: 0 0.7rem; }
input { width: 5.5rem; } #eLabel { width: 11rem; }
button { cursor: pointer; } button.primary { background: var(--st-primary-color); border-color: var(--st-primary-color); color: #fff; }
button.danger { border-color: #e5484d; color: #e5484d; }
.seg { display: inline-flex; border: 1px solid var(--st-border-color, #cbd5e1); border-radius: 0.45rem; overflow: hidden; }
.seg button { border: 0; border-radius: 0; } .seg button.on { background: var(--st-primary-color); color: #fff; }
.stage { position: relative; width: 100%; aspect-ratio: 8 / 5; background: var(--st-secondary-background-color); border-radius: 0.5rem; overflow: hidden; }
svg { width: 100%; height: 100%; display: block; touch-action: none; user-select: none; }
.three { position: absolute; inset: 0; }
.status { font-size: 0.85rem; opacity: 0.85; min-height: 1.2rem; }
"""

_JS = r"""
const S = new WeakMap()
const NS = "http://www.w3.org/2000/svg"
const snap = (v) => Math.round(v * 2) / 2

export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const root = parentElement
  let s = S.get(root)
  if (!s) { s = { rev: -1, layout: null, sel: null, mode: "2d", dirty: false, three: null }; S.set(root, s) }
  const d = data || {}
  if (d.rev !== s.rev) { s.rev = d.rev; s.layout = JSON.parse(JSON.stringify(d.layout)); s.dirty = false; s.sel = null }
  const types = d.types || []
  const tmap = Object.fromEntries(types.map((t) => [t.type, t]))
  const $ = (id) => root.querySelector("#" + id)
  const svg = $("svg"), status = $("status"), edit = $("edit"), three = $("three")
  if (!svg) return
  const L = s.layout

  const DEF_CAP = 1000
  const binFill = () => {  // 앞 보관함 100% → 다음 보관함. 전체 용량을 넘는 양은 마지막 보관함에(초과)
    const out = {}, list = L.items.filter((i) => i.type === "scrap_bin")
    if (!d.live || d.live.scrap_kg == null) return out
    let left = Math.max(0, d.live.scrap_kg)
    list.forEach((b, n) => { const cap = Number(b.capacity_kg) || DEF_CAP, k = Math.min(left, cap); left -= k; out[b.id] = { kg: k, cap, n: n + 1 } })
    if (list.length && left > 0) out[list[list.length - 1].id].kg += left
    Object.values(out).forEach((o) => { o.fill = o.kg / o.cap })
    return out
  }
  const alertRatio = () => (d.live && d.live.alert_ratio) || 0.8
  const binColor = (f) => f >= alertRatio() ? "#e5484d" : f >= 0.5 ? "#f5a524" : "#22a06b"
  const totalCap = () => L.items.filter((i) => i.type === "scrap_bin").reduce((a, b) => a + (Number(b.capacity_kg) || DEF_CAP), 0)
  const say = (msg) => { status.textContent = msg || (s.dirty ? "저장하지 않은 변경이 있습니다." : (d.notice || "")) }
  const fillSel = (sel, val) => {
    if (!sel.options.length) types.forEach((t) => { const o = document.createElement("option"); o.value = t.type; o.textContent = t.name; sel.appendChild(o) })
    if (val) sel.value = val
  }
  fillSel($("addType")); fillSel($("eType"))

  // ---------- 2D ----------
  const toLocal = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()) }
  const draw2d = () => {
    svg.setAttribute("viewBox", `-2 -2 ${L.width_m + 4} ${L.height_m + 4}`)
    svg.textContent = ""
    const bf = binFill()
    const floor = document.createElementNS(NS, "rect")
    Object.entries({ x: 0, y: 0, width: L.width_m, height: L.height_m, fill: "rgba(148,163,184,0.12)", stroke: "#94a3b8", "stroke-width": 0.3 }).forEach(([k, v]) => floor.setAttribute(k, v))
    svg.appendChild(floor)
    for (let gx = 5; gx < L.width_m; gx += 5) { const l = document.createElementNS(NS, "line"); Object.entries({ x1: gx, y1: 0, x2: gx, y2: L.height_m, stroke: "rgba(148,163,184,0.25)", "stroke-width": 0.08 }).forEach(([k, v]) => l.setAttribute(k, v)); svg.appendChild(l) }
    for (let gy = 5; gy < L.height_m; gy += 5) { const l = document.createElementNS(NS, "line"); Object.entries({ x1: 0, y1: gy, x2: L.width_m, y2: gy, stroke: "rgba(148,163,184,0.25)", "stroke-width": 0.08 }).forEach(([k, v]) => l.setAttribute(k, v)); svg.appendChild(l) }
    const cap = document.createElementNS(NS, "text")
    Object.entries({ x: 0, y: -0.6, "font-size": 1.4, fill: "currentColor" }).forEach(([k, v]) => cap.setAttribute(k, v))
    cap.textContent = `공장 ${L.width_m}m × ${L.height_m}m (격자 5m)`
    svg.appendChild(cap)
    L.items.forEach((it) => {
      const t = tmap[it.type] || { color: "#4fb3bf" }
      const g = document.createElementNS(NS, "g")
      g.setAttribute("transform", `translate(${it.x + it.w / 2} ${it.y + it.h / 2}) rotate(${it.rot})`)
      g.style.cursor = "move"
      const r = document.createElementNS(NS, "rect")
      const on = s.sel === it.id
      let fill = t.color
      if (it.type === "scrap_bin" && bf[it.id]) fill = binColor(bf[it.id].fill)
      Object.entries({ x: -it.w / 2, y: -it.h / 2, width: it.w, height: it.h, rx: 0.4, fill, "fill-opacity": it.type === "aisle" ? 0.35 : 0.75, stroke: on ? "#ffffff" : "rgba(0,0,0,0.35)", "stroke-width": on ? 0.35 : 0.12 }).forEach(([k, v]) => r.setAttribute(k, v))
      g.appendChild(r)
      const tx = document.createElementNS(NS, "text")
      const b2 = it.type === "scrap_bin" ? bf[it.id] : null
      const fs = Math.max(0.6, Math.min(1.6, it.w / Math.max(6, it.label.length * 0.9)))
      Object.entries({ x: 0, y: b2 ? -fs * 0.25 : fs * 0.35, "font-size": fs, "text-anchor": "middle", fill: "#ffffff", "font-weight": 600 }).forEach(([k, v]) => tx.setAttribute(k, v))
      tx.textContent = it.label
      if (b2) {  // 보관함: 이름 아래 줄에 보관량/용량
        const t2 = document.createElementNS(NS, "text")
        Object.entries({ x: 0, y: fs * 1.05, "font-size": fs * 0.9, "text-anchor": "middle", fill: "#ffffff", "font-weight": 700 }).forEach(([k, v]) => t2.setAttribute(k, v))
        t2.textContent = `${Math.round(b2.kg)}/${Math.round(b2.cap)}kg`
        g.appendChild(t2)
      }
      g.appendChild(tx)
      if (on) {
        const h = document.createElementNS(NS, "rect")
        Object.entries({ x: it.w / 2 - 0.9, y: it.h / 2 - 0.9, width: 1.8, height: 1.8, fill: "#ffffff", stroke: "#111", "stroke-width": 0.1 }).forEach(([k, v]) => h.setAttribute(k, v))
        h.style.cursor = "nwse-resize"
        h.addEventListener("pointerdown", (e) => startDrag(e, it, "resize"))
        g.appendChild(h)
      }
      g.addEventListener("pointerdown", (e) => startDrag(e, it, "move"))
      svg.appendChild(g)
    })
    syncEdit()
  }
  const startDrag = (e, it, kind) => {
    e.stopPropagation(); e.preventDefault()
    s.sel = it.id
    const p0 = toLocal(e), x0 = it.x, y0 = it.y, w0 = it.w, h0 = it.h
    const rad = (it.rot * Math.PI) / 180, cos = Math.cos(rad), sin = Math.sin(rad)
    const move = (ev) => {
      const p = toLocal(ev), dx = p.x - p0.x, dy = p.y - p0.y
      if (kind === "move") {
        it.x = snap(Math.min(Math.max(x0 + dx, -it.w / 2), L.width_m - it.w / 2))
        it.y = snap(Math.min(Math.max(y0 + dy, -it.h / 2), L.height_m - it.h / 2))
      } else {
        const lx = dx * cos + dy * sin, ly = -dx * sin + dy * cos  // 회전된 구역 기준 변화량
        it.w = Math.max(0.5, snap(w0 + lx)); it.h = Math.max(0.5, snap(h0 + ly))
        it.x = x0; it.y = y0
      }
      s.dirty = true; draw2d(); say()
    }
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up) }
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up)
    draw2d(); say()
  }
  svg.onpointerdown = () => { s.sel = null; draw2d(); say() }
  const selItem = () => L.items.find((i) => i.id === s.sel)
  const syncEdit = () => {
    const it = selItem()
    edit.hidden = !it || s.mode !== "2d"
    if (!it) return
    const set = (id, v) => { const el = $(id); if (root.ownerDocument.activeElement !== el) el.value = v }
    set("eLabel", it.label); fillSel($("eType"), it.type); set("eW", it.w); set("eH", it.h); set("eRot", it.rot)
    $("eCapWrap").hidden = it.type !== "scrap_bin"
    if (it.type === "scrap_bin") set("eCap", Number(it.capacity_kg) || DEF_CAP)
  }
  const upd = (fn) => { const it = selItem(); if (!it) return; fn(it); s.dirty = true; draw2d(); say() }
  $("eLabel").oninput = (e) => upd((it) => { it.label = e.target.value.slice(0, 30) })
  $("eType").onchange = (e) => upd((it) => { it.type = e.target.value; if (it.type === "scrap_bin" && !it.capacity_kg) it.capacity_kg = DEF_CAP })
  $("eCap").onchange = (e) => {
    upd((it) => { const v = Number(e.target.value); if (v >= 1 && v <= 100000) it.capacity_kg = Math.round(v * 10) / 10 })
    const it = selItem(); if (it) say(`보관함 용량 ${Math.round(it.capacity_kg).toLocaleString()}kg · 보관함 용량 합계 ${Math.round(totalCap()).toLocaleString()}kg — 저장해야 앱 전체에 반영됩니다.`)
  }
  $("eW").onchange = (e) => upd((it) => { it.w = Math.max(0.5, snap(Number(e.target.value) || it.w)) })
  $("eH").onchange = (e) => upd((it) => { it.h = Math.max(0.5, snap(Number(e.target.value) || it.h)) })
  $("eRot").onchange = (e) => upd((it) => { it.rot = ((Number(e.target.value) || 0) % 360 + 360) % 360 })
  $("rl").onclick = () => upd((it) => { it.rot = ((it.rot - 15) % 360 + 360) % 360 })
  $("rr").onclick = () => upd((it) => { it.rot = (it.rot + 15) % 360 })
  $("del").onclick = () => { const it = selItem(); if (!it) return; L.items = L.items.filter((i) => i !== it); s.sel = null; s.dirty = true; draw2d(); say("구역을 삭제했습니다. 저장해야 반영됩니다.") }
  $("add").onclick = () => {
    const t = $("addType").value, ty = tmap[t] || { name: "기타 구역" }
    const id = t + "_" + Date.now().toString(36)
    const nb = L.items.filter((i) => i.type === "scrap_bin").length
    const it = { id, type: t, label: t === "scrap_bin" && nb ? `${ty.name} ${nb + 1}` : ty.name, x: snap(L.width_m / 2 - 4), y: snap(L.height_m / 2 - 3), w: 8, h: 6, rot: 0 }
    if (t === "scrap_bin") it.capacity_kg = DEF_CAP
    L.items.push(it)
    s.sel = id; s.dirty = true; if (s.mode !== "2d") setMode("2d"); draw2d(); say("구역을 추가했습니다. 끌어서 옮기세요.")
  }
  $("save").onclick = () => { setTriggerValue("save", JSON.parse(JSON.stringify(L))); say("저장 중…") }
  $("reset").onclick = () => { if (root.ownerDocument.defaultView.confirm("기본 배치로 되돌릴까요? 저장한 내 배치가 지워집니다.")) setTriggerValue("reset", Date.now()) }

  // ---------- 3D ----------
  const draw3d = async () => {
    if (!s.three) {
      say("3D 엔진을 불러오는 중…")
      let THREE
      try { THREE = await import("https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js") }
      catch (e) { say("3D 엔진을 불러오지 못했습니다(인터넷 연결 필요): " + e); return }
      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setPixelRatio(Math.min(2, root.ownerDocument.defaultView.devicePixelRatio || 1))
      three.appendChild(renderer.domElement)
      const scene = new THREE.Scene(); scene.background = new THREE.Color(0x1b1f27)
      const cam = new THREE.PerspectiveCamera(45, 1.6, 0.5, 2000)
      scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 1.1))
      const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(40, 80, 30); scene.add(sun)
      const group = new THREE.Group(); scene.add(group)
      const orbit = { theta: 0.7, phi: 0.9, r: 95, tx: 0, tz: 0 }
      const place = () => { cam.position.set(orbit.tx + orbit.r * Math.sin(orbit.phi) * Math.sin(orbit.theta), orbit.r * Math.cos(orbit.phi), orbit.tz + orbit.r * Math.sin(orbit.phi) * Math.cos(orbit.theta)); cam.lookAt(orbit.tx, 0, orbit.tz) }
      const el = renderer.domElement
      el.addEventListener("contextmenu", (e) => e.preventDefault())
      el.addEventListener("pointerdown", (e) => {
        const x0 = e.clientX, y0 = e.clientY, o = { ...orbit }, pan = e.button === 2
        const mv = (ev) => {
          const dx = ev.clientX - x0, dy = ev.clientY - y0
          if (pan) { orbit.tx = o.tx - (dx * Math.cos(o.theta) + dy * Math.sin(o.theta)) * 0.12; orbit.tz = o.tz + (dx * Math.sin(o.theta) - dy * Math.cos(o.theta)) * 0.12 }
          else { orbit.theta = o.theta - dx * 0.008; orbit.phi = Math.min(1.45, Math.max(0.15, o.phi - dy * 0.006)) }
          place()
        }
        const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up) }
        window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up)
      })
      el.addEventListener("wheel", (e) => { e.preventDefault(); orbit.r = Math.min(400, Math.max(15, orbit.r * (e.deltaY > 0 ? 1.1 : 0.9))); place() }, { passive: false })
      const label = (text) => {
        const c = root.ownerDocument.createElement("canvas"); c.width = 512; c.height = 96
        const g = c.getContext("2d"); g.font = "bold 44px sans-serif"; g.fillStyle = "rgba(15,23,42,0.75)"; g.fillRect(0, 0, 512, 96)
        g.fillStyle = "#ffffff"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, 256, 50)
        const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false })); sp.scale.set(10, 1.9, 1); return sp
      }
      s.three = { THREE, renderer, scene, cam, group, place, label, orbit, raf: 0 }
      place()
    }
    const { THREE, renderer, scene, cam, group, label } = s.three
    group.clear()
    const W = L.width_m, H = L.height_m
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshStandardMaterial({ color: 0x475569 }))
    floor.rotation.x = -Math.PI / 2; group.add(floor)
    const grid = new THREE.GridHelper(Math.max(W, H), Math.max(W, H) / 5, 0x64748b, 0x334155); grid.position.y = 0.01; group.add(grid)
    const bf = binFill()
    L.items.forEach((it) => {
      const t = tmap[it.type] || { height_m: 2, color: "#4fb3bf" }
      let h = t.height_m, color = t.color
      const b = it.type === "scrap_bin" ? bf[it.id] : null
      if (b) {
        color = binColor(b.fill)
        const shell = new THREE.Mesh(new THREE.BoxGeometry(it.w, t.height_m, it.h), new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.35 }))
        shell.position.set(it.x + it.w / 2 - W / 2, t.height_m / 2, it.y + it.h / 2 - H / 2); shell.rotation.y = -it.rot * Math.PI / 180; group.add(shell)
        h = Math.max(0.1, t.height_m * Math.min(b.fill, 1))
      }
      const m = new THREE.Mesh(new THREE.BoxGeometry(it.w, h, it.h), new THREE.MeshStandardMaterial({ color, transparent: it.type === "aisle", opacity: it.type === "aisle" ? 0.5 : 1 }))
      m.position.set(it.x + it.w / 2 - W / 2, h / 2, it.y + it.h / 2 - H / 2); m.rotation.y = -it.rot * Math.PI / 180; group.add(m)
      if (it.type !== "aisle") {
        const sp = label(b ? `${it.label} ${Math.round(b.kg)}/${Math.round(b.cap)}kg` : it.label)
        sp.position.set(m.position.x, Math.max(h, t.height_m) + 1.6, m.position.z); group.add(sp)
      }
    })
    const resize = () => { const w = three.clientWidth || 800, hh = three.clientHeight || 500; renderer.setSize(w, hh, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%"; cam.aspect = w / hh; cam.updateProjectionMatrix() }
    resize()
    cancelAnimationFrame(s.three.raf)
    const loop = () => { if (s.mode !== "3d") return; renderer.render(scene, cam); s.three.raf = requestAnimationFrame(loop) }
    loop()
    say(d.live && d.live.scrap_kg != null ? `폐플라스틱 보관량 합계: ${Math.round(d.live.scrap_kg)}kg / 보관함 용량 합계 ${Math.round(totalCap()).toLocaleString()}kg (시연 기준값, 앞 보관함부터 채움) · 끌기 = 회전, 휠 = 확대, 오른쪽 끌기 = 이동` : "끌기 = 회전, 휠 = 확대, 오른쪽 끌기 = 이동")
  }

  const setMode = (m) => {
    s.mode = m
    $("m2d").classList.toggle("on", m === "2d"); $("m3d").classList.toggle("on", m === "3d")
    svg.style.display = m === "2d" ? "block" : "none"; three.hidden = m !== "3d"
    syncEdit()
    if (m === "3d") draw3d(); else { if (s.three) cancelAnimationFrame(s.three.raf); draw2d(); say() }
  }
  $("m2d").onclick = () => setMode("2d")
  $("m3d").onclick = () => setMode("3d")
  setMode(s.mode)
  return () => { if (s.three) cancelAnimationFrame(s.three.raf) }
}
"""

_FACTORY = st.components.v2.component("resource_factory_layout", html=_HTML, css=_CSS, js=_JS)


def factory_layout_view(*, key: str, layout: dict, rev: int, live: dict, types: list[dict], notice: str | None,
                        on_save: Callable[[], None], on_reset: Callable[[], None]):
    """배치도 표시. rev가 바뀌면(저장·초기화 후) 브라우저의 편집 사본을 서버 배치로 다시 맞춘다."""
    return _FACTORY(key=key, data={"layout": layout, "rev": rev, "live": live, "types": types, "notice": notice},
                    on_save_change=on_save, on_reset_change=on_reset)
