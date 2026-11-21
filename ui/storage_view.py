"""보관창고 그림: 2D 평면도(SVG, 서버에서 생성) · 3D 입체(three.js, Custom Components v2).

포대 1개 = 25kg, 칸 하나 = 최대 2단(50kg). 방 용량을 넘는 무게는 통로에 '넘침'으로 그린다(팀원 앱 예시 반영).
"""
from __future__ import annotations

import html
import math

import streamlit as st

COLORS = {"정상": ("#1f9d55", "#e7f6ec"), "경고": ("#f08c00", "#fff4e0"), "초과": ("#d9480f", "#ffe8e0")}
BADGE = {"정상": "#1f9d55", "경고": "#f08c00", "초과": "#e03131"}


def plan_svg(rooms: list[dict], day: dict, title: str, bag_kg: float = 25.0, stack: int = 2) -> str:
    """방별 적재 2D 평면도 SVG 문자열."""
    cell, gap, pad = 44, 10, 18
    cols_per_room = 4
    room_w = pad * 2 + cols_per_room * cell + (cols_per_room - 1) * gap
    W = room_w * len(rooms) + 12
    H = 400
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H + 70}" width="100%" role="img" '
             f'aria-label="{html.escape(title)}" style="font-family:sans-serif">',
             f'<rect x="0" y="0" width="{W}" height="{H + 70}" rx="14" fill="#ffffff"/>',
             f'<text x="16" y="30" font-size="18" font-weight="700" fill="#1b1f27">{html.escape(title)}</text>',
             f'<rect x="6" y="46" width="{W - 12}" height="{H - 60}" fill="#f6f7f9" stroke="#4a5160" stroke-width="6"/>']
    per_cell = bag_kg * stack
    for i, r in enumerate(rooms):
        x0 = 6 + i * room_w
        st_ = day["rooms"][r["id"]]
        stroke, fill = COLORS[st_["state"]]
        if i:
            parts.append(f'<line x1="{x0}" y1="46" x2="{x0}" y2="{H - 140}" stroke="#4a5160" stroke-width="6"/>')
        parts.append(f'<line x1="{x0 + 4}" y1="{H - 140}" x2="{x0 + 50}" y2="{H - 140}" stroke="#4a5160" stroke-width="6"/>')
        parts.append(f'<line x1="{x0 + room_w - 60}" y1="{H - 140}" x2="{x0 + room_w - 2}" y2="{H - 140}" stroke="#4a5160" stroke-width="6"/>')
        parts.append(f'<text x="{x0 + pad}" y="80" font-size="16" font-weight="700" fill="#1b1f27">{html.escape(r["name"])}</text>')
        bw = 46
        parts.append(f'<rect x="{x0 + room_w - bw - 14}" y="62" width="{bw}" height="24" rx="12" fill="{BADGE[st_["state"]]}"/>'
                     f'<text x="{x0 + room_w - bw / 2 - 14}" y="79" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">{st_["state"]}</text>')
        inside = min(st_["kg"], r["capacity_kg"])
        cap_cells = math.ceil(r["capacity_kg"] / per_cell)
        full = int(inside // per_cell)
        rest = inside - full * per_cell
        for c in range(cap_cells):
            cx, cy = x0 + pad + (c % cols_per_room) * (cell + gap), 100 + (c // cols_per_room) * (cell + gap)
            parts.append(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="6" fill="none" stroke="#c4c9d2" stroke-dasharray="4 3"/>')
            n = 2 if c < full else (math.ceil(rest / bag_kg) if c == full and rest > 0 else 0)
            if n:
                parts.append(f'<rect x="{cx + 3}" y="{cy + 3}" width="{cell - 6}" height="{cell - 6}" rx="5" fill="{stroke}"/>'
                             f'<rect x="{cx + 12}" y="{cy + 12}" width="{cell - 24}" height="{cell - 24}" rx="3" fill="rgba(0,0,0,0.18)"/>'
                             f'<text x="{cx + cell / 2}" y="{cy + cell / 2 + 6}" font-size="15" font-weight="700" fill="#fff" text-anchor="middle">{n}</text>')
        if st_["overflow_kg"] > 0:  # 통로 쪽 넘침
            bags = math.ceil(st_["overflow_kg"] / bag_kg)
            for b in range(min(bags, 12)):
                parts.append(f'<rect x="{x0 + pad + b * 18}" y="{H - 122}" width="15" height="24" rx="3" fill="#e03131" opacity="{0.95 if b < bags - 1 else 0.7}"/>')
            parts.append(f'<text x="{x0 + pad}" y="{H - 78}" font-size="16" font-weight="700" fill="#e03131">넘침 {st_["overflow_kg"]:.0f}kg</text>')
        # 아래 막대
        by = H + 6
        pct = st_["kg"] / r["capacity_kg"]
        parts.append(f'<text x="{x0 + 12}" y="{by}" font-size="15" font-weight="700" fill="#1b1f27">{st_["kg"]:.0f} / {r["capacity_kg"]:.0f} kg</text>'
                     f'<text x="{x0 + room_w - 12}" y="{by}" font-size="15" font-weight="700" fill="{BADGE[st_["state"]]}" text-anchor="end">{pct:.0%}</text>'
                     f'<rect x="{x0 + 12}" y="{by + 10}" width="{room_w - 24}" height="10" rx="5" fill="#e6e8ec"/>'
                     f'<rect x="{x0 + 12}" y="{by + 10}" width="{(room_w - 24) * min(pct, 1):.1f}" height="10" rx="5" fill="{BADGE[st_["state"]]}"/>')
    parts.append(f'<text x="{W - 20}" y="{H - 24}" font-size="14" fill="#5b6270" text-anchor="end">통로 · 반출입구</text></svg>')
    return "".join(parts)


_JS3D = r"""
const S = new WeakMap()
export default function (component) {
  const { data, parentElement } = component
  const root = parentElement
  let s = S.get(root); if (!s) { s = {}; S.set(root, s) }
  const box = root.querySelector("#v3")
  const d = data || {}
  const draw = async () => {
    if (!s.ctx) {
      let THREE
      try { THREE = await import("https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js") }
      catch (e) { box.textContent = "3D 엔진을 불러오지 못했습니다(인터넷 연결 필요)"; return }
      const r = new THREE.WebGLRenderer({ antialias: true }); box.appendChild(r.domElement)
      const scene = new THREE.Scene(); scene.background = new THREE.Color(0x1b1f27)
      const cam = new THREE.PerspectiveCamera(45, 1.8, 0.1, 500)
      scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 1.1)); const sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(10, 20, 8); scene.add(sun)
      const group = new THREE.Group(); scene.add(group)
      const orbit = { t: 0.6, p: 0.95, r: 26 }
      const place = () => { cam.position.set(orbit.r * Math.sin(orbit.p) * Math.sin(orbit.t), orbit.r * Math.cos(orbit.p), orbit.r * Math.sin(orbit.p) * Math.cos(orbit.t)); cam.lookAt(0, 0, 0) }
      r.domElement.addEventListener("pointerdown", (e) => { const x0 = e.clientX, y0 = e.clientY, o = { ...orbit }
        const mv = (ev) => { orbit.t = o.t - (ev.clientX - x0) * 0.008; orbit.p = Math.min(1.45, Math.max(0.2, o.p - (ev.clientY - y0) * 0.006)); place() }
        const up = () => { removeEventListener("pointermove", mv); removeEventListener("pointerup", up) }
        addEventListener("pointermove", mv); addEventListener("pointerup", up) })
      r.domElement.addEventListener("wheel", (e) => { e.preventDefault(); orbit.r = Math.min(80, Math.max(8, orbit.r * (e.deltaY > 0 ? 1.1 : 0.9))); place() }, { passive: false })
      s.ctx = { THREE, r, scene, cam, group, place }; place()
    }
    const { THREE, r, scene, cam, group } = s.ctx
    group.clear()
    const rooms = d.rooms || [], n = rooms.length, rw = 6, gapx = 0.6, depth = 5
    const totalW = n * rw + (n - 1) * gapx
    const floor = new THREE.Mesh(new THREE.BoxGeometry(totalW + 2, 0.1, depth + 4), new THREE.MeshStandardMaterial({ color: 0x9aa3b2 })); floor.position.y = -0.05; group.add(floor)
    rooms.forEach((rm, i) => {
      const x0 = -totalW / 2 + i * (rw + gapx)
      const wallM = new THREE.MeshStandardMaterial({ color: 0x4a5160, transparent: true, opacity: 0.55 })
      // 뒷벽 · 왼쪽 벽 · 오른쪽 벽 (앞쪽은 통로로 열어 둠)
      ;[[x0 + rw / 2, -depth / 2, rw, 0.1], [x0, 0, 0.1, depth], [x0 + rw, 0, 0.1, depth]].forEach(([x, z, w, dd]) => {
        const wall = new THREE.Mesh(new THREE.BoxGeometry(w, 1.4, dd), wallM); wall.position.set(x, 0.7, z); group.add(wall) })
      const color = { 정상: 0x1f9d55, 경고: 0xf08c00, 초과: 0xd9480f }[rm.state] || 0x1f9d55
      const bags = Math.round(Math.min(rm.kg, rm.capacity_kg) / d.bag_kg), per = d.stack, perRow = 4
      for (let b = 0; b < bags; b++) {
        const cellI = Math.floor(b / per), level = b % per
        const bx = x0 + 0.9 + (cellI % perRow) * 1.4, bz = -depth / 2 + 0.9 + Math.floor(cellI / perRow) * 1.3
        const m = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.5, 1.0), new THREE.MeshStandardMaterial({ color }))
        m.position.set(bx, 0.27 + level * 0.52, bz); group.add(m)
      }
      const over = Math.ceil(rm.overflow_kg / d.bag_kg)
      for (let b = 0; b < over; b++) {
        const m = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.5, 0.6), new THREE.MeshStandardMaterial({ color: 0xe03131 }))
        m.position.set(x0 + 0.5 + (b % 10) * 0.55, 0.25 + Math.floor(b / 10) * 0.5, depth / 2 + 0.9); group.add(m)
      }
      const c = document.createElement("canvas"); c.width = 512; c.height = 96; const g = c.getContext("2d")
      g.fillStyle = "rgba(15,23,42,0.8)"; g.fillRect(0, 0, 512, 96); g.fillStyle = "#fff"; g.font = "bold 36px sans-serif"; g.textAlign = "center"
      g.fillText(`${rm.name.split("·")[0].trim()} ${Math.round(rm.kg)}/${rm.capacity_kg}kg`, 256, 60)
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false })); sp.scale.set(5, 0.95, 1); sp.position.set(x0 + rw / 2, 2.6, -depth / 2); group.add(sp)
    })
    const w = box.clientWidth || 800, h = box.clientHeight || 440; r.setSize(w, h, false); r.domElement.style.width = "100%"; r.domElement.style.height = "100%"; cam.aspect = w / h; cam.updateProjectionMatrix()
    cancelAnimationFrame(s.raf); const loop = () => { r.render(scene, cam); s.raf = requestAnimationFrame(loop) }; loop()
  }
  draw()
  return () => cancelAnimationFrame(s.raf)
}
"""

_STORAGE3D = st.components.v2.component(
    "resource_storage_rooms_3d", html='<div id="v3" style="width:100%;aspect-ratio:16/9;border-radius:10px;overflow:hidden"></div>',
    js=_JS3D)


def rooms_3d(*, key: str, rooms: list[dict], day: dict, bag_kg: float, stack: int):
    data = [{"name": r["name"], "capacity_kg": r["capacity_kg"], **day["rooms"][r["id"]]} for r in rooms]
    return _STORAGE3D(key=key, data={"rooms": data, "bag_kg": bag_kg, "stack": stack})
