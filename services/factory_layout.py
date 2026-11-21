"""가상 공장 배치도(2D 편집 · 3D 보기) 저장소 (사용자 요청 2026-11-14).

- 실제 공장 크기를 모르므로 시연용 기본 배치(80m x 50m, DEMO_ASSUMED)를 베이스로 둔다.
- 사람마다 따로 저장(사용자 결정): 화면의 작업자 ID를 소유자로 DB factory_layout 테이블에 저장. 저장 전에는 기본 배치.
- 구역은 옮기기·크기 조절·회전·추가·삭제가 가능하다. 저장 시 값 범위를 검사한다(공장 밖·너무 작은 구역·개수 초과 거부).
- 3D의 폐플라스틱 보관함 높이·색은 보관량(storage_projection, 시연 기준값 + 확정 폐기 장부)에서 계산한다.
"""
from __future__ import annotations

import json

from services import db
from services.config import ServiceError

TYPES = {  # 종류 → 이름, 3D 높이(m), 색
    "line": ("사출 라인", 3.0, "#5b8def"),
    "inspection": ("검사대", 1.5, "#22a06b"),
    "scrap_bin": ("폐플라스틱 보관함", 2.5, "#e5484d"),
    "raw_storage": ("원료 창고", 4.0, "#a07d4f"),
    "shipping": ("출하·상차장", 1.0, "#8e6fd8"),
    "truck_gate": ("화물차 출입구", 4.5, "#f5a524"),
    "office": ("사무·휴게", 3.0, "#7d8590"),
    "aisle": ("통로", 0.05, "#c7ccd4"),
    "custom": ("기타 구역", 2.0, "#4fb3bf"),
}
MAX_ITEMS = 60


def default_layout() -> dict:
    """시연용 기본 배치(가상). 단위 m, (x, y) = 구역 왼쪽 위, rot = 시계 방향 각도."""
    it = [
        ("line_door", "line", "도어 사출 라인", 6, 6, 26, 6),
        ("line_bumper", "line", "범퍼 사출 라인", 6, 15, 26, 6),
        ("line_fender", "line", "휀더 사출 라인", 6, 24, 26, 6),
        ("inspect", "inspection", "검사대 (이미지 검사)", 36, 12, 8, 12),
        ("scrap_bin", "scrap_bin", "폐플라스틱 보관함", 52, 35, 10, 8),
        ("raw", "raw_storage", "원료 창고", 6, 36, 22, 10),
        ("ship", "shipping", "출하·상차장", 50, 4, 24, 10),
        ("gate", "truck_gate", "화물차 출입구", 72, 30, 6, 12),
        ("office", "office", "사무·휴게", 32, 38, 14, 8),
        ("aisle_main", "aisle", "주 통로", 4, 31.5, 66, 2.5),
    ]
    return {"version": 1, "width_m": 80, "height_m": 50, "provenance": "DEMO_ASSUMED",
            "items": [{"id": i, "type": t, "label": lab, "x": x, "y": y, "w": w, "h": h, "rot": 0}
                      for i, t, lab, x, y, w, h in it]}


def validate(layout: dict) -> dict:
    try:
        W, H = float(layout["width_m"]), float(layout["height_m"])
        items = layout["items"]
    except (KeyError, TypeError, ValueError) as e:
        raise ServiceError("INVALID_INPUT", f"배치 형식 오류: {e}") from e
    if not (10 <= W <= 1000 and 10 <= H <= 1000):
        raise ServiceError("INVALID_INPUT", "공장 크기는 10~1000m 범위여야 합니다.")
    if len(items) > MAX_ITEMS:
        raise ServiceError("INVALID_INPUT", f"구역은 최대 {MAX_ITEMS}개입니다.")
    clean, ids = [], set()
    for it in items:
        t = it.get("type") if it.get("type") in TYPES else "custom"
        x, y, w, h, rot = (float(it.get(k, 0)) for k in ("x", "y", "w", "h", "rot"))
        if w < 0.5 or h < 0.5 or w > W or h > H:
            raise ServiceError("INVALID_INPUT", f"구역 '{it.get('label')}' 크기가 범위를 벗어났습니다.")
        cx, cy = x + w / 2, y + h / 2
        if not (0 <= cx <= W and 0 <= cy <= H):
            raise ServiceError("INVALID_INPUT", f"구역 '{it.get('label')}'의 중심이 공장 밖에 있습니다.")
        iid = str(it.get("id") or "")[:40] or f"item{len(clean)}"
        if iid in ids:
            iid = f"{iid}_{len(clean)}"
        ids.add(iid)
        clean.append({"id": iid, "type": t, "label": str(it.get("label") or TYPES[t][0])[:30],
                      "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2), "rot": round(rot % 360, 1)})
    return {"version": 1, "width_m": W, "height_m": H, "provenance": "USER_EDITED", "items": clean}


def get_layout(owner: str) -> dict:
    with db.connect() as con:
        r = con.execute("SELECT layout_json, updated_at FROM factory_layout WHERE owner=?", (owner,)).fetchone()
    if not r:
        return {**default_layout(), "saved": False}
    return {**json.loads(r[0]), "saved": True, "updated_at": r[1]}


def save_layout(owner: str, layout: dict) -> dict:
    if not (owner or "").strip():
        raise ServiceError("INVALID_INPUT", "작업자 ID가 필요합니다(배치는 사람마다 따로 저장).")
    clean = validate(layout)
    with db.connect() as con:
        con.execute("INSERT INTO factory_layout (owner, layout_json, updated_at) VALUES (?,?,?) "
                    "ON CONFLICT(owner) DO UPDATE SET layout_json=excluded.layout_json, updated_at=excluded.updated_at",
                    (owner.strip(), json.dumps(clean, ensure_ascii=False), db.now()))
    return clean


def reset_layout(owner: str) -> dict:
    with db.connect() as con:
        con.execute("DELETE FROM factory_layout WHERE owner=?", (owner,))
    return default_layout()


def live_status(as_of: str | None = None, day: str | None = None, days: int = 14) -> dict:
    """3D 표시용 보관량: day(예측 날짜)의 하루 끝 예상 보관량, 없으면 기준일 현재 보관량(시연 기준값 + 확정 폐기 장부)."""
    try:
        from services.storage_projection import get_storage_projection
        s = get_storage_projection(as_of, days=days)
        cap, alert = s["max_capacity_kg"], s["alert_kg"]
        cur = s["current_stock_kg"] if not day else next(r["end_kg"] for r in s["projection"] if r["date"] == day)
        return {"scrap_kg": round(cur, 1), "capacity_kg": cap, "alert_kg": alert, "day": day,
                "fill": max(0.0, min(cur / cap, 1.2)), "dates": [r["date"] for r in s["projection"]],
                "provenance": "DEMO_ASSUMED"}
    except Exception as e:  # noqa: BLE001 - 배치도는 보관량 없이도 표시
        return {"scrap_kg": None, "capacity_kg": None, "alert_kg": None, "fill": None, "error": str(e)[:120]}


def type_table() -> list[dict]:
    return [{"type": k, "name": v[0], "height_m": v[1], "color": v[2]} for k, v in TYPES.items()]
