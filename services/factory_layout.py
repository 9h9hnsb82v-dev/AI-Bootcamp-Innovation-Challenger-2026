"""가상 공장 배치도(2D 편집 · 3D 보기) 저장소 (사용자 요청 2026-11-14).

- 실제 공장 크기를 모르므로 시연용 기본 배치(80m x 50m, DEMO_ASSUMED)를 베이스로 둔다.
- 공장 배치 하나만 저장(2026-11-25 사용자 결정, 이전: 작업자 ID별). DB factory_layout 의 owner='factory' 한 줄.
  예전 작업자별 배치가 있으면 가장 최근 것을 공장 배치로 쓴다. 저장 전에는 기본 배치.
- 폐플라스틱 보관함마다 용량(capacity_kg)을 둔다(2026-11-25). 보관함 용량 합계 = 공장 보관 용량(config.facility()).
  쌓인 양은 배치 순서대로 앞 보관함을 100%까지 채운 뒤 다음 보관함으로 넘긴다(split_bins).
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
OWNER = "factory"
MAX_BIN_KG = 100_000.0


def default_bin_kg() -> float:
    """보관함 기본 용량 = 승인된 가상 창고 용량(configs/facility_demo.json, 1,000kg)."""
    from services.config import facility_base
    return float(facility_base()["max_capacity_kg"])


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
    items = [{"id": i, "type": t, "label": lab, "x": x, "y": y, "w": w, "h": h, "rot": 0} for i, t, lab, x, y, w, h in it]
    for i in items:
        if i["type"] == "scrap_bin":
            i["capacity_kg"] = default_bin_kg()
    return {"version": 1, "width_m": 80, "height_m": 50, "provenance": "DEMO_ASSUMED", "items": items}


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
        row = {"id": iid, "type": t, "label": str(it.get("label") or TYPES[t][0])[:30],
               "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2), "rot": round(rot % 360, 1)}
        if t == "scrap_bin":
            try:
                cap = float(it.get("capacity_kg") or default_bin_kg())
            except (TypeError, ValueError) as e:
                raise ServiceError("INVALID_INPUT", f"보관함 '{row['label']}' 용량이 숫자가 아닙니다.") from e
            if not (1 <= cap <= MAX_BIN_KG):
                raise ServiceError("INVALID_INPUT", f"보관함 '{row['label']}' 용량은 1~{MAX_BIN_KG:,.0f} kg 범위여야 합니다.")
            row["capacity_kg"] = round(cap, 1)
        clean.append(row)
    return {"version": 1, "width_m": W, "height_m": H, "provenance": "USER_EDITED", "items": clean}


def get_layout() -> dict:
    """공장 배치. 공장 배치가 없으면 예전 작업자별 배치 중 가장 최근 것, 그것도 없으면 기본 배치."""
    with db.connect() as con:
        r = con.execute("SELECT layout_json, updated_at FROM factory_layout WHERE owner=?", (OWNER,)).fetchone() \
            or con.execute("SELECT layout_json, updated_at FROM factory_layout ORDER BY updated_at DESC LIMIT 1").fetchone()
    if not r:
        return {**default_layout(), "saved": False}
    lay = json.loads(r[0])
    for it in lay.get("items", []):  # 용량 기능 이전에 저장된 보관함 → 기본 용량
        if it.get("type") == "scrap_bin" and not it.get("capacity_kg"):
            it["capacity_kg"] = default_bin_kg()
    return {**lay, "saved": True, "updated_at": r[1]}


def save_layout(layout: dict) -> dict:
    clean = validate(layout)
    with db.connect() as con:
        con.execute("INSERT INTO factory_layout (owner, layout_json, updated_at) VALUES (?,?,?) "
                    "ON CONFLICT(owner) DO UPDATE SET layout_json=excluded.layout_json, updated_at=excluded.updated_at",
                    (OWNER, json.dumps(clean, ensure_ascii=False), db.now()))
    return clean


def reset_layout() -> dict:
    """기본 배치로(예전 작업자별 배치까지 지운다 — 남겨 두면 다시 불러와지므로)."""
    with db.connect() as con:
        con.execute("DELETE FROM factory_layout")
    return default_layout()


def bins(layout: dict | None = None) -> list[dict]:
    """보관함 목록(채우는 순서 = 배치 순서): [{id, label, capacity_kg}]."""
    lay = layout or get_layout()
    return [{"id": i["id"], "label": i["label"], "capacity_kg": float(i.get("capacity_kg") or default_bin_kg())}
            for i in lay.get("items", []) if i.get("type") == "scrap_bin"]


def split_bins(total_kg: float, caps: list[float]) -> list[float]:
    """앞 보관함을 100%까지 채우고 남는 양을 다음 보관함으로. 전체 용량을 넘는 양은 마지막 보관함에 더한다(초과 표시)."""
    out, left = [], max(float(total_kg), 0.0)
    for c in caps:
        k = min(left, c)
        out.append(k)
        left -= k
    if out and left > 0:
        out[-1] += left
    return out


def live_status(as_of: str | None = None, day: str | None = None, days: int = 14) -> dict:
    """3D 표시용 보관량: day(예측 날짜)의 하루 끝 예상 보관량, 없으면 기준일 현재 보관량(시연 기준값 + 확정 폐기 장부)."""
    try:
        from services.storage_projection import get_storage_projection
        s = get_storage_projection(as_of, days=days)
        cap, alert = s["max_capacity_kg"], s["alert_kg"]
        cur = s["current_stock_kg"] if not day else next(r["end_kg"] for r in s["projection"] if r["date"] == day)
        bs = bins()
        per = [{**b, "kg": round(k, 1), "fill": k / b["capacity_kg"]}
               for b, k in zip(bs, split_bins(cur, [b["capacity_kg"] for b in bs]))]
        return {"scrap_kg": round(cur, 1), "capacity_kg": cap, "alert_kg": alert, "alert_ratio": alert / cap, "day": day,
                "fill": max(0.0, min(cur / cap, 1.2)), "dates": [r["date"] for r in s["projection"]],
                "bins": per, "provenance": "DEMO_ASSUMED"}
    except Exception as e:  # noqa: BLE001 - 배치도는 보관량 없이도 표시
        return {"scrap_kg": None, "capacity_kg": None, "alert_kg": None, "fill": None, "bins": [], "error": str(e)[:120]}


def type_table() -> list[dict]:
    return [{"type": k, "name": v[0], "height_m": v[1], "color": v[2]} for k, v in TYPES.items()]
