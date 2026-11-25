"""보관창고 방별 적재 예측 (사용자 요청 2026-11-14: 팀원 앱 '방별 적재 모습' 예시 반영).

- 방 구성(가상, DEMO_ASSUMED): A실 CN7 몰딩 350kg · B실 RG3 몰딩 300kg · C실 도어·범퍼·휀더 350kg (합 1,000kg).
  창고 용량(보관함 용량 합계)이 바뀌면 같은 비율(35:30:35)로 방 용량을 늘리거나 줄인다(2026-11-25).
- 품목이 있는 발생량(확정 폐기 장부, 생산계획의 남은 예상 kg)은 해당 방에, 품목 정보가 없는
  시연 기준 보관량·수거·조정은 방 용량 비율로 나눈다. 방별 합계 = 전체 보관량 예측(storage_projection)과 같다.
- 방별 경고선 = 방 용량 x 경보 비율(80%). 그림은 포대 1개 = 25kg, 칸 하나에 최대 2단(= 50kg)으로 무게를 칸으로 환산한 것이며
  실제 부피·배치가 아니다(부피 미추정).
"""
from __future__ import annotations

from datetime import date, timedelta

from services import db
from services.config import facility

ROOMS = [
    {"id": "A", "name": "A실 · CN7 몰딩", "products": ["CN7_LH", "CN7_RH"], "capacity_kg": 350.0},
    {"id": "B", "name": "B실 · RG3 몰딩", "products": ["RG3_LH", "RG3_RH"], "capacity_kg": 300.0},
    {"id": "C", "name": "C실 · 도어·범퍼·휀더", "products": ["DOOR", "BUMPER", "FENDER"], "capacity_kg": 350.0},
]
BAG_KG = 25.0
STACK = 2


def _room_of(product_id: str | None) -> str | None:
    return next((r["id"] for r in ROOMS if product_id in r["products"]), None)


def room_projection(start_date: str | None = None, days: int = 14) -> dict:
    from services.production_plan import list_plans
    from services.waste_forecast_update import compute_update
    start = date.fromisoformat(start_date) if start_date else date.today()
    end = start + timedelta(days=days - 1)
    fac = facility()
    k = float(fac["max_capacity_kg"]) / sum(r["capacity_kg"] for r in ROOMS)
    rooms_ = [{**r, "capacity_kg": round(r["capacity_kg"] * k, 1)} for r in ROOMS]
    total_cap = sum(r["capacity_kg"] for r in rooms_)
    share = {r["id"]: r["capacity_kg"] / total_cap for r in rooms_}
    stock = {k: float(fac["current_stock_kg"]) * v for k, v in share.items()}
    flow: dict[date, dict[str, float]] = {}

    def add(d: date, room: str | None, kgv: float) -> None:
        targets = {room: 1.0} if room else share
        for k, w in targets.items():
            if d < start:
                stock[k] += kgv * w
            elif d <= end:
                flow.setdefault(d, {}).setdefault(k, 0.0)
                flow[d][k] += kgv * w

    with db.connect() as con:
        led = db.rows(con, "SELECT * FROM stock_ledger")
    for e in led:
        sign = -1 if e["event_type"] == "COLLECTION_OUT" else 1
        add(date.fromisoformat(e["date"]), _room_of(e.get("product_id")) if sign > 0 else None, sign * float(e["kg"]))
    for p in list_plans():
        u = compute_update(p["plan_id"])
        if u["remaining_expected_kg"] > 0:
            d = max(date.fromisoformat(p["plan_date"]), start)
            if d <= end:
                add(d, _room_of(p["product_id"]), u["remaining_expected_kg"])
    alert_ratio = float(fac["alert_ratio"])
    out, s = [], dict(stock)
    for i in range(days):
        d = start + timedelta(days=i)
        for k in s:
            s[k] = max(s[k] + flow.get(d, {}).get(k, 0.0), 0.0)
        rooms = {}
        for r in rooms_:
            kgv, cap = s[r["id"]], r["capacity_kg"]
            ratio = kgv / cap
            state = "초과" if kgv > cap else "경고" if ratio >= alert_ratio else "정상"
            rooms[r["id"]] = {"kg": round(kgv, 1), "ratio": ratio, "state": state,
                              "overflow_kg": round(max(kgv - cap, 0.0), 1)}
        out.append({"date": d.isoformat(), "rooms": rooms, "total_kg": round(sum(s.values()), 1)})
    return {"rooms": rooms_, "days": out, "alert_ratio": alert_ratio, "bag_kg": BAG_KG, "stack": STACK,
            "provenance": ["DEMO_ASSUMED", "DEMO_SYNTHETIC_WEIGHT"],
            "note": "방 구성·용량은 가정값. 품목 없는 기준 보관량·수거는 방 용량 비율로 배분. 포대 1개 = 25kg, 최대 2단."}


def schedule_collection(collect_date: str, kg: float, note: str = "") -> dict:
    """권장 수거 일정 예약: 해당 날짜에 수거(COLLECTION_OUT) 이벤트를 장부에 넣는다(가상 시연, 같은 날 재예약은 덮어씀)."""
    from services.config import ServiceError
    if kg <= 0:
        raise ServiceError("INVALID_INPUT", "수거량은 0보다 커야 합니다.")
    d = date.fromisoformat(collect_date)
    src = f"COLLECT:{d.isoformat()}"
    with db.connect() as con:
        con.execute("DELETE FROM stock_ledger WHERE source_id=?", (src,))
        con.execute("INSERT INTO stock_ledger VALUES (?,?,?,?,?,?,?,?)",
                    (db.new_id("EVT"), None, None, float(kg), "COLLECTION_OUT", d.isoformat(), 1, src))
    return {"date": d.isoformat(), "kg": round(float(kg), 1), "status": "SCHEDULED", "source_id": src, "note": note}


def list_collections() -> list[dict]:
    with db.connect() as con:
        return db.rows(con, "SELECT date, kg, source_id FROM stock_ledger WHERE event_type='COLLECTION_OUT' ORDER BY date")


def cancel_collection(source_id: str) -> None:
    with db.connect() as con:
        con.execute("DELETE FROM stock_ledger WHERE source_id=? AND event_type='COLLECTION_OUT'", (source_id,))
