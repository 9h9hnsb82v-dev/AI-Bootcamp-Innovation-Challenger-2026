"""수거 권장일 추천 (승인된 가상 정책: configs/facility_demo.json).

규칙
1. 수거 없이 예측했을 때 처음으로 내부 경보(용량×alert_ratio) 이상이 되는 날 = 경보일
2. 수거 가능일(요일) 중 '오늘+리드타임' 이후, 경보일 이전(포함)인 가장 늦은 날 = 권장 수거일
   (경보일 당일 수거 시 그날 발생분까지 전량 반출)
3. 그런 날이 없으면 리드타임 이후 가장 이른 수거 가능일을 URGENT로 제시(경보/초과 위험 명시)
4. 요청 마감일 = 권장 수거일 - 리드타임
"""
from __future__ import annotations

from datetime import date, timedelta

from services.config import facility
from services.storage_projection import _inputs, project_storage


def recommend_collection(start_date: str | None = None, days: int = 7) -> dict:
    start = date.fromisoformat(start_date) if start_date else date.today()
    fac, current, gen, col, _ = _inputs(start, days)
    cap = float(fac["max_capacity_kg"])
    alert = cap * float(fac["alert_ratio"])
    lead = int(fac["collection_lead_time_days"])
    weekdays = set(fac["collection_weekdays"])
    base = project_storage(current, gen, col, start, days)
    alert_day = next((date.fromisoformat(r["date"]) for r in base if r["end_kg"] >= alert), None)
    over_day = next((date.fromisoformat(r["date"]) for r in base if r["end_kg"] > cap), None)
    res = {"start_date": start.isoformat(), "alert_kg": alert, "max_capacity_kg": cap, "lead_time_days": lead,
           "projection_without_collection": base, "alert_date": alert_day and alert_day.isoformat(),
           "over_capacity_date": over_day and over_day.isoformat(), "unit": "kg",
           "provenance": ["DEMO_ASSUMED"], "policy_note": "가상 시연 정책이며 법령/실사업장 기준이 아님"}
    if alert_day is None:
        res.update(status="NO_COLLECTION_NEEDED", recommended_collection_date=None, request_by_date=None,
                   message=f"{days}일 내 경보 기준({alert:.0f}kg) 도달 예상 없음")
        return res
    earliest = start + timedelta(days=lead)
    cands = [earliest + timedelta(days=i) for i in range((alert_day - earliest).days + 1)]
    cands = [d for d in cands if d.weekday() in weekdays]
    if cands:
        pick, status = cands[-1], "RECOMMENDED"
    else:
        pick = next(earliest + timedelta(days=i) for i in range(14) if (earliest + timedelta(days=i)).weekday() in weekdays)
        status = "URGENT"
    col2 = dict(col)
    pre = next(r for r in project_storage(current, gen, col, start, (pick - start).days + 1) if r["date"] == pick.isoformat())
    if fac.get("collection_removes_all_stock", True):
        col2[pick] = col2.get(pick, 0.0) + pre["start_kg"] + pre["generation_kg"] - pre["collection_kg"]
    res.update(status=status, recommended_collection_date=pick.isoformat(),
               request_by_date=(pick - timedelta(days=lead)).isoformat(),
               expected_collection_kg=col2[pick] - col.get(pick, 0.0),
               projection_with_collection=project_storage(current, gen, col2, start, days),
               message=("경보일 이전 수거 가능" if status == "RECOMMENDED"
                        else "리드타임 때문에 경보일 이전 수거 불가 — 가장 빠른 수거 가능일 제시"))
    return res
