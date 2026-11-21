"""화물차 도로 경로 — TMAP (사용자 결정 2026-11-14).

- TMAP API 중 화물차 조건(높이·중량)을 받는 것은 '다중 경유지 안내 30'(POST /tmap/routes/routeSequential30)뿐이다
  (공식 레퍼런스 확인 2026-11-14; '자동차 경로안내'에는 화물차 항목이 없음). 경유지 = 도착지 1개로 출발→도착만 요청한다.
- 앱 키: 환경변수 TMAP_APP_KEY 또는 .streamlit/secrets.toml 의 TMAP_APP_KEY (저장소·업로드에서 제외). 헤더 'appKey'.
- 화물차: 기본 5톤 카고(가정) — 높이 340cm·폭 249cm·길이 900cm·적재 5,000kg·총중량 11,500kg. 화면에서 수정 가능.
- 전송 정보: 공장 좌표(시연용 공개 좌표)와 업체 좌표만. 결과는 DB route_cache에 저장해 같은 구간은 다시 요청하지 않는다.
- 키가 없거나 호출이 실패하면 None을 돌려주고 화면은 직선거리로 동작한다.
"""
from __future__ import annotations

import json
import os
import tomllib
from datetime import datetime

from services import db
from services.config import ROOT

URL = "https://apis.openapi.sk.com/tmap/routes/routeSequential30?version=1"
# 기본 화물차 제원(5톤 카고트럭 가정, 팀원 앱과 같은 값). 화면에서 바꿀 수 있고, 제원이 다르면 경로를 따로 조회·저장한다.
DEFAULT_TRUCK = {"height_cm": 340, "width_cm": 249, "length_cm": 900, "load_kg": 5000, "total_kg": 11500}
LIMITS = {"height_cm": (100, 600), "width_cm": (100, 300), "length_cm": (200, 4000), "load_kg": (500, 600000),
          "total_kg": (500, 600000)}  # TMAP 레퍼런스 허용 범위


def truck_params(spec: dict | None = None) -> dict:
    from services.config import ServiceError
    t = {**DEFAULT_TRUCK, **(spec or {})}
    for k, (lo, hi) in LIMITS.items():
        if not lo <= int(t[k]) <= hi:
            raise ServiceError("INVALID_INPUT", f"화물차 {k}는 {lo}~{hi} 범위여야 합니다(TMAP 기준).")
    return {"truckType": "1", "truckHeight": str(int(t["height_cm"])), "truckWidth": str(int(t["width_cm"])),
            "truckLength": str(int(t["length_cm"])), "truckWeight": str(int(t["load_kg"])),
            "truckTotalWeight": str(int(t["total_kg"]))}


def _spec_key(tp: dict) -> str:
    return "truck_h{truckHeight}_w{truckWidth}_l{truckLength}_ld{truckWeight}_t{truckTotalWeight}".format(**tp)
_LAST_CALL = 0.0
SCHEMA = """CREATE TABLE IF NOT EXISTS route_cache (
  cache_key TEXT PRIMARY KEY, distance_m REAL, time_s REAL, coords_json TEXT, provider TEXT, fetched_at TEXT NOT NULL)"""


def app_key() -> str | None:
    k = os.environ.get("TMAP_APP_KEY")
    if k:
        return k.strip()
    p = ROOT / ".streamlit/secrets.toml"
    if p.exists():
        try:
            return (tomllib.loads(p.read_text(encoding="utf-8")).get("TMAP_APP_KEY") or "").strip() or None
        except tomllib.TOMLDecodeError:
            return None
    return None


def _key(a: tuple[float, float], b: tuple[float, float], spec: str | None = None) -> str:
    return f"tmap:{spec or _spec_key(truck_params())}:{a[0]:.5f},{a[1]:.5f}:{b[0]:.5f},{b[1]:.5f}"


def _cached(ck: str) -> dict | None:
    with db.connect() as con:
        con.execute(SCHEMA)
        r = con.execute("SELECT distance_m, time_s, coords_json FROM route_cache WHERE cache_key=?", (ck,)).fetchone()
    return {"distance_km": r[0] / 1000, "minutes": r[1] / 60, "coords": json.loads(r[2]), "cached": True} if r else None


def parse(resp: dict) -> dict:
    """routeSequential30 응답 → 총거리(m)·총시간(s)·경로선 좌표[[경도, 위도], ...]."""
    props = resp.get("properties") or {}
    feats = resp.get("features") or []
    if "totalDistance" not in props:  # 일부 응답은 첫 feature에 합계가 있음
        props = next((f.get("properties", {}) for f in feats if "totalDistance" in f.get("properties", {})), {})
    coords = [c for f in feats if (f.get("geometry") or {}).get("type") == "LineString" for c in f["geometry"]["coordinates"]]
    if "totalDistance" not in props or not coords:
        raise ValueError("TMAP 응답에서 거리 또는 경로를 찾지 못함")
    return {"distance_m": float(props["totalDistance"]), "time_s": float(props["totalTime"]), "coords": coords}


QUOTA_EXCEEDED = False  # 일일 사용량 초과를 받으면 프로세스가 다시 시작될 때까지 새 호출을 하지 않음


def route(start: tuple[float, float], end: tuple[float, float], start_name: str = "공장", end_name: str = "업체",
          timeout: float = 10.0, truck: dict | None = None, cache_only: bool = False) -> dict | None:
    """start/end = (위도, 경도), truck = 화물차 제원(DEFAULT_TRUCK 키). 반환 {distance_km, minutes, coords, cached} 또는 None."""
    tp = truck_params(truck)
    ck = _key(start, end, _spec_key(tp))
    hit = _cached(ck)
    if hit or cache_only:
        return hit
    global QUOTA_EXCEEDED
    if QUOTA_EXCEEDED:
        return {"error": "TMAP 사용량 한도 초과(QUOTA_EXCEEDED) - 오늘은 저장된 경로만 표시"}
    key = app_key()
    if not key:
        return None
    import requests
    body = {"startName": start_name, "startX": f"{start[1]}", "startY": f"{start[0]}",
            "startTime": datetime.now().strftime("%Y%m%d%H%M"), "endName": end_name,
            "endX": f"{end[1]}", "endY": f"{end[0]}",
            # 경유지를 비우면 1101 오류(2026-11-14 실호출 확인) → 도착지와 같은 지점 1개(경로 불변, 출발지로 둬도 같은 결과)
            "viaPoints": [{"viaPointId": "v1", "viaPointName": end_name[:20] or "도착", "viaX": f"{end[1]}", "viaY": f"{end[0]}"}],
            "reqCoordType": "WGS84GEO",
            "resCoordType": "WGS84GEO", "searchOption": "0", **tp}
    import time
    global _LAST_CALL
    try:
        for attempt in range(3):  # 초당 호출 제한(429) 대비: 호출 간격 0.4초, 429면 1.5초·3초 기다렸다 재시도
            wait = 0.4 - (time.monotonic() - _LAST_CALL)
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL = time.monotonic()
            r = requests.post(URL, headers={"appKey": key, "Content-Type": "application/json", "Accept": "application/json"},
                              json=body, timeout=timeout)
            if r.status_code != 429:
                break
            if "QUOTA_EXCEEDED" in r.text:  # 일일 한도 초과는 재시도해도 소용없음
                QUOTA_EXCEEDED = True
                return {"error": "TMAP 사용량 한도 초과(QUOTA_EXCEEDED) - 오늘은 저장된 경로만 표시"}
            time.sleep(1.5 * (attempt + 1))
        r.raise_for_status()
        out = parse(r.json())
    except Exception as e:  # noqa: BLE001 - 화면은 직선거리로 계속 동작
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    with db.connect() as con:
        con.execute(SCHEMA)
        con.execute("INSERT OR REPLACE INTO route_cache VALUES (?,?,?,?,?,?)",
                    (ck, out["distance_m"], out["time_s"], json.dumps(out["coords"]), "TMAP_routeSequential30", db.now()))
    return {"distance_km": out["distance_m"] / 1000, "minutes": out["time_s"] / 60, "coords": out["coords"], "cached": False}
