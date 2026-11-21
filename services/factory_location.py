"""사용자 공장 위치 (2026-11-17 사용자 요청: 고정 위치 대신 내 위치·직접 입력으로 바꿀 수 있게).

- 기본값: 계명대학교 성서캠퍼스(시연용, waste_processors.FACTORY).
- 바꾼 위치는 DB app_settings 의 'factory_location' 에 저장해 다시 열어도 유지한다. reset() 으로 기본값 복귀.
- 정하는 방법: ① 브라우저 위치(화면 컴포넌트) ② TMAP 장소·주소 검색 ③ 위도·경도 직접 입력.
- TMAP 로 보내는 것: 검색어 또는 좌표만. 키는 truck_route.app_key() 로 읽고 출력·기록하지 않는다.
"""
from __future__ import annotations

import json

from services import db
from services.config import ServiceError

KEY = "factory_location"
SCHEMA = "CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
LAT, LON = (33.0, 38.9), (124.5, 132.0)  # 대한민국 범위(처리업체 자료가 국내뿐)
SOURCES = {"DEFAULT": "기본값(시연용)", "BROWSER_GPS": "브라우저 위치", "TMAP_SEARCH": "TMAP 장소·주소 검색", "MANUAL": "직접 입력"}


def default() -> dict:
    from services.waste_processors import FACTORY
    return {**FACTORY, "source": "DEFAULT"}


def get() -> dict:
    with db.connect() as con:
        con.execute(SCHEMA)
        row = con.execute("SELECT value FROM app_settings WHERE key=?", (KEY,)).fetchone()
    if not row:
        return default()
    try:
        v = json.loads(row[0])
        return v if _ok(v.get("lat"), v.get("lon")) else default()
    except (ValueError, TypeError):
        return default()


def _ok(lat, lon) -> bool:
    try:
        return LAT[0] <= float(lat) <= LAT[1] and LON[0] <= float(lon) <= LON[1]
    except (TypeError, ValueError):
        return False


def set_location(lat: float, lon: float, name: str = "", address: str = "", source: str = "MANUAL",
                 accuracy_m: float | None = None) -> dict:
    if source not in SOURCES:
        raise ServiceError("INVALID_INPUT", f"알 수 없는 위치 출처: {source}")
    if not _ok(lat, lon):
        raise ServiceError("INVALID_INPUT", f"대한민국 범위 밖 좌표입니다(위도 {LAT[0]}~{LAT[1]}, 경도 {LON[0]}~{LON[1]}). "
                                            "처리업체 자료가 국내 업체뿐이라 국내 위치만 받습니다.")
    v = {"name": (name or "사용자 지정 공장").strip()[:60], "address": (address or "").strip()[:120],
         "lat": round(float(lat), 6), "lon": round(float(lon), 6), "source": source,
         "accuracy_m": round(float(accuracy_m)) if accuracy_m else None, "provenance": "USER_SET_FACTORY_LOCATION"}
    with db.connect() as con:
        con.execute(SCHEMA)
        con.execute("INSERT OR REPLACE INTO app_settings VALUES (?,?,?)", (KEY, json.dumps(v, ensure_ascii=False), db.now()))
    return v


def reset() -> dict:
    with db.connect() as con:
        con.execute(SCHEMA)
        con.execute("DELETE FROM app_settings WHERE key=?", (KEY,))
    return default()


# ── TMAP 장소·주소 검색 / 좌표→주소 ─────────────────────────────────────────
def _tmap_get(path: str, params: dict) -> dict:
    from services.truck_route import app_key
    key = app_key()
    if not key:
        raise ServiceError("TMAP_KEY_MISSING", "TMAP 앱 키가 없어 검색할 수 없습니다. 위도·경도를 직접 입력하세요.")
    import requests
    try:
        r = requests.get(f"https://apis.openapi.sk.com{path}", params={"version": 1, **params},
                         headers={"appKey": key, "Accept": "application/json"}, timeout=10)
    except requests.RequestException as e:
        raise ServiceError("TMAP_ERROR", f"TMAP 연결 실패: {type(e).__name__}") from None
    if r.status_code == 204:
        return {}
    if r.status_code == 429:
        raise ServiceError("TMAP_QUOTA", "TMAP 사용량 한도에 걸렸습니다. 잠시 뒤 다시 하거나 좌표를 직접 입력하세요.")
    if r.status_code in (401, 403):
        raise ServiceError("TMAP_FORBIDDEN", "이 TMAP 키로 검색 API를 쓸 수 없습니다(SK open API 에서 해당 상품 사용 신청 필요). "
                                             "좌표 직접 입력이나 내 위치를 쓰세요.")
    if not r.ok:
        raise ServiceError("TMAP_ERROR", f"TMAP 검색 오류 HTTP {r.status_code}")
    return r.json()


def search_place(query: str, count: int = 6) -> list[dict]:
    """장소 이름·주소로 검색(TMAP 장소 통합검색). 반환 [{name, address, lat, lon}]."""
    q = (query or "").strip()
    if len(q) < 2:
        raise ServiceError("INVALID_INPUT", "검색어를 두 글자 이상 넣으세요.")
    data = _tmap_get("/tmap/pois", {"searchKeyword": q, "count": count, "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO"})
    pois = ((data.get("searchPoiInfo") or {}).get("pois") or {}).get("poi") or []
    out = []
    for p in pois:
        lat, lon = p.get("frontLat") or p.get("noorLat"), p.get("frontLon") or p.get("noorLon")
        road = ""
        na = (p.get("newAddressList") or {}).get("newAddress") or []
        if na:
            road = na[0].get("fullAddressRoad", "")
        addr = road or " ".join(x for x in (p.get("upperAddrName"), p.get("middleAddrName"), p.get("lowerAddrName"),
                                            p.get("detailAddrName")) if x)
        if _ok(lat, lon):
            out.append({"name": p.get("name", ""), "address": addr, "lat": float(lat), "lon": float(lon)})
    return out


def reverse_geocode(lat: float, lon: float) -> str | None:
    """좌표 → 주소 한 줄. 실패하면 None (위치 설정은 주소 없이도 된다)."""
    try:
        data = _tmap_get("/tmap/geo/reversegeocoding", {"lat": lat, "lon": lon, "coordType": "WGS84GEO", "addressType": "A10"})
    except ServiceError:
        return None
    return ((data.get("addressInfo") or {}).get("fullAddress") or "").split(",")[0] or None
