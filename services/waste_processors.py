"""폐플라스틱 위탁 처리업체 추천 — 공공데이터 「전국폐기물처리업소표준데이터」 (사용자 요청 2026-11-14).

- 원본: data/raw/waste_processors/전국폐기물처리업소표준데이터.csv (CP949, 원본 수정 없음)
- 대상: 처리폐기물정보에 51-03-01(폐합성수지류, 폐염화비닐수지류 제외)이 있는 업체 중 재활용업 업종(기본값).
  허가 ≠ 현재 수탁 가능 — 위탁 전 업체 확인 필요. 자료 미제출 지자체 업체는 빠져 있을 수 있음.
- 공장 위치: 기본은 계명대학교 성서캠퍼스(시연용) 35.8557, 128.4889. 2026-11-17부터 사용자가 바꿀 수 있다
  (services.factory_location — 내 위치·검색·직접 입력, DB에 저장).
- 화물차 조건(사용자 결정): 5톤 카고, 높이 3.4m, 총중량 11.5t — 도로 경로 서비스 연결 시 사용.
- 1차 순위 = 직선거리(하버사인). 도로거리·시간은 경로 서비스(사용자 선택 대기)를 붙이면 상위 N곳만 조회해 DB에 저장한다.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import pandas as pd

from services.config import ROOT, ServiceError

CSV = ROOT / "data/raw/waste_processors/전국폐기물처리업소표준데이터.csv"
FACTORY = {"name": "계명대학교 성서캠퍼스 (시연용 공장 위치)", "address": "대구광역시 달서구 달구벌대로 1095",
           "lat": 35.8557, "lon": 128.4889, "provenance": "USER_SPECIFIED_DEMO_LOCATION"}
TRUCK = {"name": "5톤 카고(가정)", "height_m": 3.4, "total_weight_t": 11.5}
WASTE_CODE = "51-03-01"
RECYCLE_TYPES = ("폐기물종합재활용업", "폐기물중간재활용업", "폐기물최종재활용업", "폐기물중간재활용업+폐기물종합재활용업")


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    if not CSV.exists():
        raise ServiceError("DATA_NOT_AVAILABLE", f"처리업체 자료 없음: {CSV.relative_to(ROOT)}")
    d = pd.read_csv(CSV, encoding="cp949", dtype=str)
    d["위도"], d["경도"] = pd.to_numeric(d["위도"], errors="coerce"), pd.to_numeric(d["경도"], errors="coerce")
    return d.dropna(subset=["위도", "경도"]).reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), pd.Series(lat2).map(math.radians)
    dphi = p2 - p1
    dl = pd.Series(lon2).map(math.radians) - math.radians(lon1)
    a = (dphi / 2).map(math.sin) ** 2 + math.cos(p1) * p2.map(math.cos) * (dl / 2).map(math.sin) ** 2
    return 2 * r * a.map(lambda v: math.asin(math.sqrt(min(1.0, v))))


def industries() -> list[str]:
    return sorted(_load()["업종명"].dropna().unique())


def recommend(types: tuple[str, ...] = RECYCLE_TYPES, n: int = 10, waste_code: str = WASTE_CODE,
              origin: dict | None = None) -> dict:
    """공장에서 직선거리가 가까운 순. origin 이 없으면 저장된 공장 위치(factory_location.get()). 반환: 공장·화물차 조건·후보 수·추천 목록."""
    if origin is None:
        from services.factory_location import get
        origin = get()
    if not 1 <= int(n) <= 50:
        raise ServiceError("INVALID_INPUT", "추천 수는 1~50")
    d = _load()
    pool = d[d["처리폐기물정보"].fillna("").str.contains(waste_code, regex=False) & d["업종명"].isin(types)].copy()
    pool["직선거리_km"] = haversine_km(origin["lat"], origin["lon"], pool["위도"].values, pool["경도"].values).values
    top = pool.nsmallest(int(n), "직선거리_km").reset_index(drop=True)
    top.insert(0, "순위", range(1, len(top) + 1))
    cols = ["순위", "시설명", "업종명", "소재지도로명주소", "소재지지번주소", "위도", "경도", "직선거리_km", "시설·장비명",
            "전화번호", "영업구역", "허가일자", "데이터기준일자", "관리기관명"]
    return {"factory": origin, "truck": TRUCK, "waste_code": waste_code, "candidates": int(len(pool)),
            "total_rows": int(len(d)), "data_dates": (d["데이터기준일자"].min(), d["데이터기준일자"].max()),
            "items": top[cols].to_dict("records"),
            "provenance": ["PUBLIC_DATA_WASTE_PROCESSORS", origin.get("provenance", "USER_SPECIFIED_DEMO_LOCATION")],
            "note": "허가 ≠ 현재 수탁 가능 — 위탁 전 업체 확인 필요. 도로거리는 경로 서비스 연결 후 표시."}


# ── 챗봇 조건 검색 (2026-11-18 사용자 승인, RAG ①) ─────────────────────────
CODE_DOC = ROOT / "data/docs/법령/폐기물관리법 시행규칙 별표4 폐기물 분류번호.md"


@lru_cache(maxsize=1)
def waste_codes() -> dict[str, str]:
    """폐기물관리법 시행규칙 별표 4의 분류번호 → 이름(예: '51-03-01' → '폐합성수지류(폐염화비닐수지류는 제외한다)')."""
    if not CODE_DOC.exists():
        return {}
    out, last = {}, None
    for ln in CODE_DOC.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\d{2}-\d{2}(?:-\d{2})?)\s+(.+)", ln)
        if m:
            last = m.group(1)
            out[last] = m.group(2).strip()
        elif last and ln.strip() and not ln.strip().startswith(("#", "-", "`", "■", "1.", "2.", "3.")) and ln.startswith(" ") is False:
            out[last] += ln.strip()  # PDF 줄바꿈으로 잘린 이름 이어 붙이기
    return out


def _resolve_codes(waste: str) -> list[str]:
    w = waste.strip()
    if re.fullmatch(r"\d{2}-\d{2}(-\d{2})?", w):
        return [w]
    key = w.replace(" ", "")
    return [c for c, n in waste_codes().items() if key and key in n.replace(" ", "")]


def search_processors(region: str | None = None, waste: str | None = None, business: str | None = None,
                      limit: int = 10, origin: dict | None = None) -> dict:
    """지역(주소 일부)·폐기물(분류번호 또는 이름 일부)·업종(일부)으로 처리업체를 찾는다. 공장 위치에서 가까운 순."""
    d = _load()
    limit = max(1, min(int(limit or 10), 20))
    m = pd.Series(True, index=d.index)
    applied = {}
    if region and region.strip():
        r = region.strip()
        m &= d["소재지도로명주소"].fillna("").str.contains(r, regex=False) | d["소재지지번주소"].fillna("").str.contains(r, regex=False)
        applied["지역"] = r
    if waste and waste.strip():
        codes = _resolve_codes(waste)
        if not codes:
            raise ServiceError("INVALID_INPUT", f"'{waste}'에 맞는 폐기물 분류를 찾지 못했습니다(예: 51-03-01, 폐합성수지).")
        info = d["처리폐기물정보"].fillna("")
        hit = pd.Series(False, index=d.index)
        for c in codes[:30]:
            hit |= info.str.contains(c if c.count("-") == 2 else c + "-", regex=False)
        m &= hit
        applied["폐기물"] = {c: waste_codes().get(c, "") for c in codes[:8]}
    if business and business.strip():
        m &= d["업종명"].fillna("").str.contains(business.strip(), regex=False)
        applied["업종"] = business.strip()
    pool = d[m].copy()
    if origin is None:
        from services.factory_location import get
        origin = get()
    pool["직선거리_km"] = haversine_km(origin["lat"], origin["lon"], pool["위도"].values, pool["경도"].values).values.round(2) \
        if len(pool) else []
    top = pool.nsmallest(limit, "직선거리_km") if len(pool) else pool
    cols = ["시설명", "업종명", "소재지도로명주소", "직선거리_km", "전화번호", "처리폐기물정보", "데이터기준일자", "관리기관명"]
    items = top[cols].fillna("").to_dict("records")
    for it in items:  # 분류번호를 이름과 함께(앞 5개만)
        cs = re.findall(r"\d{2}-\d{2}-\d{2}", it.pop("처리폐기물정보"))
        it["처리폐기물(일부)"] = [f"{c} {waste_codes().get(c, '')}".strip() for c in cs[:5]] + (["…"] if len(cs) > 5 else [])
    return {"conditions": applied, "matched": int(len(pool)), "shown": len(items), "factory": origin.get("name"),
            "items": items, "provenance": ["PUBLIC_DATA_WASTE_PROCESSORS"],
            "note": "허가 정보이며 현재 수탁 가능 여부·처리 단가는 아님 — 위탁 전 업체 확인 필요. 자료 미제출 지자체 업체는 빠져 있을 수 있음."}


# ── 자료 누락 지역 경고 (팀원 앱 반영 2026-11-14) ──────────────────────────
# 시군구 중심점: 통계청 SGIS 행정경계(공공누리 1유형) 가공본 vuski/admdongkor '통계청MDIS인구용_행정경계중심점'(CC BY 4.0)
CENTROIDS = ROOT / "data/raw/admin_centroids/coordinate_UTMK_names.tsv"


def utmk_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """EPSG:5179(UTM-K, GRS80, 중앙경선 127.5°, 원점위도 38°, k0 0.9996, FE 1,000,000, FN 2,000,000) → (위도, 경도)."""
    a, f = 6378137.0, 1 / 298.257222101
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    k0, lat0, lon0, fe, fn = 0.9996, math.radians(38.0), math.radians(127.5), 1_000_000.0, 2_000_000.0

    def mer(phi):
        return a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
                    + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi) - (35 * e2**3 / 3072) * math.sin(6 * phi))
    m = mer(lat0) + (y - fn) / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p1 = mu + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu) + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu) \
        + (151 * e1**3 / 96) * math.sin(6 * mu) + (1097 * e1**4 / 512) * math.sin(8 * mu)
    c1, t1 = ep2 * math.cos(p1) ** 2, math.tan(p1) ** 2
    n1 = a / math.sqrt(1 - e2 * math.sin(p1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * math.sin(p1) ** 2) ** 1.5
    d = (x - fe) / (n1 * k0)
    lat = p1 - (n1 * math.tan(p1) / r1) * (d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
                                           + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720)
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d**3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120) / math.cos(p1)
    return math.degrees(lat), math.degrees(lon)


@lru_cache(maxsize=1)
def sgg_centroids() -> pd.DataFrame:
    """시군구 중심점(위도·경도). 행정코드 끝 5자리가 0이고 시도(끝 8자리 0)가 아닌 행."""
    if not CENTROIDS.exists():
        return pd.DataFrame(columns=["name", "lat", "lon"])
    t = pd.read_csv(CENTROIDS, sep="\t", dtype={"ADMCD": str})
    t = t[t.ADMCD.str.endswith("00000") & ~t.ADMCD.str.endswith("00000000")].copy()
    ll = [utmk_to_wgs84(float(x), float(y)) for x, y in zip(t.X, t.Y)]
    return pd.DataFrame({"name": t.ADMNM.values, "lat": [p[0] for p in ll], "lon": [p[1] for p in ll]})


def missing_regions(radius_km: float, origin: dict | None = None) -> list[dict]:
    """공장에서 radius_km 안의 시군구 중 처리업소 자료(제공기관)가 한 곳도 없는 지역 = 자료 미제출 가능 지역."""
    c = sgg_centroids()
    if c.empty:
        return []
    d = _load()
    have = set(d["제공기관명"].dropna().str.strip()) | set(d["관리기관명"].dropna().str.replace("청$", "", regex=True).str.strip())
    o = origin or FACTORY
    c = c.assign(km=haversine_km(o["lat"], o["lon"], c.lat.values, c.lon.values).values)
    near = c[c.km <= radius_km]
    out = []
    for r in near.itertuples():
        nm = r.name.strip()
        if not any(nm == h or nm.replace(" ", "") == h.replace(" ", "") for h in have):
            out.append({"name": nm, "lat": r.lat, "lon": r.lon, "km": round(r.km, 1)})
    return sorted(out, key=lambda x: x["km"])
