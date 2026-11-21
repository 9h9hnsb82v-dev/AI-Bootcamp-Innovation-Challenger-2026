"""재활용가능자원 가격 조회 (챗봇 도구, 2026-11-19 사용자 요청: 2025-01 ~ 2026-10 간행물 PDF).

- 자료: data/processed/recycling_price/price_long.csv (scripts/build_price_docs.py 가 간행물 22호에서 만든 품목·기간·권역·가격)
- 원/kg, VAT 포함·물류비 제외. 대부분 가공업체의 '판매' 가격(폐지·철스크랩은 '매입' 가격) — 공장이 받는 값·처리비가 아니다.
- 작은 LLM 이 넓은 표에서 칸을 잘못 읽는 일이 있어(실측) 숫자 질문은 이 도구로 정확히 꺼낸다.
"""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from services.config import ROOT, ServiceError

CSV = ROOT / "data/processed/recycling_price/price_long.csv"
REGIONS = ["전국", "수도권", "강원", "충북", "충남", "전북", "전남", "경북", "경남"]
STAGE = {"폐신문지": "매입", "폐골판지": "매입", "철스크랩": "매입"}


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    if not CSV.exists():
        raise ServiceError("DATA_NOT_AVAILABLE", "재활용 가격 자료 없음: python -m scripts.build_price_docs 로 만드세요.")
    return pd.read_csv(CSV, encoding="utf-8-sig")


def _key(s: str) -> str:
    return re.sub(r"[\s\-_()·,]|잡색", "", str(s)).upper().replace("펠릿", "펠렛").replace("PELLET", "펠렛").replace("FLAKE", "플레이크")


def items() -> list[str]:
    return sorted(_load()["item"].unique())


def _match_item(q: str) -> str:
    k = _key(q)
    names = items()
    for n in names:
        if _key(n) == k:
            return n
    # 순서를 바꿔 쓴 경우(압축 PP ↔ PP 압축, 신문지 ↔ 폐신문지)
    toks = [t for t in re.split(r"[\s\-_]+", str(q).upper().replace("잡색", "")) if t]
    for n in names:
        nk = _key(n)
        if toks and all(_key(t) in nk for t in toks):
            return n
    raise ServiceError("INVALID_INPUT", f"'{q}' 품목을 찾지 못했습니다. 품목: {', '.join(names)}")


def _period(p: str | None) -> str | None:
    if not p:
        return None
    s = str(p).strip()
    m = re.search(r"(20\d\d)\D{0,3}(\d{1,2})?", s)
    if not m:
        raise ServiceError("INVALID_INPUT", f"기간 '{p}'을 알 수 없습니다(예: 2026-10, 2025).")
    return f"{m.group(1)}-{int(m.group(2)):02d}" if m.group(2) else m.group(1)


def get_price(item: str, period: str | None = None, region: str | None = None, start: str | None = None,
              end: str | None = None) -> dict:
    """품목 가격. period 하나(2026-10 · 2025=연평균) 또는 start~end 월 범위. region 없으면 전국+8개 권역."""
    d = _load()
    it = _match_item(item)
    g = d[d["item"] == it]
    if region and str(region).strip() not in ("", "전체"):
        reg = str(region).strip().replace("서울", "수도권").replace("경기", "수도권").replace("인천", "수도권")
        reg = {"대구": "경북", "부산": "경남", "울산": "경남", "광주": "전남", "대전": "충남", "세종": "충남"}.get(reg, reg)
        if reg not in REGIONS:
            raise ServiceError("INVALID_INPUT", f"권역은 {', '.join(REGIONS)} 중 하나(제주는 조사 제외).")
        g = g[g["region"] == reg]
    p, s, e = _period(period), _period(start), _period(end)
    if s or e:  # 범위가 있으면 범위 우선(period 와 함께 오면 범위)
        g = g[(g["period"].str.len() == 7) & (g["period"] >= (s or "0000")) & (g["period"] <= (e or "9999"))]
    elif p:
        g = g[g["period"] == p]
    elif False:
        g = g[(g["period"].str.len() == 7) & (g["period"] >= (s or "0000")) & (g["period"] <= (e or "9999"))]
    else:
        last = d[d["period"].str.len() == 7]["period"].max()
        g = g[g["period"] == last]
    if g.empty:
        raise ServiceError("DATA_NOT_AVAILABLE", f"{it} {period or ''} 자료 없음(2020~2025 연평균, 2024-01 ~ 2026-10 월별만 있음).")
    g = g.assign(_o=g["region"].map(REGIONS.index)).sort_values(["period", "_o"])
    rows = [{"기간": r.period, "권역": r.region, "원/kg": (None if pd.isna(r.price) else float(r.price))} for r in g.itertuples()][:60]
    return {"item": it, "rows": rows, "unit": "원/kg (VAT 포함, 물류비 제외)",
            "stage": ("중간가공업체 매입 가격" if it in STAGE else "가공업체가 다음 단계 업체에 판매하는 가격"),
            "source": "한국환경공단 재활용가능자원 가격조사 간행물(2025-01 ~ 2026-10호)", "provenance": ["PUBLIC_RECYCLING_PRICE_SURVEY"],
            "note": "None = 미공표. 공장이 불량품을 넘길 때 받는 값이나 처리비가 아니며 실제 거래가는 업체 견적 필요."}
