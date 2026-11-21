"""KAMP ID 40 「사출성형 공급망최적화 AI 데이터셋」 로더 및 일 단위 스냅샷 구성.

데이터 경계(중요)
- 이 모듈은 비식별 `Product_Number`만 다룬다. CN7/RG3/DOOR/BUMPER/FENDER 등
  제조 7종 품목 ID를 import하거나 참조하지 않는다.
- 타깃 `T일 예정 수주량`은 "예정(계획) 수주량"이며 확정 주문 실적이 아니다.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = ROOT / "data/raw/kamp_supply/Dataset_사출성형 공급망최적화 AI 데이터셋.zip"
PROCESSED_DIR = ROOT / "data/processed/kamp_supply"

KEY = "Product_Number"
TARGET = "T일 예정 수주량"
TARGET_SEMANTICS = "PLANNED_ORDER_QUANTITY_NOT_REALIZED_ORDERS"
HORIZONS = (0, 1, 2, 3, 4)


def planned_col(k: int) -> str:
    return "T일 예정 수주량" if k == 0 else f"T+{k}일 예정 수주량"


def last_year_col(k: int) -> str:
    return "작년 T일 예정 수주량" if k == 0 else f"작년 T+{k}일 예정 수주량"


def company_fc_col(k: int) -> str:
    return "T일 예상 수주량" if k == 0 else f"T+{k}일 예상 수주량"


PLANNED_COLS = [planned_col(k) for k in HORIZONS]
LAST_YEAR_COLS = [last_year_col(k) for k in HORIZONS]
COMPANY_FC_COLS = [company_fc_col(k) for k in HORIZONS]
QTY_COLS = PLANNED_COLS + LAST_YEAR_COLS + COMPANY_FC_COLS
EXPECTED_COLUMNS = [KEY, *QTY_COLS, "DateTime", "DoW", "Temperature", "Humidity"]


def _csv_member(z: zipfile.ZipFile) -> str:
    members = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if len(members) != 1:
        raise FileNotFoundError(f"ID40 ZIP에서 CSV를 1개로 특정할 수 없음: {members}")
    return members[0]


def raw_sha256(path: Path = RAW_ZIP) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path = RAW_ZIP) -> pd.DataFrame:
    """원본 CSV를 그대로(행 순서 유지) 읽고 `DateTime`을 혼합 서식으로 파싱한다.

    `DateTime_raw`에 원문 문자열을 보존하며, 파싱 실패는 NaT로 남겨 감사에서 보고한다.
    시각은 타임존 없는 현지 시각(naive)으로 취급한다.
    """
    if not path.exists():
        raise FileNotFoundError("DATA_NOT_AVAILABLE: " + str(path))
    with zipfile.ZipFile(path) as z:
        with z.open(_csv_member(z)) as f:
            df = pd.read_csv(f, encoding="utf-8-sig", dtype={"DateTime": str})
    df["DateTime_raw"] = df["DateTime"]
    df["DateTime"] = pd.to_datetime(df["DateTime_raw"], format="mixed", errors="coerce")
    df["Date"] = df["DateTime"].dt.normalize()
    return df


def tag_snapshot_provenance(snap: pd.DataFrame) -> pd.DataFrame:
    """제품별로 15개 수량 컬럼이 이전 날짜 스냅샷과 완전히 같은 행을 복제본으로 표시한다.

    감사 결과(2026-11-10): 2022-03-24~05-09는 2022-01-26~03-13을 57일 뒤로 재생한 복제
    구간이고, 연휴 등 일부 날짜는 전날 스냅샷 복사본이다. 복제 행은 새로운 관측이 아니다.
    snapshot_status: ORIGINAL | REPEAT_OF_EARLIER | ALL_ZERO
    """
    s = snap.sort_values([KEY, "Date"], kind="mergesort").copy()
    sig = s[QTY_COLS].astype("int64").astype(str).agg(",".join, axis=1)
    first_date = s.assign(_sig=sig).groupby([KEY, "_sig"])["Date"].transform("min")
    all_zero = (s[QTY_COLS] == 0).all(axis=1)
    s["source_date"] = first_date
    s["snapshot_status"] = "ORIGINAL"
    s.loc[first_date < s["Date"], "snapshot_status"] = "REPEAT_OF_EARLIER"
    s.loc[all_zero, "snapshot_status"] = "ALL_ZERO"
    return s.reset_index(drop=True)


def daily_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """(Product_Number, Date)당 하루 마지막 로그 1행만 남긴다.

    가이드북 [코드 12]와 같은 규칙(같은 날 마지막 로그가 그날 정보를 대표)이며,
    여기서는 하루 마지막 시점(일 마감)에 스냅샷이 확보된다고 가정한다.
    """
    d = df.dropna(subset=["DateTime"]).sort_values([KEY, "DateTime"], kind="mergesort")
    d = d.drop_duplicates([KEY, "Date"], keep="last").reset_index(drop=True)
    return d
