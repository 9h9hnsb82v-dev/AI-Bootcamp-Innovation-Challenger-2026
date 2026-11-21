"""audit_supply_dataset(): KAMP ID 40 원본 스키마·수량·연속성·시점 감사.

실행: .venv/Scripts/python -m ml.supply.audit
출력: reports/audit/kamp_id40_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.supply import data as D

OUT = D.ROOT / "reports/audit/kamp_id40_audit.json"


def _revision_check(snap: pd.DataFrame, col_fn, k: int) -> dict:
    """D-k일 스냅샷의 `T+k` 값과 D일 스냅샷의 `T` 값이 같은지(사후 수정 여부) 비교."""
    a = snap[[D.KEY, "Date", col_fn(k)]].copy()
    a["Date"] = a["Date"] + pd.Timedelta(days=k)
    b = snap[[D.KEY, "Date", col_fn(0)]]
    m = a.merge(b, on=[D.KEY, "Date"], how="inner")
    if m.empty:
        return {"pairs": 0}
    x, y = m[col_fn(k)].to_numpy(float), m[col_fn(0)].to_numpy(float)
    return {
        "pairs": int(len(m)),
        "exact_equal_ratio": round(float(np.mean(x == y)), 4),
        "mean_abs_diff": round(float(np.mean(np.abs(x - y))), 3),
        "corr": round(float(np.corrcoef(x, y)[0, 1]), 4) if np.std(x) > 0 and np.std(y) > 0 else None,
    }


def audit_supply_dataset(path: Path = D.RAW_ZIP) -> dict:
    if not path.exists():
        return {"status": "DATA_NOT_AVAILABLE", "path": str(path)}
    df = D.load_raw(path)
    rep: dict = {"status": "OK", "source_zip": str(path.relative_to(D.ROOT)),
                 "source_sha256": D.raw_sha256(path)}
    rep["shape"] = list(df.drop(columns=["DateTime_raw", "Date"]).shape)
    rep["columns"] = [c for c in df.columns if c not in ("DateTime_raw", "Date")]
    rep["columns_match_expected"] = rep["columns"] == D.EXPECTED_COLUMNS
    rep["n_products"] = int(df[D.KEY].nunique())
    rep["target"] = {"column": D.TARGET, "semantics": D.TARGET_SEMANTICS}

    # DateTime
    raw = df["DateTime_raw"].astype(str)
    rep["datetime"] = {
        "parse_failures": int(df["DateTime"].isna().sum()),
        "format_with_seconds": int(raw.str.match(r"^\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}$").sum()),
        "format_without_seconds": int(raw.str.match(r"^\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}$").sum()),
        "min": str(df["DateTime"].min()), "max": str(df["DateTime"].max()),
        "n_dates": int(df["Date"].nunique()),
        "outside_guide_period_2022-01-26_2022-05-11": int(
            ((df["Date"] < "2022-01-26") | (df["Date"] > "2022-05-11")).sum()),
    }
    dow_calc = df["DateTime"].dt.day_name()
    rep["dow_mismatch_rows"] = int((dow_calc != df["DoW"]).sum())

    # duplicates
    rep["duplicates"] = {
        "exact_duplicate_rows": int(df.duplicated(subset=rep["columns"]).sum()),
        "dup_product_datetime": int(df.duplicated([D.KEY, "DateTime"]).sum()),
        "dup_product_datetime_conflicting_values": int(
            df.groupby([D.KEY, "DateTime"])[D.QTY_COLS].nunique().gt(1).any(axis=1).sum()),
        "rows_per_product_date": df.groupby([D.KEY, "Date"]).size().value_counts().sort_index()
        .rename(lambda x: str(x)).to_dict(),
    }

    # quantities
    q = df[D.QTY_COLS]
    rep["quantities"] = {
        "negative_values": int((q < 0).sum().sum()),
        "missing_values": int(q.isna().sum().sum()),
        "non_integer_values": int((q.fillna(0) % 1 != 0).sum().sum()),
        "zero_ratio": {c: round(float((df[c] == 0).mean()), 4) for c in D.QTY_COLS},
        "target_describe": {k: round(float(v), 3) for k, v in df[D.TARGET].describe().items()},
    }
    rep["weather"] = {
        "temperature_range": [float(df.Temperature.min()), float(df.Temperature.max())],
        "humidity_range": [float(df.Humidity.min()), float(df.Humidity.max())],
        "humidity_gt_100_rows": int((df.Humidity > 100).sum()),
    }

    # daily snapshot & continuity
    snap = D.daily_snapshot(df)
    days = snap.groupby(D.KEY)["Date"].agg(["count", "min", "max"])
    span = (days["max"] - days["min"]).dt.days + 1
    rep["daily_snapshot"] = {
        "rows": int(len(snap)),
        "products_with_95_days": int((days["count"] == 95).sum()),
        "products_not_95_days": sorted(days.index[days["count"] != 95].tolist()),
        "days_per_product_distribution": days["count"].value_counts().sort_index()
        .rename(lambda x: str(x)).to_dict(),
        "products_with_gaps": int((span != days["count"]).sum()),
        "global_first_date": str(snap["Date"].min().date()),
        "global_last_date": str(snap["Date"].max().date()),
    }

    # 복제(재생) 스냅샷 감지
    tagged = D.tag_snapshot_provenance(snap)
    st = tagged["snapshot_status"].value_counts().to_dict()
    per_date = tagged.groupby("Date").apply(
        lambda g: pd.Series({"n": len(g), "original_ratio": (g.snapshot_status == "ORIGINAL").mean(),
                             "top_source": g.loc[g.snapshot_status == "REPEAT_OF_EARLIER", "source_date"]
                             .mode().astype(str).tolist()[:1]}), include_groups=False)
    genuine_dates = per_date.index[per_date["original_ratio"] >= 0.5]
    rep["replay_duplication"] = {
        "snapshot_status_counts": {k: int(v) for k, v in st.items()},
        "genuine_dates_count": int(len(genuine_dates)),
        "genuine_dates": [str(d.date()) for d in genuine_dates],
        "replayed_or_stale_dates": {str(d.date()): (r["top_source"][0][:10] if r["top_source"] else None)
                                    for d, r in per_date.iterrows() if r["original_ratio"] < 0.5},
        "finding": "2022-03-24~05-09는 2022-01-26~03-13의 일별 스냅샷을 57일 뒤로 재생한 복제본; "
                   "연휴 등 일부 날짜는 이전일 스냅샷의 복사본. 시간순 홀드아웃 시험구간이 학습구간 복제가 되므로 "
                   "원본(ORIGINAL) 스냅샷만으로 평가해야 함.",
    }
    orig = tagged[tagged.snapshot_status == "ORIGINAL"]
    rep["revision_check_planned_original_only"] = {
        f"T+{k}@D-{k} vs T@D": _revision_check(orig, D.planned_col, k) for k in range(1, 5)}

    # 시점 일관성: 과거 스냅샷의 T+k가 이후 T와 같은가? (사후 수정 여부)
    rep["revision_check_planned"] = {f"T+{k}@D-{k} vs T@D": _revision_check(snap, D.planned_col, k)
                                     for k in range(1, 5)}
    rep["revision_check_last_year"] = {f"T+{k}@D-{k} vs T@D": _revision_check(snap, D.last_year_col, k)
                                       for k in range(1, 5)}
    rep["revision_check_company_fc"] = {f"T+{k}@D-{k} vs T@D": _revision_check(snap, D.company_fc_col, k)
                                        for k in range(1, 5)}
    # 하루 내 여러 로그에서 타깃 값이 바뀌는가?
    multi = df.dropna(subset=["DateTime"]).groupby([D.KEY, "Date"])[D.TARGET].nunique()
    rep["intraday_target_changes"] = {
        "product_dates_with_multiple_logs": int((df.groupby([D.KEY, "Date"]).size() > 1).sum()),
        "product_dates_where_target_changed_intraday": int((multi > 1).sum()),
    }
    rep["feature_availability"] = FEATURE_AVAILABILITY
    rep["guidebook_issues"] = GUIDEBOOK_ISSUES
    return rep


FEATURE_AVAILABILITY = {
    "forecast_definition": "as-of일 A의 일 마감 스냅샷까지 보고, 목표일 D=A+h의 `T일 예정 수주량`(D일 마지막 로그)을 예측",
    "allowed_at_as_of_A_for_target_D": [
        "D-j일(j>=h) 스냅샷의 `T+j일 예정 수주량` (D에 대한 j일 전 예정값; j<=4)",
        "D-j일(j>=h) 스냅샷의 `작년 T+j일 예정 수주량`",
        "D-j일(j>=h) 스냅샷의 `T+j일 예상 수주량` (기업 예상값, 해당 스냅샷 기록 시점에 존재했다고 가정)",
        "A일 이전(포함) 스냅샷의 `T일 예정 수주량` 시차값(lag)과 그 통계",
        "D의 요일(달력 정보)",
    ],
    "prohibited": [
        "D일 스냅샷의 어떤 값(h>=1인 경우): `T일 예상 수주량`@D 포함",
        "A 이후 스냅샷의 모든 값, 전체 기간 통계(평균/스케일러를 전체 데이터로 적합)",
        "Temperature/Humidity의 D일 값(D일 관측 기상은 A 시점에 미확보; 가이드북도 제외)",
        "다른 제품 경계를 넘는 shift",
    ],
    "unverifiable_assumptions": [
        "`T+k일 예상 수주량`이 해당 스냅샷 기록 시각에 이미 생성되어 있었는지는 원본/가이드로 검증 불가 → 기록시점 가용 가정으로 사용하고 UI에 표기",
        "h=0(당일 nowcast)는 같은 날 스냅샷의 `T일 예상 수주량`이 타깃과 동시 생성될 수 있어 운영 지원 대상에서 제외(연구용 참고만)",
    ],
}

GUIDEBOOK_ISSUES = [
    "가이드북 [코드 23] train_test_split(random_state=42)은 shuffle=True 기본값 → 시간순 분할이 아님(시간 누출). 본 프로젝트는 날짜 기준 분할 사용.",
    "가이드북 [코드 16] shift는 전체 프레임에 적용 후 제품별 앞 4행 삭제 → 95일 연속 제품에만 안전. 본 프로젝트는 제품별 groupby + 날짜 병합으로 재구성.",
    "가이드북은 D일 `T일 예상 수주량`을 피처로 사용 → 생성 시점 미확인으로 h>=1 운영 모델에서 제외.",
]


def main() -> dict:
    rep = audit_supply_dataset()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return rep


if __name__ == "__main__":
    main()
    print("written", OUT)
