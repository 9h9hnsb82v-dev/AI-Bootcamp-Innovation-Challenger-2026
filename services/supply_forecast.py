"""트랙 S — KAMP ID 40 미래 수주량(예정 수주량) 예측 서비스.

- 결과 단위는 '개(units)'이며 폐기물 kg가 아니다. 7종 제조품목/kg ledger와 연결하지 않는다.
- 지원 as-of: 학습 데이터 기간(2022-01-26~2022-03-14)의 ORIGINAL 스냅샷 기반. 그 밖의 시점(예: 2026년)은
  입력 피처가 없으므로 수치를 만들지 않고 UNSUPPORTED_AS_OF 오류를 낸다.
- available_inputs를 사용자가 직접 넣으면 계산은 하되 UNVERIFIED_FUTURE_INPUT 태그를 붙인다.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from services import db
from services.config import ROOT, ServiceError

MODEL_DIR = ROOT / "models/supply"
TARGET_DESC = "T일 예정 수주량 (예정/계획 수주량 — 확정 주문 실적 아님)"


@lru_cache(maxsize=1)
def _metrics() -> dict:
    p = MODEL_DIR / "metrics.json"
    if not p.exists():
        raise ServiceError("NOT_TRAINED", "수주량 모델 미학습: `python -m ml.supply.train` 실행 필요")
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _snap() -> pd.DataFrame:
    from ml.supply.features import original_snapshots
    try:
        return original_snapshots()
    except FileNotFoundError as e:
        raise ServiceError("DATA_NOT_AVAILABLE", str(e)) from e


@lru_cache(maxsize=4)
def _model(h: int):
    import joblib
    p = MODEL_DIR / f"model_h{h}.joblib"
    return joblib.load(p) if p.exists() else None


def model_status() -> dict:
    try:
        m = _metrics()
    except ServiceError as e:
        return {"status": e.code, "message": e.message}
    return {"status": "TRAINED", "model_version": m["model_version"], "split": m["split"],
            "serving": {h: r["serving"] for h, r in m["horizons"].items()}}


def list_supply_products() -> list[dict]:
    s = _snap()
    g = s.groupby("Product_Number")["Date"].agg(["count", "min", "max"]).reset_index()
    return [{"product_number": r.Product_Number, "original_snapshot_days": int(r["count"]),
             "first_date": str(r["min"].date()), "last_date": str(r["max"].date())} for _, r in g.iterrows()]


def product_history(product_number: str) -> pd.DataFrame:
    from ml.supply import data as D
    s = _snap()
    h = s[s.Product_Number == product_number]
    if h.empty:
        raise ServiceError("UNKNOWN_SUPPLY_PRODUCT", f"{product_number} 없음")
    return h[["Date", D.TARGET] + [D.planned_col(k) for k in range(1, 5)]].sort_values("Date")


def _split_of(target: pd.Timestamp) -> str:
    sp = _metrics()["split"]
    for k, v in sp.items():
        if str(target.date()) in v:
            return k
    return "not_in_split"


def forecast_order_quantity(product_number: str, forecast_as_of: str, horizon_days: int,
                            available_inputs: dict | None = None) -> dict:
    from ml.supply.features import MAX_H, baseline_predictions, build_features, feature_names
    m = _metrics()
    if horizon_days not in range(1, MAX_H + 1):
        raise ServiceError("UNSUPPORTED_HORIZON", f"horizon_days는 1~{MAX_H}만 지원(원본 T+1~T+4 구조)")
    as_of = pd.Timestamp(date.fromisoformat(str(forecast_as_of)))
    target = as_of + pd.Timedelta(days=horizon_days)
    snap = _snap()
    if product_number not in set(snap.Product_Number):
        raise ServiceError("UNKNOWN_SUPPLY_PRODUCT", f"{product_number}는 KAMP ID40 제품번호가 아님")
    lo = pd.Timestamp(m["split"]["train"][0])
    hi = pd.Timestamp(m["split"]["test"][-1])
    provenance = ["KAMP_ID40_PLANNED_ORDER_FORECAST"]
    warnings = []
    feats = feature_names(horizon_days)
    if available_inputs is not None:
        row = pd.DataFrame([{f: available_inputs.get(f, np.nan) for f in feats}])
        provenance.append("UNVERIFIED_FUTURE_INPUT")
        warnings.append("사용자 입력 피처 사용 — 데이터셋으로 검증되지 않은 입력")
        actual = None
    else:
        if not (lo <= as_of <= hi - pd.Timedelta(days=horizon_days)):
            raise ServiceError("UNSUPPORTED_AS_OF",
                               f"forecast_as_of={as_of.date()}는 지원 범위({lo.date()}~{(hi - pd.Timedelta(days=horizon_days)).date()}, "
                               "2022 데이터 백테스트) 밖입니다. 해당 시점의 입력 피처가 없어 수치를 생성하지 않습니다.")
        keys = pd.DataFrame({"Product_Number": [product_number], "Date": [target]})
        row = build_features(snap, keys, horizon_days)
        hit = snap[(snap.Product_Number == product_number) & (snap.Date == target)]
        actual = float(hit["T일 예정 수주량"].iloc[0]) if len(hit) else None
        if row[feats[:3]].isna().all(axis=1).iloc[0] and np.isnan(row["last_T"].iloc[0]):
            raise ServiceError("DATA_NOT_AVAILABLE", "as-of 시점 이전 ORIGINAL 스냅샷이 없어 예측 불가")
        provenance.append("KAMP_ID40_BACKTEST")
    serving = m["horizons"][str(horizon_days)]["serving"]
    mdl = _model(horizon_days) if serving["kind"] == "ML" else None
    if mdl is not None:
        pred = float(np.clip(mdl["model"].predict(row[feats])[0], 0, None))
    else:
        rule = serving["name"] if serving["kind"] == "BASELINE" else "BASELINE_LATEST_PLAN"
        pred = float(baseline_predictions(row, horizon_days)[rule][0])
    split = _split_of(target) if available_inputs is None else "user_input"
    if split in ("train", "val"):
        warnings.append(f"목표일이 {split} 구간 — 표본 내(in-sample) 결과로 일반화 성능 아님")
    b = next(x for x in serving["residual_bins"] if x["pred_upper"] is None or pred <= x["pred_upper"])
    lo_i, hi_i = max(pred - b["q90"], 0.0), max(pred - b["q10"], 0.0)
    rec = {"forecast_id": db.new_id("SF"), "product_number": product_number,
           "forecast_as_of": str(as_of.date()), "target_date": str(target.date()), "horizon": horizon_days,
           "predicted_planned_order_units": pred, "model_version": m["model_version"],
           "data_split": split, "provenance": ",".join(provenance), "created_at": db.now()}
    with db.connect() as con:
        con.execute("INSERT INTO supply_forecast VALUES (:forecast_id,:product_number,:forecast_as_of,:target_date,"
                    ":horizon,:predicted_planned_order_units,:model_version,:data_split,:provenance,:created_at)", rec)
    test_metric = m["horizons"][str(horizon_days)]["test"][serving["name"]]
    return {**rec, "provenance": provenance, "unit": "units (개) — 폐기물 kg 아님", "target": TARGET_DESC,
            "model_used": serving["name"], "interval_80_from_val_residuals": [lo_i, hi_i],
            "actual_planned_order_units": actual, "test_metrics_for_this_horizon": test_metric,
            "warnings": warnings + ["KAMP ID40 비식별 제품 — 7종 제조품목과 실제 매핑 없음"]}


def get_supply_backtest(product_number: str | None = None, date_range: tuple[str, str] | None = None,
                        horizon: int | None = None) -> dict:
    m = _metrics()
    p = MODEL_DIR / "backtest_predictions.csv"
    df = pd.read_csv(p, parse_dates=["Date", "as_of"], encoding="utf-8-sig")
    if product_number:
        df = df[df.Product_Number == product_number]
    if horizon:
        df = df[df.horizon == horizon]
    if date_range:
        df = df[df.Date.between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))]
    sel = {int(h): r["serving"]["name"] for h, r in m["horizons"].items()}
    df = df.assign(selected_model=df.horizon.map(sel))
    df["predicted"] = [r[s] for s, (_, r) in zip(df.selected_model, df.iterrows())]
    return {"rows": df.rename(columns={"y": "actual_planned_order_units"}), "split": m["split"],
            "metrics": {h: r["test"] for h, r in m["horizons"].items()},
            "selected": sel, "provenance": ["KAMP_ID40_BACKTEST"], "target": TARGET_DESC}
