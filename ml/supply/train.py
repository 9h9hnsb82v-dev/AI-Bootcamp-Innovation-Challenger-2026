"""train_supply_forecast(): KAMP ID 40 `T일 예정 수주량` horizon별 회귀 학습·시간순 백테스트.

- 데이터: ORIGINAL 스냅샷만(복제/정체 제외), 목표일 2022-01-26~03-14
  (05-10은 47일 재생 구간 뒤 고립된 원본이라 lag 피처가 모두 결측 → 제외, 보고서에 명시)
- 목표일: 원본 비율 50% 이상인 날짜(genuine date)만 사용
- 분할: 목표일 기준 시간순 train / val / test(≈70/15/15). horizon h 모델은 평가구간 시작일 S에 대해
  학습 목표일 D <= S - h 인 행만 사용(purge) → 평가 as-of 시점 이후의 정답이 학습에 들어가지 않음
- 모델: 누출 없는 기준선 4종 vs HistGradientBoosting vs XGBoost (val WAPE로 선택)
- 최종 모델은 train+val로 재적합 후 test에서 1회 평가. 예측값은 0 이상으로 클리핑한 뒤 평가.

실행: .venv/Scripts/python -m ml.supply.train   (로컬 CPU, 유료 리소스 미사용)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from ml.supply import data as D
from ml.supply.features import MAX_H, baseline_predictions, build_training_table, feature_names
from ml.tracking import log_run

MODEL_DIR = D.ROOT / "models/supply"
REPORT_DIR = D.ROOT / "reports/supply"
SEED = 42
TARGET_START, TARGET_END = pd.Timestamp("2022-01-26"), pd.Timestamp("2022-03-14")
SPLIT_FRACTIONS = (8 / 11, 2 / 11, 1 / 11)  # 8:2:1 (사용자 결정 2026-11-11, 이전 70/15/15)
MODEL_VERSION = "kamp40_planned_order_v1"


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, None)
    err = p - y
    denom = np.abs(y) + np.abs(p)
    smape = np.where(denom == 0, 0.0, 2 * np.abs(err) / np.where(denom == 0, 1, denom))
    return {"n": int(len(y)), "MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "WAPE": float(np.sum(np.abs(err)) / max(np.sum(np.abs(y)), 1e-9)), "sMAPE": float(np.mean(smape)),
            "bias": float(np.mean(err))}


def make_split(dates: list[pd.Timestamp]) -> dict[str, list[pd.Timestamp]]:
    """목표일 리스트를 시간순 train/val/test로 분할한다(purge는 horizon별로 별도 적용)."""
    dates = sorted(dates)
    n = len(dates)
    # 최대 잔여 배분(largest remainder): 반올림으로 시험 기간이 과소해지지 않게 정수 일수를 배분
    raw = [n * f for f in SPLIT_FRACTIONS]
    cnt = [int(x) for x in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - cnt[i], reverse=True)[: n - sum(cnt)]:
        cnt[i] += 1
    n_tr, n_va = cnt[0], cnt[1]
    return {"train": dates[:n_tr], "val": dates[n_tr:n_tr + n_va], "test": dates[n_tr + n_va:]}


def genuine_dates(snap_tagged: pd.DataFrame) -> list[pd.Timestamp]:
    r = snap_tagged.groupby("Date")["snapshot_status"].apply(lambda s: (s == "ORIGINAL").mean())
    return sorted(r.index[r >= 0.5])


def _models():
    return {
        "HGB": lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05,
                                                     max_leaf_nodes=15, min_samples_leaf=20, random_state=SEED),
        "XGB": lambda: XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9,
                                    colsample_bytree=0.8, objective="reg:absoluteerror", random_state=SEED,
                                    n_jobs=4),
    }


def train_supply_forecast() -> dict:
    tagged = D.tag_snapshot_provenance(D.daily_snapshot(D.load_raw()))
    snap = tagged[tagged.snapshot_status == "ORIGINAL"].reset_index(drop=True)
    snap = snap[snap.Date <= TARGET_END]
    target_dates = [d for d in genuine_dates(tagged) if TARGET_START <= d <= TARGET_END]
    split = make_split(target_dates)
    which = {d: k for k, v in split.items() for d in v}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {"model_version": MODEL_VERSION, "target": D.TARGET, "target_semantics": D.TARGET_SEMANTICS,
               "provenance": "KAMP_ID40_BACKTEST", "unit": "units (개)", "seed": SEED,
               "source_sha256": D.raw_sha256(), "data_policy": "ORIGINAL snapshots only (replay/stale excluded)",
               "split": {k: [str(d.date()) for d in v] for k, v in split.items()},
               "purge_rule": "horizon h: training target dates <= eval_start - h", "prediction_postprocess": "clip(pred, 0, None) before evaluation",
               "excluded_target_dates": ["2022-05-10 (isolated original after replay segment; no lag features)"],
               "horizons": {}, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    all_preds = []
    for h in range(1, MAX_H + 1):
        t = build_training_table(snap, h)
        t = t[t.Date.between(TARGET_START, TARGET_END)].copy()
        t["split"] = t.Date.map(which)
        t = t[t.split.isin(["train", "val", "test"])]
        feats = feature_names(h)
        val_start, test_start = split["val"][0], split["test"][0]
        tr = t[(t.split == "train") & (t.Date <= val_start - pd.Timedelta(days=h))]
        va, te = t[t.split == "val"], t[t.split == "test"]
        trva = t[t.split.isin(["train", "val"]) & (t.Date <= test_start - pd.Timedelta(days=h))]

        res = {"rows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                        "train_plus_val_refit": int(len(trva))}, "val": {}, "test": {}}
        val_preds = baseline_predictions(va, h)
        for name, mk in _models().items():
            val_preds[name] = np.clip(mk().fit(tr[feats], tr.y).predict(va[feats]), 0, None)
        for name, p in val_preds.items():
            res["val"][name] = metrics(va.y, p)
        best = min(res["val"], key=lambda k: res["val"][k]["WAPE"])
        res["selected_by_val_WAPE"] = best
        final = {}
        for name, mk in _models().items():
            final[name] = mk().fit(trva[feats], trva.y)
        test_preds = baseline_predictions(te, h)
        for name, m in final.items():
            test_preds[name] = np.clip(m.predict(te[feats]), 0, None)
        for name, p in test_preds.items():
            res["test"][name] = metrics(te.y, p)

        # 서비스용 모델: val에서 선택된 ML 모델(기준선이 선택되면 기준선 규칙을 그대로 사용)
        serve = best if best in final else None
        vp = val_preds[best]
        resid = vp - va.y.to_numpy(float)
        edges = list(np.quantile(vp, [1 / 3, 2 / 3]))
        bins = []
        for lo_e, hi_e in zip([-np.inf] + edges, edges + [np.inf]):
            msk = (vp > lo_e) & (vp <= hi_e)
            r = resid[msk] if msk.sum() >= 20 else resid
            bins.append({"pred_upper": None if np.isinf(hi_e) else float(hi_e),
                         "q10": float(np.quantile(r, 0.1)), "q90": float(np.quantile(r, 0.9)), "n": int(msk.sum())})
        res["serving"] = {"kind": "ML" if serve else "BASELINE", "name": best,
                          "residual = pred - actual (val)": True, "residual_bins": bins}
        if serve:
            joblib.dump({"model": final[serve], "features": feats, "horizon": h, "name": serve},
                        MODEL_DIR / f"model_h{h}.joblib")
        pr = te[[D.KEY, "Date", "as_of", "horizon", "y"]].copy()
        for name, p in test_preds.items():
            pr[name] = p
        pr["split"] = "test"
        all_preds.append(pr)
        # 제품별 test 지표(선택 모델)
        pr["_sel"] = test_preds[best]
        per_prod = pr.groupby(D.KEY).apply(lambda g: pd.Series(metrics(g.y, g._sel)), include_groups=False)
        per_prod.to_csv(REPORT_DIR / f"per_product_test_h{h}.csv", encoding="utf-8-sig")
        res["feature_importance_note"] = "HGB/XGB tree models; see joblib for details"
        summary["horizons"][str(h)] = res
        summary["horizons"][str(h)]["mlflow_run_id"] = log_run(
            "kamp_id40_supply", f"{MODEL_VERSION}_h{h}", {"horizon": h, "features": feats, "seed": SEED,
                                                          "selected": best},
            {f"test_{k}_{m}": v[m] for k, v in res["test"].items() for m in ("MAE", "WAPE")})

    preds = pd.concat(all_preds).drop(columns=["_sel"], errors="ignore")
    preds.to_csv(MODEL_DIR / "backtest_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([(d, k) for k, v in split.items() for d in v], columns=["target_date", "split"]) \
        .to_csv(MODEL_DIR / "split_index.csv", index=False)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = train_supply_forecast()
    print("split:", {k: len(v) for k, v in s["split"].items()}, s["split"])
    for h, r in s["horizons"].items():
        print(f"h={h} rows={r['rows']} selected={r['selected_by_val_WAPE']}")
        for name, m in r["test"].items():
            print(f"   test {name:24s} MAE={m['MAE']:7.2f} RMSE={m['RMSE']:7.2f} WAPE={m['WAPE']:.3f} sMAPE={m['sMAPE']:.3f}")
