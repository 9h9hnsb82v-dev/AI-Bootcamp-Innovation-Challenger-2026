"""누출 없는 horizon별 학습 테이블 구성 (KAMP ID 40).

정의
- 관측 단위: (Product_Number, 목표일 D, horizon h), as-of일 A = D - h (달력일).
- 타깃 y = D일 ORIGINAL 스냅샷의 `T일 예정 수주량` (예정 수주량; 확정 주문 아님).
- 피처는 날짜 <= A 인 ORIGINAL 스냅샷에서만 만든다(복제/정체 스냅샷 제외 — 사용자 결정 2026-11-10).
  * plan_j / ly_j / cfc_j (j = h..4): D-j일 스냅샷의 T+j 예정/작년/기업예상 값 (D에 대한 값)
  * last_T, last_T_age: A 이전 가장 최근 ORIGINAL 스냅샷의 `T일 예정 수주량`과 경과일
  * mean3_T: A 이전 최근 3개 ORIGINAL 스냅샷 `T일 예정 수주량` 평균
  * prod_mean_T: A 이전 해당 제품의 `T일 예정 수주량` 누적 평균
  * dow_D: 목표일 요일(달력 정보)
- D일 스냅샷 값, A 이후 값, 기상값은 사용하지 않는다.
- 모든 shift/병합은 Product_Number 그룹 내에서만 수행한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.supply import data as D

MAX_H = 4


def original_snapshots(df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    df_raw = D.load_raw() if df_raw is None else df_raw
    tagged = D.tag_snapshot_provenance(D.daily_snapshot(df_raw))
    return tagged[tagged.snapshot_status == "ORIGINAL"].reset_index(drop=True)


def feature_names(h: int) -> list[str]:
    cols = []
    for j in range(h, MAX_H + 1):
        cols += [f"plan_{j}", f"ly_{j}", f"cfc_{j}"]
    return cols + ["last_T", "last_T_age", "mean3_T", "prod_mean_T", "dow_D"]


def _lookup(snap: pd.DataFrame, keys: pd.DataFrame, col: str, offset_days: int) -> np.ndarray:
    """keys(Product_Number, Date=D)에 대해 D-offset 스냅샷의 col 값을 가져온다."""
    q = keys[[D.KEY]].copy()
    q["Date"] = keys["Date"] - pd.Timedelta(days=offset_days)
    m = q.merge(snap[[D.KEY, "Date", col]], on=[D.KEY, "Date"], how="left")
    return m[col].to_numpy(float)


def _history_feats(snap: pd.DataFrame, keys: pd.DataFrame, as_of: pd.Series) -> pd.DataFrame:
    """as-of(포함) 이전 ORIGINAL 스냅샷의 T일 예정 수주량 통계 (제품 그룹 내 asof 병합)."""
    hist = snap[[D.KEY, "Date", D.TARGET]].sort_values("Date")
    g = hist.groupby(D.KEY, group_keys=False)
    hist = hist.assign(
        mean3_T=g[D.TARGET].transform(lambda s: s.rolling(3, min_periods=1).mean()),
        prod_mean_T=g[D.TARGET].transform(lambda s: s.expanding().mean()),
        hist_date=hist["Date"])
    q = keys[[D.KEY]].assign(as_of=as_of.values, _i=np.arange(len(keys))).sort_values("as_of")
    m = pd.merge_asof(q, hist.rename(columns={"Date": "as_of"}), on="as_of", by=D.KEY,
                      direction="backward", allow_exact_matches=True).sort_values("_i")
    return pd.DataFrame({
        "last_T": m[D.TARGET].to_numpy(float),
        "last_T_age": (m["as_of"] - m["hist_date"]).dt.days.to_numpy(float),
        "mean3_T": m["mean3_T"].to_numpy(float),
        "prod_mean_T": m["prod_mean_T"].to_numpy(float),
    })


def build_features(snap: pd.DataFrame, keys: pd.DataFrame, h: int) -> pd.DataFrame:
    """keys: columns [Product_Number, Date(=목표일 D)] → horizon h 피처 테이블."""
    keys = keys.reset_index(drop=True)
    out = keys[[D.KEY, "Date"]].copy()
    out["horizon"] = h
    out["as_of"] = out["Date"] - pd.Timedelta(days=h)
    for j in range(h, MAX_H + 1):
        out[f"plan_{j}"] = _lookup(snap, keys, D.planned_col(j), j)
        out[f"ly_{j}"] = _lookup(snap, keys, D.last_year_col(j), j)
        out[f"cfc_{j}"] = _lookup(snap, keys, D.company_fc_col(j), j)
    out = pd.concat([out, _history_feats(snap, keys, out["as_of"])], axis=1)
    out["dow_D"] = out["Date"].dt.day_of_week.astype(float)
    return out


def build_training_table(snap: pd.DataFrame, h: int) -> pd.DataFrame:
    keys = snap[[D.KEY, "Date", D.TARGET]].rename(columns={D.TARGET: "y"})
    feats = build_features(snap, keys, h)
    feats["y"] = keys["y"].to_numpy(float)
    return feats


def baseline_predictions(f: pd.DataFrame, h: int) -> dict[str, np.ndarray]:
    """누출 없는 단순 기준선."""
    latest_plan = f[f"plan_{h}"].copy()
    for j in range(h + 1, MAX_H + 1):          # 가장 최근 예정값이 없으면 더 오래된 예정값
        latest_plan = latest_plan.fillna(f[f"plan_{j}"])
    latest_plan = latest_plan.fillna(f["last_T"]).fillna(0.0)
    return {
        "BASELINE_LATEST_PLAN": latest_plan.to_numpy(float),
        "BASELINE_LAST_KNOWN_T": f["last_T"].fillna(0.0).to_numpy(float),
        "BASELINE_MEAN3_T": f["mean3_T"].fillna(0.0).to_numpy(float),
        "BASELINE_COMPANY_FC": f[f"cfc_{h}"].fillna(f["last_T"]).fillna(0.0).to_numpy(float),
    }
