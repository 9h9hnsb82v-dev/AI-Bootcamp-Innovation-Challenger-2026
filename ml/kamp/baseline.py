"""KAMP ID 4 학습구간 제품별 불량률 기준선 (`KAMP_TRAIN_BASELINE`).

확정 규칙(사용자 승인 2026-11-10)
- 단위: `_id` 중복 제거 후 레코드 1건 = 사출 1샷 기록. `ERR_FACT_QTY` 미사용.
- 라벨: 원본 `PassOrFail` Y=양품, N=불량 (파생본 0/1과 `_id` 대조로 N→1 확인).
- 분할: 제품별 샷 단위 층화 무작위 8:2:1, seed 42 (사용자 결정 2026-11-11, 가이드북 「사출성형기 AI 데이터셋」 72쪽의
  층화 추출 방식). 양품·불량 비율이 학습/검증/시험에서 같게 유지된다.
  주의: 무작위 분할이라 학습·시험 불량률이 설계상 비슷해지고 인접 샷이 구간에 섞인다(미래 불량률 예측 검증력은 약함).
  (이전: 제품별 날짜 단위 시간순 70% 학습 / 30% 홀드아웃 — 생산일이 5~7일뿐이라 날짜 단위 8:2:1 불가)
- 추정: Jeffreys 사후평균 (k+0.5)/(n+1), 95% Jeffreys 구간, 원시 비율 병기.

실행: .venv/Scripts/python -m ml.kamp.baseline
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import beta

from ml.kamp.audit import RAW_ZIP, ROOT, load_labeled

OUT = ROOT / "models/kamp/baseline_rates.json"
SPLIT_OUT = ROOT / "data/processed/kamp/split_index.csv"
PART_NAME_TO_PRODUCT = {
    "CN7 W/S SIDE MLD'G LH": "CN7_LH",
    "CN7 W/S SIDE MLD'G RH": "CN7_RH",
    "RG3 MOLD'G W/SHLD, LH": "RG3_LH",
    "RG3 MOLD'G W/SHLD, RH": "RG3_RH",
}


def prepared() -> pd.DataFrame:
    d = load_labeled().drop_duplicates("_id")
    d = d[d["PART_NAME"].isin(PART_NAME_TO_PRODUCT)].copy()
    d["product_id"] = d["PART_NAME"].map(PART_NAME_TO_PRODUCT)
    d["is_defect"] = (d["PassOrFail"] == "N").astype(int)
    d["date"] = d["TimeStamp"].dt.normalize()
    return d.sort_values("TimeStamp").reset_index(drop=True)


SEED = 42


def assign_split(d: pd.DataFrame) -> pd.DataFrame:
    """제품별·불량여부별 층화 무작위 8:2:1 (시험 1/11 → 나머지에서 검증 2/10)."""
    from sklearn.model_selection import train_test_split
    d = d.copy()
    d["split"] = "train"
    for pid, g in d.groupby("product_id"):
        rest, te = train_test_split(g.index, test_size=1 / 11, stratify=g.is_defect, random_state=SEED)
        tr, va = train_test_split(rest, test_size=2 / 10, stratify=g.loc[rest, "is_defect"], random_state=SEED)
        d.loc[va, "split"] = "val"
        d.loc[te, "split"] = "test"
    return d


def jeffreys(k: int, n: int) -> dict:
    return {
        "estimated_defect_rate": (k + 0.5) / (n + 1),
        "raw_rate": k / n if n else None,
        "ci95_low": float(beta.ppf(0.025, k + 0.5, n - k + 0.5)),
        "ci95_high": float(beta.ppf(0.975, k + 0.5, n - k + 0.5)),
    }


def main() -> dict:
    d = assign_split(prepared())
    res = {"provenance": "KAMP_TRAIN_BASELINE", "method": "JEFFREYS_POSTERIOR_MEAN",
           "unit_of_count": "deduplicated _id record (one shot log)",
           "label_mapping": {"Y": "PASS", "N": "DEFECT"},
           "split_rule": "per-product shot-level stratified (by defect label) random 8:2:1, seed 42 (KAMP guidebook p.72 style)",
           "source_zip": str(RAW_ZIP.relative_to(ROOT)),
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "products": {}}
    for pid, g in d.groupby("product_id"):
        tr, va, ho = g[g.split == "train"], g[g.split == "val"], g[g.split == "test"]
        k, n = int(tr.is_defect.sum()), int(len(tr))
        r = jeffreys(k, n)
        r.update({
            "train_n": n, "train_defects": k,
            "train_dates": [str(x.date()) for x in sorted(tr.date.unique())],
            "val_raw_rate": float(va.is_defect.mean()) if len(va) else None,
            "val_n": int(len(va)), "val_defects": int(va.is_defect.sum()),
            "val_dates": [str(x.date()) for x in sorted(va.date.unique())],
            "test_n": int(len(ho)), "test_defects": int(ho.is_defect.sum()),
            "test_raw_rate": float(ho.is_defect.mean()) if len(ho) else None,
            "test_dates": [str(x.date()) for x in sorted(ho.date.unique())],
            "warnings": (["ZERO_DEFECT_IN_TRAIN"] if k == 0 else []) + (["SMALL_SAMPLE"] if n < 1000 else []),
        })
        res["products"][pid] = r
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    d[["_id", "product_id", "TimeStamp", "split", "is_defect"]].to_csv(SPLIT_OUT, index=False)
    return res


if __name__ == "__main__":
    out = main()
    for pid, r in out["products"].items():
        print(pid, f"rate={r['estimated_defect_rate']:.5f} raw={r['raw_rate']:.5f} k/n={r['train_defects']}/{r['train_n']}"
              f" CI=[{r['ci95_low']:.5f},{r['ci95_high']:.5f}] val={r['val_defects']}/{r['val_n']} test={r['test_defects']}/{r['test_n']} {r['warnings']}")
