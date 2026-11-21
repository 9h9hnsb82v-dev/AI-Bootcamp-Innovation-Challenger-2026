"""KAMP ID 4 공정 센서 기반 불량확률 분류기 (`XGBOOST_SENSOR`).

생산 중/후 센서값이 확보된 경우에만 쓰는 보조 모델이다. 출력은 샷 단위 불량 *확률*이며
폐기물 kg가 아니다. 분할은 ml.kamp.baseline 과 동일한 제품별 층화 무작위 8:2:1. 지표는 val·test 각각 보고.
인접 샷이 구간에 섞이므로 시간순 분할보다 낙관적인 지표일 수 있다.

실행: .venv/Scripts/python -m ml.kamp.train_xgb
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score)
from xgboost import XGBClassifier

from ml.kamp.audit import ROOT, SENSOR_EXCLUDE
from ml.kamp.baseline import assign_split, prepared
from ml.tracking import log_run

MODEL_DIR = ROOT / "models/kamp"
SEED = 42


def feature_columns(d) -> list[str]:
    extra = {"product_id", "is_defect", "date", "split", "_sensor_sig"}
    return [c for c in d.columns if c not in SENSOR_EXCLUDE | extra and d[c].dtype.kind in "fi"]


def main() -> dict:
    d = assign_split(prepared())
    feats = feature_columns(d)
    for pid in sorted(d.product_id.unique()):
        d[f"is_{pid}"] = (d.product_id == pid).astype(int)
    feats += [f"is_{p}" for p in sorted(d.product_id.unique())]
    tr, va, te = d[d.split == "train"], d[d.split == "val"], d[d.split == "test"]
    pos = max(int(tr.is_defect.sum()), 1)
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                          colsample_bytree=0.8, scale_pos_weight=(len(tr) - pos) / pos,
                          eval_metric="aucpr", random_state=SEED)
    model.fit(tr[feats], tr.is_defect)
    def scores(df, prefix):
        p = model.predict_proba(df[feats])[:, 1]
        yhat = (p >= 0.5).astype(int)
        y = df.is_defect.to_numpy()
        return {f"{prefix}_n": int(len(df)), f"{prefix}_defects": int(y.sum()),
                f"{prefix}_pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
                f"{prefix}_precision@0.5": float(precision_score(y, yhat, zero_division=0)),
                f"{prefix}_recall@0.5": float(recall_score(y, yhat, zero_division=0)),
                f"{prefix}_f1@0.5": float(f1_score(y, yhat, zero_division=0)),
                f"{prefix}_brier": float(brier_score_loss(y, p)), f"{prefix}_defect_prevalence": float(y.mean())}
    metrics = {**scores(va, "val"), **scores(te, "test")}
    meta = {"model_version": "kamp_xgb_sensor_v1", "provenance": "XGBOOST_SENSOR",
            "output": "per-shot defect probability (NOT kg)", "seed": SEED, "features": feats,
            "train_n": int(len(tr)), "train_defects": int(tr.is_defect.sum()), "metrics": metrics,
            "split": "8:2:1 per-product stratified random (seed 42)", "note": "검증·시험 불량 표본이 매우 적어 지표 신뢰도가 낮음(참고용).",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / "xgb_sensor.json")
    (MODEL_DIR / "xgb_sensor_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["usable_for_kg"] = False
    # 신뢰 조건: 검증·시험 불량 표본이 각각 30건 이상이고 검증 PR-AUC >= 0.3 (표본이 적으면 재현율 0/1이 우연에 좌우됨)
    reliable = (metrics["val_defects"] >= 30 and metrics["test_defects"] >= 30
                and (metrics["val_pr_auc"] or 0) >= 0.3 and metrics["test_recall@0.5"] > 0)
    meta["status"] = "TRAINED" if reliable else "TRAINED_NOT_RELIABLE"
    meta["mlflow_run_id"] = log_run("kamp_id4_sensor", meta["model_version"],
                                    {"seed": SEED, "n_features": len(feats), "max_depth": 3},
                                    {k: v for k, v in metrics.items() if v is not None})
    (MODEL_DIR / "xgb_sensor_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    m = main()
    print(json.dumps(m["metrics"], indent=1), m.get("mlflow_run_id"), m.get("mlflow_error"))
