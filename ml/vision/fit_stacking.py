"""2차 판정 모델(stacking) 학습: 부품별 로지스틱 회귀 [logit(EfficientNet 점수), logit(YOLO11m 점수)] → 최종 불량 확률.

- 학습: 검증셋(val 9,241장)만 사용. 임계값은 검증셋 5겹 교차예측의 Youden J. 시험셋·holdout 미사용.
- 서비스 규칙(사용자 결정 2026-11-13): 두 모델 판정 일치 + 2차 판정 모델도 같은 판정 + 2차 확률이 임계값 ±UNCERTAIN_MARGIN 밖
  → 자동 판정. 하나라도 아니면 AI_DISAGREE(작업자 확인).
- 결과를 models/vision/ensemble_full/meta.json 의 "stacking" 에 기록하고 평가셋(test+holdout) 성능을 함께 남긴다.
실행: .venv/Scripts/python -m ml.vision.fit_stacking
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ml.vision.compare_yolo_effnet import youden
from ml.vision.ensemble_check import load, logit
from ml.vision.prepare_subset import ROOT

EFF = ROOT / "models/vision/effnet_full_e20/named-outputs/results/effnet_scores.csv"
YOLO = ROOT / "models/vision/yolo11m_cls_full_e30/named-outputs/results/yolo11m-cls_scores.csv"
META = ROOT / "models/vision/ensemble_full/meta.json"
UNCERTAIN_MARGIN = 0.10  # ±0.20부터는 범퍼(임계값 0.81)에서 자동 판정이 무너짐(검증셋 확인)
PARTS = ("DOOR", "BUMPER", "FENDER")


def main() -> dict:
    A, B = load(str(EFF)), load(str(YOLO))
    df = A.merge(B[["subset", "stem", "score"]], on=["subset", "stem"], suffixes=("_a", "_b"), validate="1:1")
    df = df.rename(columns={"score_a": "a", "score_b": "b"})
    df["y"] = (df.quality == "DEFECT").astype(int)
    meta = json.loads(META.read_text(encoding="utf-8"))
    th_member = {"effnet": json.loads((META.parent / "effnet/meta.json").read_text(encoding="utf-8"))["thresholds"],
                 "yolo": json.loads((META.parent / "yolo11m_cls/meta.json").read_text(encoding="utf-8"))["thresholds"]}
    stack, ev = {}, {}
    for part in PARTS:
        g = df[df.part == part]
        v, t = g[g.subset == "val"], g[g.subset.isin(["test", "holdout"])]
        Xv = np.c_[logit(v.a), logit(v.b)]
        lr = LogisticRegression(C=1.0, max_iter=1000)
        oof = cross_val_predict(lr, Xv, v.y, cv=5, method="predict_proba")[:, 1]
        th = round(youden(oof, v.y.to_numpy()), 2)
        lr.fit(Xv, v.y)
        stack[part] = {"coef": [float(c) for c in lr.coef_[0]], "intercept": float(lr.intercept_[0]), "threshold": th}
        ps = lr.predict_proba(np.c_[logit(t.a), logit(t.b)])[:, 1]
        da, db = t.a.to_numpy() >= th_member["effnet"][part], t.b.to_numpy() >= th_member["yolo"][part]
        auto = (da == db) & ((ps >= th) == da) & (np.abs(ps - th) >= UNCERTAIN_MARGIN)
        y = t.y.to_numpy().astype(bool)
        ev[part] = {"n": int(len(t)), "auto_rate": float(auto.mean()), "auto_accuracy": float((da[auto] == y[auto]).mean()),
                    "stack_accuracy_all": float(((ps >= th) == y).mean())}
    n = sum(e["n"] for e in ev.values())
    meta["stacking"] = {"model": "부품별 로지스틱 회귀 [logit(EfficientNet), logit(YOLO11m)], 검증셋 학습", "parts": stack,
                        "uncertain_margin": UNCERTAIN_MARGIN}
    meta["rule"] = ("세 판정(EfficientNet, YOLO11m, 2차 판정 모델)이 모두 같고 2차 판정 확률이 기준값 ±"
                    f"{UNCERTAIN_MARGIN:.2f} 밖이면 자동 판정, 아니면 AI_DISAGREE(작업자 확인). 부품 유형은 두 모델 6클래스 확률 평균.")
    meta["evaluation"] = {"set": "test+holdout 5,779장(공식 Validation)",
                          "auto_rate": sum(e["auto_rate"] * e["n"] for e in ev.values()) / n,
                          "auto_mean_accuracy": float(np.mean([e["auto_accuracy"] for e in ev.values()])),
                          "by_part": ev, "report": "reports/vision/full_data_retrain.md"}
    meta["test_part_metrics"] = {p: {"자동판정_비율": ev[p]["auto_rate"], "자동판정_정확도": ev[p]["auto_accuracy"],
                                     "2차판정_단독_정확도": ev[p]["stack_accuracy_all"],
                                     **{k: v for k, v in meta["test_part_metrics"][p].items() if k.endswith("_정확도") and "일치" not in k}}
                                 for p in PARTS}
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"stacking": stack, "evaluation": meta["evaluation"]}, ensure_ascii=False, indent=1))
    return meta


if __name__ == "__main__":
    main()
