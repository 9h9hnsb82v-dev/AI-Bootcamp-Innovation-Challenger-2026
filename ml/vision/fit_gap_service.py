"""휀더 '틈(단차)' 촬영 판정 규칙 확정 + 서비스 파일 작성 (사용자 결정 2026-11-14: 2안 = 같은 화면에서 촬영 부위 전환).

- 단차 모델: models/vision/gap_effb0 (작업 tender_candle_1fy6x2p4by). 서비스 입력 = 사진 가운데 가이드 박스(짧은 변의 GUIDE_FRAC) 영역.
  학습·평가의 '조준 오차' 방식(박스 긴 변의 1.2~1.8배 crop, 중심 ±0.25L)이 이 촬영 조건을 흉내 낸 것이다.
- 2차 판정(단차): 검증셋에서 [logit EfficientNet, logit YOLO11m, logit 단차(조준 오차)] 로지스틱 회귀, 5겹 교차예측 Youden 임계값.
- 자동 판정 규칙 후보를 검증셋으로 고르고 평가셋(test+holdout)에 한 번 적용해 보고한다.
출력: models/vision/gap/{gap_effb0.pt, meta.json}
실행: .venv/Scripts/python -m ml.vision.fit_gap_service
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ml.vision.compare_yolo_effnet import ROOT, youden
from ml.vision.ensemble_check import load, logit

SRC = ROOT / "models/vision/gap_effb0/named-outputs/results"
DST = ROOT / "models/vision/gap"
ENS = ROOT / "models/vision/ensemble_full"
GUIDE_FRAC = 0.5   # 가이드 박스 한 변 = 사진 짧은 변 × 0.5
MARGIN = 0.10


def rules(d: pd.DataFrame, th: dict, p2: np.ndarray, th2: float) -> dict:
    de, dy, dg, ds = d.eff >= th["eff"], d.yolo >= th["yolo"], d.score_jitter >= th["gap"], p2 >= th2
    sure = np.abs(p2 - th2) >= MARGIN
    y = d.y.astype(bool).to_numpy()
    out = {}
    for name, auto in {"네 판정 모두 일치 + 애매하지 않음": (de == dy) & (dy == dg) & (dg == ds) & sure,
                       "단차 모델 = 2차 판정 + 애매하지 않음": (dg == ds) & sure,
                       "2차 판정만 (전부 자동)": np.ones(len(d), bool)}.items():
        auto = np.asarray(auto)
        out[name] = {"auto_rate": float(auto.mean()), "auto_accuracy": float((np.asarray(ds)[auto] == y[auto]).mean())}
    return out


def main() -> dict:
    g = pd.read_csv(SRC / "gap_scores.csv")
    E = load(str(ROOT / "models/vision/effnet_full_e20/named-outputs/results/effnet_scores.csv"))
    Y = load(str(ROOT / "models/vision/yolo11m_cls_full_e30/named-outputs/results/yolo11m-cls_scores.csv"))
    d = g.merge(E[["subset", "stem", "score"]].rename(columns={"score": "eff"}), on=["subset", "stem"], validate="1:1") \
         .merge(Y[["subset", "stem", "score"]].rename(columns={"score": "yolo"}), on=["subset", "stem"], validate="1:1")
    d["y"] = (d.quality == "DEFECT").astype(int)
    v, t = d[d.subset == "val"], d[d.subset != "val"]
    th = {"eff": json.loads((ENS / "effnet/meta.json").read_text(encoding="utf-8"))["thresholds"]["FENDER"],
          "yolo": json.loads((ENS / "yolo11m_cls/meta.json").read_text(encoding="utf-8"))["thresholds"]["FENDER"],
          "gap": round(youden(v.score_jitter.to_numpy(), v.y.to_numpy()), 2)}
    X = lambda df: np.c_[logit(df.eff), logit(df.yolo), logit(df.score_jitter)]
    lr = LogisticRegression(max_iter=1000)
    oof = cross_val_predict(lr, X(v), v.y, cv=5, method="predict_proba")[:, 1]
    th2 = round(youden(oof, v.y.to_numpy()), 2)
    lr.fit(X(v), v.y)
    res_val = rules(v, th, oof, th2)
    res_eval = rules(t, th, lr.predict_proba(X(t))[:, 1], th2)
    chosen = "단차 모델 = 2차 판정 + 애매하지 않음"
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "gap_effb0.pt", DST / "gap_effb0.pt")
    src_meta = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))
    meta = {"model_version": "aihub_gap_effb0_img448_guided", "framework": "torchvision EfficientNet-B0 (BSD-3)",
            "azure_job": "tender_candle_1fy6x2p4by", "img": src_meta["img"], "best_epoch": src_meta["best_epoch"],
            "guide_frac": GUIDE_FRAC, "gap_threshold": th["gap"], "member_thresholds": th,
            "stacking": {"inputs": ["logit EfficientNet", "logit YOLO11m", "logit 단차"], "coef": [float(c) for c in lr.coef_[0]],
                         "intercept": float(lr.intercept_[0]), "threshold": th2, "uncertain_margin": MARGIN},
            "rule": f"틈(단차) 촬영: {chosen} → 자동 판정, 아니면 AI_DISAGREE(작업자 확인). EfficientNet·YOLO11m은 참고 표시.",
            "rule_candidates": {"val": res_val, "test+holdout": res_eval}, "chosen_rule": chosen,
            "caveat": "평가는 AI-Hub 사진에서 단차 박스 위치를 흔들어 가이드 촬영을 흉내 낸 것. 실제 카메라 가이드 촬영은 미검증(GAP_GUIDE_UNVALIDATED)."}
    (DST / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"thresholds": th, "stack_threshold": th2, "val": res_val, "eval": res_eval}, ensure_ascii=False, indent=1))
    return meta


if __name__ == "__main__":
    main()
