"""휀더 단차 전용 모델 평가: 현재 서비스 모델(EfficientNet·YOLO11m·2차 판정)과 같은 단차 사진으로 비교.

평가셋: 휀더 단차 사진 test 260 / holdout 1,403 (공식 Validation, 모든 모델 학습 미사용).
임계값: 방식마다 검증셋(val 2,743, 공식 Training 촬영일 그룹)에서 Youden J. 시험셋은 1회 평가.
조합: 검증셋에서 [logit EfficientNet, logit YOLO11m, logit 단차모델] 로지스틱 회귀(5겹 교차예측으로 임계값) → 평가셋.
차이: 현재 2차 판정 모델 대비 정확도 차이, 부트스트랩 2,000회 95% CI.
실행: .venv/Scripts/python -m ml.vision.eval_gap <gap_scores.csv>
출력: reports/vision/gap/gap_eval.json, .md
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from ml.vision.compare_yolo_effnet import ROOT, youden
from ml.vision.ensemble_check import load, logit

OUT = ROOT / "reports/vision/gap"
ENS = ROOT / "models/vision/ensemble_full"


def metrics(pred, y, score=None) -> dict:
    pred, y = np.asarray(pred).astype(bool), np.asarray(y).astype(bool)
    r = {"accuracy": float((pred == y).mean()), "recall": float(pred[y].mean()), "specificity": float((~pred[~y]).mean())}
    if score is not None:
        r["auc"] = float(roc_auc_score(y, score))
    return r


def main(gap_csv: str) -> dict:
    g = pd.read_csv(gap_csv)
    E = load(str(ROOT / "models/vision/effnet_full_e20/named-outputs/results/effnet_scores.csv"))
    Y = load(str(ROOT / "models/vision/yolo11m_cls_full_e30/named-outputs/results/yolo11m-cls_scores.csv"))
    d = g.merge(E[["subset", "stem", "score"]].rename(columns={"score": "eff"}), on=["subset", "stem"], validate="1:1") \
         .merge(Y[["subset", "stem", "score"]].rename(columns={"score": "yolo"}), on=["subset", "stem"], validate="1:1")
    assert len(d) == len(g), "점수 결합 누락"
    d["y"] = (d.quality == "DEFECT").astype(int)
    meta = json.loads((ENS / "meta.json").read_text(encoding="utf-8"))
    c = meta["stacking"]["parts"]["FENDER"]
    d["stack"] = 1 / (1 + np.exp(-(c["coef"][0] * logit(d.eff) + c["coef"][1] * logit(d.yolo) + c["intercept"])))
    th_fixed = {"eff": json.loads((ENS / "effnet/meta.json").read_text(encoding="utf-8"))["thresholds"]["FENDER"],
                "yolo": json.loads((ENS / "yolo11m_cls/meta.json").read_text(encoding="utf-8"))["thresholds"]["FENDER"],
                "stack": c["threshold"]}
    v = d[d.subset == "val"]
    th = dict(th_fixed)
    for col in ("score_gt", "score_jitter", "score_tiles"):
        th[col] = youden(v[col].to_numpy(), v.y.to_numpy())
    X = lambda df, cols: np.c_[tuple(logit(df[k]) for k in cols)]
    combos = {"combo_jitter": ["eff", "yolo", "score_jitter"], "combo_tiles": ["eff", "yolo", "score_tiles"]}
    for name, cols in combos.items():
        lr = LogisticRegression(max_iter=1000)
        oof = cross_val_predict(lr, X(v, cols), v.y, cv=5, method="predict_proba")[:, 1]
        th[name] = youden(oof, v.y.to_numpy())
        lr.fit(X(v, cols), v.y)
        d[name] = lr.predict_proba(X(d, cols))[:, 1]
        th[f"{name}_coef"] = lr.coef_[0].round(3).tolist()
    labels = {"eff": "현재 EfficientNet-B0", "yolo": "현재 YOLO11m-cls", "stack": "현재 2차 판정 모델 (기준)",
              "score_gt": "단차 모델 · 박스 위치 그대로 (상한)", "score_jitter": "단차 모델 · 조준 오차 (촬영 안내 가정)",
              "score_tiles": "단차 모델 · 조각 탐색 (위치 모름)",
              "combo_jitter": "조합: 현재 2모델 + 단차 모델(조준 오차)", "combo_tiles": "조합: 현재 2모델 + 단차 모델(조각 탐색)"}
    res = {"thresholds": th, "n": d.subset.value_counts().to_dict(), "sets": {}}
    rng = np.random.default_rng(42)
    lines = ["# 휀더 단차 전용 모델 평가", ""]
    for key, subs in (("test", ["test"]), ("test+holdout", ["test", "holdout"])):
        t = d[d.subset.isin(subs)].reset_index(drop=True)
        idx = rng.integers(0, len(t), (2000, len(t)))
        base_ok = (t["stack"] >= th["stack"]).to_numpy() == t.y.astype(bool).to_numpy()
        rows = {}
        for col, lab in labels.items():
            pred = (t[col] >= th[col]).to_numpy()
            r = metrics(pred, t.y, t[col])
            ok = pred == t.y.astype(bool).to_numpy()
            diff = (ok.astype(float) - base_ok)[idx].mean(1)
            r["diff_vs_stack"] = float(ok.mean() - base_ok.mean())
            r["ci95"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
            rows[col] = r
        res["sets"][key] = rows
        res["sets"][key]["_tiles_hit_box_rate"] = float(t.tiles_hit_box.mean())
        lines += [f"## 평가셋: {key} ({len(t):,}장)", "",
                  "| 방식 | 정확도 | 재현율(불량 검출) | 특이도(양품 통과) | AUC | 2차 판정 모델 대비 (95% CI) |",
                  "|---|---:|---:|---:|---:|---|"]
        for col, lab in labels.items():
            r = rows[col]
            ci = "기준" if col == "stack" else f"{r['diff_vs_stack']*100:+.1f}%p ({r['ci95'][0]*100:+.1f} ~ {r['ci95'][1]*100:+.1f})"
            lines.append(f"| {lab} | **{r['accuracy']:.3f}** | {r['recall']:.3f} | {r['specificity']:.3f} | {r['auc']:.3f} | {ci} |")
        lines += ["", f"- 조각 탐색에서 최고 점수 조각이 실제 단차 박스와 겹친 비율: {res['sets'][key]['_tiles_hit_box_rate']:.1%}", ""]
    lines += [f"- 임계값(검증셋): " + ", ".join(f"{k} {v:.2f}" for k, v in th.items() if isinstance(v, float)),
              f"- 조합 가중치 [EfficientNet, YOLO11m, 단차]: 조준 오차 {th['combo_jitter_coef']}, 조각 탐색 {th['combo_tiles_coef']}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gap_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "gap_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
