"""세션 비겹침 재평가 결과 정리 (사용자 승인 2026-11-14). 같은 방식으로 '촬영일 통째' 분할에서 다시 학습한 모델들의 시험 점수.

- 서비스 규칙을 그대로 재현: 부품별 검증 Youden 임계값 → 두 모델 판정, 2차 판정 로지스틱 회귀(검증 5겹 교차예측 임계값),
  세 판정 일치 + 2차 확률이 기준 ±0.10 밖이면 자동 판정. 단차는 '단차 모델 = 2차 판정(세 점수)' + 애매 폭.
- 비교 대상: 같은 규칙의 기존(세션 겹침) 시험 결과. 현재 서비스 모델은 새 시험 사진 대부분을 학습에 봤으므로 새 시험셋으로 평가하지 않는다.
실행: .venv/Scripts/python -m ml.vision.eval_sd <effnet_scores.csv> <yolo_scores.csv> <gap_scores.csv> <det_summary.json>
출력: reports/vision/sd_eval.json, .md
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

PARTS = ("DOOR", "BUMPER", "FENDER")
MARGIN = 0.10
OUT = ROOT / "reports/vision"


def stack_fit(v: pd.DataFrame, cols: list[str]):
    X = lambda df: np.c_[tuple(logit(df[c]) for c in cols)]  # noqa: E731
    lr = LogisticRegression(max_iter=1000)
    oof = cross_val_predict(lr, X(v), v.y, cv=5, method="predict_proba")[:, 1]
    th = youden(oof, v.y.to_numpy())
    lr.fit(X(v), v.y)
    return (lambda df: lr.predict_proba(X(df))[:, 1]), th


def surface(eff_csv: str, yolo_csv: str) -> dict:
    E, Y = load(eff_csv), load(yolo_csv)
    d = E.merge(Y[["subset", "stem", "score"]], on=["subset", "stem"], suffixes=("_e", "_y"), validate="1:1")
    d = d.rename(columns={"score_e": "e", "score_y": "yv"})
    d["y"] = (d.quality == "DEFECT").astype(int)
    out = {}
    for part in PARTS:
        g = d[d.part == part]
        v, t = g[g.subset == "val"], g[g.subset == "test"]
        the, thy = youden(v.e.to_numpy(), v.y.to_numpy()), youden(v.yv.to_numpy(), v.y.to_numpy())
        f, ths = stack_fit(v, ["e", "yv"])
        sp = f(t)
        y = t.y.to_numpy().astype(bool)
        de, dy, ds = t.e.to_numpy() >= the, t.yv.to_numpy() >= thy, sp >= ths
        auto = (de == dy) & (ds == de) & (np.abs(sp - ths) >= MARGIN)
        out[part] = {"n_test": int(len(t)), "effnet_acc": float((de == y).mean()), "yolo_acc": float((dy == y).mean()),
                     "stack_acc": float((ds == y).mean()), "auto_rate": float(auto.mean()),
                     "auto_acc": float((ds[auto] == y[auto]).mean()) if auto.any() else None,
                     "effnet_auc": float(roc_auc_score(y, t.e)), "yolo_auc": float(roc_auc_score(y, t.yv))}
    ttl = d[d.subset == "test"]
    out["_mean"] = {k: float(np.mean([out[p][k] for p in PARTS])) for k in ("effnet_acc", "yolo_acc", "stack_acc", "auto_acc")}
    out["_auto_rate_overall"] = float(sum(out[p]["auto_rate"] * out[p]["n_test"] for p in PARTS) / len(ttl))
    Yr = pd.read_csv(yolo_csv)
    out["_yolo_top1_6cls_test"] = float((Yr[Yr.subset == "test"].true == Yr[Yr.subset == "test"].pred).mean())
    return out, d


def gap(gap_csv: str, d: pd.DataFrame) -> dict:
    g = pd.read_csv(gap_csv).merge(d[["subset", "stem", "e", "yv"]], on=["subset", "stem"], how="inner")
    g["y"] = (g.quality == "DEFECT").astype(int)
    v, t = g[g.subset == "val"], g[g.subset == "test"]
    thg = youden(v.score_jitter.to_numpy(), v.y.to_numpy())
    f, ths = stack_fit(v, ["e", "yv", "score_jitter"])
    sp, y = f(t), t.y.to_numpy().astype(bool)
    dg, ds = t.score_jitter.to_numpy() >= thg, sp >= ths
    auto = (dg == ds) & (np.abs(sp - ths) >= MARGIN)
    f2, th2 = stack_fit(v, ["e", "yv"])
    base = f2(t) >= th2
    return {"n_test": int(len(t)), "gap_jitter_acc": float((dg == y).mean()), "gap_gt_acc":
            float(((t.score_gt.to_numpy() >= youden(v.score_gt.to_numpy(), v.y.to_numpy())) == y).mean()),
            "surface_stack_acc_on_gap_photos": float((base == y).mean()), "gap_rule_auto_rate": float(auto.mean()),
            "gap_rule_auto_acc": float((ds[auto] == y[auto]).mean()) if auto.any() else None}


def main(eff_csv: str, yolo_csv: str, gap_csv: str, det_json: str) -> dict:
    s, d = surface(eff_csv, yolo_csv)
    gp = gap(gap_csv, d)
    det = json.loads(open(det_json, encoding="utf-8").read())
    old_ens = json.loads((ROOT / "models/vision/ensemble_full/meta.json").read_text(encoding="utf-8"))["evaluation"]
    split = json.loads((ROOT / "reports/vision/split_sd.json").read_text(encoding="utf-8"))
    res = {"split": split, "surface": s, "gap": gp, "det_test": det.get("test"),
           "old_overlapping": {"auto_rate": old_ens["auto_rate"], "auto_mean_accuracy": old_ens["auto_mean_accuracy"],
                               "by_part": old_ens["by_part"]}}
    L = ["# 세션 비겹침 재평가 (촬영일 통째 분할)", "",
         f"- 시험 {split['images']['test']:,}장(촬영일 {split['days']['test']}일) · 학습·검증과 60초 이내 겹침 "
         f"{split['test_within_60s_of_trainval']:.1%} (기존 시험셋 98%)", "",
         "| 부품 | 시험 장수 | EfficientNet | YOLO11m-cls | 2차 판정 | 자동 판정 비율 | 자동 판정 정확도 | (기존·세션 겹침) 자동 판정 정확도 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in PARTS:
        r, o = s[p], old_ens["by_part"][p]
        L.append(f"| {p} | {r['n_test']:,} | {r['effnet_acc']:.3f} | {r['yolo_acc']:.3f} | {r['stack_acc']:.3f} | {r['auto_rate']:.1%} | "
                 f"{r['auto_acc']:.3f} | {o['auto_accuracy']:.3f} |")
    m = s["_mean"]
    L += [f"| 평균 | | {m['effnet_acc']:.3f} | {m['yolo_acc']:.3f} | {m['stack_acc']:.3f} | {s['_auto_rate_overall']:.1%} | "
          f"{m['auto_acc']:.3f} | {old_ens['auto_mean_accuracy']:.3f} |", "",
          f"- YOLO11m-cls 6클래스 top-1(시험): {s['_yolo_top1_6cls_test']:.3f}",
          f"- 휀더 틈(단차) 사진 {gp['n_test']:,}장: 단차 모델(조준 오차) {gp['gap_jitter_acc']:.3f} · 박스 위치 그대로 {gp['gap_gt_acc']:.3f} · "
          f"표면 2차 판정 {gp['surface_stack_acc_on_gap_photos']:.3f} · 단차 규칙 자동 {gp['gap_rule_auto_rate']:.1%}, 정확도 {gp['gap_rule_auto_acc']:.3f}"]
    if det.get("test"):
        pc = det["test"]["per_class_mAP50"]
        L.append(f"- 결함 위치 검출(시험 mAP50): 전체 {det['test']['mAP50']:.3f} · 스크래치 {pc.get('scratch', float('nan')):.3f} · "
                 f"외관 손상 {pc.get('exterior_damage', float('nan')):.3f} · 단차 불량 {pc.get('gap_defect', float('nan')):.3f}")
    (OUT / "sd_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "sd_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:5])
