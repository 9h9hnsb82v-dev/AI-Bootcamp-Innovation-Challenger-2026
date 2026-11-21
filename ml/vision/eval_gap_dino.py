"""휀더 단차 판정: EfficientNet 단차 모델 vs DINOv2 단차 모델 비교 (사용자 승인 2026-11-19).

같은 시험 3,058장(촬영일 통째 분할)에서
1) 단차 모델 단독: gt(박스 중심) · jitter(작업자 조준 오차 흉내) 정확도·AUC — 기준값은 검증셋 Youden
2) 앱 규칙: 단차 모델 + 2차 판정(표면 EfficientNet · 표면 DINOv2 · 단차 점수, 검증셋 로지스틱 회귀),
   단차 판정과 2차 판정이 같고 기준 ±0.10 밖이면 자동 판정
3) DINO − EfficientNet 정확도 차이 95% 부트스트랩 신뢰구간
실행: .venv/Scripts/python -m ml.vision.eval_gap_dino <gap_dino_scores.csv>
출력: reports/vision/gap_dino_eval.{json,md}
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.vision.common import ROOT
from ml.vision.compare_yolo_effnet import youden
from ml.vision.ensemble_check import load
from ml.vision.eval_dino_sd import boot_diff
from ml.vision.fit_dino_ensemble import MARGIN, _apply, _fit

RUNS = ROOT / "reports/vision/sd_runs"
OUT = ROOT / "reports/vision"


def main(gap_dino_csv: str) -> dict:
    E = load(str(RUNS / "effnet/effnet_scores.csv")).rename(columns={"score": "e"})
    D = load(str(ROOT / "models/vision/dinov2_vitb14_sd_e15/dinov2_vitb14_scores.csv")).rename(columns={"score": "dv"})
    surf = E.merge(D[["subset", "stem", "dv"]], on=["subset", "stem"])
    ge = pd.read_csv(RUNS / "gap/gap_scores.csv")[["subset", "stem", "quality", "score_gt", "score_jitter"]]
    gd = pd.read_csv(gap_dino_csv)[["subset", "stem", "score_gt", "score_jitter"]]
    g = ge.merge(gd, on=["subset", "stem"], suffixes=("_E", "_D"), validate="1:1")
    g = g.merge(surf[["subset", "stem", "e", "dv"]], on=["subset", "stem"], how="inner")
    g["y"] = (g.quality == "DEFECT").astype(int)
    v, t = g[g.subset == "val"], g[g.subset == "test"]
    y = t.y.to_numpy().astype(bool)
    res: dict = {"n_val": int(len(v)), "n_test": int(len(t)), "single": {}, "app_rule": {}, "dino_minus_effnet": {}}
    correct = {}
    for name, k in (("EfficientNet-B0", "E"), ("DINOv2 ViT-B/14", "D")):
        res["single"][name] = {}
        for mode in ("gt", "jitter"):
            col = f"score_{mode}_{k}"
            th = youden(v[col].to_numpy(), v.y.to_numpy())
            ok = (t[col].to_numpy() >= th) == y
            correct[(k, mode)] = ok
            res["single"][name][mode] = {"acc": float(ok.mean()), "auc": float(roc_auc_score(y, t[col])), "threshold": float(th)}
        # 앱 규칙(jitter = 실제 촬영과 가까운 쪽)
        col = f"score_jitter_{k}"
        thg = youden(v[col].to_numpy(), v.y.to_numpy())
        sk = _fit(v, ["e", "dv", col])
        sp = _apply(sk, t, ["e", "dv", col])
        dg, ds = t[col].to_numpy() >= thg, sp >= sk["threshold"]
        auto = (dg == ds) & (np.abs(sp - sk["threshold"]) >= MARGIN)
        res["app_rule"][name] = {"stack_acc": float((ds == y).mean()), "auto_rate": float(auto.mean()),
                                 "auto_acc": float((ds[auto] == y[auto]).mean()) if auto.any() else None,
                                 "worker_check": int((~auto).sum())}
    for mode in ("gt", "jitter"):
        a, b = correct[("D", mode)].astype(float), correct[("E", mode)].astype(float)
        res["dino_minus_effnet"][mode] = {"diff": float(a.mean() - b.mean()), "ci95": boot_diff(a, b)}
    pct = lambda x: "—" if x is None else f"{x * 100:.1f}%"  # noqa: E731
    L = ["# 휀더 단차 판정 — EfficientNet vs DINOv2 단차 모델 (세션 비겹침 시험)", "",
         f"- 시험 {res['n_test']:,}장 · 검증 {res['n_val']:,}장(기준값·2차 판정은 검증셋으로만). 앱의 표면 모델은 EfficientNet+DINOv2(현재 앱).", "",
         "## 단차 모델 단독", "", "| 모델 | gt 정확도 (AUC) | jitter 정확도 (AUC) |", "|---|---:|---:|"]
    for name, r in res["single"].items():
        L.append(f"| {name} | {pct(r['gt']['acc'])} ({r['gt']['auc']:.3f}) | {pct(r['jitter']['acc'])} ({r['jitter']['auc']:.3f}) |")
    L += ["", "## DINO − EfficientNet (95% 부트스트랩 CI)", "", "| 자르기 | 차이 | 95% CI |", "|---|---:|---|"]
    for mode, r in res["dino_minus_effnet"].items():
        L.append(f"| {mode} | {r['diff'] * 100:+.1f}%p | {r['ci95'][0] * 100:+.1f} ~ {r['ci95'][1] * 100:+.1f} |")
    L += ["", "## 앱 규칙 (단차 모델 + 2차 판정, jitter)", "", "| 단차 모델 | 2차 판정 정확도 | 자동 판정 비율 | 자동 판정 정확도 | 작업자 확인 장수 |",
          "|---|---:|---:|---:|---:|"]
    for name, r in res["app_rule"].items():
        L.append(f"| {name} | {pct(r['stack_acc'])} | {pct(r['auto_rate'])} | {pct(r['auto_acc'])} | {r['worker_check']:,} |")
    (OUT / "gap_dino_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "gap_dino_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
