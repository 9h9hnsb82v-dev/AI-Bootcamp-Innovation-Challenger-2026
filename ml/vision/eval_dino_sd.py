"""DINOv2 vs YOLO11m-cls vs EfficientNet-B0 — 세션 비겹침 분할 비교 (사용자 요청 2026-11-18).

eval_sd.py 와 같은 규칙: 부품별 검증셋 Youden 임계값으로 양품/불량 판정 → 시험셋 1회 평가.
2차 판정(로지스틱 회귀, 검증 5겹 교차예측 임계값)과 자동 판정(세 판정 일치 + 기준 ±0.10 밖)도 조합별로 같은 방식.
차이 신뢰구간: 부품 안 시험 사진 부트스트랩 2,000회 (DINO 정확도 - YOLO 정확도).
실행: .venv/Scripts/python -m ml.vision.eval_dino_sd <effnet_scores.csv> <yolo_scores.csv> <dino_scores.csv> [<dino_summary.json>]
출력: reports/vision/dino_sd_eval.json, .md
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.vision.compare_yolo_effnet import youden
from ml.vision.ensemble_check import load
from ml.vision.eval_sd import MARGIN, OUT, PARTS, stack_fit

NAMES = {"e": "EfficientNet-B0", "yv": "YOLO11m-cls", "dv": "DINOv2 ViT-B/14"}
COMBOS = {"현재 앱 (EfficientNet+YOLO)": ["e", "yv"], "EfficientNet+DINO": ["e", "dv"],
          "YOLO+DINO": ["yv", "dv"], "세 모델": ["e", "yv", "dv"]}


def merged(eff: str, yolo: str, dino: str) -> pd.DataFrame:
    E, Y, D = load(eff), load(yolo), load(dino)
    d = E.rename(columns={"score": "e"}).merge(Y[["subset", "stem", "score"]].rename(columns={"score": "yv"}),
                                               on=["subset", "stem"], validate="1:1")
    d = d.merge(D[["subset", "stem", "score"]].rename(columns={"score": "dv"}), on=["subset", "stem"], validate="1:1")
    d["y"] = (d.quality == "DEFECT").astype(int)
    return d


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diff = a[idx].mean(1) - b[idx].mean(1)
    return [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]


def evaluate(d: pd.DataFrame) -> dict:
    res: dict = {"single": {}, "combo": {}, "dino_minus_yolo": {}}
    for part in PARTS:
        g = d[d.part == part]
        v, t = g[g.subset == "val"], g[g.subset == "test"]
        y = t.y.to_numpy().astype(bool)
        correct = {}
        for c, nm in NAMES.items():
            th = youden(v[c].to_numpy(), v.y.to_numpy())
            correct[c] = (t[c].to_numpy() >= th) == y
            res["single"].setdefault(nm, {})[part] = {"acc": float(correct[c].mean()), "auc": float(roc_auc_score(y, t[c]))}
        res["dino_minus_yolo"][part] = {"diff": float(correct["dv"].mean() - correct["yv"].mean()),
                                        "ci95": boot_diff(correct["dv"].astype(float), correct["yv"].astype(float))}
        for nm, cols in COMBOS.items():
            f, ths = stack_fit(v, cols)
            sp = f(t)
            ds = sp >= ths
            singles = [t[c].to_numpy() >= youden(v[c].to_numpy(), v.y.to_numpy()) for c in cols]
            auto = np.all([s == ds for s in singles], axis=0) & (np.abs(sp - ths) >= MARGIN)
            res["combo"].setdefault(nm, {})[part] = {
                "stack_acc": float((ds == y).mean()), "auto_rate": float(auto.mean()),
                "auto_acc": float((ds[auto] == y[auto]).mean()) if auto.any() else None, "n_test": int(len(t))}
    for grp in ("single", "combo"):
        for nm, r in res[grp].items():
            k = "acc" if grp == "single" else "stack_acc"
            r["_mean"] = float(np.mean([r[p][k] for p in PARTS]))
    return res


def main(eff: str, yolo: str, dino: str, summary: str | None = None) -> dict:
    res = evaluate(merged(eff, yolo, dino))
    if summary:
        s = json.loads(open(summary, encoding="utf-8").read())
        res["dino_train"] = {k: s.get(k) for k in ("model", "license", "img", "epochs", "batch", "best_epoch",
                                                   "best_val_top1", "test_top1_6cls", "train_seconds")}
    p = lambda x: "—" if x is None else f"{x * 100:.1f}%"  # noqa: E731
    L = ["# DINOv2 vs YOLO11m-cls vs EfficientNet-B0 (세션 비겹침 시험셋)", "",
         "같은 이미지·분할·448px. 판정 임계값은 부품별 검증셋 Youden, 시험셋은 1회 평가.", "",
         "## 단일 모델 정확도 (AUC)", "", "| 모델 | 도어 | 범퍼 | 휀더 | 평균 |", "|---|---:|---:|---:|---:|"]
    for nm, r in res["single"].items():
        L.append(f"| {nm} | " + " | ".join(f"{p(r[x]['acc'])} ({r[x]['auc']:.3f})" for x in PARTS) + f" | {p(r['_mean'])} |")
    L += ["", "## DINO - YOLO 정확도 차이 (95% 부트스트랩 CI)", "", "| 부품 | 차이 | 95% CI |", "|---|---:|---|"]
    for x in PARTS:
        r = res["dino_minus_yolo"][x]
        L.append(f"| {x} | {r['diff'] * 100:+.1f}%p | {r['ci95'][0] * 100:+.1f} ~ {r['ci95'][1] * 100:+.1f} |")
    L += ["", "## 조합(2차 판정) — 정확도 / 자동 판정 비율 / 자동 판정 정확도", "",
          "| 조합 | 도어 | 범퍼 | 휀더 | 평균 2차 정확도 |", "|---|---|---|---|---:|"]
    for nm, r in res["combo"].items():
        L.append(f"| {nm} | " + " | ".join(f"{p(r[x]['stack_acc'])} / {p(r[x]['auto_rate'])} / {p(r[x]['auto_acc'])}"
                                            for x in PARTS) + f" | {p(r['_mean'])} |")
    if "dino_train" in res:
        L += ["", f"DINO 학습: {json.dumps(res['dino_train'], ensure_ascii=False)}"]
    (OUT / "dino_sd_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "dino_sd_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:5])
