"""앱 판정 조합을 EfficientNet-B0 + DINOv2 로 구성 (사용자 결정 2026-11-18: 'EfficientNet+DINO', YOLO 대체).

- 세 모델 모두 같은 세션 비겹침 분할(촬영일 통째, manifest_sd.csv)로 학습된 것을 쓴다:
  EfficientNet-B0 SD 5에폭(bright_kumquat) · DINOv2 ViT-B/14 SD 15에폭(good_bean) · 단차 EfficientNet SD(good_fish).
  (현재 앱의 '전체 재학습' EfficientNet 은 분할이 달라 DINO 와 섞으면 검증·시험 사진이 학습에 섞일 수 있어 쓰지 않는다.)
- 정하는 값은 모두 검증셋으로만: 부품별 Youden 임계값, 2차 판정(로지스틱 회귀) 계수와 기준(5겹 교차예측), 단차 임계값·2차 판정.
- 시험셋은 성적표에만 1회 쓴다(eval_sd 와 같은 규칙: 세 판정 일치 + 2차 확률이 기준 ±0.10 밖이면 자동 판정).
실행: .venv/Scripts/python -m ml.vision.fit_dino_ensemble <effnet_sd_dir> <dino_sd_dir> <gap_sd_dir>
출력: models/vision/ensemble_sd_dino/{meta.json, effnet/, dinov2/, gap/}
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ml.vision.common import MODEL_DIR, ROOT
from ml.vision.compare_yolo_effnet import youden
from ml.vision.ensemble_check import load, logit

OUT = MODEL_DIR / "ensemble_sd_dino"
PARTS = ("DOOR", "BUMPER", "FENDER")
MARGIN = 0.10
GUIDE_FRAC = 0.5  # 기존 단차 서비스와 같은 가이드 박스(짧은 변 x 0.5)


def _fit(v: pd.DataFrame, cols: list[str]) -> dict:
    X = np.c_[tuple(logit(v[c]) for c in cols)]
    lr = LogisticRegression(max_iter=1000)
    oof = cross_val_predict(lr, X, v.y, cv=5, method="predict_proba")[:, 1]
    th = youden(oof, v.y.to_numpy())
    lr.fit(X, v.y)
    return {"coef": [float(c) for c in lr.coef_[0]], "intercept": float(lr.intercept_[0]), "threshold": float(th)}


def _apply(sk: dict, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    z = sum(c * logit(df[k]) for c, k in zip(sk["coef"], cols)) + sk["intercept"]
    return 1 / (1 + np.exp(-z))


def main(eff_dir: str, dino_dir: str, gap_dir: str) -> dict:
    eff_dir, dino_dir, gap_dir = Path(eff_dir), Path(dino_dir), Path(gap_dir)
    E, D = load(str(eff_dir / "effnet_scores.csv")), load(str(dino_dir / "dinov2_vitb14_scores.csv"))
    d = E.rename(columns={"score": "e"}).merge(D[["subset", "stem", "score"]].rename(columns={"score": "dv"}),
                                               on=["subset", "stem"], validate="1:1")
    d["y"] = (d.quality == "DEFECT").astype(int)
    th_e, th_d, stack, test = {}, {}, {}, {}
    for part in PARTS:
        g = d[d.part == part]
        v, t = g[g.subset == "val"], g[g.subset == "test"]
        th_e[part] = youden(v.e.to_numpy(), v.y.to_numpy())
        th_d[part] = youden(v.dv.to_numpy(), v.y.to_numpy())
        stack[part] = _fit(v, ["e", "dv"])
        sp = _apply(stack[part], t, ["e", "dv"])
        y = t.y.to_numpy().astype(bool)
        de, dd, ds = t.e.to_numpy() >= th_e[part], t.dv.to_numpy() >= th_d[part], sp >= stack[part]["threshold"]
        auto = (de == dd) & (ds == de) & (np.abs(sp - stack[part]["threshold"]) >= MARGIN)
        test[part] = {"n": int(len(t)), "EfficientNet_정확도": float((de == y).mean()), "DINOv2_정확도": float((dd == y).mean()),
                      "2차판정_단독_정확도": float((ds == y).mean()), "자동판정_비율": float(auto.mean()),
                      "자동판정_정확도": float((ds[auto] == y[auto]).mean()) if auto.any() else None}
    # 단차(휀더 틈) — 단차 점수 + 두 표면 점수의 2차 판정
    G = pd.read_csv(gap_dir / "gap_scores.csv").merge(d[["subset", "stem", "e", "dv"]], on=["subset", "stem"], how="inner")
    G["y"] = (G.quality == "DEFECT").astype(int)
    gv, gt = G[G.subset == "val"], G[G.subset == "test"]
    th_g = youden(gv.score_jitter.to_numpy(), gv.y.to_numpy())
    gsk = _fit(gv, ["e", "dv", "score_jitter"])
    sp = _apply(gsk, gt, ["e", "dv", "score_jitter"])
    y = gt.y.to_numpy().astype(bool)
    dg, ds = gt.score_jitter.to_numpy() >= th_g, sp >= gsk["threshold"]
    auto = (dg == ds) & (np.abs(sp - gsk["threshold"]) >= MARGIN)
    gap_test = {"n": int(len(gt)), "단차모델_정확도": float((dg == y).mean()), "2차판정_정확도": float((ds == y).mean()),
                "자동판정_비율": float(auto.mean()), "자동판정_정확도": float((ds[auto] == y[auto]).mean()) if auto.any() else None}

    split = json.loads((ROOT / "reports/vision/split_sd.json").read_text(encoding="utf-8"))
    dsum = json.loads((dino_dir / "summary.json").read_text(encoding="utf-8"))
    emeta = json.loads((eff_dir / "meta.json").read_text(encoding="utf-8"))
    gmeta = json.loads((gap_dir / "meta.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("effnet", "dinov2", "gap"):
        (OUT / sub).mkdir(exist_ok=True)
    shutil.copy2(eff_dir / "efficientnet_b0_6cls.pt", OUT / "effnet/efficientnet_b0_6cls.pt")
    shutil.copy2(dino_dir / "best.pt", OUT / "dinov2/best.pt")
    shutil.copy2(gap_dir / "gap_effb0.pt", OUT / "gap/gap_effb0.pt")
    common = {"split": f"세션 비겹침(촬영일 통째) train {split['images']['train']:,} / val {split['images']['val']:,} / test {split['images']['test']:,}",
              "threshold_rule": "부품별 검증셋 Youden J (시험셋 미사용)"}
    (OUT / "effnet/meta.json").write_text(json.dumps({
        "model_version": emeta["model_version"], "backend": "effnet_sd", "framework": "torchvision EfficientNet-B0 (BSD-3)",
        "classes": emeta["classes"], "img_size": 448, "epochs": emeta.get("epochs"), "azure_job": "bright_kumquat_cxk5xx4lcv",
        "thresholds": th_e, "test_part_metrics": {p: {"accuracy": test[p]["EfficientNet_정확도"]} for p in PARTS}, **common},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "dinov2/meta.json").write_text(json.dumps({
        "model_version": "aihub_dinov2_vitb14_6cls_img448_sd_e15", "backend": "dino_sd",
        "framework": f"timm {dsum.get('timm')} {dsum['model']} (DINOv2, Apache-2.0)", "timm_name": "vit_base_patch14_dinov2",
        "classes": ["BUMPER_DEFECT", "BUMPER_GOOD", "DOOR_DEFECT", "DOOR_GOOD", "FENDER_DEFECT", "FENDER_GOOD"],
        "img_size": dsum["img"], "epochs": dsum["epochs"], "best_epoch": dsum["best_epoch"], "azure_job": "good_bean_wcbrhhynb9",
        "thresholds": th_d, "test_part_metrics": {p: {"accuracy": test[p]["DINOv2_정확도"]} for p in PARTS}, **common},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "gap/meta.json").write_text(json.dumps({
        "model_version": gmeta["model_version"] + "_sd", "framework": "torchvision EfficientNet-B0 (BSD-3)", "azure_job": "good_fish_54nz80x30g",
        "img": gmeta["img"], "guide_frac": GUIDE_FRAC, "gap_threshold": th_g,
        "stacking": {"inputs": ["logit EfficientNet", "logit DINOv2", "logit 단차"], **gsk, "uncertain_margin": MARGIN},
        "test": gap_test, "caveat": "AI-Hub 사진의 단차 박스로 가이드 촬영을 흉내 낸 평가. 실제 카메라 가이드 촬영은 미검증(GAP_GUIDE_UNVALIDATED).", **common},
        ensure_ascii=False, indent=1), encoding="utf-8")
    n_test = sum(test[p]["n"] for p in PARTS)
    meta = {"model_version": "aihub_ensemble_effb0_dinov2_sd", "backend": "ensemble_dino", "members": ["effnet_sd", "dino_sd"],
            "member_names": ["EfficientNet-B0", "DINOv2 ViT-B/14"],
            "rule": "세 판정(EfficientNet, DINOv2, 2차 판정 모델)이 모두 같고 2차 판정 확률이 기준값 ±0.10 밖이면 자동 판정, 아니면 AI_DISAGREE(작업자 확인).",
            "license_note": "EfficientNet-B0 BSD-3, DINOv2 Apache-2.0 (YOLO/AGPL 미사용). 결함 위치 박스(YOLO11 검출)는 별도·표시 전용.",
            "decided_by": "사용자 2026-11-18 ('EfficientNet+DINO', 미리보기는 EfficientNet 단독)", **common,
            "test_part_metrics": test,
            "evaluation": {"set": f"세션 비겹침 시험 {n_test:,}장(처음 보는 촬영일 {split['days']['test']}일)", "honest": True,
                           "by_part": {p: {"auto_rate": test[p]["자동판정_비율"], "auto_accuracy": test[p]["자동판정_정확도"]} for p in PARTS},
                           "gap": gap_test, "report": "reports/vision/dino_sd_eval.md"},
            "stacking": {"model": "부품별 로지스틱 회귀 [logit(EfficientNet), logit(DINOv2)], 검증셋 학습", "parts": stack,
                         "uncertain_margin": MARGIN},
            "created": str(date.today())}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"thresholds_effnet": th_e, "thresholds_dino": th_d, "test": test, "gap": gap_test}, ensure_ascii=False, indent=1))
    return meta


if __name__ == "__main__":
    main(*sys.argv[1:4])
