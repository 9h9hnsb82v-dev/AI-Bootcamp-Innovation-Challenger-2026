"""앱 휀더 단차 판정을 DINOv2 단차 모델로 교체 (사용자 결정 2026-11-19: '단차 DINO로 바꿔서 앱에 적용').

- 단차 모델: DINOv2 ViT-B/14 (crimson_hamster_g7c4qxkk13, 세션 비겹침 분할, 448px, 10에폭) — 기존 EfficientNet 단차(gap/)는 되돌리기용으로 남긴다.
- 정하는 값은 검증셋으로만: 단차 임계값(jitter 점수 Youden), 2차 판정(로지스틱 회귀: 표면 EfficientNet · 표면 DINOv2 · 단차 DINO, 5겹 교차예측 기준).
- 시험셋은 성적표에만 1회(앱 규칙: 단차 판정 = 2차 판정이고 기준 ±0.10 밖이면 자동 판정).
실행: .venv/Scripts/python -m ml.vision.fit_gap_dino <gap_dino 결과 폴더(gap_dinov2.pt · gap_scores.csv · meta.json)>
출력: models/vision/ensemble_sd_dino/gap_dino/{gap_dinov2.pt, meta.json} + ensemble_sd_dino/meta.json 의 evaluation.gap 갱신
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ml.vision.common import ROOT
from ml.vision.compare_yolo_effnet import youden
from ml.vision.ensemble_check import load
from ml.vision.fit_dino_ensemble import GUIDE_FRAC, MARGIN, OUT, _apply, _fit

RUNS = ROOT / "reports/vision/sd_runs"


def main(src: str) -> dict:
    src = Path(src)
    E = load(str(RUNS / "effnet/effnet_scores.csv")).rename(columns={"score": "e"})
    D = load(str(ROOT / "models/vision/dinov2_vitb14_sd_e15/dinov2_vitb14_scores.csv")).rename(columns={"score": "dv"})
    G = pd.read_csv(src / "gap_scores.csv").merge(E[["subset", "stem", "e"]].merge(D[["subset", "stem", "dv"]], on=["subset", "stem"]),
                                                   on=["subset", "stem"], how="inner")
    G["y"] = (G.quality == "DEFECT").astype(int)
    v, t = G[G.subset == "val"], G[G.subset == "test"]
    th = youden(v.score_jitter.to_numpy(), v.y.to_numpy())
    sk = _fit(v, ["e", "dv", "score_jitter"])
    sp = _apply(sk, t, ["e", "dv", "score_jitter"])
    y = t.y.to_numpy().astype(bool)
    dg, ds = t.score_jitter.to_numpy() >= th, sp >= sk["threshold"]
    auto = (dg == ds) & (np.abs(sp - sk["threshold"]) >= MARGIN)
    test = {"n": int(len(t)), "단차모델_정확도": float((dg == y).mean()), "2차판정_정확도": float((ds == y).mean()),
            "자동판정_비율": float(auto.mean()), "자동판정_정확도": float((ds[auto] == y[auto]).mean())}
    src_meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    dst = OUT / "gap_dino"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "gap_dinov2.pt", dst / "gap_dinov2.pt")
    meta = {"model_version": src_meta["model_version"] + "_sd", "backbone": "dino", "weights": "gap_dinov2.pt",
            "timm_name": "vit_base_patch14_dinov2", "framework": "timm DINOv2 ViT-B/14 (Apache-2.0)", "azure_job": "crimson_hamster_g7c4qxkk13",
            "img": src_meta["img"], "best_epoch": src_meta["best_epoch"], "guide_frac": GUIDE_FRAC, "gap_threshold": th,
            "stacking": {"inputs": ["logit EfficientNet", "logit DINOv2", "logit 단차 DINOv2"], **sk, "uncertain_margin": MARGIN},
            "test": test, "replaces": "gap/ (EfficientNet-B0 단차, 되돌리기용으로 남김)", "decided_by": "사용자 2026-11-19",
            "caveat": "AI-Hub 사진의 단차 박스로 가이드 촬영을 흉내 낸 평가. 실제 카메라 가이드 촬영은 미검증(GAP_GUIDE_UNVALIDATED).",
            "created": str(date.today())}
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    em_p = OUT / "meta.json"
    em = json.loads(em_p.read_text(encoding="utf-8"))
    em["evaluation"]["gap_previous_effnet"] = em["evaluation"].get("gap")
    em["evaluation"]["gap"] = test
    em["gap_model"] = "gap_dino"
    em_p.write_text(json.dumps(em, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"gap_threshold": th, "stack_threshold": sk["threshold"], "test": test}, ensure_ascii=False, indent=1))
    return meta


if __name__ == "__main__":
    main(sys.argv[1])
