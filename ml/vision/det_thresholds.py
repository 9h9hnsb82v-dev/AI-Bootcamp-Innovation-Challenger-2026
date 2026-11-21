"""YOLO11 결함 위치 박스 표시 기준(클래스별 신뢰도) 결정 — 검증셋 무작위 600장, IoU>=0.5 매칭, F1 최대 지점.

Ultralytics val은 Windows 한글 경로의 데이터 yaml을 깨뜨려 읽어서, 사진을 직접 열어 예측하고 정답 박스와 매칭한다.
결과: models/vision/yolo11m_det/meta.json 의 conf(표시 클래스만: scratch, exterior_damage) + 검증 지표
실행: .venv/Scripts/python -m ml.vision.det_thresholds [weights]
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ml.vision.prepare_subset import ROOT

NAMES = ["scratch", "exterior_damage", "gap_defect", "gap_normal"]
SHOW = ("scratch", "exterior_damage")  # 단차는 검출이 약해(mAP50 0.10~0.30) 표시하지 않음
SRC = ROOT / "models/vision/yolo11m_det_full/named-outputs/results"
DST = ROOT / "models/vision/yolo11m_det"


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main(weights: str) -> dict:
    import os
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "hub": False, "mlflow": False})
    full = ROOT / "data/processed/aihub_full"
    imgs = sorted((full / "images/val").glob("*/*.jpg"))
    random.Random(42).shuffle(imgs)
    imgs = imgs[:600]
    m = YOLO(weights)
    preds, gts = {c: [] for c in range(4)}, {c: 0 for c in range(4)}
    for k in range(0, len(imgs), 16):
        batch = imgs[k:k + 16]
        res = m.predict([Image.open(p).convert("RGB") for p in batch], imgsz=640, conf=0.05, verbose=False, device="cpu")
        for p, r in zip(batch, res):
            lab = full / "labels" / p.relative_to(full / "images").with_suffix(".txt")
            g = [list(map(float, ln.split())) for ln in lab.read_text().splitlines() if ln.strip()]
            gb = {c: [[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for cc, x, y, w, h in g if int(cc) == c] for c in range(4)}
            for c in range(4):
                gts[c] += len(gb[c])
            used = {c: set() for c in range(4)}
            for c, s, b in sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist()), key=lambda t: -t[1]):
                c = int(c)
                best, bj = 0.0, -1
                for j, gbx in enumerate(gb[c]):
                    if j not in used[c] and (v := iou(b, gbx)) > best:
                        best, bj = v, j
                if best >= 0.5:
                    used[c].add(bj)
                preds[c].append((s, best >= 0.5))
        if k % 320 == 0:
            print(f"{k}/{len(imgs)}", flush=True)
    out = {}
    for c in range(4):
        best = None
        for t in np.arange(0.05, 0.95, 0.01):
            tp = sum(1 for s, ok in preds[c] if s >= t and ok)
            fp = sum(1 for s, ok in preds[c] if s >= t and not ok)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / gts[c] if gts[c] else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if best is None or f1 > best[1]:
                best = (round(float(t), 2), round(f1, 3), round(prec, 3), round(rec, 3))
        out[NAMES[c]] = {"conf": best[0], "f1": best[1], "precision": best[2], "recall": best[3], "gt_boxes": gts[c]}
    summary = json.loads((SRC / "summary.json").read_text())
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "best.pt", DST / "best.pt")
    meta = {"model_version": "aihub_yolo11m_det_full_e20", "framework": "Ultralytics 8.3.40 YOLO11m detect (AGPL-3.0)",
            "azure_job": "upbeat_boot_xkg8yt7pdt", "img": 640, "names": NAMES,
            "conf": {n: out[n]["conf"] for n in SHOW}, "val_threshold_search": out,
            "test": summary["test"], "holdout": summary["holdout"],
            "use": "결함 위치 표시 전용(판정·kg에 쓰지 않음). 단차 클래스는 검출이 약해 표시 안 함.",
            "license_note": "AGPL-3.0 - 내부 사용 전용. 사용자 허락 전 코드·모델 외부 공개/배포 금지."}
    (DST / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return meta


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(SRC / "best.pt"))
