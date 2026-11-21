"""YOLOv7 가중치로 val/test 이미지별 클래스 최대 신뢰도를 계산한다(이미지 단위 비교용).

YOLOv7 저장소 루트에서 실행(저장소 모듈 import 필요):
  cd yolov7 && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python ../ml/vision/yolo_image_scores.py \
      --weights runs/train/<exp>/weights/best.pt --data-root ./data/aihub_yolo --img 640 --out ../yolo_scores.csv
출력 CSV: subset, stem, max_conf_<class>, n_det, infer_ms
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from models.experimental import attempt_load  # noqa: E402  (YOLOv7 저장소)
from utils.datasets import letterbox  # noqa: E402
from utils.general import non_max_suppression  # noqa: E402



def main(a) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(a.weights, map_location=dev).eval()
    half = dev.type == "cuda"
    if half:
        model.half()
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, "module") else model.names  # 학습 data.yaml 클래스(3 또는 4)
    rows = []
    for subset in ("val", "test"):
        for p in sorted((Path(a.data_root) / "images" / subset).glob("*.jpg")):
            img0 = cv2.imread(str(p))
            img = letterbox(img0, a.img, stride=stride)[0][:, :, ::-1].transpose(2, 0, 1)
            x = torch.from_numpy(np.ascontiguousarray(img)).to(dev)
            x = (x.half() if half else x.float()) / 255.0
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                pred = model(x[None])[0]
            det = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.6)[0]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            r = {"subset": subset, "stem": p.stem, "n_det": int(len(det)), "infer_ms": round(ms, 2)}
            for i, n in enumerate(names):
                c = det[det[:, 5] == i][:, 4] if len(det) else det
                r[f"max_conf_{n}"] = float(c.max()) if len(c) else 0.0
            rows.append(r)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("written", a.out, len(rows), "mean infer ms", np.mean([r["infer_ms"] for r in rows]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
