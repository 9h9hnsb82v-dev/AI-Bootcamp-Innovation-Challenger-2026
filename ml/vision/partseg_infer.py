"""1단계 추론: 차량 전체 사진 → 부품 영역(Mask R-CNN). 모델이 없으면 MODEL_NOT_READY.

결과는 근접 촬영 안내용이며 결함 판정·kg에 쓰지 않는다(services.part_locator 참고).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from ml.vision.imageio import open_oriented

MODEL_DIR = Path(__file__).resolve().parents[2] / "models/vision/partseg"
WEIGHTS = MODEL_DIR / "maskrcnn_partseg.pt"
META = MODEL_DIR / "meta.json"
SCORE_THRESHOLD = 0.5
INFER_LONG_SIDE = 1024  # 학습 데이터 준비와 같은 긴 변
COLORS = {"DOOR": (0, 200, 0), "BUMPER": (255, 200, 0), "FENDER": (255, 0, 0)}


def status() -> dict:
    if not (WEIGHTS.exists() and META.exists()):
        return {"status": "MODEL_NOT_READY", "message": "부품 분할 모델 미학습: azureml/job_partseg_h100.yml"}
    meta = json.loads(META.read_text(encoding="utf-8"))
    return {"status": "OK", "model_version": meta.get("model_version", "partseg_maskrcnn"), "meta": meta}


@lru_cache(maxsize=1)
def _load():
    import torch
    from ml.vision.train_partseg import build_model
    meta = json.loads(META.read_text(encoding="utf-8"))
    want = os.environ.get("RESOURCE_VISION_DEVICE", "cpu")
    dev = torch.device("cuda" if want == "cuda" and torch.cuda.is_available() else "cpu")
    m = build_model(len(meta["classes"]), meta["min_size"], meta["max_size"])
    m.load_state_dict(torch.load(WEIGHTS, map_location=dev, weights_only=True))
    m.eval().to(dev)
    return m, dev, meta


def locate(image_bytes: bytes, score_threshold: float = SCORE_THRESHOLD) -> dict:
    """서비스 부품(DOOR/BUMPER/FENDER)별 존재 여부·최고 점수·면적 비율과 색 표시 이미지를 반환."""
    st = status()
    if st["status"] != "OK":
        return st
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    m, dev, meta = _load()
    img = open_oriented(image_bytes, INFER_LONG_SIDE)
    img.thumbnail((INFER_LONG_SIDE, INFER_LONG_SIDE))
    with torch.no_grad():
        p = m([TF.to_tensor(img).to(dev)])[0]
    classes, service = meta["classes"], meta["service_parts"]
    to_service = {c: sp for sp, cs in service.items() for c in cs}
    W, H = img.size
    parts = {sp: {"present": False, "max_score": 0.0, "area_frac": 0.0} for sp in service}
    dets, union = [], {sp: np.zeros((H, W), bool) for sp in service}
    for mask, score, label, box in zip(p["masks"], p["scores"].tolist(), p["labels"].tolist(), p["boxes"].tolist()):
        cls = classes[label - 1]
        sp = to_service.get(cls)
        if score < score_threshold:
            continue
        dets.append({"label": cls, "service_part": sp, "score": round(score, 4), "box": [round(v, 1) for v in box]})
        if sp:
            mk = (mask[0] > 0.5).cpu().numpy()
            union[sp] |= mk
            parts[sp]["present"] = True
            parts[sp]["max_score"] = max(parts[sp]["max_score"], round(score, 4))
    arr = np.array(img).astype(float)
    for sp, mk in union.items():
        parts[sp]["area_frac"] = round(float(mk.mean()), 4)
        if mk.any():
            arr[mk] = arr[mk] * 0.45 + np.array(COLORS.get(sp, (0, 0, 255))) * 0.55
    return {"status": "OK", "model_version": st["model_version"], "provenance": "PARTSEG_GUIDE_ONLY",
            "score_threshold": score_threshold, "parts": parts, "detections": dets,
            "overlay": Image.fromarray(arr.astype(np.uint8)), "image_size": [W, H]}
