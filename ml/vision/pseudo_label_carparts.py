"""3단계: carparts-seg 똑바른 원본을 절반은 학습 추가(정답 + 의사 라벨), 절반은 외부 검증으로 유지.

- 원본 단위 50/50 분할(seed 42). 똑바름 판정은 정답 라벨 위치(prepare_partseg.is_upright).
- 학습 절반(cp_train): carparts 정답을 HITL 클래스로 매핑해 사용.
  carparts에 없는 클래스(Fender, Quarter-panel, Rocker-panel, Roof, Grille, License-plate, Front/Back-window, Front/Back-wheel)는
  1차 모델(HITL 전용) 예측 중 점수 >= 0.7, 정답 마스크와 50% 미만 겹치는 것만 의사 라벨로 추가.
- 외부 검증 절반(ext_cp_holdout): 범퍼·도어 정답만(prepare_partseg.CARPARTS_MAP) — 학습에 쓰지 않음.
실행: .venv/Scripts/python -m ml.vision.pseudo_label_carparts
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from ml.vision.prepare_partseg import CARPARTS_MAP, CARPARTS_ZIP, OUT, is_upright, save_resized
from ml.vision.train_partseg import rasterize

PSEUDO_SCORE = 0.7
MAX_OVERLAP = 0.5
TRAIN_MAP = {**CARPARTS_MAP, "hood": "Hood", "front_glass": "Windshield", "back_glass": "Back-windshield",
             "front_light": "Headlight", "front_left_light": "Headlight", "front_right_light": "Headlight",
             "back_light": "Tail-light", "back_left_light": "Tail-light", "back_right_light": "Tail-light",
             "left_mirror": "Mirror", "right_mirror": "Mirror", "trunk": "Trunk", "tailgate": "Trunk"}
PSEUDO_CLASSES = {"Fender", "Quarter-panel", "Rocker-panel", "Roof", "Grille", "License-plate",
                  "Front-window", "Back-window", "Front-wheel", "Back-wheel"}


def _mask_to_polygons(mk: np.ndarray) -> list[list[list[float]]]:
    import cv2
    cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(float).tolist() for c in cs if cv2.contourArea(c) >= 50 and len(c) >= 3]


def main() -> dict:
    import torch
    import torchvision.transforms.functional as TF
    import yaml
    from ml.vision.partseg_infer import _load
    model, dev, meta = _load()
    classes = meta["classes"]
    z = zipfile.ZipFile(CARPARTS_ZIP)
    names = yaml.safe_load(z.read("carparts-seg.yaml"))["names"]
    by_base = defaultdict(list)
    for m in sorted(n for n in z.namelist() if n.startswith("labels/") and n.endswith(".txt")):
        by_base[re.sub(r"\.rf\.[0-9a-f]+\.txt$", "", m.split("/")[-1])].append(m)
    chosen = []
    for base, ms in sorted(by_base.items()):
        up = [m for m in ms if is_upright(z.read(m).decode(), names)]
        if up:
            chosen.append((base, up[0]))
    # 원본 이름 해시로 결정적 50/50 분할
    half = {b: ("cp_train" if int(hashlib.md5(b.encode()).hexdigest(), 16) % 2 == 0 else "ext_cp_holdout") for b, _ in chosen}
    recs = {"cp_train": [], "ext_cp_holdout": []}
    st = Counter()
    for base, m in chosen:
        sp = half[base]
        img_m = f"images/{m.split('/')[1]}/{m.split('/')[-1][:-4]}.jpg"
        im = Image.open(io.BytesIO(z.read(img_m))).convert("RGB")
        stem = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_")
        s = save_resized(im, OUT / "images" / sp / f"{stem}.jpg")
        W, H = round(im.width * min(1, s)), round(im.height * min(1, s))
        rec = {"file": f"images/{sp}/{stem}.jpg", "width": W, "height": H, "source": "carparts-seg", "objects": []}
        cmap = TRAIN_MAP if sp == "cp_train" else CARPARTS_MAP
        for line in z.read(m).decode().splitlines():
            v = line.split()
            if len(v) < 7 or names[int(v[0])] not in cmap:
                continue
            xy = list(map(float, v[1:]))
            rec["objects"].append({"label": cmap[names[int(v[0])]], "origin": "gt",
                                   "exterior": [[xy[i] * W, xy[i + 1] * H] for i in range(0, len(xy), 2)], "interiors": []})
            st[f"{sp}/gt/{cmap[names[int(v[0])]]}"] += 1
        if sp == "cp_train":
            gt_union = np.zeros((H, W), bool)
            for o in rec["objects"]:
                gt_union |= rasterize(o, W, H).astype(bool)
            img = Image.open(OUT / rec["file"]).convert("RGB")
            with torch.no_grad():
                p = model([TF.to_tensor(img).to(dev)])[0]
            for mk, sc, lb in zip(p["masks"], p["scores"].tolist(), p["labels"].tolist()):
                cls = classes[lb - 1]
                if cls not in PSEUDO_CLASSES or sc < PSEUDO_SCORE:
                    continue
                mk = (mk[0] > 0.5).cpu().numpy()
                if mk.sum() == 0 or (mk & gt_union).sum() / mk.sum() >= MAX_OVERLAP:
                    st["cp_train/pseudo_dropped_overlap"] += 1
                    continue
                for poly in _mask_to_polygons(mk):
                    rec["objects"].append({"label": cls, "origin": "pseudo", "score": round(sc, 3), "exterior": poly, "interiors": []})
                    st[f"cp_train/pseudo/{cls}"] += 1
        recs[sp].append(rec)
        st[f"img/{sp}"] += 1
    for sp, r in recs.items():
        (OUT / f"ann_{sp}.json").write_text(json.dumps(r), encoding="utf-8")
    out = {"pseudo_score": PSEUDO_SCORE, "max_overlap": MAX_OVERLAP, "pseudo_model": meta.get("model_version"), **st}
    (OUT / "stats_pseudo.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    main()
