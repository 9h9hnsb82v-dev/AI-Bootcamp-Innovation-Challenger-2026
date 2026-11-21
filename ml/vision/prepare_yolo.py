"""YOLOv7 시범 비교용 데이터셋 구성 (EfficientNet v2와 같은 이미지·같은 train/val/test 분할).

- 이미지: 긴 변 1024px (휀더는 prepare_fender_hr 산출물 재사용, 도어/범퍼는 원천 tar에서 추출)
- 라벨: AI-Hub COCO bbox → YOLO txt (정규화 cx cy w h). 리사이즈는 종횡비 유지라 정규화 좌표 불변
- 클래스(부품 공통):
    0 scratch          (101 스크래치, 불량품)
    1 exterior_damage  (102 외관 손상, 불량품)
    2 gap_defect       (203 단차, 불량품)
    3 gap_normal       (203 단차, 양품 — 휀더 양품 사진에도 틈 영역 bbox가 있음)
  양품 이미지의 101/102 bbox가 있으면 제외하고 개수만 보고
- 이미지 단위 결함 점수 = 클래스 0~2 중 최대 신뢰도 (평가 스크립트에서 사용)
출력: data/processed/aihub/yolo/{images,labels}/{train,val,test}/, data.yaml, stats.json
실행: .venv/Scripts/python -m ml.vision.prepare_yolo
"""
from __future__ import annotations

import io
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from ml.vision.audit_aihub_labels import AIHUB_DIR, LABEL_TARS
from ml.vision.imageio import open_oriented
from ml.vision.prepare_subset import MANIFEST, ROOT

Image.MAX_IMAGE_PIXELS = None
LONG_SIDE = 1024
HR_DIR = ROOT / "data/processed/aihub/images_hr1024"
YOLO_DIR = ROOT / "data/processed/aihub/yolo"
NAMES = ["scratch", "exterior_damage", "gap_defect", "gap_normal"]
DEFECT_CLASS_IDS = [0, 1, 2]


def box_class(cat: int, quality: str) -> int | None:
    if cat == 203:
        return 2 if quality == "불량품" else 3
    if quality != "불량품":
        return None
    return {101: 0, 102: 1}.get(cat)


def hr_path(r) -> Path:
    stem = r.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    folder = r["orig_subset"] if "orig_subset" in r and isinstance(r["orig_subset"], str) else r.subset  # 파일 위치는 최초 분할 기준
    return HR_DIR / folder / r["class"] / f"{stem}.jpg"


def ensure_hr(m: pd.DataFrame) -> None:
    todo = m[[not hr_path(r).exists() for _, r in m.iterrows()]]
    for tar_rel, g in todo.groupby("source_tar"):
        want = {r.member: hr_path(r) for _, r in g.iterrows()}
        done = 0
        with tarfile.open(AIHUB_DIR / tar_rel, "r:") as t:
            for mem in t:
                if mem.name not in want:
                    continue
                im = open_oriented(t.extractfile(mem).read(), LONG_SIDE)  # EXIF 회전 적용(라벨 좌표계)
                im.thumbnail((LONG_SIDE, LONG_SIDE))
                want[mem.name].parent.mkdir(parents=True, exist_ok=True)
                im.save(want[mem.name], quality=92)
                done += 1
                if done >= len(want):
                    break
        print(f"HR {tar_rel}: {done}/{len(want)}", flush=True)


def load_labels(m: pd.DataFrame) -> dict[str, dict]:
    need = set(m.json_member)
    out = {}
    for (_, _), rel in LABEL_TARS.items():
        with tarfile.open(AIHUB_DIR / rel) as t:
            for mem in t:
                if mem.name in need:
                    out[mem.name] = json.loads(t.extractfile(mem).read().decode("utf-8-sig"))
    return out


def main() -> dict:
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    m = m[m.subset != "unused"].reset_index(drop=True)  # 8:2:1 재분할에서 제외된 공식 Validation 잔여분
    ensure_hr(m)
    labels = load_labels(m)
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    stats = Counter()
    rows = []
    for _, r in m.iterrows():
        d = labels[r.json_member]
        im = d["images"][0]
        W, H = float(im["width"]), float(im["height"])
        with Image.open(hr_path(r)) as hr:  # EXIF 적용 후 저장된 이미지와 라벨 좌표계 종횡비 일치 확인
            iw, ih = hr.size
        if abs((iw / ih) - (W / H)) / (W / H) > 0.02:
            stats["aspect_mismatch_images"] += 1
            stats.setdefault("aspect_mismatch_examples", [])
            if len(stats["aspect_mismatch_examples"]) < 5:
                stats["aspect_mismatch_examples"].append(r.member)
        lines = []
        for a in d["annotations"]:
            if "bbox" not in a:
                continue
            c = box_class(a["category_id"], a.get("attributes", {}).get("quality", ""))
            if c is None:
                stats["skipped_good_image_defect_box"] += 1
                continue
            x, y, w, h = a["bbox"]
            cx, cy = min(max((x + w / 2) / W, 0), 1), min(max((y + h / 2) / H, 0), 1)
            lines.append(f"{c} {cx:.6f} {cy:.6f} {min(w / W, 1):.6f} {min(h / H, 1):.6f}")
            stats[f"box_{r.subset}_{NAMES[c]}"] += 1
        stem = f"{r.part}_{r.quality}_{r.path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}"
        img_dst = YOLO_DIR / "images" / r.subset / f"{stem}.jpg"
        lab_dst = YOLO_DIR / "labels" / r.subset / f"{stem}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hr_path(r), img_dst)
        lab_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[f"img_{r.subset}"] += 1
        rows.append({"subset": r.subset, "part": r.part, "quality": r.quality, "stem": stem,
                     "subtype": r.json_member.split("/")[1], "n_boxes": len(lines)})
    pd.DataFrame(rows).to_csv(YOLO_DIR / "index.csv", index=False, encoding="utf-8-sig")
    (YOLO_DIR / "data.yaml").write_text(
        "train: ./data/aihub_yolo/images/train\nval: ./data/aihub_yolo/images/val\ntest: ./data/aihub_yolo/images/test\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8", newline="\n")
    out = dict(sorted(stats.items()))
    (YOLO_DIR / "stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
