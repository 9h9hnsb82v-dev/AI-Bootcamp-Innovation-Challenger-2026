"""YOLO11 결함 위치 검출용 라벨: AI-Hub 원본 전체(aihub_full, 640px) 사진에 박스 라벨(txt)을 만든다 (사용자 결정 2026-11-14 (b)).

- 사진은 data/processed/aihub_full/images/<subset>/<CLASS>/<stem>.jpg 를 그대로 쓰고,
  라벨은 같은 구조의 data/processed/aihub_full/labels/<subset>/<CLASS>/<stem>.txt 에 쓴다(Ultralytics 규칙: /images/ → /labels/).
- 클래스(prepare_yolo.py와 동일): 0 scratch(101 불량) · 1 exterior_damage(102 불량) · 2 gap_defect(203 불량) · 3 gap_normal(203 양품)
  양품 사진의 101/102 박스는 제외(개수만 보고). 박스가 없는 사진은 빈 라벨(배경 사진)로 둔다.
- 라벨 좌표는 EXIF 회전이 적용된 화면 기준(2026-11-10 검증)이며 추출 사진도 EXIF 적용본이라 정규화 좌표가 그대로 맞는다.
- 분할은 aihub_full과 동일: train/val(공식 Training, 촬영일 그룹) · test 1,080 · holdout 4,699(공식 Validation).
출력: labels/, data_det.yaml, det_stats.json, det_index.csv
실행: .venv/Scripts/python -m ml.vision.prepare_yolo_full
"""
from __future__ import annotations

import json
import tarfile
from collections import Counter

import pandas as pd

from ml.vision.audit_aihub_labels import AIHUB_DIR, LABEL_TARS
from ml.vision.prepare_subset import ROOT
from ml.vision.prepare_yolo import NAMES, box_class

FULL = ROOT / "data/processed/aihub_full"


def main() -> dict:
    m = pd.read_csv(FULL / "manifest.csv", encoding="utf-8-sig")
    need = set(m.json_member)
    labels = {}
    for (_, _), rel in LABEL_TARS.items():
        with tarfile.open(AIHUB_DIR / rel) as t:
            for mem in t:
                if mem.name in need and mem.name not in labels:
                    labels[mem.name] = json.loads(t.extractfile(mem).read().decode("utf-8-sig"))
        print(rel, len(labels), flush=True)
    stats, rows = Counter(), []
    for r in m.itertuples():
        d = labels.get(r.json_member)
        if d is None:
            stats["missing_label_json"] += 1
            continue
        im = d["images"][0]
        W, H = float(im["width"]), float(im["height"])
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
        dst = FULL / "labels" / r.subset / r._2 / f"{r.stem}.txt"  # r._2 = class 열
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[f"img_{r.subset}"] += 1
        stats[f"img_{r.subset}_with_box"] += bool(lines)
        rows.append({"subset": r.subset, "stem": r.stem, "part": r.part, "quality": r.quality,
                     "subtype": r.json_member.split("/")[1], "n_boxes": len(lines)})
    pd.DataFrame(rows).to_csv(FULL / "det_index.csv", index=False, encoding="utf-8-sig")
    (FULL / "data_det.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8", newline="\n")
    out = dict(sorted(stats.items()))
    (FULL / "det_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
