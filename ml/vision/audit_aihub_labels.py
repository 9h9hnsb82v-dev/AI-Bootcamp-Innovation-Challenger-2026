"""AI-Hub 「부품 품질 검사 영상 데이터(자동차)」 라벨 tar 감사 (도어·범퍼·휀더).

원본 tar는 압축 해제하지 않고 스트리밍으로 읽는다. 결과는
reports/audit/aihub_label_audit.json 과 data/processed/aihub/label_index.csv 로 저장.

실행: .venv/Scripts/python -m ml.vision.audit_aihub_labels
"""
from __future__ import annotations

import collections
import csv
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AIHUB_DIR = ROOT / "data/raw/aihub/162.부품 품질 검사 영상 데이터(자동차)/01.데이터"
LABEL_TARS = {
    ("train", "DOOR"): "1.Training/라벨링데이터/TL_1.도어.tar",
    ("train", "BUMPER"): "1.Training/라벨링데이터/TL_5.범퍼.tar",
    ("train", "FENDER"): "1.Training/라벨링데이터/TL_11.휀더.tar",
    ("val", "DOOR"): "2.Validation/라벨링데이터/VL_1.도어.tar",
    ("val", "BUMPER"): "2.Validation/라벨링데이터/VL_5.범퍼.tar",
    ("val", "FENDER"): "2.Validation/라벨링데이터/VL_11.휀더.tar",
}
OUT_JSON = ROOT / "reports/audit/aihub_label_audit.json"
OUT_INDEX = ROOT / "data/processed/aihub/label_index.csv"


def iter_labels(tar_path: Path):
    with tarfile.open(tar_path) as t:
        for m in t:
            if m.isfile() and m.name.lower().endswith(".json"):
                f = t.extractfile(m)
                if f is None:
                    continue
                yield m.name, json.loads(f.read().decode("utf-8-sig"))


def main() -> dict:
    summary: dict = {"source": str(AIHUB_DIR.relative_to(ROOT)), "splits": {}}
    rows = []
    sample_json = None
    for (split, part), rel in LABEL_TARS.items():
        p = AIHUB_DIR / rel
        if not p.exists():
            summary["splits"][f"{split}/{part}"] = "DATA_NOT_AVAILABLE"
            continue
        c = collections.Counter()
        folders = collections.Counter()
        quality = collections.Counter()
        cats = collections.Counter()
        work_part = collections.Counter()
        prefix = collections.Counter()
        has_bbox = collections.Counter()
        n_ann = collections.Counter()
        for name, d in iter_labels(p):
            c["json"] += 1
            folder = "/".join(name.split("/")[:-1])
            folders[folder] += 1
            img = d["images"][0]
            fname = img["file_name"]
            prefix["_".join(fname.split("_")[:3])] += 1
            anns = d.get("annotations", [])
            n_ann[len(anns)] += 1
            cat_names = {x["id"]: x["name"] for x in d.get("categories", [])}
            q = sorted({a.get("attributes", {}).get("quality", "?") for a in anns}) or ["NO_ANN"]
            quality["|".join(q)] += 1
            for a in anns:
                cats[f'{a.get("category_id")}:{cat_names.get(a.get("category_id"), "?")}'] += 1
                at = a.get("attributes", {})
                work_part[f'{at.get("work")}/{at.get("part")}'] += 1
                has_bbox["bbox" if "bbox" in a else "no_bbox"] += 1
            if sample_json is None and any("bbox" in a for a in anns):
                s = json.loads(json.dumps(d))
                for a in s["annotations"]:
                    if "segmentation" in a:
                        a["segmentation"] = f"<{len(a['segmentation'])} values>"
                sample_json = s
            rows.append({
                "split": split, "part": part, "json_member": name, "file_name": fname,
                "width": img.get("width"), "height": img.get("height"),
                "date_captured": img.get("date_captured"),
                "quality": "|".join(q), "n_annotations": len(anns),
                "category_ids": "|".join(sorted({str(a.get("category_id")) for a in anns})),
                "has_bbox": any("bbox" in a for a in anns),
                "license": d.get("licenses", [{}])[0].get("name"),
            })
        summary["splits"][f"{split}/{part}"] = {
            "n_json": c["json"], "folders": dict(folders), "quality_per_image": dict(quality),
            "annotation_categories": dict(cats), "work/part": dict(work_part),
            "bbox_presence": dict(has_bbox), "annotations_per_image": dict(sorted(n_ann.items())),
            "filename_prefix": dict(prefix.most_common(20)),
        }
    summary["sample_defect_json"] = sample_json
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with OUT_INDEX.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return summary


if __name__ == "__main__":
    main()
    print("written", OUT_JSON)
