"""AI-Hub 원천 tar에서 학습용 서브셋을 무작위 추출·축소 저장한다(원본 tar는 수정하지 않음).

- 클래스 = (부품, 품질) 6종: DOOR/BUMPER/FENDER × GOOD/DEFECT  (라벨 JSON의 attributes.quality 기준)
- 공식 Training → train/val(촬영일 그룹 분할), 공식 Validation → test
- 저장: data/processed/aihub/images/<split>/<class>/<stem>.jpg (긴 변 512px), manifest.csv
- tar는 헤더만 순회하고 선택된 파일만 읽는다(seek).

실행: .venv/Scripts/python -m ml.vision.prepare_subset --per-class-train 1200 --per-class-test 250
"""
from __future__ import annotations

import argparse
import csv
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 신뢰된 AI-Hub 원본(최대 1.08억 화소) 경고 억제

from ml.vision.audit_aihub_labels import AIHUB_DIR, OUT_INDEX, ROOT
from ml.vision.imageio import open_oriented

IMG_TARS = {
    ("train", "DOOR"): "1.Training/원천데이터/TS_1.도어.tar",
    ("train", "BUMPER"): "1.Training/원천데이터/TS_5.범퍼.tar",
    ("train", "FENDER"): "1.Training/원천데이터/TS_11.휀더.tar",
    ("val", "DOOR"): "2.Validation/원천데이터/VS_1.도어.tar",
    ("val", "BUMPER"): "2.Validation/원천데이터/VS_5.범퍼.tar",
    ("val", "FENDER"): "2.Validation/원천데이터/VS_11.휀더.tar",
}
OUT_DIR = ROOT / "data/processed/aihub/images"
MANIFEST = ROOT / "data/processed/aihub/manifest.csv"
LONG_SIDE = 512
SEED = 42


def quality_class(q: str) -> str | None:
    if q == "양품":
        return "GOOD"
    if q == "불량품":
        return "DEFECT"
    return None  # 혼재/미상 → 제외


def select(index: pd.DataFrame, n_train: int, n_test: int, part_train: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = index.copy()
    idx["qclass"] = idx["quality"].map(quality_class)
    idx = idx.dropna(subset=["qclass"])
    idx["stem"] = idx["file_name"].str.rsplit(".", n=1).str[0].str.lower()
    idx["capture_day"] = idx["date_captured"].astype(str).str[:10]
    picked = []
    for (split, part, q), g in idx.groupby(["split", "part", "qclass"]):
        n = (part_train or {}).get(part, n_train) if split == "train" else n_test
        picked.append(g.iloc[rng.permutation(len(g))[: min(n, len(g))]])
    sel = pd.concat(picked)
    # 공식 Training 내부: 촬영일 그룹 단위로 약 15%를 val로 (같은 날 촬영 이미지가 train/val에 섞이지 않게)
    sel["subset"] = np.where(sel["split"] == "train", "train", "test")
    for part, g in sel[sel.split == "train"].groupby("part"):
        days = np.array(sorted(g.capture_day.unique()))
        days = days[rng.permutation(len(days))]
        counts = g.capture_day.value_counts()
        acc, val_days = 0, []
        for d in days:
            if acc >= 0.15 * len(g):
                break
            val_days.append(d)
            acc += counts[d]
        sel.loc[(sel.part == part) & (sel.split == "train") & sel.capture_day.isin(val_days), "subset"] = "val"
    return sel


def main(n_train: int, n_test: int, part_train: dict | None = None) -> None:
    index = pd.read_csv(OUT_INDEX, encoding="utf-8-sig")
    sel = select(index, n_train, n_test, part_train)
    rows = []
    for (split, part), rel in IMG_TARS.items():
        want = sel[(sel.split == split) & (sel.part == part)].set_index("stem")
        if want.empty:
            continue
        tar_path = AIHUB_DIR / rel
        done = 0
        with tarfile.open(tar_path, "r:") as t:
            for m in t:
                if not m.isfile():
                    continue
                stem = Path(m.name).stem.lower()
                if stem not in want.index:
                    continue
                r = want.loc[stem]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                cls = f"{part}_{r.qclass}"
                dst = OUT_DIR / r.subset / cls / f"{stem}.jpg"
                if not dst.exists():
                    im = open_oriented(t.extractfile(m).read(), LONG_SIDE)  # EXIF 회전 적용(라벨 좌표계와 일치)
                    im.thumbnail((LONG_SIDE, LONG_SIDE))
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    im.save(dst, quality=92)
                rows.append({"subset": r.subset, "class": cls, "part": part, "quality": r.qclass,
                             "path": dst.relative_to(ROOT).as_posix(), "source_tar": rel, "member": m.name,
                             "capture_day": r.capture_day, "json_member": r.json_member})
                done += 1
                if done >= len(want):
                    break
        print(f"{split}/{part}: {done}/{len(want)}", flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("manifest", MANIFEST, len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class-train", type=int, default=1200)
    ap.add_argument("--per-class-test", type=int, default=250)
    ap.add_argument("--fender-train", type=int, default=None, help="휀더만 클래스당 학습 표본 수 별도 지정")
    a = ap.parse_args()
    main(a.per_class_train, a.per_class_test, {"FENDER": a.fender_train} if a.fender_train else None)
