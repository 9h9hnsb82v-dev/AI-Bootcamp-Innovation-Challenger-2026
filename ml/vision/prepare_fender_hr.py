"""휀더 전용 고해상도 서브셋: 기존 매니페스트의 FENDER 행(같은 이미지·같은 분할)을 긴 변 1024px로 재추출.

단차(패널 틈) 결함은 512px 축소본에서 디테일이 사라진다(원본 bbox 약 800×700px → 512px에서 약 100px).
출력: data/processed/aihub/images_hr1024/<subset>/<class>/<stem>.jpg, manifest_fender_hr1024.csv
실행: .venv/Scripts/python -m ml.vision.prepare_fender_hr
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pandas as pd
from PIL import Image

from ml.vision.audit_aihub_labels import AIHUB_DIR
from ml.vision.imageio import open_oriented
from ml.vision.prepare_subset import MANIFEST, ROOT

Image.MAX_IMAGE_PIXELS = None
LONG_SIDE = 1024
OUT_DIR = ROOT / "data/processed/aihub/images_hr1024"
OUT_MANIFEST = ROOT / "data/processed/aihub/manifest_fender_hr1024.csv"


def main() -> pd.DataFrame:
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    f = m[m.part == "FENDER"].copy()
    f["subtype"] = f.json_member.str.split("/").str[1]
    f["stem"] = f.path.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
    f["hr_path"] = [f"{OUT_DIR.relative_to(ROOT).as_posix()}/{s}/{c}/{t}.jpg" for s, c, t in zip(f.subset, f["class"], f.stem)]
    for tar_rel, g in f.groupby("source_tar"):
        want = dict(zip(g.member, g.hr_path))
        done = 0
        with tarfile.open(AIHUB_DIR / tar_rel, "r:") as t:
            for mem in t:
                if mem.name not in want:
                    continue
                dst = ROOT / want[mem.name]
                if not dst.exists():
                    im = open_oriented(t.extractfile(mem).read(), LONG_SIDE)  # EXIF 회전 적용
                    im.thumbnail((LONG_SIDE, LONG_SIDE))
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    im.save(dst, quality=92)
                done += 1
                if done >= len(want):
                    break
        print(f"{tar_rel}: {done}/{len(want)}", flush=True)
    out = f.drop(columns=["path"]).rename(columns={"hr_path": "path"})
    missing = [p for p in out.path if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"누락 {len(missing)}개")
    out.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")
    print("manifest", OUT_MANIFEST, len(out), out.groupby(["subset", "quality"]).size().to_dict())
    return out


if __name__ == "__main__":
    main()
