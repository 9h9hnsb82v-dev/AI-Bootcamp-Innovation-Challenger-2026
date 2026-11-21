"""휀더 단차 전용 모델용 데이터: 원본 해상도에서 단차 박스(category 203) 주변을 잘라 저장한다 (2026-11-13, 사용자 요청).

- 대상: aihub_full manifest의 휀더 '단차' 사진 전체(공식 Training 13,298 + 공식 Validation 1,663). 사진마다 단차 박스가 정확히 1개이며
  박스 품질(양품/불량품)은 사진 품질과 같다(reports/vision/gap/gap_boxes.csv 측정).
- 분할은 aihub_full과 동일(train/val/test/holdout) → 기존 모델과 같은 평가셋으로 비교 가능.
- crops/: 박스 중심, 한 변 = 2 × 박스 긴 변(맥락 포함, 학습 시 위치·크기 흔들기용) → 768px 정사각형.
  원본 경계에 닿으면 잘린 부분은 회색으로 채운다. 박스 중심의 crop 내 좌표(cx, cy)와 박스 긴 변(L, crop 픽셀)을 manifest에 기록.
- full/: 평가셋(val/test/holdout)만 원본 전체를 긴 변 1536px로 저장(위치를 모를 때 조각 탐색 평가용) + 박스 좌표(같은 축척).
실행: .venv/Scripts/python -m ml.vision.prepare_gap_crops [--workers 12]
"""
from __future__ import annotations

import argparse
import tarfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import pandas as pd

from ml.vision.audit_aihub_labels import AIHUB_DIR
from ml.vision.prepare_subset import IMG_TARS, ROOT

OUT = ROOT / "data/processed/aihub_gap"
CROP = 768
FULL = 1536
CTX = 2.0  # crop 한 변 = CTX × 박스 긴 변


def _work(data: bytes, box: dict, crop_dst: str, full_dst: str | None) -> dict:
    from PIL import Image
    from ml.vision.imageio import open_oriented
    im = open_oriented(data)
    iw, ih = im.size
    sx, sy = iw / box["W"], ih / box["H"]  # 라벨 좌표계(EXIF 적용 화면 기준) → 실제 픽셀
    x, y, w, h = box["x"] * sx, box["y"] * sy, box["w"] * sx, box["h"] * sy
    cx, cy, L = x + w / 2, y + h / 2, max(w, h)
    side = CTX * L
    x0, y0 = cx - side / 2, cy - side / 2
    canvas = Image.new("RGB", (round(side), round(side)), (128, 128, 128))
    src = im.crop((max(0, round(x0)), max(0, round(y0)), min(iw, round(x0 + side)), min(ih, round(y0 + side))))
    canvas.paste(src, (max(0, -round(x0)), max(0, -round(y0))))
    canvas = canvas.resize((CROP, CROP), Image.BICUBIC)
    Path(crop_dst).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(crop_dst, quality=93)
    out = {"crop_cx": CROP / 2, "crop_cy": CROP / 2, "crop_L": L * CROP / side, "img_w": iw, "img_h": ih}
    if full_dst:
        k = FULL / max(iw, ih)
        f = im.resize((round(iw * k), round(ih * k)), Image.BICUBIC)
        Path(full_dst).parent.mkdir(parents=True, exist_ok=True)
        f.save(full_dst, quality=93)
        out.update(full_x=x * k, full_y=y * k, full_w=w * k, full_h=h * k)
    return out


def main(workers: int) -> None:
    m = pd.read_csv(ROOT / "data/processed/aihub_full/manifest.csv", encoding="utf-8-sig")
    g = m[(m.part == "FENDER") & m.json_member.str.contains("/단차/")].copy()
    boxes = pd.read_csv(ROOT / "reports/vision/gap/gap_boxes.csv", encoding="utf-8-sig")
    g = g.merge(boxes[["json_member", "W", "H", "x", "y", "w", "h"]], on="json_member", how="inner", validate="1:1")
    g["crop_path"] = [f"data/processed/aihub_gap/crops/{s}/{q}/{st}.jpg" for s, q, st in zip(g.subset, g.quality, g.stem)]
    g["full_path"] = [f"data/processed/aihub_gap/full/{s}/{q}/{st}.jpg" if s != "train" else ""
                      for s, q, st in zip(g.subset, g.quality, g.stem)]
    print(g.subset.value_counts().to_dict(), flush=True)
    results = {}
    with ProcessPoolExecutor(workers) as ex:
        for sp, rel in ((("train", "FENDER"), IMG_TARS[("train", "FENDER")]), (("val", "FENDER"), IMG_TARS[("val", "FENDER")])):
            want = g[g.official_split == sp[0]].set_index("stem")
            todo = {st: r for st, r in want.iterrows()}
            pending, done = {}, 0
            with tarfile.open(AIHUB_DIR / rel, "r:") as t:
                for mem in t:
                    st = Path(mem.name).stem.lower()
                    if not mem.isfile() or st not in todo:
                        continue
                    r = todo.pop(st)
                    fut = ex.submit(_work, t.extractfile(mem).read(), {k: r[k] for k in ("W", "H", "x", "y", "w", "h")},
                                    str(ROOT / r.crop_path), str(ROOT / r.full_path) if r.full_path else None)
                    pending[fut] = st
                    if len(pending) >= workers * 3:
                        fin, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for f in fin:
                            results[pending.pop(f)] = f.result()
                        done += len(fin)
                        if done % 2000 < len(fin):
                            print(f"  {rel}: {done}", flush=True)
                    if not todo:
                        break
            for f, st in pending.items():
                results[st] = f.result()
            print(f"{rel}: 완료, 미발견 {len(todo)}", flush=True)
    info = pd.DataFrame.from_dict(results, orient="index")
    g = g.set_index("stem").join(info, how="inner").reset_index()
    cols = ["subset", "quality", "stem", "crop_path", "full_path", "crop_cx", "crop_cy", "crop_L",
            "full_x", "full_y", "full_w", "full_h", "img_w", "img_h", "capture_day", "official_split"]
    g[cols].to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")
    print("manifest", len(g), g.groupby(["subset", "quality"]).size().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    main(ap.parse_args().workers)
