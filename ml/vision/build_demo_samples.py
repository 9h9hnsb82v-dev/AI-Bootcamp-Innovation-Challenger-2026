"""이미지 검사 'AI-Hub 시험 샘플' 모드용 샘플 세트 (사용자 승인 2026-11-14, 팀원 화면 장점 1번).

- 출처: AI-Hub 공식 Validation 중 서비스 시험셋(test 1,080장) — 어느 모델의 학습·임계값 결정에도 쓰지 않은 사진.
- 표면 촬영: 부품 x 양품/불량 클래스마다 SAMPLES_PER_CLASS장(seed 42), 긴 변 1280px.
- 휀더 틈 촬영: 단차 test 사진에서 틈 박스 중심, 한 변 = 3 x 박스 긴 변으로 잘라 '가이드 박스 가운데 틈' 촬영을 재현(양품·불량 각 GAP_SAMPLES장).
- 원본 해상도 사진이 필요하므로 원천 tar가 있는 PC에서 만들고, 결과 폴더(data/demo/aihub_samples)를 앱과 함께 배포한다.
출력: data/demo/aihub_samples/<file>.jpg, index.csv (file, part, quality, subtype, shot, stem)
실행: .venv/Scripts/python -m ml.vision.build_demo_samples
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
from PIL import Image

from ml.vision.audit_aihub_labels import AIHUB_DIR
from ml.vision.imageio import open_oriented
from ml.vision.prepare_subset import IMG_TARS, ROOT

OUT = ROOT / "data/demo/aihub_samples"
SAMPLES_PER_CLASS = 15
GAP_SAMPLES = 10
SEED = 42


def main() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data/processed/aihub_full/manifest.csv", encoding="utf-8-sig")
    t = m[m.subset == "test"].copy()
    t["subtype"] = t.json_member.str.split("/").str[1]
    pick = t.groupby("class").sample(SAMPLES_PER_CLASS, random_state=SEED)
    gap = pd.read_csv(ROOT / "data/processed/aihub_gap/manifest.csv", encoding="utf-8-sig")
    gap = gap[gap.subset == "test"].groupby("quality").sample(GAP_SAMPLES, random_state=SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    want = {s: ("surface", r) for s, r in pick.set_index("stem").iterrows()}
    rows = []
    for (sp, part), rel in IMG_TARS.items():
        if sp != "val":
            continue
        todo = {s: v for s, v in want.items() if v[1].part == part}
        gtodo = {s: r for s, r in gap.set_index("stem").iterrows()} if part == "FENDER" else {}
        with tarfile.open(AIHUB_DIR / rel) as tf:
            for mem in tf:
                st = Path(mem.name).stem.lower()
                if st not in todo and st not in gtodo:
                    continue
                im = open_oriented(tf.extractfile(mem).read(), 2048)
                if st in todo:
                    r = todo.pop(st)[1]
                    s = im.copy()
                    s.thumbnail((1280, 1280))
                    f = f"{part.lower()}_{r.quality.lower()}_{st[:20]}.jpg"
                    s.save(OUT / f, quality=90)
                    rows.append({"file": f, "part": part, "quality": r.quality, "subtype": r.subtype, "shot": "surface", "stem": st})
                if st in gtodo:
                    g = gtodo.pop(st)
                    k = im.width / g.img_w  # full_* 좌표는 긴 변 1536 축척 → 원본 → 현재 디코딩 축척
                    k0 = g.img_w / max(g.img_w, g.img_h) * 0 + max(g.img_w, g.img_h) / 1536
                    cx, cy = (g.full_x + g.full_w / 2) * k0 * k, (g.full_y + g.full_h / 2) * k0 * k
                    side = 3 * max(g.full_w, g.full_h) * k0 * k
                    canvas = Image.new("RGB", (round(side), round(side)), (128, 128, 128))
                    x0, y0 = cx - side / 2, cy - side / 2
                    canvas.paste(im.crop((max(0, round(x0)), max(0, round(y0)), min(im.width, round(x0 + side)),
                                          min(im.height, round(y0 + side)))), (max(0, -round(x0)), max(0, -round(y0))))
                    canvas.thumbnail((1280, 1280))
                    f = f"fender_gap_{g.quality.lower()}_{st[:20]}.jpg"
                    canvas.save(OUT / f, quality=90)
                    rows.append({"file": f, "part": "FENDER", "quality": g.quality, "subtype": "단차", "shot": "gap", "stem": st})
                if not todo and not gtodo:
                    break
    idx = pd.DataFrame(rows).sort_values(["part", "shot", "quality", "file"])
    idx.to_csv(OUT / "index.csv", index=False, encoding="utf-8-sig")
    print(idx.groupby(["part", "shot", "quality"]).size().to_string())
    return idx


if __name__ == "__main__":
    main()
