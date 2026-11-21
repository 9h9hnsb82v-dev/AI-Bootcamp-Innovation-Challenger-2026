"""1단계(차량 전체 사진 → 부품 영역) 분할 데이터 준비.

- 학습/검증/시험: Humans in the Loop "Car parts" (CC0 1.0, 998장, 21클래스, 폴리곤)
  * 주의: 배포 zip의 폴더 이름이 뒤바뀌어 있다. 부품 라벨은 "Car damages dataset/" 폴더에 있다(meta.json 클래스로 판별).
  * 8:2:1 무작위 분할(seed 42). 지각 해시(dHash)가 같은 사진은 같은 분할로 묶는다.
- 외부 검증: Ultralytics carparts-seg (CC BY 4.0, Gianmarco Russo/Roboflow) 원본 585장 중 1장씩.
  * 휀더 라벨이 없으므로 학습에는 쓰지 않고, 범퍼·도어 클래스만 외부 평가한다.
출력: data/processed/partseg/{images/<split>/*.jpg, ann_<split>.json, classes.json, stats.json}
  ann JSON: [{"file", "width", "height", "objects": [{"label", "exterior": [[x,y]...], "interiors": [[[x,y]...]]}]}]
실행: .venv/Scripts/python -m ml.vision.prepare_partseg
"""
from __future__ import annotations

import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
HITL_ZIP = ROOT / "data/raw/hitl_carparts/Car parts and car damages dataset.zip"
CARPARTS_ZIP = ROOT / "data/raw/carparts_seg/carparts-seg.zip"
OUT = ROOT / "data/processed/partseg"
LONG_SIDE = 1024
SEED = 42

# 서비스 품목 매핑(1단계 결과 → 생산계획 품목)
SERVICE_PARTS = {"DOOR": ["Front-door", "Back-door"], "BUMPER": ["Front-bumper", "Back-bumper"], "FENDER": ["Fender"]}
# carparts-seg 클래스 → HITL 클래스(외부 평가용, 대응이 명확한 것만)
CARPARTS_MAP = {"front_bumper": "Front-bumper", "back_bumper": "Back-bumper",
                "front_door": "Front-door", "front_left_door": "Front-door", "front_right_door": "Front-door",
                "back_door": "Back-door", "back_left_door": "Back-door", "back_right_door": "Back-door"}


def dhash(im: Image.Image, n: int = 8) -> int:
    g = im.convert("L").resize((n + 1, n), Image.BILINEAR)
    px = list(g.getdata())
    return sum(1 << i for i in range(n * n) if px[(i // n) * (n + 1) + i % n] > px[(i // n) * (n + 1) + i % n + 1])


def save_resized(im: Image.Image, path: Path) -> float:
    s = min(1.0, LONG_SIDE / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=92)
    return s


def hitl_parts_folder(z: zipfile.ZipFile) -> str:
    for d in ("Car damages dataset", "Car parts dataset"):
        titles = {c["title"] for c in json.loads(z.read(f"{d}/meta.json"))["classes"]}
        if "Fender" in titles:
            return d
    raise RuntimeError("부품 클래스(Fender)가 있는 폴더를 찾지 못함")


def prepare_hitl() -> tuple[list[str], dict]:
    z = zipfile.ZipFile(HITL_ZIP)
    d = hitl_parts_folder(z)
    classes = sorted(c["title"] for c in json.loads(z.read(f"{d}/meta.json"))["classes"])
    items = []
    for m in sorted(n for n in z.namelist() if n.startswith(f"{d}/File1/ann/") and n.endswith(".json")):
        a = json.loads(z.read(m))
        img_name = m.split("/")[-1][:-5]
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(f"{d}/File1/img/{img_name}")))).convert("RGB")
        assert im.size == (a["size"]["width"], a["size"]["height"]), f"크기 불일치 {img_name}"
        items.append((img_name, im, a["objects"], dhash(im)))
    # 동일 dHash 그룹 → 같은 분할
    groups = defaultdict(list)
    for it in items:
        groups[it[3]].append(it)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    n = len(keys)
    # 8:2:1 (사용자 결정 2026-11-11) = 학습 8/11, 검증 2/11, 시험 1/11
    split_of = {k: ("train" if i < 8 / 11 * n else "val" if i < 10 / 11 * n else "test") for i, k in enumerate(keys)}
    anns = defaultdict(list)
    stats = Counter()
    for k in keys:
        for img_name, im, objs, _ in groups[k]:
            sp = split_of[k]
            stem = re.sub(r"[^0-9A-Za-z]+", "_", Path(img_name).stem).strip("_")
            s = save_resized(im, OUT / "images" / sp / f"{stem}.jpg")
            rec = {"file": f"images/{sp}/{stem}.jpg", "width": round(im.width * s) if s < 1 else im.width,
                   "height": round(im.height * s) if s < 1 else im.height, "source": "HITL", "objects": []}
            for o in objs:
                ext = [[x * s, y * s] for x, y in o["points"]["exterior"]]
                if len(ext) < 3:
                    continue
                rec["objects"].append({"label": o["classTitle"], "exterior": ext,
                                       "interiors": [[[x * s, y * s] for x, y in h] for h in o["points"]["interior"]]})
                stats[f"{sp}/{o['classTitle']}"] += 1
            anns[sp].append(rec)
            stats[f"img/{sp}"] += 1
    stats["dhash_groups_multi"] = sum(len(v) > 1 for v in groups.values())
    for sp, recs in anns.items():
        (OUT / f"ann_{sp}.json").write_text(json.dumps(recs), encoding="utf-8")
    return classes, dict(stats)


# 정답 라벨 위치로 사진 방향 판정: (위쪽 부품, 아래쪽 부품) 쌍이 모두 세로 방향 위→아래면 '똑바름'
UPRIGHT_PAIRS = [("front_glass", "front_bumper"), ("back_glass", "back_bumper"), ("hood", "front_bumper"),
                 ("front_glass", "hood"), ("front_glass", "front_light"), ("back_glass", "back_light"),
                 ("front_door", "wheel"), ("back_door", "wheel"), ("left_mirror", "wheel"), ("right_mirror", "wheel")]


def is_upright(label_txt: str, names: list[str]) -> bool:
    import numpy as np
    c = defaultdict(list)
    for line in label_txt.splitlines():
        v = line.split()
        if len(v) >= 7:
            c[names[int(v[0])]].append(np.array(list(map(float, v[1:]))).reshape(-1, 2).mean(0))
    c = {k: np.mean(v, 0) for k, v in c.items()}
    votes = []
    for u, d in UPRIGHT_PAIRS:
        if u in c and d in c:
            dx, dy = c[d] - c[u]
            votes.append(bool(dy > 0 and abs(dy) > abs(dx)))
    return bool(votes) and all(votes)


def prepare_carparts_external(mode: str = "upright") -> dict:
    """carparts-seg 원본별 1장(증강 사본 중 선택), YOLO 폴리곤(정규화) → 절대 좌표, 매핑 클래스만.

    mode="upright": 정답 라벨 위치로 판정한 똑바른 사본만(없으면 그 원본 제외) → ext_carparts (대표 지표, 서비스 조건)
    mode="first":   이름순 첫 사본(회전 증강 포함 가능) → ext_carparts_any (회전 강건성 참고)
    """
    import yaml
    z = zipfile.ZipFile(CARPARTS_ZIP)
    names = yaml.safe_load(z.read("carparts-seg.yaml"))["names"]
    split_name = "ext_carparts" if mode == "upright" else "ext_carparts_any"
    by_base = defaultdict(list)
    for m in sorted(n for n in z.namelist() if n.startswith("labels/") and n.endswith(".txt")):
        by_base[re.sub(r"\.rf\.[0-9a-f]+\.txt$", "", m.split("/")[-1])].append(m)
    chosen = []
    for base, ms in sorted(by_base.items()):
        if mode == "upright":
            ms = [m for m in ms if is_upright(z.read(m).decode(), names)]
        if ms:
            chosen.append((base, ms[0]))
    recs, stats = [], Counter()
    for base, m in chosen:
        sp = m.split("/")[1]
        img_m = f"images/{sp}/{m.split('/')[-1][:-4]}.jpg"
        if img_m not in z.NameToInfo:
            continue
        im = Image.open(io.BytesIO(z.read(img_m))).convert("RGB")
        stem = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_")
        s = save_resized(im, OUT / "images" / split_name / f"{stem}.jpg")
        W, H = round(im.width * min(1, s)), round(im.height * min(1, s))
        rec = {"file": f"images/{split_name}/{stem}.jpg", "width": W, "height": H, "source": "carparts-seg", "objects": []}
        for line in z.read(m).decode().splitlines():
            v = line.split()
            if len(v) < 7 or names[int(v[0])] not in CARPARTS_MAP:
                continue
            xy = list(map(float, v[1:]))
            rec["objects"].append({"label": CARPARTS_MAP[names[int(v[0])]],
                                   "exterior": [[xy[i] * W, xy[i + 1] * H] for i in range(0, len(xy), 2)], "interiors": []})
            stats[CARPARTS_MAP[names[int(v[0])]]] += 1
        recs.append(rec)
    (OUT / f"ann_{split_name}.json").write_text(json.dumps(recs), encoding="utf-8")
    stats["images"] = len(recs)
    stats["bases_total"] = len(by_base)
    return dict(stats)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    classes, st = prepare_hitl()
    ext = {"upright": prepare_carparts_external("upright"), "any": prepare_carparts_external("first")}
    (OUT / "classes.json").write_text(json.dumps({"classes": classes, "service_parts": SERVICE_PARTS,
                                                  "external_eval_classes": sorted(set(CARPARTS_MAP.values()))},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {"hitl": st, "ext_carparts": ext, "long_side": LONG_SIDE, "seed": SEED,
             "licenses": {"HITL": "CC0 1.0", "carparts-seg": "CC BY 4.0 (Gianmarco Russo / Roboflow, via Ultralytics)"}}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


if __name__ == "__main__":
    main()
