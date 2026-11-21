"""AI-Hub 도어·범퍼·휀더 원본 전체(51,986장)를 긴 변 640px로 추출한다 (원본 tar는 수정하지 않음).

분할 (기존 서비스 모델과 같은 시험 기준으로 비교하기 위함)
- 공식 Training(46,207) → 부품별 촬영일 그룹 단위로 약 20%를 val, 나머지 train (resplit_821과 같은 규칙)
- 공식 Validation 중 기존 시험셋 1,080장(클래스당 180) → test (그대로 고정)
- 공식 Validation 나머지(약 4,700장) → holdout (학습·임계값 결정에 쓰지 않는 추가 시험셋)
  공식 Validation은 기존 모델도 학습에 쓰지 않았으므로 test+holdout 전체로 모든 모델을 공정하게 비교할 수 있다.
- 라벨: JSON attributes.quality (양품=GOOD, 불량품=DEFECT). 640px는 448px 학습 시 짧은 변(480)이 448 이상이 되게 고른 값.

출력: data/processed/aihub_full/images/<subset>/<PART_QUALITY>/<stem>.jpg, manifest.csv, split_stats.json
이미 있는 파일은 건너뛴다(중단 후 재실행 가능).
실행: .venv/Scripts/python -m ml.vision.prepare_full [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
import pandas as pd

from ml.vision.audit_aihub_labels import AIHUB_DIR, OUT_INDEX, ROOT
from ml.vision.prepare_subset import IMG_TARS, MANIFEST as OLD_MANIFEST, quality_class

OUT = ROOT / "data/processed/aihub_full"
LONG_SIDE = 640
SEED = 42
VAL_FRAC = 0.20


def split(index: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    d = index.copy()
    d["qclass"] = d["quality"].map(quality_class)
    d = d.dropna(subset=["qclass"]).copy()
    d["stem"] = d["file_name"].str.rsplit(".", n=1).str[0].str.lower()
    d["capture_day"] = d["date_captured"].astype(str).str[:10]
    d["class"] = d["part"] + "_" + d["qclass"]
    d["subset"] = np.where(d.split == "train", "train", "holdout")
    tr = d.split == "train"
    for part, g in d[tr].groupby("part"):
        days = np.array(sorted(g.capture_day.unique()))
        days = days[rng.permutation(len(days))]
        counts = g.capture_day.value_counts()
        target, acc, val_days = VAL_FRAC * len(g), 0, []
        for day in days:
            if acc + counts[day] <= target:
                val_days.append(day)
                acc += counts[day]
        rest = [x for x in days if x not in val_days]
        if rest:
            best = min(rest, key=lambda x: abs(acc + counts[x] - target))
            if abs(acc + counts[best] - target) < abs(acc - target):
                val_days.append(best)
        d.loc[tr & (d.part == part) & d.capture_day.isin(val_days), "subset"] = "val"
    old = pd.read_csv(OLD_MANIFEST, encoding="utf-8-sig")
    test_stems = set(old.loc[old.subset == "test", "path"].str.rsplit("/", n=1).str[-1].str[:-4].str.lower())
    d.loc[(d.split == "val") & d.stem.isin(test_stems), "subset"] = "test"
    assert (d.subset == "test").sum() == len(test_stems), "기존 시험셋 1,080장을 모두 찾지 못함"
    return d


def _convert(data: bytes, dst: str) -> str:
    from ml.vision.imageio import open_oriented
    im = open_oriented(data, LONG_SIDE)
    im.thumbnail((LONG_SIDE, LONG_SIDE))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=92)
    return dst


def main(workers: int) -> dict:
    d = split(pd.read_csv(OUT_INDEX, encoding="utf-8-sig"))
    d["path"] = [f"data/processed/aihub_full/images/{s}/{c}/{st}.jpg" for s, c, st in zip(d.subset, d["class"], d.stem)]
    stats = {"counts": d.subset.value_counts().to_dict(),
             "by_class": d.groupby(["subset", "class"]).size().unstack(0).fillna(0).astype(int).to_dict(),
             "long_side": LONG_SIDE, "seed": SEED}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats["counts"]), flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for (sp, part), rel in IMG_TARS.items():
            want = d[(d.split == sp) & (d.part == part)].set_index("stem")
            todo = {st: ROOT / p for st, p in want.path.items() if not (ROOT / p).exists()}
            print(f"{rel}: 대상 {len(want)}, 남은 {len(todo)}", flush=True)
            if not todo:
                continue
            pending, done = set(), 0
            with tarfile.open(AIHUB_DIR / rel, "r:") as t:
                for m in t:
                    st = Path(m.name).stem.lower()
                    if not m.isfile() or st not in todo:
                        continue
                    pending.add(ex.submit(_convert, t.extractfile(m).read(), str(todo.pop(st))))
                    if len(pending) >= workers * 4:  # 메모리 보호: 대기 작업 수 제한
                        fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in fin:
                            f.result()
                        done += len(fin)
                        if done % 2000 < len(fin):
                            print(f"  {rel}: {done}", flush=True)
                    if not todo:
                        break
            for f in pending:
                f.result()
            print(f"{rel}: 완료, 미발견 {len(todo)}", flush=True)
    missing = [p for p in d.path if not (ROOT / p).exists()]
    d[["subset", "class", "part", "qclass", "stem", "path", "capture_day", "json_member", "split"]].rename(
        columns={"qclass": "quality", "split": "official_split"}).to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")
    stats["missing"] = len(missing)
    (OUT / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print("missing", len(missing), "manifest", len(d))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    main(ap.parse_args().workers)
