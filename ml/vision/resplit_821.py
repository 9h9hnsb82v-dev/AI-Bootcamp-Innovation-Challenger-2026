"""AI-Hub 근접 사진 서브셋을 8:2:1(학습 72.7 : 검증 18.2 : 시험 9.1)로 재분할 (사용자 결정 2026-11-11).

- 이미지 파일은 그대로 두고 manifest.csv 의 subset 만 다시 지정한다(경로 path·orig_subset 유지).
- 공식 Training(10,800) → 부품별 촬영일 그룹 단위로 약 20%를 val, 나머지 train (같은 날 사진이 train/val에 섞이지 않게)
- 공식 Validation → 클래스당 180장 test (seed 42), 나머지는 'unused' (어디에도 쓰지 않음)
- 이전 분할은 manifest_split_v1.csv 로 보관
실행: .venv/Scripts/python -m ml.vision.resplit_821
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from ml.vision.prepare_subset import MANIFEST, ROOT

SEED = 42
VAL_FRAC_OF_TRAINING = 0.20  # train:val = 8:2
TEST_PER_CLASS = 180         # 6 x 180 = 1,080 = 9.1% of 11,880
BACKUP = MANIFEST.with_name("manifest_split_v1.csv")


def resplit(m: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    m = m.copy()
    if "orig_subset" not in m:
        m["orig_subset"] = m["subset"]  # 파일 위치(path)를 만든 원래 분할
    official_train = m.source_tar.str.startswith("1.Training")
    m.loc[official_train, "subset"] = "train"
    for part, g in m[official_train].groupby("part"):
        days = np.array(sorted(g.capture_day.astype(str).unique()))
        days = days[rng.permutation(len(days))]
        counts = g.capture_day.astype(str).value_counts()
        target, acc, val_days = VAL_FRAC_OF_TRAINING * len(g), 0, []
        for d in days:  # 촬영일을 섞은 순서로 보며 목표를 넘지 않는 날만 val에 추가
            if acc + counts[d] <= target:
                val_days.append(d)
                acc += counts[d]
        rest = [d for d in days if d not in val_days]  # 남은 차이는 목표에 가장 가깝게 만드는 하루로 보정
        if rest:
            best = min(rest, key=lambda d: abs(acc + counts[d] - target))
            if abs(acc + counts[best] - target) < abs(acc - target):
                val_days.append(best)
                acc += counts[best]
        m.loc[official_train & (m.part == part) & m.capture_day.astype(str).isin(val_days), "subset"] = "val"
    m.loc[~official_train, "subset"] = "unused"
    for cls, g in m[~official_train].groupby("class"):
        pick = g.index[rng.permutation(len(g))[:TEST_PER_CLASS]]
        m.loc[pick, "subset"] = "test"
    return m


def main() -> dict:
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    if not BACKUP.exists():
        shutil.copy(MANIFEST, BACKUP)
    base = pd.read_csv(BACKUP, encoding="utf-8-sig")
    new = resplit(base)
    new.to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    used = new[new.subset != "unused"]
    n = used.subset.value_counts()
    stats = {"counts": n.to_dict(), "ratio_pct": (n / n.sum() * 100).round(1).to_dict(),
             "by_class": used.groupby(["subset", "class"]).size().unstack(0).to_dict(),
             "unused": int((new.subset == "unused").sum()), "backup": BACKUP.relative_to(ROOT).as_posix()}
    (ROOT / "reports/audit/aihub_split_821.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("counts", "ratio_pct", "unused")}, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
