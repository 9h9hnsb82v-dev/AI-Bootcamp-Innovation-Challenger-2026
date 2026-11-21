"""촬영 세션이 겹치지 않는 재분할 'sd' (사용자 승인 2026-11-14: 세션 비겹침 재평가, 검출 모델 포함).

배경: 공식 Training/Validation 분할은 같은 촬영 세션에서 나뉘어 시험 사진의 약 98%가 학습 사진과 60초 이내 촬영
(reports/vision/session_overlap.json). → 공식 분할을 버리고 51,986장 전체를 '촬영일 통째'로 나눈다.
- 촬영일(capture_day) 114일을 무작위로 섞어(seed 42) 앞에서부터 시험 ≈10%, 검증 ≈15%, 나머지 학습.
  부품x품질 6클래스가 시험·검증에 각각 일정 비율 이상 들어가도록 확인(부족하면 날짜를 더 넣음).
- 같은 날짜는 한 분할에만 속한다(부품이 달라도). 자정을 넘는 세션 대비로 분할 간 60초 이내 사진 수를 함께 보고.
출력: data/processed/aihub_full/manifest_sd.csv(subset 교체), data/processed/aihub_gap/manifest_sd.csv,
      data/processed/aihub_full/sd_{train,val,test}.txt(검출용 이미지 목록, 데이터 폴더 기준 상대경로), reports/vision/split_sd.json
실행: .venv/Scripts/python -m ml.vision.resplit_session
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml.vision.prepare_subset import ROOT

FULL = ROOT / "data/processed/aihub_full"
GAP = ROOT / "data/processed/aihub_gap"
SEED = 42
TEST_FRAC, VAL_FRAC, MIN_CLASS_FRAC = 0.10, 0.15, 0.06


def assign(m: pd.DataFrame) -> dict[str, str]:
    rng = np.random.default_rng(SEED)
    days = np.array(sorted(m.capture_day.astype(str).unique()))
    days = list(days[rng.permutation(len(days))])
    cls_n = m["class"].value_counts()
    out, i = {}, 0
    for name, frac in (("test", TEST_FRAC), ("val", VAL_FRAC)):
        picked = []
        while i < len(days):
            picked.append(days[i])
            i += 1
            sub = m[m.capture_day.astype(str).isin(picked)]
            share = sub["class"].value_counts().reindex(cls_n.index, fill_value=0) / cls_n
            if len(sub) >= frac * len(m) and share.min() >= MIN_CLASS_FRAC:
                break
        out.update({d: name for d in picked})
    out.update({d: "train" for d in days[i:]})
    return out


def cross_60s(m: pd.DataFrame) -> dict:
    li = pd.read_csv(ROOT / "data/processed/aihub/label_index.csv", encoding="utf-8-sig")
    li["stem"] = li.file_name.str.rsplit(".", n=1).str[0].str.lower()
    x = m.merge(li[["stem", "date_captured"]], on="stem", how="left")
    x["ts"] = pd.to_datetime(x.date_captured, errors="coerce")
    tr = x[x.subset != "test"]
    res = {}
    for sub in ("test",):
        t = x[x.subset == sub]
        near = []
        for part, g in t.groupby("part"):
            ref = np.sort(tr[tr.part == part].ts.dropna().values)
            v = g.ts.values
            j = np.searchsorted(ref, v)
            d = np.minimum(np.abs(v - ref[np.clip(j - 1, 0, len(ref) - 1)]), np.abs(ref[np.clip(j, 0, len(ref) - 1)] - v))
            near += list(d <= np.timedelta64(60, "s"))
        res[sub] = float(np.mean(near))
    return res


def main() -> dict:
    m = pd.read_csv(FULL / "manifest.csv", encoding="utf-8-sig")
    day2 = assign(m)
    m["subset"] = m.capture_day.astype(str).map(day2)
    m.to_csv(FULL / "manifest_sd.csv", index=False, encoding="utf-8-sig")
    rel = m.path.str.replace("data/processed/aihub_full/", "", regex=False)
    for s in ("train", "val", "test"):
        (FULL / f"sd_{s}.txt").write_text("\n".join(rel[m.subset == s]) + "\n", encoding="utf-8")
    g = pd.read_csv(GAP / "manifest.csv", encoding="utf-8-sig")
    g["subset"] = g.capture_day.astype(str).map(day2)
    g.to_csv(GAP / "manifest_sd.csv", index=False, encoding="utf-8-sig")
    stats = {"rule": "촬영일 통째 분할(같은 날짜는 한 분할에만), seed 42, 시험 >=10%·검증 >=15%, 각 클래스 >=6%",
             "days": {s: int(sum(1 for v in day2.values() if v == s)) for s in ("train", "val", "test")},
             "images": m.subset.value_counts().to_dict(),
             "by_class": m.groupby(["subset", "class"]).size().unstack(0).to_dict(),
             "gap_images": g.subset.value_counts().to_dict(),
             "test_within_60s_of_trainval": cross_60s(m)["test"],
             "test_from_official_training": float((m[m.subset == "test"].official_split == "train").mean())}
    (ROOT / "reports/vision/split_sd.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


if __name__ == "__main__":
    main()
