"""평가셋 촬영 세션 겹침 감사 (2026-11-14, 팀원 앱 '근거·성능' 지적으로 확인).

AI-Hub 공식 Training(학습·검증)과 공식 Validation(시험·holdout)이 같은 촬영 세션에서 나뉘어 있다.
→ 시험 점수는 '처음 보는 차량'보다 높게 나올 수 있다(모델 간 상대 비교는 같은 조건이라 유효).
기준: 같은 촬영일 / 같은 부품의 학습 사진과 60초 이내 촬영 / 촬영 시각 완전 동일.
출력: reports/vision/session_overlap.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml.vision.prepare_subset import ROOT


def main() -> dict:
    m = pd.read_csv(ROOT / "data/processed/aihub_full/manifest.csv", encoding="utf-8-sig")
    li = pd.read_csv(ROOT / "data/processed/aihub/label_index.csv", encoding="utf-8-sig")
    li["stem"] = li.file_name.str.rsplit(".", n=1).str[0].str.lower()
    m = m.merge(li[["stem", "date_captured"]], on="stem", how="left")
    m["ts"] = pd.to_datetime(m.date_captured, errors="coerce")
    tr = m[m.subset.isin(["train", "val"])]
    out = {"rule": "60초 이내 = 같은 부품 학습 사진과 촬영 간격 60초 이하", "sets": {}}
    for sub in ("test", "holdout"):
        t = m[m.subset == sub]
        near = []
        for part, g in t.groupby("part"):
            ref = np.sort(tr[tr.part == part].ts.dropna().values)
            x = g.ts.values
            i = np.searchsorted(ref, x)
            d = np.minimum(np.abs(x - ref[np.clip(i - 1, 0, len(ref) - 1)]), np.abs(ref[np.clip(i, 0, len(ref) - 1)] - x))
            near += list(d <= np.timedelta64(60, "s"))
        out["sets"][sub] = {"n": int(len(t)), "same_day": float(t.capture_day.isin(set(tr.capture_day)).mean()),
                            "within_60s": float(np.mean(near)),
                            "identical_timestamp": float(t.date_captured.isin(set(tr.date_captured)).mean())}
    out["conclusion"] = ("시험·holdout 사진의 약 98%가 학습 사진과 같은 촬영 세션(60초 이내)이다. 보고한 절대 정확도는 낙관적일 수 있고, "
                         "모델 간 비교(같은 시험셋)는 유효하다. 세션이 겹치지 않는 시험셋으로 재평가가 필요하다.")
    (ROOT / "reports/vision/session_overlap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    main()
