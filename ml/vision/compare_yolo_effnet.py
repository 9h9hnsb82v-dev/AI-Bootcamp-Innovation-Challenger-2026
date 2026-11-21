"""EfficientNet-B0(v2) vs YOLOv7 이미지 단위 결함 판정 비교 (같은 시험셋, 같은 규칙).

규칙
- 부품(계획 품목)을 안다고 가정한 이미지 단위 이진 판정(양품/불량)
- EfficientNet 점수: P(part_DEFECT)/P(part)   · YOLO 점수: 결함 클래스(scratch/exterior_damage/gap_defect) 최대 신뢰도
- 임계값: 부품별로 검증셋 Youden J 최대값 → 시험셋 1회 평가 (YOLO 기본 conf 0.25 결과도 병기)
- 지표: 재현율·특이도·정밀도·AUC, 휀더 하위유형(단차/외관 손상)
실행: .venv/Scripts/python -m ml.vision.compare_yolo_effnet <yolo_scores.csv> [--yolo-meta <json>]
출력: reports/vision/yolo_vs_effnet.json, .md
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ml.vision.prepare_subset import ROOT
from ml.vision.train_fender import binary_metrics

YOLO_DIR = ROOT / "data/processed/aihub/yolo"
OUT = ROOT / "reports/vision"
GRID = np.linspace(0.01, 0.99, 99)


def youden(p, y):
    return float(GRID[np.argmax([binary_metrics(p, y, t)["recall"] + binary_metrics(p, y, t)["specificity"] for t in GRID])])


def effnet_scores(idx: pd.DataFrame) -> np.ndarray:
    from ml.vision.infer import predict
    return np.array([predict((YOLO_DIR / "images" / s / f"{st}.jpg").read_bytes(), expected_part=p,
                             with_cam=False, backend_name="effnet")["defect_prob"] for s, st, p in zip(idx.subset, idx.stem, idx.part)])


def evaluate(df: pd.DataFrame, col: str, fixed: float | None = None) -> dict:
    out = {}
    for part, g in df.groupby("part"):
        v, t = g[g.subset == "val"], g[g.subset == "test"]
        th = fixed if fixed is not None else youden(v[col].to_numpy(), v.y.to_numpy())
        r = {"threshold": th, "test": binary_metrics(t[col].to_numpy(), t.y.to_numpy(), th)}
        if part == "FENDER":
            r["test_by_subtype"] = {s: binary_metrics(x[col].to_numpy(), x.y.to_numpy(), th) for s, x in t.groupby("subtype")}
        out[part] = r
    return out


def main(scores_csv: str, yolo_meta: str | None) -> dict:
    from ml.vision.infer import META
    eff = "EfficientNet-B0 " + json.loads(META.read_text(encoding="utf-8"))["model_version"]
    idx = pd.read_csv(YOLO_DIR / "index.csv", encoding="utf-8-sig")
    idx = idx[idx.subset.isin(["val", "test"])].copy()
    ys = pd.read_csv(scores_csv)
    df = idx.merge(ys, on=["subset", "stem"], how="left", validate="1:1")
    assert df.n_det.notna().all(), "YOLO 점수 누락 이미지"
    df["y"] = (df.quality == "DEFECT").astype(int)
    df["yolo"] = df[["max_conf_scratch", "max_conf_exterior_damage", "max_conf_gap_defect"]].max(axis=1)
    df["effnet"] = effnet_scores(df)
    res = {"protocol": "per-part image-level; threshold = val Youden J; test evaluated once",
           "test_n": int((df.subset == "test").sum()), "val_n": int((df.subset == "val").sum()),
           eff: evaluate(df, "effnet"),
           "YOLOv7": evaluate(df, "yolo"), "YOLOv7 @conf0.25": evaluate(df, "yolo", 0.25),
           "yolo_mean_infer_ms_gpu": float(df.infer_ms.mean())}
    if yolo_meta:
        res["yolo_meta"] = json.loads(open(yolo_meta, encoding="utf-8").read())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "yolo_vs_effnet.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    df.to_csv(OUT / "yolo_vs_effnet_scores.csv", index=False, encoding="utf-8-sig")
    lines = ["| 모델 | 부품 | 임계값 | 재현율 | 특이도 | 정밀도 | AUC |", "|---|---|---:|---:|---:|---:|---:|"]
    for name in (eff, "YOLOv7", "YOLOv7 @conf0.25"):
        for part, r in res[name].items():
            t = r["test"]
            lines.append(f"| {name} | {part} | {r['threshold']:.2f} | {t['recall']:.3f} | {t['specificity']:.3f} | "
                         f"{t['precision']:.3f} | {t['auc']:.3f} |")
            for s, x in r.get("test_by_subtype", {}).items():
                lines.append(f"| {name} | FENDER·{s} | | {x['recall']:.3f} | {x['specificity']:.3f} | {x['precision']:.3f} | {x['auc']:.3f} |")
    (OUT / "yolo_vs_effnet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_csv")
    ap.add_argument("--yolo-meta")
    a = ap.parse_args()
    main(a.scores_csv, a.yolo_meta)
