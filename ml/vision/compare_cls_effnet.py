"""YOLO11 분류(6클래스) vs EfficientNet-B0 v3 이미지 단위 판정 비교 (같은 val/test, 같은 규칙).

점수: 계획 부품을 안다고 가정, P(part_DEFECT) / (P(part_DEFECT) + P(part_GOOD))  - EfficientNet과 같은 정의
임계값: 부품별 검증셋 Youden J -> 시험셋 1회 평가. 6클래스 top-1 정확도도 함께 보고.
EfficientNet 점수는 reports/vision/effnet_split821_scores.csv(8:2:1 분할, 서비스 모델) 재사용.
실행: .venv/Scripts/python -m ml.vision.compare_cls_effnet <scores_dir> [출력 접미사]   (예: models/vision/yolo11_cls _split821)
출력: reports/vision/yolo11cls_vs_effnet.json, .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from ml.vision.compare_yolo_effnet import OUT, evaluate

EFF_SCORES = OUT / "effnet_split821_scores.csv"  # 8:2:1 재분할 후 서비스 EfficientNet 점수


def main(scores_dir: str, tag: str = "") -> dict:
    sd = Path(scores_dir)
    base = pd.read_csv(EFF_SCORES, encoding="utf-8-sig")[["subset", "stem", "part", "quality", "subtype", "effnet"]]
    base["y"] = (base.quality == "DEFECT").astype(int)
    summary = json.loads((sd / "summary.json").read_text(encoding="utf-8"))
    res = {"protocol": "per-part image-level; score = P(part_DEFECT)/(P(part_DEFECT)+P(part_GOOD)); threshold = val Youden J; test once",
           "EfficientNet-B0 split821": evaluate(base, "effnet"), "job_summary": summary}
    lines = ["| 모델 | 부품 | 임계값 | 정확도 | 재현율 | 특이도 | 정밀도 | AUC |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    names = ["EfficientNet-B0 split821"]
    for f in sorted(sd.glob("*_scores.csv")):
        m = f.name.replace("_scores.csv", "")
        s = pd.read_csv(f)
        df = base.merge(s, on=["subset", "stem"], how="left", validate="1:1")
        assert df.true.notna().all(), f"{m}: 점수 누락"
        assert (df.true == df.part + "_" + df.quality).all(), f"{m}: 클래스 불일치"
        df[m] = [r[f"p_{r.part}_DEFECT"] / (r[f"p_{r.part}_DEFECT"] + r[f"p_{r.part}_GOOD"]) for _, r in df.iterrows()]
        res[m] = evaluate(df, m)
        t = df[df.subset == "test"]
        res[m]["_test_top1_6cls"] = float((t.pred == t.true).mean())
        res[m]["_test_part_mismatch_rate"] = float((t.pred.str.split("_").str[0] != t.part).mean())
        res[m]["_mean_infer_ms_gpu"] = float(df.infer_ms.mean())
        names.append(m)
    for n in names:
        for part, r in res[n].items():
            if part.startswith("_"):
                continue
            t = r["test"]
            lines.append(f"| {n} | {part} | {r['threshold']:.2f} | {t['accuracy']:.3f} | {t['recall']:.3f} | {t['specificity']:.3f} | {t['precision']:.3f} | {t['auc']:.3f} |")
            for st, x in r.get("test_by_subtype", {}).items():
                lines.append(f"| {n} | FENDER·{st} | | {x['accuracy']:.3f} | {x['recall']:.3f} | {x['specificity']:.3f} | {x['precision']:.3f} | {x['auc']:.3f} |")
    for n in names[1:]:
        lines.append(f"\n{n}: 시험셋 6클래스 top-1 {res[n]['_test_top1_6cls']:.3f}, 부품 오인식률 {res[n]['_test_part_mismatch_rate']:.3f}")
    (OUT / f"yolo11cls_vs_effnet{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"yolo11cls_vs_effnet{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
