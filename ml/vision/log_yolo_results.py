"""YOLOv7 결과(results.txt, test mAP, 이미지 점수 요약)를 MLflow에 기록. Azure ML 작업 안에서 실행.

results.txt 열: epoch, gpu_mem, box, obj, cls, total, labels, img_size, P, R, mAP@.5, mAP@.5:.95, val_box, val_obj, val_cls
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main(out: Path) -> dict:
    import mlflow
    meta = json.loads((out / "yolo_meta.json").read_text())
    rows = [l.split() for l in (out / "run/results.txt").read_text().splitlines() if l.strip()]
    text = (out / "test_map.txt").read_text(errors="ignore")
    m = re.search(r"^\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text, re.M)
    test = dict(zip(["images", "labels", "P", "R", "mAP50", "mAP50_95"], map(float, m.groups()))) if m else {}
    per_class = {c: float(v) for c, v in re.findall(r"^\s*(scratch|exterior_damage|gap_defect|gap_normal)\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)",
                                                   text, re.M)}
    with mlflow.start_run(nested=bool(mlflow.active_run())):
        mlflow.log_params(meta)
        for i, r in enumerate(rows):
            mlflow.log_metrics({"val_P": float(r[8]), "val_R": float(r[9]), "val_mAP50": float(r[10]),
                                "val_mAP50_95": float(r[11]), "train_total_loss": float(r[5])}, step=i)
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items() if k not in ("images", "labels")})
        mlflow.log_metrics({f"test_mAP50_{k}": v for k, v in per_class.items()})
        mlflow.log_metric("train_seconds", meta["train_seconds"])
    summary = {"test": test, "test_mAP50_per_class": per_class, "epochs_logged": len(rows), **meta}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main(Path(sys.argv[1]))
