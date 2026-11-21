"""완료된 Azure ML 작업에 에폭별 학습 곡선 지표와 그래프를 사후 기록한다(작업 상태·결과는 바꾸지 않음).

학습 당시에는 Ultralytics MLflow 연동을 꺼 두어(settings mlflow=False) 에폭별 값이 작업 메트릭 탭에 없었다.
원본 값은 각 작업 출력의 results.csv / results.txt 에 있으므로 그대로 옮긴다.
실행: .venv-aml/Scripts/python azureml/backfill_epoch_metrics.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import csv
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_external_vision_run import tracking_uri  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MV, CURVES = ROOT / "models/vision", ROOT / "reports/vision/curves"
CLS_COLS = {"train/loss": "train_loss", "val/loss": "val_loss", "metrics/accuracy_top1": "val_accuracy_top1",
            "metrics/accuracy_top5": "val_accuracy_top5"}
V7_COLS = ["ep", "mem", "box", "obj", "cls", "total", "labels", "img", "P", "R", "mAP50", "mAP50_95", "vbox", "vobj", "vcls"]
V7_LOG = {"box": "train_box_loss", "obj": "train_obj_loss", "cls": "train_cls_loss", "P": "val_precision",
          "R": "val_recall", "mAP50": "val_mAP50", "mAP50_95": "val_mAP50_95", "vbox": "val_box_loss",
          "vobj": "val_obj_loss", "vcls": "val_cls_loss"}

JOBS = {  # 작업 이름: [(지표 접두사, 결과 파일, 형식)], 그래프
    "dynamic_melon_32s4hz0mfc": ([("yolo11m-cls", MV / "yolo11m_cls_e50_split821/results.csv", "cls")],
                                 ["yolo11m-cls_50ep_split821.png", "yolo11m-cls_30_vs_50ep_split821.png"]),
    "dreamy_brain_29f7wfqkb5": ([("yolo11m-cls", MV / "yolo11_cls_split821/yolo11m-cls/results.csv", "cls"),
                                 ("yolo11s-cls", MV / "yolo11_cls_split821/yolo11s-cls/results.csv", "cls")],
                                ["yolo11m-cls_30ep_split821.png", "yolo11s-cls_30ep_split821.png"]),
    "teal_avocado_jn1z0gjfq5": ([("yolov7", MV / "yolov7_h100_e100_split821/run/results.txt", "v7")],
                                ["yolov7_e100_split821.png"]),
}


def metrics(prefix: str, path: Path, kind: str) -> list[Metric]:
    ts = int(time.time() * 1000)
    out = []
    if kind == "cls":
        for r in csv.DictReader(path.open(encoding="utf-8")):
            r = {k.strip(): v for k, v in r.items()}
            out += [Metric(f"{prefix}/{v}", float(r[k]), ts, int(r["epoch"])) for k, v in CLS_COLS.items()]
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = dict(zip(V7_COLS, line.split()))
            ep = int(r["ep"].split("/")[0]) + 1
            out += [Metric(f"{prefix}/{v}", float(r[k]), ts, ep) for k, v in V7_LOG.items()]
    return out


def main() -> None:
    import mlflow
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    for job, (sources, pngs) in JOBS.items():
        ms = [m for p, f, k in sources if f.exists() for m in metrics(p, f, k)]
        for i in range(0, len(ms), 1000):  # log_batch 1회 최대 1000개
            client.log_batch(job, metrics=ms[i:i + 1000])
        for png in pngs:
            if (CURVES / png).exists():
                client.log_artifact(job, str(CURVES / png), "training_curves")
        print(job, len(ms), "metrics,", len(pngs), "plots")


if __name__ == "__main__":
    main()
