"""모델 성능표를 Azure ML 메트릭 탭에 기록한다.

1) 각 학습 작업(run)에 해당 모델의 시험셋 지표(test/*)를 추가한다(작업 상태·결과는 바꾸지 않음).
   EfficientNet 작업에는 에폭별 학습 손실·검증 macro F1(선 그래프)도 추가한다.
2) 비교 전용 run 'resource-vision-scoreboard-split821'을 만들어 모델별 지표(타일)와 비교 그래프(이미지 탭)를 기록한다.
   스텝 = 순위(1~5)인 'ranked/mean_accuracy' 선 그래프로 순위도 한눈에 보이게 한다.
먼저 실행: .venv/Scripts/python -m ml.vision.model_scoreboard
실행: .venv-aml/Scripts/python azureml/log_model_scoreboard.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_external_vision_run import tracking_uri  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports/vision"
EXPERIMENT = "resource-predict-ai-vision"
JOB_OF = {  # 성능표 모델 이름 -> (작업, 지표 접두사)
    "EfficientNet-B0 (8ep, 서비스)": ("wheat_worm_c84pn8qwvr", "efficientnet_b0"),
    "YOLO11m-cls (30ep)": ("dreamy_brain_29f7wfqkb5", "yolo11m-cls"),
    "YOLO11s-cls (30ep)": ("dreamy_brain_29f7wfqkb5", "yolo11s-cls"),
    "YOLO11m-cls (50ep)": ("dynamic_melon_32s4hz0mfc", "yolo11m-cls"),
    "YOLOv7 (100ep, 검출)": ("teal_avocado_jn1z0gjfq5", "yolov7"),
}
SHORT = {"EfficientNet-B0 (8ep, 서비스)": "effnet_b0_8ep", "YOLO11m-cls (30ep)": "yolo11m_30ep",
         "YOLO11s-cls (30ep)": "yolo11s_30ep", "YOLO11m-cls (50ep)": "yolo11m_50ep", "YOLOv7 (100ep, 검출)": "yolov7_100ep"}
KEYS = ["mean_accuracy", "DOOR_accuracy", "BUMPER_accuracy", "FENDER_accuracy", "FENDER_gap_accuracy",
        "mean_auc", "DOOR_auc", "BUMPER_auc", "FENDER_auc", "top1_6cls", "rank"]


def main() -> None:
    import mlflow
    mlflow.set_tracking_uri(tracking_uri())
    c = MlflowClient()
    rows = json.loads((REP / "model_scoreboard.json").read_text(encoding="utf-8"))["rows"]
    ts = int(time.time() * 1000)
    for r in rows:  # 1) 각 작업에 시험 지표
        job, pre = JOB_OF[r["model"]]
        c.log_batch(job, metrics=[Metric(f"{pre}/test_{k}", float(r[k]), ts, 0) for k in KEYS if r.get(k) is not None])
    hist = json.loads((ROOT / "models/vision/meta.json").read_text(encoding="utf-8"))["history"]
    c.log_batch("wheat_worm_c84pn8qwvr", metrics=[Metric(f"efficientnet_b0/{k}", float(h[k]), ts, int(h["epoch"]))
                                                   for h in hist for k in ("train_loss", "val_macro_f1") if k in h])
    exp = c.get_experiment_by_name(EXPERIMENT)  # 2) 비교 run
    run = c.create_run(exp.experiment_id, run_name="resource-vision-scoreboard-split821",
                       tags={"purpose": "모델 성능 비교(8:2:1 시험셋 1,080장)", "best_model": rows[0]["model"],
                             "serving_model": "EfficientNet-B0 (8ep, 서비스)",
                             "note": "YOLO11은 AGPL-3.0 내부 실험 전용"})
    rid = run.info.run_id
    ms = [Metric(f"{SHORT[r['model']]}/{k}", float(r[k]), ts, 0) for r in rows for k in KEYS if r.get(k) is not None]
    ms += [Metric(f"ranked/{k}", float(r[k]), ts, r["rank"]) for r in rows
           for k in ("mean_accuracy", "FENDER_accuracy", "mean_auc")]
    c.log_batch(rid, metrics=ms)
    for f in ["curves/model_scoreboard_accuracy.png", "curves/model_scoreboard_val_by_epoch.png",
              "curves/yolo11m-cls_30_vs_50ep_split821.png", "model_scoreboard.md", "model_scoreboard.csv"]:
        c.log_artifact(rid, str(REP / f), "scoreboard")
    c.set_terminated(rid, "FINISHED")
    print("run", rid, len(ms), "metrics")


if __name__ == "__main__":
    main()
