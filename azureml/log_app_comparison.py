"""이전 앱 모델 vs 현재 앱 모델 비교표를 Azure ML 메트릭 탭에 기록한다.

- 비교 run 'resource-vision-app-models-prev-vs-current': 모델별 지표(타일) + 시간순(1~6) 선 그래프 'timeline/*'
  + 표·그래프 파일(scoreboard 폴더). 새로 학습한 두 작업에는 시험셋 지표(test/*)를 추가한다(작업 결과는 바꾸지 않음).
먼저 실행: .venv/Scripts/python -m ml.vision.app_model_comparison
실행: .venv-aml/Scripts/python azureml/log_app_comparison.py
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

REP = Path(__file__).resolve().parents[1] / "reports/vision"
SHORT = ["prev1_effnet_8ep", "prev2_yolo11m_30ep", "cur_effnet_full_20ep", "cur_yolo11m_full_30ep",
         "cur_stacking", "cur_service_auto"]
JOBS = {2: "tidy_rainbow_nvlmbc9cbm", 3: "funny_engine_sv1gczvg12"}
KEYS = ["mean_accuracy", "DOOR_accuracy", "BUMPER_accuracy", "FENDER_accuracy", "FENDER_gap_accuracy", "mean_auc",
        "top1_6cls", "part_false_block_rate", "auto_rate"]


def main() -> None:
    import mlflow
    mlflow.set_tracking_uri(tracking_uri())
    c = MlflowClient()
    rows = json.loads((REP / "app_model_comparison.json").read_text(encoding="utf-8"))["rows"]
    ts = int(time.time() * 1000)
    for i, job in JOBS.items():
        c.log_batch(job, metrics=[Metric(f"test/{k}", float(rows[i][k]), ts, 0) for k in KEYS if rows[i].get(k) is not None])
    exp = c.get_experiment_by_name("resource-predict-ai-vision")
    run = c.create_run(exp.experiment_id, run_name="resource-vision-app-models-prev-vs-current",
                       tags={"purpose": "이전 앱 모델 vs 현재 앱 모델 (같은 시험셋 1,080장)",
                             "order": " → ".join(r["model"] for r in rows),
                             "serving_model": rows[-1]["model"],
                             "note": "현재 서비스 행 정확도 = 자동 판정분(나머지는 작업자 확인). YOLO11은 AGPL-3.0 내부 전용"})
    rid = run.info.run_id
    ms = [Metric(f"{s}/{k}", float(r[k]), ts, 0) for s, r in zip(SHORT, rows) for k in KEYS if r.get(k) is not None]
    ms += [Metric(f"timeline/{k}", float(r[k]), ts, i) for i, r in enumerate(rows, 1)
           for k in ("mean_accuracy", "DOOR_accuracy", "BUMPER_accuracy", "FENDER_accuracy", "auto_rate")]
    c.log_batch(rid, metrics=ms)
    for f in ["curves/app_model_comparison.png", "curves/app_model_comparison_lines.png", "app_model_comparison.md", "app_model_comparison.csv",
              "curves/full_data_curves.png"]:
        c.log_artifact(rid, str(REP / f), "scoreboard")
    c.set_terminated(rid, "FINISHED")
    print("run", rid, len(ms), "metrics")


if __name__ == "__main__":
    main()
