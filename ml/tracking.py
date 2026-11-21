"""로컬 MLflow(SQLite 백엔드) 기록 헬퍼. 실패해도 학습 결과에는 영향을 주지 않는다."""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def log_run(experiment: str, run_name: str, params: dict, metrics: dict, tags: dict | None = None) -> str | None:
    try:
        import mlflow
        in_azureml = bool(os.environ.get("AZUREML_RUN_ID"))
        if in_azureml:  # Azure ML 작업: 작업이 지정한 워크스페이스 MLflow에 하위 run으로 기록
            parent = mlflow.active_run() or mlflow.start_run()
            tags = {**(tags or {}), "azureml_parent_run": parent.info.run_id, "experiment_key": experiment}
        else:
            mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
            mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name, nested=in_azureml) as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float)) and v == v})
            mlflow.set_tags({"python": sys.version.split()[0], "platform": platform.platform(),
                             **(tags or {})})
            return run.info.run_id
    except Exception as e:  # noqa: BLE001
        print("MLflow logging skipped:", repr(e)[:200])
        return None
