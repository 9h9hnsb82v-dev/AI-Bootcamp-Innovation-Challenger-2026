"""팀 VM(AICD, A10)에서 학습한 이미지 모델 결과를 Azure ML 워크스페이스에 기록·등록한다.

- 워크스페이스 MLflow에 run 1개 생성: 하이퍼파라미터, 시험 지표, 학습 이력, meta.json 첨부
- 모델 가중치+meta를 MLflow 아티팩트로 올리고 Azure ML 모델 레지스트리에 등록
- 학습 컴퓨트는 Azure ML 작업이 아니라 외부 VM임을 태그로 명시(compute=AICD-A10-VM)

실행(분리된 venv, azureml-mlflow는 mlflow-skinny<=3.15 필요):
  .venv-aml/Scripts/python azureml/log_external_vision_run.py <모델 폴더>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WS = {"workspace": "AIBIC_10_24_workspace", "rg": "stu_AIBIC2026_15_RG_EUS2"}
MODEL_NAME = "resource-aihub-vision-effb0"


def tracking_uri() -> str:
    az = "az.cmd" if os.name == "nt" else "az"
    return subprocess.check_output([az, "ml", "workspace", "show", "-n", WS["workspace"], "-g", WS["rg"],
                                    "--query", "mlflow_tracking_uri", "-o", "tsv"], text=True).strip()


def main(model_dir: Path) -> dict:
    import mlflow
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("resource-predict-ai-vision")
    rep = meta["test_report"]
    with mlflow.start_run(run_name=meta["model_version"] + "_aicd") as run:
        mlflow.set_tags({"compute": "AICD-A10-VM (team VM, outside Azure ML job)", "provenance": "AIHUB_IMAGE_AI",
                         "test_source": meta["test_source"], "note": meta["note"]})
        mlflow.log_params({"img_size": meta["img_size"], "epochs": meta["epochs"], "seed": meta["seed"],
                           "classes": ",".join(meta["classes"]), "val_n": meta["val_n"], "test_n": meta["test_n"],
                           "train_counts": json.dumps(meta["train_counts"])})
        mlflow.log_metrics({"test_macro_f1": rep["macro avg"]["f1-score"], "test_accuracy": rep["accuracy"],
                            **{f"{p}_defect_recall": v["defect_recall"] for p, v in meta["test_part_metrics"].items()},
                            **{f"{p}_defect_precision": v["defect_precision"] for p, v in meta["test_part_metrics"].items()},
                            **{f"{p}_part_type_accuracy": v["part_type_accuracy"] for p, v in meta["test_part_metrics"].items()}})
        for h in meta["history"]:
            mlflow.log_metrics({"train_loss": h["train_loss"], "val_macro_f1": h["val_macro_f1"]}, step=h["epoch"])
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        run_id = run.info.run_id
    print("logged run", run_id, flush=True)
    # mlflow.register_model(3.x)은 Azure ML이 지원하지 않는 logged-models API를 호출하므로 az CLI로 등록
    az = "az.cmd" if os.name == "nt" else "az"
    out = subprocess.check_output(
        [az, "ml", "model", "create", "-n", MODEL_NAME, "--type", "custom_model",
         "--path", f"azureml://jobs/{run_id}/outputs/artifacts/paths/model/",
         "--description", f"AI-Hub door/bumper/fender 6-class EfficientNet-B0 ({meta['model_version']}), trained on AICD A10 VM",
         "--tags", f"img_size={meta['img_size']}", "compute=AICD-A10-VM", f"mlflow_run_id={run_id}",
         f"test_macro_f1={meta['test_report']['macro avg']['f1-score']:.4f}",
         "-w", WS["workspace"], "-g", WS["rg"], "--query", "version", "-o", "tsv"], text=True).strip()
    return {"run_id": run_id, "model": MODEL_NAME, "version": out}


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
