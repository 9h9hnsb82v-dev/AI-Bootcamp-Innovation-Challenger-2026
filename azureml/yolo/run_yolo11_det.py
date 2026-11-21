"""Azure ML 작업 본문: YOLO11 결함 위치 검출(4클래스) 학습 → test/holdout 평가 → 이미지별 박스 저장 (사용자 결정 2026-11-14 (b)).

클래스: 0 scratch · 1 exterior_damage · 2 gap_defect · 3 gap_normal (ml/vision/prepare_yolo_full.py)
서비스에서는 판정이 아니라 '결함 위치 표시'에만 쓴다. Ultralytics AGPL-3.0 - 내부 전용(사용자 허락 전 공개·배포 금지).
사용: python azureml/yolo/run_yolo11_det.py <aihub_full 폴더> <labels.tar 폴더> <출력> [--model yolo11m.pt --epochs 30 --img 640 --batch 32]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
import time
from pathlib import Path

NAMES = ["scratch", "exterior_damage", "gap_defect", "gap_normal"]


def main(a) -> None:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import numpy as np
    if not hasattr(np, "trapz"):  # numpy 2.x에서 삭제 → Ultralytics 8.3.40 mAP 계산이 실패(2차 작업 teal_cassava_cfr54n3qd2 원인)
        np.trapz = np.trapezoid
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "mlflow": False, "tensorboard": False})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Ultralytics는 사진의 실제 경로에서 /images/ → /labels/ 로 라벨을 찾는다(심볼릭 링크를 따라감).
    # 그래서 라벨을 내려받은 사진 폴더(쓰기 가능, download 모드) 안에 직접 푼다. (1차 작업 honest_napa_kp5kjbggtl 실패 원인)
    ds = Path(a.images).resolve()
    with tarfile.open(Path(a.labels) / "labels.tar") as t:
        t.extractall(ds)  # → <images 입력>/labels/<subset>/<CLASS>/<stem>.txt
    from ultralytics.data.utils import img2label_paths
    imgs = sorted((ds / "images" / "train").glob("*/*.jpg"))
    found = sum(Path(lp).exists() for lp in img2label_paths([str(i) for i in imgs[:2000]]))
    nonempty = sum(Path(lp).stat().st_size > 0 for lp in img2label_paths([str(i) for i in imgs[:2000]]) if Path(lp).exists())
    print(f"label check: {found}/2000 found, {nonempty} with boxes", flush=True)
    assert found == min(2000, len(imgs)) and nonempty > 0, "라벨을 찾지 못함 - 학습 중단"
    if a.splits:  # 재분할: sd_{train,val,test}.txt(images/... 상대경로)를 데이터 폴더에 두고 목록으로 학습·평가
        for sname in ("train", "val", "test"):
            lines = (Path(a.splits) / f"sd_{sname}.txt").read_text().split()
            (ds / f"sd_{sname}.txt").write_text("\n".join("./" + x for x in lines) + "\n")
        (ds / "data_main.yaml").write_text(f"path: {ds}\ntrain: sd_train.txt\nval: sd_val.txt\ntest: sd_test.txt\n"
                                          f"nc: {len(NAMES)}\nnames: {NAMES}\n")
        evals = ("main",)
    else:
        for split_name, test_dir in (("main", "images/test"), ("holdout", "images/holdout")):
            (ds / f"data_{split_name}.yaml").write_text(
                f"path: {ds}\ntrain: images/train\nval: images/val\ntest: {test_dir}\nnc: {len(NAMES)}\nnames: {NAMES}\n")
        evals = ("main", "holdout")
    t0 = time.time()
    model = YOLO(a.model)
    model.train(data=str(ds / "data_main.yaml"), imgsz=a.img, epochs=a.epochs, batch=a.batch, workers=8, seed=42,
                project="/tmp/runs", name="det", exist_ok=True, plots=True)
    train_s = time.time() - t0
    run = Path("/tmp/runs/det")
    # 학습 직후 가중치·곡선을 먼저 출력에 저장(3차 작업 gifted_coconut_9byr4y2yy5는 후처리 오류로 가중치를 잃음)
    (out / "best.pt").write_bytes((run / "weights/best.pt").read_bytes())
    for f in ["results.csv", "args.yaml", "results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
        if (run / f).exists():
            (out / f).write_bytes((run / f).read_bytes())
    best = YOLO(str(run / "weights/best.pt"))
    summary = {"model": a.model, "epochs": a.epochs, "img": a.img, "train_seconds": round(train_s, 1), "names": NAMES}
    for split_name in evals:
        m = best.val(data=str(ds / f"data_{split_name}.yaml"), split="test", imgsz=a.img, batch=a.batch, plots=False,
                     project="/tmp/runs", name=f"val_{split_name}", exist_ok=True)
        key = "test" if split_name == "main" else "holdout"
        summary[key] = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precision": float(m.box.mp),
                        "recall": float(m.box.mr),
                        "per_class_mAP50": {NAMES[i]: float(v) for i, v in zip(m.box.ap_class_index, m.box.ap50)}}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))  # 평가마다 바로 저장
    rows = []  # 이미지별 예측 박스(신뢰도 0.05 이상) - 선택(--predict). 4차 작업에서 GPU 메모리 부족 이력
    for subset in (("val", "test", "holdout") if a.predict else ()):
        files = sorted(str(f) for f in (ds / "images" / subset).glob("*/*.jpg"))  # 클래스 하위 폴더까지(폴더 지정은 하위를 안 봄)
        for r in best.predict(source=files, imgsz=a.img, conf=0.05, stream=True, verbose=False, batch=64):
            p = Path(r.path)
            boxes = [{"cls": int(c), "conf": round(float(s), 4), "xyxyn": [round(float(v), 4) for v in b]}
                     for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxyn)]
            rows.append({"subset": subset, "cls_dir": p.parent.name, "stem": p.stem, "boxes": json.dumps(boxes)})
    if rows:
        with open(out / "det_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    try:
        import mlflow
        names = {"train/box_loss": "train_box_loss", "train/cls_loss": "train_cls_loss", "val/box_loss": "val_box_loss",
                 "val/cls_loss": "val_cls_loss", "metrics/mAP50(B)": "val_mAP50", "metrics/mAP50-95(B)": "val_mAP50_95",
                 "metrics/precision(B)": "val_precision", "metrics/recall(B)": "val_recall"}
        for row in csv.DictReader((run / "results.csv").open()):
            row = {k.strip(): v for k, v in row.items()}
            mlflow.log_metrics({v: float(row[k]) for k, v in names.items() if k in row}, step=int(float(row["epoch"])))
        for key in ("test", "holdout"):
            if key in summary:
                mlflow.log_metrics({f"{key}_{k}": v for k, v in summary[key].items() if isinstance(v, float)})
        for png in ("results.png", "confusion_matrix.png", "PR_curve.png"):
            if (out / png).exists():
                mlflow.log_artifact(str(out / png), "det_plots")
    except Exception as e:  # noqa: BLE001
        print("mlflow skip:", e)
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--splits", default=None, help="sd_{train,val,test}.txt 가 있는 폴더(재분할)")
    ap.add_argument("--predict", action="store_true", help="이미지별 예측 박스 저장")
    main(ap.parse_args())
