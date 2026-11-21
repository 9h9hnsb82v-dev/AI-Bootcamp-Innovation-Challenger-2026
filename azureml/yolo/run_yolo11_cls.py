"""Azure ML 작업 본문: YOLO11 분류(6클래스 = 부품 x 양품/불량) 학습 -> val/test 이미지별 확률 -> 출력 저장.

EfficientNet v3와 같은 이미지·같은 분할(data/processed/aihub/yolo 의 index.csv)을 쓴다.
Ultralytics(AGPL-3.0)는 내부 비교 실험 전용 - 결과 모델을 배포·서비스하지 않는다(사용자 승인 전).
사용: python azureml/yolo/run_yolo11_cls.py <yolo_dataset_dir> <output_dir> [--models yolo11s-cls,yolo11m-cls] [--epochs 30] [--img 448]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path


def build_imagefolder(data_in: Path, root: Path) -> dict:
    """index.csv 기준으로 root/{train,val,test}/{PART_QUALITY}/stem.jpg 심볼릭 링크 구성."""
    counts: dict = {}
    with open(data_in / "index.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cls = f"{r['part']}_{r['quality']}"
            d = root / r["subset"] / cls
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{r['stem']}.jpg").symlink_to(data_in / "images" / r["subset"] / f"{r['stem']}.jpg")
            counts[f"{r['subset']}/{cls}"] = counts.get(f"{r['subset']}/{cls}", 0) + 1
    return counts


def score_images(weights: Path, root: Path, img: int, out_csv: Path) -> None:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    names = model.names  # {idx: "DOOR_DEFECT", ...}
    rows = []
    for subset in ("val", "test", "holdout"):  # holdout: 전체 데이터 추출본의 추가 시험셋(없으면 건너뜀)
        for p in sorted((root / subset).glob("*/*.jpg")):
            t0 = time.time()
            res = model.predict(str(p), imgsz=img, verbose=False)[0]
            ms = (time.time() - t0) * 1000
            prob = res.probs.data.float().cpu().numpy()
            r = {"subset": subset, "stem": p.stem, "true": p.parent.name,
                 "pred": names[int(prob.argmax())], "infer_ms": round(ms, 2)}
            r.update({f"p_{names[i]}": float(prob[i]) for i in range(len(prob))})
            rows.append(r)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(a) -> None:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "mlflow": False, "tensorboard": False})  # 외부 분석 전송·자동 로깅 끔
    data_in, out = Path(a.data_in), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.manifest:  # 세션 비겹침 재분할 등: manifest의 subset·class·path로 imagefolder 구성(이미지 복사 없이 링크)
        root = Path("/tmp/cls_data")
        counts = {}
        with open(a.manifest, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rel = r["path"].split("aihub_full/", 1)[1]
                d = root / r["subset"] / r["class"]
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{r['stem']}.jpg").symlink_to(data_in / rel)
                counts[f"{r['subset']}/{r['class']}"] = counts.get(f"{r['subset']}/{r['class']}", 0) + 1
    elif (data_in / "index.csv").exists():
        root = Path("/tmp/cls_data")
        counts = build_imagefolder(data_in, root)
    else:  # 이미 {train,val,test,holdout}/{CLASS}/*.jpg 구조인 폴더(aihub_full 추출본)
        root = data_in / "images" if (data_in / "images").is_dir() else data_in
        counts = {f"{d.parent.name}/{d.name}": len(list(d.glob("*.jpg"))) for d in sorted(root.glob("*/*")) if d.is_dir()}
    print(json.dumps(counts, indent=1))
    summary = {"counts": counts, "img": a.img, "epochs": a.epochs, "ultralytics": __import__("ultralytics").__version__}
    for m in a.models.split(","):
        t0 = time.time()
        model = YOLO(f"{m}.pt")
        model.train(data=str(root), imgsz=a.img, epochs=a.epochs, batch=a.batch, workers=8, seed=42,
                    project=str(Path("/tmp/runs")), name=m, exist_ok=True, plots=True)
        train_s = time.time() - t0
        run = Path("/tmp/runs") / m
        best = run / "weights" / "best.pt"
        score_images(best, root, a.img, out / f"{m}_scores.csv")
        dst = out / m
        dst.mkdir(exist_ok=True)
        for f in ["results.csv", "args.yaml", "confusion_matrix.png", "results.png"]:
            if (run / f).exists():
                (dst / f).write_bytes((run / f).read_bytes())
        (dst / "best.pt").write_bytes(best.read_bytes())
        summary[m] = {"train_seconds": round(train_s, 1), "best": "best.pt"}
        print(m, "done", round(train_s, 1), "s")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    try:
        import mlflow
        with mlflow.start_run(nested=bool(mlflow.active_run())):
            mlflow.log_params({"img": a.img, "epochs": a.epochs, "models": a.models})
            for m in a.models.split(","):
                mlflow.log_metric(f"{m}_train_seconds", summary[m]["train_seconds"])
                # 에폭별 학습 곡선(Ultralytics MLflow 연동은 꺼 두었으므로 results.csv에서 직접 기록)
                rc = out / m / "results.csv"
                if rc.exists():
                    names = {"train/loss": "train_loss", "val/loss": "val_loss",
                             "metrics/accuracy_top1": "val_accuracy_top1", "metrics/accuracy_top5": "val_accuracy_top5"}
                    for row in csv.DictReader(rc.open()):
                        row = {k.strip(): v for k, v in row.items()}
                        mlflow.log_metrics({f"{m}/{v}": float(row[k]) for k, v in names.items()}, step=int(row["epoch"]))
                for png in ("results.png", "confusion_matrix.png"):
                    if (out / m / png).exists():
                        mlflow.log_artifact(str(out / m / png), f"{m}_plots")
    except Exception as e:  # 기록 실패는 학습 결과에 영향 없음
        print("mlflow skip:", e)
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_in")
    ap.add_argument("out")
    ap.add_argument("--models", default="yolo11s-cls,yolo11m-cls")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--img", type=int, default=448)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--manifest", default=None, help="subset/class/path 열이 있는 manifest csv(재분할용)")
    main(ap.parse_args())
