"""학습 곡선(손실·정확도) 그래프 생성. Ultralytics results.png와 같은 형식(실측 + 이동평균 점선).

실행: .venv/Scripts/python -m ml.vision.plot_training_curves
출력: reports/vision/curves/*.png
- 기록에 있는 항목만 그린다. 기록되지 않은 값(예: EfficientNet split821의 학습 정확도·검증 손실)은 추정해서 그리지 않는다.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.vision.compare_yolo_effnet import OUT, ROOT

MV = ROOT / "models/vision"
CURVES = OUT / "curves"


def smooth(y, f=0.1):
    """Ultralytics와 같은 박스 필터 이동평균."""
    y = np.asarray(y, float)
    nf = 2 * round(len(y) * f) + 1  # 홀수 길이 필터(출력 길이 = 입력 길이)
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def grid(title: str, x, panels: list[tuple[str, list]], out: str, ncols: int = 5, best: int | None = None) -> str:
    n = len(panels)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols + 0.4, 2.5 * nrows + 0.6), squeeze=False)
    for ax, (name, y) in zip(axes.flat, panels):
        y = np.asarray(y, float)
        ax.plot(x, y, marker=".", markersize=5, linewidth=1.5, label="results")
        if len(y) >= 5:
            ax.plot(x, smooth(y), ":", linewidth=1.8, label="smooth")
        if best is not None:
            ax.axvline(best, color="tab:red", linewidth=0.8, alpha=0.6, label=f"best ep {best}")
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    CURVES.mkdir(parents=True, exist_ok=True)
    path = CURVES / out
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def yolo_cls(csv, title, out) -> str:
    d = pd.read_csv(csv)
    d.columns = [c.strip() for c in d.columns]
    best = int(d.epoch[d["metrics/accuracy_top1"].idxmax()])
    return grid(title, d.epoch, [("train/loss", d["train/loss"]), ("val/loss", d["val/loss"]),
                                 ("metrics/accuracy_top1 (val)", d["metrics/accuracy_top1"]),
                                 ("metrics/accuracy_top5 (val)", d["metrics/accuracy_top5"])], out, ncols=4, best=best)


def yolov7(txt, title, out) -> str:
    cols = ["ep", "mem", "box", "obj", "cls", "total", "labels", "img", "P", "R", "mAP50", "mAP50_95", "vbox", "vobj", "vcls"]
    d = pd.read_csv(txt, sep=r"\s+", header=None, names=cols)
    x = d.ep.str.split("/").str[0].astype(int) + 1
    best = int(x[(0.1 * d.mAP50 + 0.9 * d.mAP50_95).idxmax()])  # YOLOv7 fitness
    return grid(title, x, [("train/box_loss", d.box), ("train/obj_loss", d.obj), ("train/cls_loss", d.cls),
                           ("metrics/precision", d.P), ("metrics/recall", d.R),
                           ("val/box_loss", d.vbox), ("val/obj_loss", d.vobj), ("val/cls_loss", d.vcls),
                           ("metrics/mAP_0.5", d.mAP50), ("metrics/mAP_0.5:0.95", d.mAP50_95)], out, best=best)


def history(hist, keys, title, out, best_key) -> str:
    d = pd.DataFrame(hist)
    panels = [(k, d[k]) for k in keys if k in d]
    best = int(d.epoch[d[best_key].idxmax()]) if best_key in d else None
    return grid(title, d.epoch, panels, out, ncols=4, best=best)


def overlay(runs: dict, out: str) -> str:
    """같은 모델의 에폭 설정 비교(과적합 확인용)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for label, csv in runs.items():
        d = pd.read_csv(csv)
        d.columns = [c.strip() for c in d.columns]
        for ax, k in zip(axes, ["train/loss", "val/loss", "metrics/accuracy_top1"]):
            ax.plot(d.epoch, d[k], marker=".", markersize=4, label=label)
            ax.set_title(k if k != "metrics/accuracy_top1" else "val accuracy_top1", fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("YOLO11m-cls: 30 vs 50 epochs (split 8:2:1)", fontsize=11)
    fig.tight_layout()
    path = CURVES / out
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main() -> list[str]:
    made = []
    for sub, tag in [("yolo11_cls_split821/yolo11m-cls", "yolo11m-cls 30ep"), ("yolo11_cls_split821/yolo11s-cls", "yolo11s-cls 30ep"),
                     ("yolo11m_cls_e50_split821", "yolo11m-cls 50ep")]:
        f = MV / sub / "results.csv"
        if f.exists():
            made.append(yolo_cls(f, f"{tag} (AI-Hub 6 classes, split 8:2:1)", tag.replace(" ", "_") + "_split821.png"))
    a, b = MV / "yolo11_cls_split821/yolo11m-cls/results.csv", MV / "yolo11m_cls_e50_split821/results.csv"
    if a.exists() and b.exists():
        made.append(overlay({"30 epochs": a, "50 epochs": b}, "yolo11m-cls_30_vs_50ep_split821.png"))
    f = MV / "yolov7_h100_e100_split821/run/results.txt"
    if f.exists():
        made.append(yolov7(f, "YOLOv7 100ep, 4 classes (split 8:2:1)", "yolov7_e100_split821.png"))
    m = json.loads((MV / "meta.json").read_text(encoding="utf-8"))
    made.append(history(m["history"], ["train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1"],
                        f"EfficientNet-B0 {m['model_version']}", "efficientnet_b0_split821.png", "val_macro_f1"))
    for sub, tag, out in [("partseg", "partseg v2' (serving)", "partseg_v2_split821.png"),
                          ("partseg_v1_split821", "partseg v1'", "partseg_v1_split821.png")]:
        h = json.loads((MV / sub / "metrics.json").read_text(encoding="utf-8"))["history"]
        made.append(history(h, ["train_loss", "val_mask_AP", "val_mask_AP50", "val_fender_mask_AP"],
                            f"Mask R-CNN {tag} (HITL split 8:2:1)", out, "val_mask_AP"))
    return made


if __name__ == "__main__":
    print("\n".join(main()))
