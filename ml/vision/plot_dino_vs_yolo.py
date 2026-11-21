"""YOLO11m-cls vs DINOv2 ViT-B/14 학습 곡선 비교 그림 (세션 비겹침 분할, 같은 448px·15에폭, 2026-11-18).

실행: .venv/Scripts/python -m ml.vision.plot_dino_vs_yolo <yolo results.csv> <dino summary.json>
출력: reports/vision/curves/dino_vs_yolo11m_sd.png
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ml.vision.compare_yolo_effnet import ROOT  # noqa: E402

YC, DC = "#4C78A8", "#E45756"


def main(yolo_csv: str, dino_json: str) -> str:
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    y = pd.read_csv(yolo_csv)
    y.columns = [c.strip() for c in y.columns]
    s = json.loads(open(dino_json, encoding="utf-8").read())
    d = pd.DataFrame(s["history"])
    d["time"] = d["sec"].cumsum()
    ev = json.loads((ROOT / "reports/vision/dino_sd_eval.json").read_text(encoding="utf-8"))
    n = {k: sum(v for kk, v in s["counts"].items() if kk.startswith(k + "/")) for k in ("train", "val", "test")}

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    # (1) 검증 정확도(같은 검증셋 7,953장, 6클래스 top-1)
    a = ax[0]
    a.plot(y.epoch, y["metrics/accuracy_top1"] * 100, "o-", color=YC, label="YOLO11m-cls")
    a.plot(d.epoch, d.val_accuracy_top1 * 100, "s-", color=DC, label="DINOv2 ViT-B/14")
    for df, col, c in ((y, "metrics/accuracy_top1", YC), (d, "val_accuracy_top1", DC)):
        i = df[col].idxmax()
        a.annotate(f"최고 {df[col][i] * 100:.1f}% ({int(df.epoch[i])}에폭)", (df.epoch[i], df[col][i] * 100),
                   xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color=c, weight="bold")
    a.set(title=f"검증 정확도 (6클래스 top-1, 검증 {n['val']:,}장)", xlabel="에폭", ylabel="정확도 (%)", ylim=(70, 92))
    a.set_xticks(range(1, 16))
    a.grid(alpha=0.3)
    a.legend(loc="lower right")
    # (2) 학습 시간 대비 검증 정확도(H100)
    a = ax[1]
    a.plot(y.time / 60, y["metrics/accuracy_top1"] * 100, "o-", color=YC, label=f"YOLO11m-cls (총 {y.time.iloc[-1] / 60:.0f}분)")
    a.plot(d.time / 60, d.val_accuracy_top1 * 100, "s-", color=DC, label=f"DINOv2 (총 {d.time.iloc[-1] / 60:.0f}분)")
    a.set(title="학습 시간 대비 검증 정확도 (H100)", xlabel="누적 학습 시간 (분)", ylabel="정확도 (%)", ylim=(70, 92))
    a.grid(alpha=0.3)
    a.legend(loc="lower right")
    # (3) 학습 손실(정의가 달라 절대값 비교 불가, 추세만)
    a = ax[2]
    a.plot(y.epoch, y["train/loss"], "o-", color=YC, label="YOLO11m-cls 학습 손실")
    a.plot(y.epoch, y["val/loss"], "o--", color=YC, alpha=0.55, label="YOLO11m-cls 검증 손실")
    a.plot(d.epoch, d.train_loss, "s-", color=DC, label="DINOv2 학습 손실 (라벨 스무딩 0.1)")
    a.set(title="손실 추세 (모델별 손실 정의가 달라 높이 비교는 의미 없음)", xlabel="에폭", ylabel="손실")
    a.title.set_fontsize(10)
    a.set_xticks(range(1, 16))
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

    sg = ev["single"]
    t = (f"시험셋 {n['test']:,}장 정확도 — YOLO11m: 도어 {sg['YOLO11m-cls']['DOOR']['acc'] * 100:.1f}% · 범퍼 "
         f"{sg['YOLO11m-cls']['BUMPER']['acc'] * 100:.1f}% · 휀더 {sg['YOLO11m-cls']['FENDER']['acc'] * 100:.1f}%   |   "
         f"DINOv2: 도어 {sg['DINOv2 ViT-B/14']['DOOR']['acc'] * 100:.1f}% · 범퍼 {sg['DINOv2 ViT-B/14']['BUMPER']['acc'] * 100:.1f}% · "
         f"휀더 {sg['DINOv2 ViT-B/14']['FENDER']['acc'] * 100:.1f}%  (부품별 검증셋 임계값)")
    fig.suptitle(f"YOLO11m-cls vs DINOv2 ViT-B/14 — 같은 이미지·분할(학습 {n['train']:,} / 검증 {n['val']:,} / 시험 {n['test']:,}장), "
                 "448px·15에폭", fontsize=13, weight="bold")
    fig.text(0.5, 0.015, t, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = ROOT / "reports/vision/curves/dino_vs_yolo11m_sd.png"
    fig.savefig(out, dpi=150)
    return str(out)


if __name__ == "__main__":
    print(main(*sys.argv[1:3]))
