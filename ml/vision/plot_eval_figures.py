"""학습 곡선 · 혼동행렬 · 정확도 그래프 (세션 비겹침 분할, 2026-11-19 사용자 요청).

Ultralytics results.png 처럼 '파란 점선 = 실제 값, 주황 점선 = 부드럽게 한 추세'로 그리고, 혼동행렬은 정답(열) 기준으로 정규화한다.
입력: reports/vision/sd_runs/(effnet · yolo11m_cls · det · gap), models/vision/dinov2_vitb14_sd_e15, models/vision/ensemble_sd_dino
실행: .venv/Scripts/python -m ml.vision.plot_eval_figures
출력: reports/figures/eval_*.png
"""
from __future__ import annotations

import ast
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from ml.vision.common import ROOT  # noqa: E402
from ml.vision.ensemble_check import load  # noqa: E402

RUNS = ROOT / "reports/vision/sd_runs"
DINO = ROOT / "models/vision/dinov2_vitb14_sd_e15"
ENS = ROOT / "models/vision/ensemble_sd_dino"
OUT = ROOT / "reports/figures"
BLUE, ORANGE = "#1f77b4", "#ff7f0e"
CLS = ["DOOR_GOOD", "DOOR_DEFECT", "BUMPER_GOOD", "BUMPER_DEFECT", "FENDER_GOOD", "FENDER_DEFECT"]
CLS_KO = {"DOOR_GOOD": "도어 양품", "DOOR_DEFECT": "도어 불량", "BUMPER_GOOD": "범퍼 양품", "BUMPER_DEFECT": "범퍼 불량",
          "FENDER_GOOD": "휀더 양품", "FENDER_DEFECT": "휀더 불량"}
PART_KO = {"DOOR": "도어", "BUMPER": "범퍼", "FENDER": "휀더"}


def _font():
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _hist(meta_path) -> pd.DataFrame:
    h = json.loads(meta_path.read_text(encoding="utf-8"))["history"]
    return pd.DataFrame(ast.literal_eval(h) if isinstance(h, str) else h)


def _panel(ax, x, y, title, legend=False):
    """Ultralytics results.png 모양: 실제 값(파랑 점·선) + 부드러운 추세(주황 점선)."""
    y = np.asarray(y, float)
    ax.plot(x, y, marker="o", ms=4, lw=1.8, color=BLUE, label="결과")
    if len(y) >= 3:
        ax.plot(x, gaussian_filter1d(y, sigma=max(1.0, len(y) / 12)), ":", lw=2, color=ORANGE, label="추세")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(fontsize=8)


def _blank(ax, text="기록 없음"):
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, color="#999", transform=ax.transAxes)


def curves_classification() -> str:
    """분류 모델 4종 × 지표 5칸."""
    e = _hist(RUNS / "effnet/meta.json")
    y = pd.read_csv(RUNS / "yolo11m_cls/results.csv")
    y.columns = [c.strip() for c in y.columns]
    d = pd.DataFrame(json.loads((ROOT / "models/vision/dinov2_vitb14_sd_e15_v2/summary.json").read_text(encoding="utf-8"))["history"])
    g = _hist(RUNS / "gap/meta.json")
    gd = _hist(RUNS / "gap_dino/meta.json")
    rows = [
        ("EfficientNet-B0 (5에폭)", e.epoch, [("train/loss", e.train_loss), ("val/loss", e.val_loss), ("train/accuracy", e.train_acc),
                                           ("val/accuracy (6클래스)", e.val_acc), ("val/macro F1", e.val_macro_f1)]),
        ("YOLO11m-cls (15에폭)", y.epoch, [("train/loss", y["train/loss"]), ("val/loss", y["val/loss"]), None,
                                         ("val/accuracy top1 (6클래스)", y["metrics/accuracy_top1"]),
                                         ("val/accuracy top5", y["metrics/accuracy_top5"])]),
        ("DINOv2 ViT-B/14 (15에폭)", d.epoch, [("train/loss (라벨 스무딩 0.1)", d.train_loss), ("val/loss", d.val_loss),
                                             ("train/accuracy", d.train_acc), ("val/accuracy top1 (6클래스)", d.val_accuracy_top1),
                                             ("val/macro F1", d.val_macro_f1)]),
        ("단차 EfficientNet (10에폭)", g.epoch, [("train/loss", g.train_loss), ("val/loss", g.val_loss), ("train/accuracy", g.train_acc),
                                              ("val/accuracy (단차)", g.val_acc_gt), ("val/AUC (단차)", g.val_auc_gt)]),
        ("단차 DINOv2 (10에폭)", gd.epoch, [("train/loss", gd.train_loss), ("val/loss", gd.val_loss), ("train/accuracy", gd.train_acc),
                                         ("val/accuracy (단차)", gd.val_acc_gt), ("val/AUC (단차)", gd.val_auc_gt)]),
    ]
    fig, axes = plt.subplots(len(rows), 5, figsize=(20, 3.4 * len(rows)))
    for r, (name, x, cells) in enumerate(rows):
        for c, cell in enumerate(cells):
            ax = axes[r, c]
            if cell is None:
                _blank(ax)
            else:
                _panel(ax, x, cell[1], cell[0], legend=(r == 0 and c == 0))
            if c == 0:
                ax.set_ylabel(name, fontsize=12, weight="bold")
    fig.suptitle("학습 곡선 — 이미지 판정 분류 모델 (세션 비겹침 분할: 학습 37,087 · 검증 7,953장)", fontsize=15, weight="bold")
    fig.text(0.5, 0.005, "손실(loss)은 모델마다 정의가 달라 높이를 서로 비교하지 않습니다. 빈칸 = 그 학습에서 기록하지 않은 지표. 주황 점선 = 추세. "
             "DINOv2 줄은 지표를 모두 기록한 재학습(v2, 같은 조건) — 앱 모델(v1)과 시험 성적이 ±1%p 안으로 같음.", ha="center", fontsize=10, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return _save(fig, "eval_curves_classification")


def curves_detection() -> str:
    """결함 위치 검출(YOLO11m detect) — Ultralytics results.png 와 같은 2×5 배치."""
    r = pd.read_csv(RUNS / "det/results.csv")
    r.columns = [c.strip() for c in r.columns]
    cols = ["train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/precision(B)", "metrics/recall(B)",
            "val/box_loss", "val/cls_loss", "val/dfl_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.6))
    for i, c in enumerate(cols):
        _panel(axes.flat[i], r.epoch, r[c], c, legend=(i == 1))
    s = json.loads((RUNS / "det/summary.json").read_text(encoding="utf-8"))["test"]
    fig.suptitle("학습 곡선 — 결함 위치 검출 YOLO11m (15에폭 · 640px · 표시 전용)", fontsize=15, weight="bold")
    fig.text(0.5, 0.008, f"시험셋: mAP50 {s['mAP50']:.3f} · mAP50-95 {s['mAP50_95']:.3f} · 정밀도 {s['precision']:.3f} · 재현율 {s['recall']:.3f}   "
             f"(스크래치 {s['per_class_mAP50']['scratch']:.3f} · 외관 손상 {s['per_class_mAP50']['exterior_damage']:.3f} · "
             f"단차 불량 {s['per_class_mAP50']['gap_defect']:.3f} · 단차 정상 {s['per_class_mAP50']['gap_normal']:.3f})",
             ha="center", fontsize=10.5, color="#444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    return _save(fig, "eval_curves_detection")


def _cm_plot(ax, cm, xlabels, ylabels, title, counts=None):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v < 0.005 and (counts is None or counts[i, j] == 0):
                continue
            txt = f"{v:.2f}" + (f"\n({counts[i, j]:,})" if counts is not None else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="white" if v > 0.55 else "#222")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=9)
    ax.set_xlabel("정답 (True)", fontsize=10)
    ax.set_ylabel("예측 (Predicted)", fontsize=10)
    ax.set_title(title, fontsize=11.5, weight="bold")


def _cm6(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    t = df[df.subset == "test"]
    probs = t[[f"p_{c}" for c in CLS]].to_numpy()
    pred = np.array(CLS)[probs.argmax(1)]
    cm = pd.crosstab(pd.Categorical(pred, CLS), pd.Categorical(t.true, CLS), dropna=False).to_numpy().astype(float)
    return cm / np.maximum(cm.sum(0, keepdims=True), 1), float((pred == t.true.to_numpy()).mean())


def confusion_6cls() -> str:
    srcs = [("EfficientNet-B0", RUNS / "effnet/effnet_scores.csv"), ("YOLO11m-cls", RUNS / "yolo11m_cls/yolo11m-cls_scores.csv"),
            ("DINOv2 ViT-B/14", DINO / "dinov2_vitb14_scores.csv")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 8.2))
    labels = [CLS_KO[c] for c in CLS]
    for ax, (name, p) in zip(axes, srcs):
        df = pd.read_csv(p)
        if "true" not in df:  # 서비스 EfficientNet 점수 형식
            df["true"] = df["part"] + "_" + df["quality"]
        cm, acc = _cm6(df)
        _cm_plot(ax, cm, labels, labels, f"{name} · 6클래스 top-1 {acc:.1%}")
    fig.suptitle("혼동행렬 (정답 기준 정규화) — 시험 6,946장 · 학습에 쓰지 않은 촬영일 9일", fontsize=15, weight="bold")
    fig.text(0.5, 0.01, "6클래스 = 부품(도어·범퍼·휀더) × 양품/불량. 앱 판정은 계획 부품을 알고 양품/불량만 보므로 부품을 틀려도 판정에는 덜 영향.",
             ha="center", fontsize=10, color="#555")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    return _save(fig, "eval_confusion_6class")


def _app_decisions() -> pd.DataFrame:
    """현재 앱 조합(EfficientNet + DINOv2 + 2차 판정)의 시험셋 판정: 불량 의심 / 양품 추정 / 작업자 확인."""
    meta = json.loads((ENS / "meta.json").read_text(encoding="utf-8"))
    te = json.loads((ENS / "effnet/meta.json").read_text(encoding="utf-8"))["thresholds"]
    td = json.loads((ENS / "dinov2/meta.json").read_text(encoding="utf-8"))["thresholds"]
    E, D = load(str(RUNS / "effnet/effnet_scores.csv")), load(str(DINO / "dinov2_vitb14_scores.csv"))
    d = E.rename(columns={"score": "e"}).merge(D[["subset", "stem", "score"]].rename(columns={"score": "dv"}), on=["subset", "stem"])
    d = d[d.subset == "test"].copy()
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    out = []
    for part, g in d.groupby("part"):
        sk = meta["stacking"]["parts"][part]
        sp = 1 / (1 + np.exp(-(sk["coef"][0] * lg(g.e) + sk["coef"][1] * lg(g.dv) + sk["intercept"])))
        de, dd, ds = g.e >= te[part], g.dv >= td[part], sp >= sk["threshold"]
        auto = (de == dd) & (ds == de) & (np.abs(sp - sk["threshold"]) >= meta["stacking"]["uncertain_margin"])
        out.append(g.assign(pred=np.where(~auto, "작업자 확인", np.where(ds, "불량 의심", "양품 추정"))))
    return pd.concat(out)


def confusion_app() -> str:
    d = _app_decisions()
    rows, cols = ["불량 의심", "양품 추정", "작업자 확인"], ["DEFECT", "GOOD"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, part in zip(axes, ("DOOR", "BUMPER", "FENDER")):
        g = d[d.part == part]
        cnt = pd.crosstab(pd.Categorical(g.pred, rows), pd.Categorical(g.quality, cols), dropna=False).to_numpy()
        cm = cnt / np.maximum(cnt.sum(0, keepdims=True), 1)
        auto = g[g.pred != "작업자 확인"]
        acc = ((auto.pred == "불량 의심") == (auto.quality == "DEFECT")).mean()
        _cm_plot(ax, cm, ["불량", "양품"], rows, f"{PART_KO[part]} ({len(g):,}장) · 자동 판정 정확도 {acc:.1%}", counts=cnt)
    fig.suptitle("현재 앱 판정 혼동행렬 — EfficientNet-B0 + DINOv2 + 2차 판정 (시험 6,946장)", fontsize=15, weight="bold")
    fig.text(0.5, 0.01, "'작업자 확인' = 세 판정이 엇갈리거나 2차 확률이 기준 ±0.10 안(처리 대기함). 칸 숫자: 정답 기준 비율 (장수). "
             "자동 판정 정확도 = 작업자 확인을 뺀 사진의 정확도.", ha="center", fontsize=10, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return _save(fig, "eval_confusion_app")


def accuracy_summary() -> str:
    ev = json.loads((ROOT / "reports/vision/dino_sd_eval.json").read_text(encoding="utf-8"))
    sg, combo = ev["single"], ev["combo"]
    series = [("EfficientNet-B0", [sg["EfficientNet-B0"][p]["acc"] for p in ("DOOR", "BUMPER", "FENDER")], "#9ecae1"),
              ("YOLO11m-cls", [sg["YOLO11m-cls"][p]["acc"] for p in ("DOOR", "BUMPER", "FENDER")], "#bdbdbd"),
              ("DINOv2 ViT-B/14", [sg["DINOv2 ViT-B/14"][p]["acc"] for p in ("DOOR", "BUMPER", "FENDER")], "#3182bd"),
              ("이전 앱 (EfficientNet+YOLO) 자동 판정", [combo["현재 앱 (EfficientNet+YOLO)"][p]["auto_acc"] for p in ("DOOR", "BUMPER", "FENDER")], "#fdae6b"),
              ("현재 앱 (EfficientNet+DINO) 자동 판정", [combo["EfficientNet+DINO"][p]["auto_acc"] for p in ("DOOR", "BUMPER", "FENDER")], "#2E7D3A")]
    fig, ax = plt.subplots(figsize=(15, 6.4))
    x = np.arange(3)
    w = 0.16
    for i, (name, vals, col) in enumerate(series):
        b = ax.bar(x + (i - 2) * w, np.array(vals) * 100, w, label=name, color=col, edgecolor="white")
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v * 100 + 0.4, f"{v * 100:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, ["도어 (2,269장)", "범퍼 (1,015장)", "휀더 (3,662장)"], fontsize=12)
    ax.set_ylim(70, 102)
    ax.set_ylabel("정확도 (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    ax.set_title("부품별 정확도 — 시험 6,946장 (부품별 검증셋 임계값, 자동 판정 = 작업자 확인 제외)", fontsize=14, weight="bold")
    fig.tight_layout()
    return _save(fig, "eval_accuracy_summary")


def detection_confusion_copy() -> str:
    dst = OUT / "eval_confusion_detection.png"
    shutil.copy2(RUNS / "det/confusion_matrix.png", dst)  # Ultralytics 원본(장수 기준)
    return str(dst)


def _save(fig, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


if __name__ == "__main__":
    _font()
    for fn in (curves_classification, curves_detection, confusion_6cls, confusion_app, accuracy_summary, detection_confusion_copy):
        print(fn())
