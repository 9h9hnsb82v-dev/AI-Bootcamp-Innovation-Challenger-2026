"""근접 사진 판정 모델 전체 성능표 (8:2:1 분할, 시험셋 1,080장, 같은 규칙).

규칙: 계획 부품을 안다고 가정한 부품별 양품/불량 판정. 임계값 = 검증셋 Youden J, 시험셋 1회 평가.
순위 기준(서비스 지표): 부품별 정확도의 평균(도어·범퍼·휀더 각 360장이라 전체 정확도와 같음).
차이 검정: 시험 이미지 부트스트랩 2,000회로 EfficientNet(서비스) 대비 평균 정확도 차이의 95% 신뢰구간.
실행: .venv/Scripts/python -m ml.vision.model_scoreboard
출력: reports/vision/model_scoreboard.{json,md,csv}, reports/vision/curves/model_scoreboard_*.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.vision.compare_yolo_effnet import OUT, ROOT, youden

MV = ROOT / "models/vision"
PARTS = ["DOOR", "BUMPER", "FENDER"]
BASE = "EfficientNet-B0 (8ep, 서비스)"
SEED = 42


def load() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_csv(OUT / "effnet_split821_scores.csv", encoding="utf-8-sig")
    df["y"] = (df.quality == "DEFECT").astype(int)
    cols, top1 = {BASE: "effnet"}, {}
    meta = json.loads((MV / "meta.json").read_text(encoding="utf-8"))
    top1[BASE] = meta["test_report"]["accuracy"]
    for name, f in [("YOLO11m-cls (30ep)", MV / "yolo11_cls_split821/yolo11m-cls_scores.csv"),
                    ("YOLO11s-cls (30ep)", MV / "yolo11_cls_split821/yolo11s-cls_scores.csv"),
                    ("YOLO11m-cls (50ep)", MV / "yolo11m_cls_e50_split821/yolo11m-cls-e50_scores.csv")]:
        s = pd.read_csv(f)
        m = df[["subset", "stem", "part"]].merge(s, on=["subset", "stem"], how="left", validate="1:1")
        df[name] = [r[f"p_{r.part}_DEFECT"] / (r[f"p_{r.part}_DEFECT"] + r[f"p_{r.part}_GOOD"]) for _, r in m.iterrows()]
        t = m[m.subset == "test"]
        top1[name] = float((t.pred == t.true).mean())
        cols[name] = name
    v7 = pd.read_csv(OUT / "yolo_e100_split821_vs_effnet_scores.csv", encoding="utf-8-sig")[["subset", "stem", "yolo"]]
    df = df.merge(v7, on=["subset", "stem"], how="left", validate="1:1")
    cols["YOLOv7 (100ep, 검출)"] = "yolo"
    assert df[list(cols.values())].notna().all().all(), "점수 누락"
    return df, cols, top1


def auc(p, y) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def main() -> dict:
    df, cols, top1 = load()
    test = df[df.subset == "test"].reset_index(drop=True)
    rows, correct = [], {}
    for name, c in cols.items():
        r = {"model": name}
        corr = np.zeros(len(test), bool)
        for part in PARTS:
            v = df[(df.subset == "val") & (df.part == part)]
            th = youden(v[c].to_numpy(), v.y.to_numpy())
            mask = (test.part == part).to_numpy()
            t = test[mask]
            yh = (t[c].to_numpy() >= th).astype(int)
            y = t.y.to_numpy()
            corr[mask] = yh == y
            r[f"{part}_accuracy"] = float((yh == y).mean())
            r[f"{part}_recall"] = float(yh[y == 1].mean())
            r[f"{part}_specificity"] = float(1 - yh[y == 0].mean())
            r[f"{part}_auc"] = auc(t[c].to_numpy(), y)
            r[f"{part}_threshold"] = th
            if part == "FENDER":
                g = t.subtype.to_numpy() == "단차"
                r["FENDER_gap_accuracy"] = float((yh[g] == y[g]).mean())
        r["mean_accuracy"] = float(corr.mean())
        r["mean_auc"] = float(np.mean([r[f"{p}_auc"] for p in PARTS]))
        r["top1_6cls"] = top1.get(name)
        rows.append(r)
        correct[name] = corr
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(test), (2000, len(test)))
    for r in rows:
        d = correct[r["model"]].astype(float) - correct[BASE].astype(float)
        boots = d[idx].mean(1)
        r["diff_vs_effnet"] = float(d.mean())
        r["diff_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    rows.sort(key=lambda r: -r["mean_accuracy"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    res = {"protocol": __doc__.split("\n")[2:5], "test_n": len(test), "rows": rows}
    (OUT / "model_scoreboard.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "model_scoreboard.csv", index=False, encoding="utf-8-sig")
    f = lambda x: "—" if x is None else f"{x:.3f}"
    md = ["| 순위 | 모델 | 평균 정확도 | 도어 | 범퍼 | 휀더 | 휀더·단차 | 평균 AUC | 6클래스 top-1 | EfficientNet 대비 (95% CI) |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        ci = "기준" if r["model"] == BASE else f"{r['diff_vs_effnet']*100:+.1f}%p ({r['diff_ci95'][0]*100:+.1f} ~ {r['diff_ci95'][1]*100:+.1f})"
        md.append(f"| {r['rank']} | {r['model']} | **{r['mean_accuracy']:.3f}** | {f(r['DOOR_accuracy'])} | {f(r['BUMPER_accuracy'])} | "
                  f"{f(r['FENDER_accuracy'])} | {f(r['FENDER_gap_accuracy'])} | {f(r['mean_auc'])} | {f(r['top1_6cls'])} | {ci} |")
    (OUT / "model_scoreboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    plot_bars(rows)
    plot_epochs()
    print("\n".join(md))
    return res


def _font():
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        from matplotlib import font_manager
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return


def plot_bars(rows) -> None:
    _font()
    fig, ax = plt.subplots(figsize=(13, 4.2))
    keys = [("DOOR_accuracy", "도어"), ("BUMPER_accuracy", "범퍼"), ("FENDER_accuracy", "휀더"),
            ("FENDER_gap_accuracy", "휀더·단차"), ("mean_accuracy", "평균")]
    w = 0.8 / len(rows)
    x = np.arange(len(keys))
    for i, r in enumerate(rows):
        vals = [r[k] for k, _ in keys]
        b = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=f"{r['rank']}. {r['model']}")
        ax.bar_label(b, fmt="%.3f", fontsize=6, rotation=90, padding=2)
    ax.set_xticks(x, [k for _, k in keys])
    ax.set_ylim(0.6, 1.04)
    ax.set_ylabel("시험셋 정확도")
    ax.set_title("근접 사진 양품/불량 판정 정확도 (8:2:1 시험셋 1,080장, 검증 임계값)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(OUT / "curves/model_scoreboard_accuracy.png", dpi=150)
    plt.close(fig)


def plot_epochs() -> None:
    """에폭별 검증 정확도(선 그래프). EfficientNet은 정확도 기록이 없어 macro F1로 별도 표시."""
    _font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, f in [("YOLO11m-cls 30ep", MV / "yolo11_cls_split821/yolo11m-cls/results.csv"),
                     ("YOLO11s-cls 30ep", MV / "yolo11_cls_split821/yolo11s-cls/results.csv"),
                     ("YOLO11m-cls 50ep", MV / "yolo11m_cls_e50_split821/results.csv")]:
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        axes[0].plot(d.epoch, d["metrics/accuracy_top1"], marker=".", label=label)
        axes[1].plot(d.epoch, d["val/loss"], marker=".", label=label)
    h = pd.DataFrame(json.loads((MV / "meta.json").read_text(encoding="utf-8"))["history"])
    axes[0].plot(h.epoch, h.val_macro_f1, marker="s", linestyle="--", label="EfficientNet 8ep (macro F1 - 정확도 기록 없음, 참고)")
    if "val_loss" in h:
        axes[1].plot(h.epoch, h.val_loss, marker="s", linestyle="--", label="EfficientNet 8ep")
    axes[0].set_title("에폭별 검증 6클래스 정확도(top-1)")
    axes[1].set_title("에폭별 검증 손실")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "curves/model_scoreboard_val_by_epoch.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
