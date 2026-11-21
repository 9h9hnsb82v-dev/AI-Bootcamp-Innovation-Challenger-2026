"""이전 앱 서비스 모델 vs 현재 앱 모델 정확도 비교표 (같은 시험셋 1,080장 = AI-Hub 공식 Validation, 클래스당 180장).

이전 앱: EfficientNet-B0 8ep(12,300장 서브셋, ~11/11) → YOLO11m-cls 30ep(같은 서브셋, 11/11~11/13)
현재 앱: 원본 전체 51,986장으로 재학습한 EfficientNet-B0 · YOLO11m-cls + 2차 판정 모델 → 서비스 규칙
        (세 판정 일치 + 2차 확률이 기준 ±0.10 밖이면 자동 판정, 아니면 작업자 확인)
규칙: 계획 부품을 안다고 가정, 각 모델 자기 검증셋에서 정한 부품별 임계값(시험셋 미사용). 2차 판정 모델은 meta.json 값.
차이: 이전 서비스(YOLO11m-cls 30ep) 대비 평균 정확도, 부품별 층화 부트스트랩 2,000회 95% CI.
부품 오차단: 올바른 부품 사진인데 다른 부품으로 0.95 이상 확신해 등록이 막히는 비율.
실행: .venv/Scripts/python -m ml.vision.app_model_comparison
출력: reports/vision/app_model_comparison.{json,md,csv}, reports/vision/curves/app_model_comparison.png(막대), app_model_comparison_lines.png(선)
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.vision.compare_yolo_effnet import OUT, ROOT, youden
from ml.vision.ensemble_check import logit

MV = ROOT / "models/vision"
PARTS = ["DOOR", "BUMPER", "FENDER"]
CLS = [f"{p}_{q}" for p in PARTS for q in ("GOOD", "DEFECT")]
PREV = "이전② YOLO11m-cls 30ep"
FILES = {
    "이전① EfficientNet-B0 8ep": OUT / "effnet_split821_scores.csv",
    PREV: MV / "yolo11_cls_split821/yolo11m-cls_scores.csv",
    "현재 EfficientNet-B0 전체 20ep": MV / "effnet_full_e20/named-outputs/results/effnet_scores.csv",
    "현재 YOLO11m-cls 전체 30ep": MV / "yolo11m_cls_full_e30/named-outputs/results/yolo11m-cls_scores.csv",
}
PERIOD = {"이전① EfficientNet-B0 8ep": "이전 앱 (~11/11)", PREV: "이전 앱 (11/11~11/13)",
          "현재 EfficientNet-B0 전체 20ep": "현재 앱 (구성원)", "현재 YOLO11m-cls 전체 30ep": "현재 앱 (구성원)",
          "현재 2차 판정 모델 단독": "현재 앱 (구성원)", "현재 서비스 (세 판정 + 작업자 확인)": "현재 앱 (서비스 규칙)"}
DATA = {"이전① EfficientNet-B0 8ep": "12,300장 서브셋", PREV: "12,300장 서브셋"}
SEED = 42


def key(stem: str) -> str:
    return (stem.split("_", 2)[-1] if stem.split("_")[0] in PARTS else stem).lower()


def load(path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    d["key"] = d.stem.map(key)
    if "effnet" in d:  # 이전 EfficientNet 점수 파일: 6클래스 확률 대신 부품 조건부 점수·부품 추정
        d["score"], d["pp"], d["pc"] = d.effnet, d.pred_part, d.part_conf
        d["true"] = d.part + "_" + d.quality
        d["top1"] = np.nan
        return d[["subset", "key", "true", "score", "pp", "pc", "top1"]]
    part = d.true.str.split("_").str[0]
    d["score"] = [r[f"p_{p}_DEFECT"] / max(r[f"p_{p}_DEFECT"] + r[f"p_{p}_GOOD"], 1e-12) for p, (_, r) in zip(part, d.iterrows())]
    pp = np.stack([d[f"p_{p}_GOOD"] + d[f"p_{p}_DEFECT"] for p in PARTS], 1)
    d["pp"], d["pc"] = np.array(PARTS)[pp.argmax(1)], pp.max(1)
    d["top1"] = (d.pred == d.true).astype(float)
    return d[["subset", "key", "true", "score", "pp", "pc", "top1"] + [f"p_{c}" for c in CLS]]


def main() -> dict:
    test = pd.read_csv(OUT / "effnet_split821_scores.csv", encoding="utf-8-sig")
    test = test[test.subset == "test"].assign(key=lambda x: x.stem.map(key))[["key", "part", "quality", "subtype"]]
    test["y"] = (test.quality == "DEFECT").astype(int)
    test = test.sort_values(["part", "key"]).reset_index(drop=True)
    dec, score, info = {}, {}, {}
    full = {}
    for name, f in FILES.items():
        d = load(f)
        t = test[["key"]].merge(d[d.subset == "test"], on="key", how="left", validate="1:1")
        assert t.score.notna().all(), f"{name}: 시험 점수 누락"
        yh = np.zeros(len(test), int)
        ths = {}
        for part in PARTS:
            v = d[(d.subset == "val") & (d.true.str.startswith(part))]
            ths[part] = youden(v.score.to_numpy(), v.true.str.endswith("DEFECT").astype(int).to_numpy())
            m = (test.part == part).to_numpy()
            yh[m] = t.score.to_numpy()[m] >= ths[part]
        dec[name], score[name] = yh, t.score.to_numpy()
        fb = (t.pp.to_numpy() != test.part.to_numpy()) & (t.pc.to_numpy() >= 0.95)
        info[name] = {"thresholds": ths, "false_block": float(fb.mean()),
                      "top1": float(t.top1.mean()) if t.top1.notna().all() else None}
        if name.startswith("현재"):
            full[name] = t
    # 현재 앱: 2차 판정 모델 + 서비스 규칙 (meta.json 의 계수·임계값 그대로)
    meta = json.loads((MV / "ensemble_full/meta.json").read_text(encoding="utf-8"))
    eff_n, yol_n = "현재 EfficientNet-B0 전체 20ep", "현재 YOLO11m-cls 전체 30ep"
    th_e = json.loads((MV / "ensemble_full/effnet/meta.json").read_text(encoding="utf-8"))["thresholds"]
    th_y = json.loads((MV / "ensemble_full/yolo11m_cls/meta.json").read_text(encoding="utf-8"))["thresholds"]
    sp, sth = np.zeros(len(test)), np.zeros(len(test))
    for part in PARTS:
        m = (test.part == part).to_numpy()
        c = meta["stacking"]["parts"][part]
        z = c["coef"][0] * logit(score[eff_n][m]) + c["coef"][1] * logit(score[yol_n][m]) + c["intercept"]
        sp[m], sth[m] = 1 / (1 + np.exp(-z)), c["threshold"]
    de = np.array([s >= th_e[p] for s, p in zip(score[eff_n], test.part)])
    dy = np.array([s >= th_y[p] for s, p in zip(score[yol_n], test.part)])
    ds = sp >= sth
    auto = (de == dy) & (ds == de) & (np.abs(sp - sth) >= meta["stacking"]["uncertain_margin"])
    avg = {c: (full[eff_n][f"p_{c}"].to_numpy() + full[yol_n][f"p_{c}"].to_numpy()) / 2 for c in CLS}
    ppa = np.stack([avg[f"{p}_GOOD"] + avg[f"{p}_DEFECT"] for p in PARTS], 1)
    fb_avg = float(((np.array(PARTS)[ppa.argmax(1)] != test.part.to_numpy()) & (ppa.max(1) >= 0.95)).mean())
    top1_avg = float((np.array(CLS)[np.stack([avg[c] for c in CLS], 1).argmax(1)] == (test.part + "_" + test.quality).to_numpy()).mean())
    dec["현재 2차 판정 모델 단독"], score["현재 2차 판정 모델 단독"] = ds.astype(int), sp
    info["현재 2차 판정 모델 단독"] = {"false_block": fb_avg, "top1": top1_avg}
    SVC = "현재 서비스 (세 판정 + 작업자 확인)"
    dec[SVC], score[SVC] = ds.astype(int), sp
    info[SVC] = {"false_block": fb_avg, "top1": top1_avg}

    y = test.y.to_numpy()
    gap = (test.part == "FENDER").to_numpy() & (test.subtype == "단차").to_numpy()
    rng = np.random.default_rng(SEED)
    pidx = {p: np.where(test.part == p)[0] for p in PARTS}
    boot = [np.concatenate([rng.choice(ix, len(ix)) for ix in pidx.values()]) for _ in range(2000)]
    rows = []
    for name in list(FILES) + ["현재 2차 판정 모델 단독", SVC]:
        cover = auto if name == SVC else np.ones(len(test), bool)
        ok = dec[name] == y
        r = {"model": name, "period": PERIOD[name], "data": DATA.get(name, "원본 전체 51,986장"),
             "auto_rate": float(cover.mean())}
        for p in PARTS:
            m = (test.part == p).to_numpy() & cover
            r[f"{p}_accuracy"] = float(ok[m].mean())
            r[f"{p}_auc"] = float(roc_auc_score(y[test.part == p], score[name][test.part == p]))
            r[f"{p}_auto_rate"] = float(cover[(test.part == p).to_numpy()].mean())
        r["FENDER_gap_accuracy"] = float(ok[gap & cover].mean())
        r["mean_accuracy"] = float(np.mean([r[f"{p}_accuracy"] for p in PARTS]))
        r["mean_auc"] = float(np.mean([r[f"{p}_auc"] for p in PARTS]))
        r["top1_6cls"] = info[name]["top1"]
        r["part_false_block_rate"] = info[name]["false_block"]
        if name != SVC:
            base = dec[PREV] == y
            diffs = [np.mean([ok[b][test.part.to_numpy()[b] == p].mean() - base[b][test.part.to_numpy()[b] == p].mean()
                              for p in PARTS]) for b in boot]
            r["diff_vs_prev_service"] = r["mean_accuracy"] - float(np.mean([base[test.part == p].mean() for p in PARTS]))
            r["diff_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        rows.append(r)
    if rows[0]["top1_6cls"] is None:  # 이전 EfficientNet: 6클래스 top-1은 학습 메타의 시험 보고값
        rows[0]["top1_6cls"] = json.loads((MV / "meta.json").read_text(encoding="utf-8"))["test_report"]["accuracy"]
    res = {"test_set": "AI-Hub 공식 Validation 1,080장(클래스당 180), 모든 모델 학습에 미사용", "rows": rows,
           "note": "현재 서비스 행의 정확도는 자동 판정된 사진만의 정확도(나머지는 작업자 확인)"}
    (OUT / "app_model_comparison.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "app_model_comparison.csv", index=False, encoding="utf-8-sig")
    f = lambda x: "—" if x is None else f"{x:.3f}"
    md = ["| 구분 | 모델 | 학습 데이터 | 평균 정확도 | 도어 | 범퍼 | 휀더 | 휀더·단차 | 평균 AUC | 6클래스 top-1 | 부품 오차단 | 자동 판정 | 이전 서비스 대비 (95% CI) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        ci = ("기준" if r["model"] == PREV else
              f"{r['diff_vs_prev_service']*100:+.1f}%p ({r['diff_ci95'][0]*100:+.1f} ~ {r['diff_ci95'][1]*100:+.1f})"
              if "diff_ci95" in r else "자동 판정분 기준")
        md.append(f"| {r['period']} | {r['model']} | {r['data']} | **{r['mean_accuracy']:.3f}** | {f(r['DOOR_accuracy'])} | "
                  f"{f(r['BUMPER_accuracy'])} | {f(r['FENDER_accuracy'])} | {f(r['FENDER_gap_accuracy'])} | {f(r['mean_auc'])} | "
                  f"{f(r['top1_6cls'])} | {r['part_false_block_rate']:.1%} | {r['auto_rate']:.1%} | {ci} |")
    (OUT / "app_model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    plot(rows)
    print("\n".join(md))
    return res


PART_KO = {"DOOR": "도어", "BUMPER": "범퍼", "FENDER": "휀더"}
COLORS = {"DOOR": "#2563eb", "BUMPER": "#059669", "FENDER": "#ea580c", "mean": "#334155"}


def _labels(rows) -> list[str]:
    return [r["model"].replace("현재 서비스 (세 판정 + 작업자 확인)", "현재 서비스 (자동 판정분)").replace(" ", "\n", 1) for r in rows]


def plot(rows) -> None:
    """막대: 모델별 도어·범퍼·휀더·평균 정확도(모든 막대 위에 값 표시)."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 5.6))
    x = np.arange(len(rows))
    series = [(PART_KO[p], [r[f"{p}_accuracy"] for r in rows], COLORS[p]) for p in PARTS]
    series.append(("평균", [r["mean_accuracy"] for r in rows], COLORS["mean"]))
    for i, (lab, vals, col) in enumerate(series):
        bars = ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                    rotation=90, fontweight="bold" if lab == "평균" else "normal")
    ax.axvline(1.5, ls=":", c="gray")
    ax.text(0.5, 1.035, "이전 앱", ha="center")
    ax.text(3.5, 1.035, "현재 앱", ha="center")
    ax.set_xticks(x, _labels(rows), fontsize=9)
    ax.set_ylim(0.7, 1.06)
    ax.set_yticks(np.arange(0.70, 1.001, 0.05))
    ax.set_ylabel("시험셋 정확도 (1,080장)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "curves/app_model_comparison.png", dpi=130)
    plt.close(fig)
    plot_lines(rows)


def plot_lines(rows) -> None:
    """선: 이전① → 현재 서비스 순서로 부품별·평균 정확도와 자동 판정 비율 변화(Azure ML timeline/* 와 같은 값)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [3, 1.4]})
    x = np.arange(1, len(rows) + 1)
    for p in PARTS + ["mean"]:
        vals = [r["mean_accuracy" if p == "mean" else f"{p}_accuracy"] for r in rows]
        ax.plot(x, vals, marker="o", lw=2.6 if p == "mean" else 1.8, color=COLORS[p], label=PART_KO.get(p, "평균"))
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, -14 if p in ("FENDER", "DOOR", "mean") else 7),
                        ha="center", fontsize=8, color=COLORS[p], fontweight="bold" if p == "mean" else "normal")
    ax.axvspan(0.6, 2.5, color="#e2e8f0", alpha=.5, lw=0)
    ax.text(1.55, 0.715, "이전 앱", ha="center", fontsize=10)
    ax.text(4.5, 0.715, "현재 앱", ha="center", fontsize=10)
    ax.set_xticks(x, _labels(rows), fontsize=8.5)
    ax.set_xlim(0.6, len(rows) + 0.4)
    ax.set_ylim(0.70, 1.01)
    ax.set_ylabel("시험셋 정확도 (1,080장)")
    ax.set_title("모델 변화에 따른 정확도 (현재 서비스 = 자동 판정분)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.3)
    fb = [r["part_false_block_rate"] * 100 for r in rows]
    au = [r["auto_rate"] * 100 for r in rows]
    ax2.plot(x, fb, marker="o", color="#b91c1c", label="부품 오차단 (%)")
    for xi, v in zip(x, fb):
        ax2.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#b91c1c")
    ax3 = ax2.twinx()
    ax3.plot(x, au, marker="s", color="#7c3aed", label="자동 판정 비율 (%)")
    for xi, v in zip(x, au):
        ax3.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#7c3aed")
    ax2.set_ylim(0, 4.5)
    ax3.set_ylim(85, 102)
    ax2.set_ylabel("부품 오차단 (%)", color="#b91c1c")
    ax3.set_ylabel("자동 판정 비율 (%)", color="#7c3aed")
    ax2.set_xticks(x, [f"{i}" for i in x])
    ax2.set_xlabel("모델 순서 (왼쪽 그래프와 같음)")
    ax2.set_title("부품 오차단 · 자동 판정 비율")
    ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "curves/app_model_comparison_lines.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
