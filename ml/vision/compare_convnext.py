"""표면 판정 조합 3가지 비교 (사용자 요청 2026-11-19): ① 지금 조합 EfficientNet+DINOv2 ② ConvNeXt+DINOv2 ③ EfficientNet+ConvNeXt+DINOv2.

- 세 모델 모두 같은 세션 비겹침 분할(manifest_sd.csv)로 학습: EfficientNet-B0(bright_kumquat) · DINOv2 ViT-B/14(good_bean) · ConvNeXt-Small(keen_rhubarb).
- 앱 규칙 그대로: 부품별 검증셋 Youden 임계값, 2차 판정 = 로지스틱 회귀(검증셋, 5겹 교차예측 기준값),
  모든 모델 판정과 2차 판정이 같고 2차 확률이 기준 ±0.10 밖이면 자동 판정, 아니면 작업자 확인.
- 휀더 단차도 같은 조합으로: 단차 DINOv2 점수 + 표면 점수들의 2차 판정(앱 규칙: 단차 판정 = 2차 판정 + 여유 밖이면 자동).
- 시험셋은 성적표에만 1회. 차이의 95% 신뢰구간은 시험 사진 짝지은 부트스트랩(2,000회).
실행: .venv/Scripts/python -m ml.vision.compare_convnext <convnext_scores.csv>
출력: reports/vision/convnext_compare.{json,md}
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ml.vision.common import ROOT
from ml.vision.compare_yolo_effnet import youden
from ml.vision.ensemble_check import load
from ml.vision.fit_dino_ensemble import MARGIN, PARTS, _apply, _fit

RUNS = ROOT / "reports/vision/sd_runs"
NAMES = {"e": "EfficientNet-B0", "c": "ConvNeXt-Small", "dv": "DINOv2"}
COMBOS = {"① 지금 조합 (EfficientNet+DINOv2)": ["e", "dv"],
          "② ConvNeXt+DINOv2": ["c", "dv"],
          "③ 세 모델 (EfficientNet+ConvNeXt+DINOv2)": ["e", "c", "dv"]}


def _surface(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """부품별로 검증셋에서 규칙을 정하고 시험 사진마다 (정답, 2차 판정, 자동 여부)를 돌려준다."""
    out = []
    for part in PARTS:
        g = d[d.part == part]
        v, t = g[g.subset == "val"], g[g.subset == "test"]
        sk = _fit(v, cols)
        sp = _apply(sk, t, cols)
        ds = sp >= sk["threshold"]
        agree = np.ones(len(t), bool)
        for c in cols:
            agree &= (t[c].to_numpy() >= youden(v[c].to_numpy(), v.y.to_numpy())) == ds
        out.append(pd.DataFrame({"stem": t.stem.to_numpy(), "part": part, "y": t.y.to_numpy().astype(bool), "pred": ds,
                                 "auto": agree & (np.abs(sp - sk["threshold"]) >= MARGIN)}))
    return pd.concat(out, ignore_index=True)


def _gap(G: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    v, t = G[G.subset == "val"], G[G.subset == "test"]
    th = youden(v.score_jitter.to_numpy(), v.y.to_numpy())
    sk = _fit(v, cols + ["score_jitter"])
    sp = _apply(sk, t, cols + ["score_jitter"])
    ds = sp >= sk["threshold"]
    auto = ((t.score_jitter.to_numpy() >= th) == ds) & (np.abs(sp - sk["threshold"]) >= MARGIN)
    return pd.DataFrame({"stem": t.stem.to_numpy(), "part": "FENDER_GAP", "y": t.y.to_numpy().astype(bool), "pred": ds, "auto": auto})


def _summ(r: pd.DataFrame) -> dict:
    a = r.auto.to_numpy()
    ok = r.pred.to_numpy() == r.y.to_numpy()
    return {"n": int(len(r)), "2차판정_정확도": float(ok.mean()), "자동판정_비율": float(a.mean()),
            "자동판정_정확도": float(ok[a].mean()) if a.any() else None, "작업자_확인_장수": int((~a).sum()),
            "자동판정_오판_장수": int((a & ~ok).sum())}


def _ci(base: pd.DataFrame, other: pd.DataFrame, n: int = 2000) -> dict:
    """짝지은 부트스트랩: (other - base) 의 자동 판정 비율·자동 판정 정확도·2차 판정 정확도 차이 95% 구간."""
    rng = np.random.default_rng(0)
    m = base.merge(other, on=["stem", "part"], suffixes=("_b", "_o"))
    okb, oko = (m.pred_b == m.y_b).to_numpy(), (m.pred_o == m.y_o).to_numpy()
    ab, ao = m.auto_b.to_numpy(), m.auto_o.to_numpy()
    res = {"자동판정_비율": [], "자동판정_정확도": [], "2차판정_정확도": []}
    for _ in range(n):
        i = rng.integers(0, len(m), len(m))
        res["자동판정_비율"].append(ao[i].mean() - ab[i].mean())
        res["자동판정_정확도"].append(oko[i][ao[i]].mean() - okb[i][ab[i]].mean())
        res["2차판정_정확도"].append(oko[i].mean() - okb[i].mean())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}


def main(cnx_csv: str) -> dict:
    E = load(str(RUNS / "effnet/effnet_scores.csv")).rename(columns={"score": "e"})
    D = load(str(ROOT / "models/vision/dinov2_vitb14_sd_e15/dinov2_vitb14_scores.csv"))[["subset", "stem", "score"]].rename(columns={"score": "dv"})
    C = load(cnx_csv)[["subset", "stem", "score"]].rename(columns={"score": "c"})
    d = E.merge(D, on=["subset", "stem"], validate="1:1").merge(C, on=["subset", "stem"], validate="1:1")
    d["y"] = (d.quality == "DEFECT").astype(int)
    single = {}
    for k, nm in NAMES.items():
        accs = []
        for part in PARTS:
            g = d[d.part == part]
            v, t = g[g.subset == "val"], g[g.subset == "test"]
            accs.append(((t[k].to_numpy() >= youden(v[k].to_numpy(), v.y.to_numpy())) == t.y.to_numpy().astype(bool)))
        single[nm] = float(np.concatenate(accs).mean())
    G = pd.read_csv(ROOT / "models/vision/gap_dinov2_sd/gap_scores.csv").merge(d[["subset", "stem", "e", "c", "dv"]], on=["subset", "stem"], how="inner")
    G["y"] = (G.quality == "DEFECT").astype(int)
    rows, res = {}, {}
    for name, cols in COMBOS.items():
        s, gp = _surface(d, cols), _gap(G, cols)
        rows[name] = (s, gp)
        res[name] = {"표면_전체": _summ(s), "표면_부품별": {p: _summ(s[s.part == p]) for p in PARTS}, "휀더_단차": _summ(gp)}
    base = list(COMBOS)[0]
    for name in list(COMBOS)[1:]:
        res[name]["표면_차이_95CI_vs_①"] = _ci(rows[base][0], rows[name][0])
        res[name]["단차_차이_95CI_vs_①"] = _ci(rows[base][1], rows[name][1])
    out = {"test_n_surface": int((d.subset == "test").sum()), "test_n_gap": int((G.subset == "test").sum()),
           "단독_정확도(부품별 임계값)": single, "조합": res}
    (ROOT / "reports/vision/convnext_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    pct = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    ci = lambda c: f"{c[0] * 100:+.1f} ~ {c[1] * 100:+.1f}%p"
    L = ["# 표면 판정 조합 비교 — EfficientNet · ConvNeXt · DINOv2 (세션 비겹침 시험)", "",
         f"- 표면 시험 {out['test_n_surface']:,}장 · 휀더 단차 시험 {out['test_n_gap']:,}장. 임계값·2차 판정은 검증셋으로만 정함(앱 규칙 그대로).",
         "- 학습: 같은 분할·448px. EfficientNet-B0 5에폭, DINOv2 ViT-B/14 15에폭, ConvNeXt-Small 15에폭(keen_rhubarb_zg7x46gs5g, Apache-2.0 가중치).", "",
         "## 단독 모델 (부품별 검증 임계값)", "", "| 모델 | 시험 정확도 |", "|---|---:|"]
    L += [f"| {k} | {pct(v)} |" for k, v in single.items()]
    L += ["", "## 조합별 앱 규칙 성적 (표면 전체)", "",
          "| 조합 | 2차 판정 정확도 | 자동 판정 비율 | 자동 판정 정확도 | 작업자 확인 | 자동 오판 |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in res.items():
        s = r["표면_전체"]
        L.append(f"| {name} | {pct(s['2차판정_정확도'])} | {pct(s['자동판정_비율'])} | {pct(s['자동판정_정확도'])} | {s['작업자_확인_장수']:,}장 | {s['자동판정_오판_장수']:,}장 |")
    L += ["", "## 부품별 (자동 판정 비율 / 자동 판정 정확도)", "", "| 조합 | " + " | ".join(PARTS) + " |", "|---|" + "---:|" * len(PARTS)]
    for name, r in res.items():
        L.append(f"| {name} | " + " | ".join(f"{pct(r['표면_부품별'][p]['자동판정_비율'])} / {pct(r['표면_부품별'][p]['자동판정_정확도'])}" for p in PARTS) + " |")
    L += ["", "## 휀더 단차 (단차 DINOv2 + 표면 점수 2차 판정)", "",
          "| 조합 | 2차 판정 정확도 | 자동 판정 비율 | 자동 판정 정확도 | 작업자 확인 |", "|---|---:|---:|---:|---:|"]
    for name, r in res.items():
        s = r["휀더_단차"]
        L.append(f"| {name} | {pct(s['2차판정_정확도'])} | {pct(s['자동판정_비율'])} | {pct(s['자동판정_정확도'])} | {s['작업자_확인_장수']:,}장 |")
    L += ["", "## ①과의 차이 95% 신뢰구간 (짝지은 부트스트랩)", "", "| 조합 | 대상 | 자동 판정 비율 | 자동 판정 정확도 | 2차 판정 정확도 |", "|---|---|---:|---:|---:|"]
    for name in list(COMBOS)[1:]:
        for lab, key in (("표면", "표면_차이_95CI_vs_①"), ("단차", "단차_차이_95CI_vs_①")):
            c = res[name][key]
            L.append(f"| {name} | {lab} | {ci(c['자동판정_비율'])} | {ci(c['자동판정_정확도'])} | {ci(c['2차판정_정확도'])} |")
    L += ["", "구간이 0을 포함하면 ①과 차이가 있다고 말하기 어렵다."]
    (ROOT / "reports/vision/convnext_compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
