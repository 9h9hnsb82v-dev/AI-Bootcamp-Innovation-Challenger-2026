"""두 근접 사진 판정 모델의 조합(2차 검증) 효과 검증 - 멘토링 제안(2026-11-13).

질문: 모델 A(예: EfficientNet-B0)와 모델 B(예: YOLO11m-cls)의 양품/불량 판단이 엇갈릴 때
      2차 판정을 하면 단일 모델보다 정확해지는가?

공통 규칙(기존 비교와 동일): 계획 부품을 안다고 가정, 점수 = P(부품_DEFECT)/(P(부품_DEFECT)+P(부품_GOOD)),
부품별 임계값·조합 가중치는 검증셋(val)에서만 정하고 시험셋은 한 번만 평가한다.

비교 방법
- A 단독, B 단독            : 각자 검증셋 Youden 임계값
- 평균(soft voting)         : 두 점수 평균 → 검증셋 Youden 임계값
- 확신도 우선(멘토 규칙)    : 둘이 같으면 그대로, 엇갈리면 자기 임계값에서 더 멀리 떨어진(logit 거리) 모델을 따름
- 2차 판정 모델(stacking)   : 부품별 로지스틱 회귀 [logit A, logit B] - 검증셋 5겹 교차예측으로 임계값 결정
- 엇갈림 → 작업자 확인      : 두 모델이 일치한 사진만 자동 판정, 엇갈린 사진은 작업자 확인(자동 비율 함께 보고)
평균 정확도 = 도어·범퍼·휀더 정확도의 단순 평균. 단일 최고 모델 대비 차이는 부품별 층화 부트스트랩 95% CI(2,000회).

실행: .venv/Scripts/python -m ml.vision.ensemble_check A이름=점수.csv B이름=점수.csv [--tag 접미사]
점수 CSV: (subset, stem) + [effnet 열] 또는 [true, p_<CLASS> 열]. subset ∈ val/test(/holdout)
출력: reports/vision/ensemble_check{tag}.json, .md
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ml.vision.compare_yolo_effnet import OUT, youden
from ml.vision.train_fender import binary_metrics

PARTS = ("DOOR", "BUMPER", "FENDER")
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    if "effnet" in d:  # 서비스 EfficientNet 점수 파일 형식
        return d[["subset", "stem", "part", "quality", "effnet"]].rename(columns={"effnet": "score"})
    d["part"], d["quality"] = d.true.str.split("_").str[0], d.true.str.split("_").str[1]
    d["score"] = [r[f"p_{r.part}_DEFECT"] / max(r[f"p_{r.part}_DEFECT"] + r[f"p_{r.part}_GOOD"], 1e-12) for _, r in d.iterrows()]
    return d[["subset", "stem", "part", "quality", "score"]]


def decide(v: pd.DataFrame, t: pd.DataFrame) -> dict[str, np.ndarray]:
    """한 부품의 검증셋(v)으로 규칙을 정하고 평가셋(t)의 판정(0/1)을 낸다."""
    yv = v.y.to_numpy()
    tha, thb = youden(v.a.to_numpy(), yv), youden(v.b.to_numpy(), yv)
    da, db = (t.a.to_numpy() >= tha).astype(int), (t.b.to_numpy() >= thb).astype(int)
    avg_th = youden(((v.a + v.b) / 2).to_numpy(), yv)
    ma, mb = np.abs(logit(t.a) - logit(tha)), np.abs(logit(t.b) - logit(thb))
    Xv, Xt = np.c_[logit(v.a), logit(v.b)], np.c_[logit(t.a), logit(t.b)]
    lr = LogisticRegression(C=1.0, max_iter=1000)
    oof = cross_val_predict(lr, Xv, yv, cv=5, method="predict_proba")[:, 1]
    st_th = youden(oof, yv)
    lr.fit(Xv, yv)
    return {"A": da, "B": db,
            "평균": (((t.a + t.b) / 2).to_numpy() >= avg_th).astype(int),
            "확신도 우선": np.where(da == db, da, np.where(ma >= mb, da, db)),
            "2차 판정 모델": (lr.predict_proba(Xt)[:, 1] >= st_th).astype(int),
            "_agree": (da == db), "_coef": lr.coef_[0].round(3).tolist(),
            "_th": {"A": tha, "B": thb, "평균": avg_th, "2차 판정 모델": st_th}}


def mean_acc(dec: dict, ys: dict, idx: dict | None = None) -> float:
    accs = []
    for part in PARTS:
        i = idx[part] if idx is not None else slice(None)
        accs.append(float((dec[part][i] == ys[part][i]).mean()))
    return float(np.mean(accs))


def evaluate(df: pd.DataFrame, eval_subsets: list[str], names: tuple[str, str]) -> dict:
    out, decs, ys = {"subsets": eval_subsets, "parts": {}}, {}, {}
    for part in PARTS:
        g = df[df.part == part]
        v, t = g[g.subset == "val"], g[g.subset.isin(eval_subsets)]
        d = decide(v, t)
        y = t.y.to_numpy()
        ys[part] = y
        decs[part] = d
        out["parts"][part] = {"n": int(len(t)), "thresholds": d["_th"], "stack_coef_[A,B]": d["_coef"],
                              "disagree_rate": float(1 - d["_agree"].mean()),
                              "acc_when_agree": float((d["A"][d["_agree"]] == y[d["_agree"]]).mean()),
                              "A_acc_when_disagree": float((d["A"][~d["_agree"]] == y[~d["_agree"]]).mean()) if (~d["_agree"]).any() else None,
                              **{k: binary_metrics(d[k].astype(float), y, 0.5) for k in ("A", "B", "평균", "확신도 우선", "2차 판정 모델")}}
    methods = ["A", "B", "평균", "확신도 우선", "2차 판정 모델"]
    out["mean_accuracy"] = {m: mean_acc({p: decs[p][m] for p in PARTS}, ys) for m in methods}
    best_single = max(("A", "B"), key=lambda m: out["mean_accuracy"][m])
    rng = np.random.default_rng(42)
    boots = {m: [] for m in methods if m not in ("A", "B")}
    for _ in range(2000):
        idx = {p: rng.integers(0, len(ys[p]), len(ys[p])) for p in PARTS}
        base = mean_acc({p: decs[p][best_single] for p in PARTS}, ys, idx)
        for m in boots:
            boots[m].append(mean_acc({p: decs[p][m] for p in PARTS}, ys, idx) - base)
    out["vs_best_single"] = {"best_single": f"{best_single} ({names[0] if best_single == 'A' else names[1]})",
                             **{m: {"diff": out["mean_accuracy"][m] - out["mean_accuracy"][best_single],
                                    "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]} for m, b in boots.items()}}
    agree = {p: decs[p]["_agree"] for p in PARTS}
    out["defer_to_operator"] = {"auto_rate": float(np.mean(np.concatenate(list(agree.values())))),
                                "auto_mean_accuracy": float(np.mean([(decs[p]["A"][agree[p]] == ys[p][agree[p]]).mean() for p in PARTS])),
                                "by_part_auto_rate": {p: float(agree[p].mean()) for p in PARTS}}
    return out


def main(a_spec: str, b_spec: str, tag: str = "") -> dict:
    (na, pa), (nb, pb) = a_spec.split("=", 1), b_spec.split("=", 1)
    A, B = load(pa), load(pb)
    df = A.merge(B[["subset", "stem", "score"]], on=["subset", "stem"], suffixes=("_a", "_b"), validate="1:1")
    assert len(df) == len(A) == len(B), f"점수 행 불일치 A={len(A)} B={len(B)} 결합={len(df)}"
    df = df.rename(columns={"score_a": "a", "score_b": "b"})
    df["y"] = (df.quality == "DEFECT").astype(int)
    res = {"A": na, "B": nb, "protocol": __doc__.split("비교 방법")[0].strip(),
           "n": df.subset.value_counts().to_dict()}
    sets = [["test"]] + ([["holdout"], ["test", "holdout"]] if (df.subset == "holdout").any() else [])
    res["results"] = {"+".join(s): evaluate(df, s, (na, nb)) for s in sets}
    label = {"A": f"A 단독 ({na})", "B": f"B 단독 ({nb})", "평균": "평균(soft voting)",
             "확신도 우선": "확신도 우선(멘토 규칙)", "2차 판정 모델": "2차 판정 모델(stacking)"}
    lines = [f"# 모델 조합(2차 검증) 검증: A={na}, B={nb}", ""]
    for key, r in res["results"].items():
        lines += [f"## 평가셋: {key} ({sum(p['n'] for p in r['parts'].values()):,}장)", "",
                  "| 방법 | 도어 | 범퍼 | 휀더 | 평균 | 단일 최고 대비 (95% CI) |", "|---|---:|---:|---:|---:|---|"]
        for m in ("A", "B", "평균", "확신도 우선", "2차 판정 모델"):
            accs = [r["parts"][p][m]["accuracy"] for p in PARTS]
            vs = r["vs_best_single"].get(m)
            ci = f"{vs['diff']*100:+.1f}%p ({vs['ci95'][0]*100:+.1f} ~ {vs['ci95'][1]*100:+.1f})" if vs else ""
            lines.append(f"| {label[m]} | " + " | ".join(f"{x:.3f}" for x in accs) + f" | {r['mean_accuracy'][m]:.3f} | {ci} |")
        dfr = r["defer_to_operator"]
        lines += ["", f"- 두 모델 판단 일치율 {dfr['auto_rate']:.1%} → 일치한 사진만 자동 판정 시 평균 정확도 {dfr['auto_mean_accuracy']:.3f}, "
                  f"나머지 {1-dfr['auto_rate']:.1%}는 작업자 확인 (부품별 자동 비율 " +
                  ", ".join(f"{p} {v:.1%}" for p, v in dfr["by_part_auto_rate"].items()) + ")",
                  "- 엇갈린 사진에서 A 정확도: " + ", ".join(
                      f"{p} {r['parts'][p]['A_acc_when_disagree']:.2f}" for p in PARTS if r["parts"][p]["A_acc_when_disagree"] is not None), ""]
    (OUT / f"ensemble_check{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"ensemble_check{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tag", default="")
    x = ap.parse_args()
    main(x.a, x.b, x.tag)
