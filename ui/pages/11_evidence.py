"""근거·성능 — 데이터·모델 근거와 주의 사항을 한 화면에 (2026-11-14 팀원 앱 반영). 실제 평가 결과 파일만 읽어 표시한다."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from services.config import ROOT
from services.provenance import PROVENANCE

st.title("근거·성능")
st.caption("데이터·모델 근거 (저장된 실제 평가 결과만 표시). 자세한 내용은 reports/ 폴더의 보고서를 참고하세요.")


def load(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


ov = load("reports/vision/session_overlap.json")
if ov:
    t, h = ov["sets"]["test"], ov["sets"]["holdout"]
    st.warning(f"**평가셋 주의(이전 분할):** AI-Hub 시험 {t['n']:,}장의 {t['within_60s']:.1%}, 추가 시험 {h['n']:,}장의 {h['within_60s']:.1%}가 "
               f"학습 사진과 같은 촬영 세션(60초 이내)이었습니다. 촬영 시각이 완전히 같은 사진도 약 {t['identical_timestamp']:.0%}입니다. "
               "그래서 촬영일 통째로 다시 나눠(세션 비겹침) 새로 학습·평가했고, 아래 수치와 검사 화면 안내는 모두 그 정직한 수치입니다.",
               icon="⚠️")

# 2026-11-19: 지금 서비스 중인 조합(serving.json)의 세션 비겹침 시험 성적을 먼저 보여 준다.
serving = (load("models/vision/serving.json") or {}).get("backend")
cur = load("models/vision/ensemble_sd_dino/meta.json") if serving == "ensemble_dino" else None
st.subheader("이미지 검사 모델 (현재 서비스)")
if cur:
    ev, tm = cur["evaluation"], cur["test_part_metrics"]
    st.markdown(f"**{cur['model_version']}** — {' + '.join(cur['member_names'])} + 2차 판정 로지스틱 회귀 · {cur.get('license_note', '')}")
    st.caption(cur.get("rule", ""))
    st.dataframe(pd.DataFrame([{"부품": p, "시험 장수": f"{v['n']:,}", "EfficientNet": f"{v['EfficientNet_정확도']:.3f}",
                                "DINOv2": f"{v['DINOv2_정확도']:.3f}", "2차 판정": f"{v['2차판정_단독_정확도']:.3f}",
                                "자동 판정 비율": f"{v['자동판정_비율']:.1%}", "자동 판정 정확도": f"{v['자동판정_정확도']:.3f}"}
                               for p, v in tm.items()]), hide_index=True)
    st.caption(f"평가셋: {ev['set']} · 분할: {cur.get('split', '')} · 학습 사진과 같은 촬영일 없음. 자세한 내용: {ev.get('report', '')}")
    gm = load("models/vision/ensemble_sd_dino/" + cur.get("gap_model", "gap") + "/meta.json")
    g = ev.get("gap")
    if g:
        name = gm.get("model_version", "") if gm else ""
        prev = ev.get("gap_previous_effnet")
        prev_txt = f" (이전 EfficientNet 단차: 자동 판정 {prev['자동판정_비율']:.1%}, 정확도 {prev['자동판정_정확도']:.3f})" if prev else ""
        st.markdown(f"**휀더 틈(단차) 가이드 촬영** — {name}: 시험 {g['n']:,}장 · 단차 모델 정확도 {g['단차모델_정확도']:.3f} · "
                    f"자동 판정 {g['자동판정_비율']:.1%}, 자동 판정 정확도 {g['자동판정_정확도']:.3f}{prev_txt}. "
                    "AI-Hub 사진에서 가이드 촬영을 흉내 낸 평가이며 실제 카메라는 미검증. 자세한 내용: reports/vision/gap_dino_eval.md")
else:
    st.caption("현재 서비스 조합의 세션 비겹침 평가 파일이 없어 표시하지 않습니다.")

with st.expander("이전 조합 기록 (EfficientNet + YOLO11m, 2026-11-18까지 서비스) — 참고용"):
    sd = load("reports/vision/sd_eval.json")
    if sd:
        sp = sd["split"]
        st.markdown("**세션 비겹침 재평가** (같은 분할, 이전 조합)")
        st.caption(f"촬영일 통째 분할 · 시험 {sp['images']['test']:,}장(촬영일 {sp['days']['test']}일) · "
                   f"학습 사진과 60초 이내 겹침 {sp['test_within_60s_of_trainval']:.1%}. 자세한 내용: reports/vision/sd_eval.md")
        st.dataframe(pd.DataFrame([{"부품": p, "시험 장수": v["n_test"], "EfficientNet": f"{v['effnet_acc']:.3f}",
                                    "YOLO11m-cls": f"{v['yolo_acc']:.3f}", "2차 판정": f"{v['stack_acc']:.3f}",
                                    "자동 판정 비율": f"{v['auto_rate']:.1%}", "자동 판정 정확도": f"{v['auto_acc']:.3f}"}
                                   for p, v in sd["surface"].items() if not p.startswith("_")]), hide_index=True)
    ens = load("models/vision/ensemble_full/meta.json")
    if ens:
        e = ens["evaluation"]
        st.markdown(f"**{ens['model_version']}** — EfficientNet-B0(20에폭, BSD) + YOLO11m-cls(30에폭, AGPL-3.0). "
                    "아래 수치는 학습 사진과 촬영 세션이 겹친 평가셋이라 **실제보다 높게** 나온 값입니다.")
        st.dataframe(pd.DataFrame([{"부품": p, "자동 판정 비율": f"{v['auto_rate']:.1%}", "자동 판정 정확도": f"{v['auto_accuracy']:.3f}",
                                    "2차 판정 단독 정확도": f"{v['stack_accuracy_all']:.3f}"} for p, v in e["by_part"].items()]),
                     hide_index=True)
        st.caption(f"평가셋: {e['set']} · 전체 자동 판정 {e['auto_rate']:.1%}, 자동 판정 정확도 {e['auto_mean_accuracy']:.3f}")
    gap = load("models/vision/gap/meta.json")
    if gap:
        rr = gap["rule_candidates"]["test+holdout"][gap["chosen_rule"]]
        st.markdown(f"**이전 단차 모델** — {gap['model_version']}: 자동 판정 {rr['auto_rate']:.1%}, 자동 판정 정확도 {rr['auto_accuracy']:.3f} (이전 분할)")

st.subheader("표시·안내용 모델")
det = load("models/vision/yolo11m_det/meta.json")
if det:
    st.markdown(f"**결함 위치 박스(표시 전용)** — {det['model_version']}: 시험 mAP50 스크래치 {det['test']['per_class_mAP50']['scratch']:.3f}, "
                f"외관 손상 {det['test']['per_class_mAP50']['exterior_damage']:.3f} · 단차는 검출이 약해 표시 안 함")
ps = load("models/vision/partseg/meta.json")
if ps:
    st.markdown(f"**차량 전체 사진 부품 위치(Mask R-CNN, 안내용)** — {ps.get('model_version', '')}")

st.subheader("KAMP (생산계획 사전예측 · 센서)")
base = load("models/kamp/baseline_rates.json")
if base:
    st.caption(f"계획용 불량률: {base['method']} · 분할 {base['split_rule']}")
    st.dataframe(pd.DataFrame([{"품목": k, "추정 불량률": f"{v['estimated_defect_rate']:.3%}", "관측 비율": f"{v['raw_rate']:.3%}",
                                "95% 구간": f"{v['ci95_low']:.3%} ~ {v['ci95_high']:.3%}"} for k, v in base["products"].items()]),
                 hide_index=True)
xgb = load("models/kamp/xgb_sensor_meta.json")
if xgb:
    st.markdown(f"**{xgb['model_version']}** (센서 입력 시에만, kg 사용 여부: {xgb.get('usable_for_kg')}) — {xgb.get('note', '')}")
sup = load("models/supply/metrics.json")
if sup:
    st.markdown(f"**KAMP ID40 수주량 예측 {sup['model_version']}** — 목표: {sup['target']} ({sup['unit']}), 데이터: {sup['data_policy']}. "
                "생산계획·kg에 자동 연결하지 않음.")

st.subheader("출처 태그 설명")
st.dataframe(pd.DataFrame([{"태그": k, "의미": v} for k, v in PROVENANCE.items()]), hide_index=True)
