"""발표용 시연 시나리오를 별도 DB에 재현한다(기존 앱 DB는 건드리지 않음).

실행:
  .venv/Scripts/python -m scripts.demo_scenario            # data/app/demo.db 생성(있으면 재생성)
  $env:RESOURCE_DB="data/app/demo.db"; .venv/Scripts/streamlit run ui/app.py

시나리오(모든 값은 가상 시연값):
 1. CN7_LH 5,000개 + DOOR/BUMPER/FENDER 각 1,000개 확정 계획 → 사전 예측 kg
 2. BUMPER 고유 400개 검사(폐기 12·양품 388), 600개 미생산 → 70kg → 84kg
 3. KAMP ID40 Product_c0 수주예측 → 경고 확인 → FENDER 초안(수량 직접 수정) → 확정
 4. 보관량 7일 예측·수거 추천
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEMO_DB = ROOT / "data/app/demo.db"


def main(start: date | None = None) -> dict:
    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    if DEMO_DB.exists():
        DEMO_DB.unlink()
    os.environ["RESOURCE_DB"] = str(DEMO_DB)
    from services import inspection_registry as ir
    from services import production_plan as pp
    from services import supply_to_plan_draft as sd
    from services.collection_recommendation import recommend_collection
    from services.storage_projection import get_storage_projection
    from services.supply_forecast import forecast_order_quantity
    from services.waste_forecast import predict_preproduction_waste
    from services.waste_forecast_update import update_waste_forecast

    start = start or date.today()
    d1 = (start + timedelta(days=1)).isoformat()
    log = []
    plans = {p: pp.create_production_plan(p, d1, n) for p, n in
             [("CN7_LH", 5000), ("DOOR", 1000), ("BUMPER", 1000), ("FENDER", 1000)]}
    for p, plan in plans.items():
        r = predict_preproduction_waste(plan["plan_id"])
        log.append(f"[1] {p} {plan['planned_units']}개 → 예상 {r['expected_scrap_kg']:.2f} kg ({', '.join(r['provenance'])})")

    b = plans["BUMPER"]["plan_id"]
    for i in range(400):
        uid = f"B{i:04d}"
        ir.register_inspection(b, uid, f"demo-bumper-{uid}".encode(), None)
        ir.record_operator_decision(b, uid, "CONFIRMED_SCRAP" if i < 12 else "CONFIRMED_GOOD", "demo_operator",
                                    decision_date=start.isoformat())
    u = update_waste_forecast(b)
    log.append(f"[2] BUMPER 검사 후: 최초 {u['initial_expected_kg']:.2f} kg → 갱신 {u['updated_expected_total_kg']:.2f} kg "
               f"(확인 폐기 {u['confirmed_scrap_kg']:.1f} + 남은 예상 {u['remaining_expected_kg']:.1f}) 버킷 {u['buckets']}")

    f = forecast_order_quantity("Product_c0", "2022-03-12", 1)
    log.append(f"[3] ID40 Product_c0 {f['target_date']} 예정 수주량 예측 {f['predicted_planned_order_units']:.1f}개 "
               f"(실제 {f['actual_planned_order_units']:.0f}개) — 개수, kg 아님")
    draft = sd.create_demo_plan_draft_from_order_forecast(f["forecast_id"], "FENDER", 300,
                                                          (start + timedelta(days=3)).isoformat(),
                                                          acknowledge_mapping_warning=True)
    conf = sd.confirm_demo_plan_draft(draft["draft_id"], user_confirmed=True, acknowledge_mapping_warning=True)
    r = predict_preproduction_waste(conf["plan"]["plan_id"])
    log.append(f"[3] 사용자 선택 FENDER 300개 초안 확정 → {conf['plan']['plan_id']} 예상 {r['expected_scrap_kg']:.2f} kg "
               f"({', '.join(r['provenance'])})")

    s = get_storage_projection(start.isoformat(), 7)
    c = recommend_collection(start.isoformat(), 7)
    log.append(f"[4] 보관량 현재 {s['current_stock_kg']:.1f} → 3일 {s['stock_after_3_days_kg']:.1f} → 7일 "
               f"{s['stock_after_7_days_kg']:.1f} kg; 수거: {c['status']} {c.get('recommended_collection_date') or ''}")
    log += prepare_demo_photos()
    return {"db": str(DEMO_DB), "log": log}


DEMO_PHOTOS = ROOT / "data/demo/photos"


def prepare_demo_photos(n_car: int = 3) -> list[str]:
    """[5] 2단계 검사 시연용 사진: 차량 전체(HITL 시험셋, CC0, 학습 미사용) + 부품 근접(AI-Hub 시험셋, 로컬 전용).

    선택은 정답 라벨 기준(모델 예측 아님). 결과 확인 문구에는 1단계 모델 결과를 함께 적는다.
    """
    import json
    import shutil
    from services.part_locator import PART_KO
    out = []
    ps = ROOT / "data/processed/partseg"
    if (ps / "ann_test.json").exists():
        recs = json.loads((ps / "ann_test.json").read_text(encoding="utf-8"))
        need = {"Fender", "Front-door", "Front-bumper"}
        picks = [r for r in recs if need <= {o["label"] for o in r["objects"]}][:n_car]
        (DEMO_PHOTOS / "car").mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(picks):
            dst = DEMO_PHOTOS / "car" / f"car_{i + 1}.jpg"
            shutil.copy(ps / r["file"], dst)
            try:
                from ml.vision.partseg_infer import locate
                res = locate(dst.read_bytes())
                seen = ", ".join(PART_KO[p] for p, v in res.get("parts", {}).items() if v["present"]) or res.get("status")
            except ImportError:
                seen = "모델 없음"
            out.append(f"[5] 차량 전체 사진 {dst.relative_to(ROOT)} (정답: 휀더·앞도어·앞범퍼 포함) → 1단계 인식: {seen}")
    yolo = ROOT / "data/processed/aihub/yolo"
    if (yolo / "index.csv").exists():
        import pandas as pd
        idx = pd.read_csv(yolo / "index.csv", encoding="utf-8-sig")
        (DEMO_PHOTOS / "closeup").mkdir(parents=True, exist_ok=True)
        for part in ("FENDER", "DOOR", "BUMPER"):
            for q in ("DEFECT", "GOOD"):
                r = idx[(idx.subset == "test") & (idx.part == part) & (idx.quality == q)].iloc[0]
                dst = DEMO_PHOTOS / "closeup" / f"{part.lower()}_{q.lower()}.jpg"
                shutil.copy(yolo / "images/test" / f"{r.stem}.jpg", dst)
                seen = ""
                try:
                    from ml.vision.infer import predict
                    from services.inspection_registry import PART_MISMATCH_BLOCK_CONF
                    a = predict(dst.read_bytes(), expected_part=part, with_cam=False)
                    if a.get("status") == "OK":
                        verdict = {"AI_SUSPECTED_DEFECT": "결함 의심", "AI_PASS": "양품 추정",
                                   "AI_DISAGREE": "작업자 확인(" + "; ".join(a.get("check_reasons", [])) + ")"}[a["verdict"]]
                        detail = (", ".join(f"{m['name']} {m['defect_prob']:.3f}/{m['threshold']:.2f}" for m in a["members"])
                                  if a.get("members") else f"{a['defect_prob']:.3f} (기준 {a['threshold']:.2f})")
                        seen = f" → 2단계 AI({a['backend']}): {verdict} {detail}"
                        if a["predicted_part"] != part:
                            blocked = a["part_confidence"] >= PART_MISMATCH_BLOCK_CONF
                            seen += (f", 부품 오인식 {a['predicted_part']} {a['part_confidence']:.3f} → "
                                     + ("등록 차단(오차단, 작업자 수기 판정 필요)" if blocked else "작업자 확인 표시"))
                except ImportError:
                    seen = " → 2단계 모델 없음"
                out.append(f"[5] 근접 사진 {dst.relative_to(ROOT)} (정답: {part} {q}, AI-Hub 시험셋){seen}")
    return out


if __name__ == "__main__":
    out = main()
    print("\n".join(out["log"]))
    print("demo DB:", out["db"])
