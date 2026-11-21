"""근접 사진 판정 백엔드(YOLO11m-cls / EfficientNet) 공통 출력 규칙 검증. 모델·패키지가 없으면 건너뜀."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "data/demo/photos/closeup"


def _ready(b):
    from ml.vision.infer import status
    return status(b)["status"] == "OK" and (PHOTOS / "door_defect.jpg").exists()


@pytest.mark.parametrize("b", ["yolo11m_cls", "effnet", "effnet_full", "yolo11m_full", "ensemble"])
def test_backend_output_contract(b):
    if not _ready(b):
        pytest.skip(f"{b} 모델 또는 데모 사진 없음")
    from ml.vision.infer import predict
    from PIL import Image
    for name, part in [("door_defect", "DOOR"), ("bumper_good", "BUMPER"), ("fender_defect", "FENDER")]:
        data = (PHOTOS / f"{name}.jpg").read_bytes()
        r = predict(data, expected_part=part, backend_name=b)
        assert r["status"] == "OK" and r["backend"] == b
        assert set(r["class_probs"]) == {f"{p}_{q}" for p in ("DOOR", "BUMPER", "FENDER") for q in ("GOOD", "DEFECT")}
        assert abs(sum(r["class_probs"].values()) - 1) < 1e-4
        assert 0 <= r["defect_prob"] <= 1 and 0 < r["threshold"] < 1
        assert r["evaluated_part"] == part
        cam = np.array(r["cam"])
        assert cam.ndim == 2 and cam.max() <= 1 + 1e-6
        if b.startswith("yolo"):  # 가운데 정사각형 전처리 → 히트맵 격자는 원본 비율을 따름
            w, h = Image.open(PHOTOS / f"{name}.jpg").size
            assert abs(cam.shape[1] / cam.shape[0] - w / h) < 0.15


def test_yolo11_uses_per_part_validation_thresholds():
    if not _ready("yolo11m_cls"):
        pytest.skip("YOLO11 모델 없음")
    import json
    from ml.vision.infer import YOLO_META, predict
    th = json.loads(YOLO_META.read_text(encoding="utf-8"))["thresholds"]
    data = (PHOTOS / "door_defect.jpg").read_bytes()
    for part in ("DOOR", "BUMPER", "FENDER"):
        assert predict(data, expected_part=part, with_cam=False, backend_name="yolo11m_cls")["threshold"] == th[part]


def test_ensemble_rule_agree_or_disagree():
    """조합 모델: 세 판정이 같고 2차 확률이 애매하지 않으면 그 판정, 아니면 AI_DISAGREE."""
    if not _ready("ensemble"):
        pytest.skip("조합 모델 없음")
    from ml.vision.infer import predict
    for name, part in [("door_defect", "DOOR"), ("bumper_good", "BUMPER"), ("fender_defect", "FENDER")]:
        data = (PHOTOS / f"{name}.jpg").read_bytes()
        r = predict(data, expected_part=part, with_cam=False, backend_name="ensemble")
        e, y, s = r["members"]
        margin = 0.10
        auto = e["verdict"] == y["verdict"] == s["verdict"] and abs(s["defect_prob"] - s["threshold"]) >= margin
        assert r["verdict"] == (s["verdict"] if auto else "AI_DISAGREE")
        assert r["agree"] == auto and (not auto) == bool(r["check_reasons"])
        assert r["defect_prob"] == s["defect_prob"] and r["ai_recommendation"] == s["verdict"]
        # 2차 판정 모델은 두 점수에 대해 단조 증가(가중치 양수)
        from ml.vision.infer import ENS_META, _stack
        import json
        meta = json.loads(ENS_META.read_text(encoding="utf-8"))
        assert _stack(meta, part, 0.9, 0.9)[0] > _stack(meta, part, 0.1, 0.1)[0]


def test_disagree_is_recorded_for_operator_without_kg_change():
    from services import inspection_registry as ir
    from services import production_plan as pp
    from services.waste_forecast_update import compute_update
    plan = pp.create_production_plan("FENDER", "2026-11-13", 100)
    before = compute_update(plan["plan_id"])["updated_expected_total_kg"]
    ai = {"status": "OK", "predicted_part": "FENDER", "part_confidence": 0.99, "defect_prob": 0.4, "threshold": 0.3,
          "verdict": "AI_DISAGREE", "model_version": "t"}
    rec = ir.register_inspection(plan["plan_id"], "U1", b"fender", ai)
    assert rec["review_status"] == "AI_DISAGREE"
    u = compute_update(plan["plan_id"])
    assert u["confirmed_scrap_kg"] == 0 and u["updated_expected_total_kg"] == before
    assert ir.unit_buckets(plan["plan_id"])["units_with_ai_result_awaiting_operator"] == 1


def test_defect_boxes_display_only_contract():
    """YOLO11 결함 위치 박스: 표시 클래스(스크래치·외관 손상)만, 기준 이상만, 정규화 좌표. 판정 결과에는 영향 없음."""
    from ml.vision.infer import DET_META, det_ready, detect_defects
    if not det_ready():
        pytest.skip("검출 모델 없음")
    import json
    meta = json.loads(DET_META.read_text(encoding="utf-8"))
    for name in ("door_defect", "bumper_good"):
        boxes = detect_defects((PHOTOS / f"{name}.jpg").read_bytes())
        for b in boxes:
            assert b["name"] in meta["conf"] and b["conf"] >= meta["conf"][b["name"]]
            assert all(0 <= v <= 1 for v in b["xyxyn"])
