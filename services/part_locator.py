"""1단계: 차량 전체 사진에서 계획 부품 위치 확인 → 근접 촬영 안내.

불변식
- 이 단계는 DB에 아무것도 쓰지 않는다(검사 기록·개체 버킷·stock_ledger·kg 불변).
- 결함 판정은 2단계(inspection_registry.inspect_part_image, 근접 사진)만 한다.
- 계획 부품이 안 보여도 차단하지 않고 경고만 한다(촬영 각도·가림 때문일 수 있음).
"""
from __future__ import annotations

from services.config import ServiceError
from services.inspection_registry import IMAGE_PRODUCTS, _check_plan_for_images
from services.production_plan import get_plan

PART_KO = {"DOOR": "도어", "BUMPER": "범퍼", "FENDER": "휀더"}


def _locate(image_bytes: bytes) -> dict:
    try:
        from ml.vision.partseg_infer import locate
        return locate(image_bytes)
    except ImportError:
        return {"status": "MODEL_NOT_READY", "message": "torch 미설치"}


def locate_plan_part(plan_id: str, image_bytes: bytes) -> dict:
    if not image_bytes:
        raise ServiceError("INVALID_INPUT", "차량 전체 사진이 비어 있습니다.")
    plan = get_plan(plan_id)
    _check_plan_for_images(plan)
    target = plan["product_id"]
    res = _locate(image_bytes)
    out = {"plan_id": plan_id, "plan_part": target, "status": res.get("status"), "provenance": ["PARTSEG_GUIDE_ONLY"],
           "kg_changed": False}
    if res.get("status") != "OK":
        out["message"] = res.get("message", "")
        out["guidance"] = f"부품 위치 확인 모델을 쓸 수 없습니다. {PART_KO[target]} 표면을 가까이 찍어 바로 검사하세요."
        return out
    parts = res["parts"]
    found = parts[target]["present"]
    others = [PART_KO[p] for p in IMAGE_PRODUCTS if p != target and parts[p]["present"]]
    out.update(model_version=res["model_version"], parts=parts, detections=res["detections"], overlay=res["overlay"],
               plan_part_found=found)
    if found:
        out["guidance"] = (f"사진에서 계획 부품 {PART_KO[target]} 영역을 찾았습니다(점수 {parts[target]['max_score']:.2f}). "
                           f"표시된 {PART_KO[target]} 표면을 가까이 찍어 아래 검사에 올리세요.")
    else:
        extra = f" 사진에는 {', '.join(others)} 영역이 보입니다." if others else ""
        out["guidance"] = (f"사진에서 계획 부품 {PART_KO[target]}를 찾지 못했습니다.{extra} 다른 각도에서 다시 찍거나, "
                           f"실물을 확인한 뒤 {PART_KO[target]} 표면을 가까이 찍어 검사하세요. (차단하지 않음)")
    return out
