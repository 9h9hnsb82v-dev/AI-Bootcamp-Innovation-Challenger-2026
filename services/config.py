"""설정 로더. `configs/product_master.json`이 7종 품목의 유일한 진실원본이며 수정하지 않는다."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
KAMP_BASELINE_FILE = ROOT / "models/kamp/baseline_rates.json"
SCRAP_FRACTION = 1.0  # 시뮬레이션 가정: 생산 전 불량 완제품 전량 폐기 (runner/sprue/purge 미포함)


class ServiceError(Exception):
    """사용자에게 그대로 보여줄 수 있는 업무 오류. code 는 상태 코드(MODEL_NOT_READY 등)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def product_master() -> dict:
    return _load("product_master.json")


def supply_policy() -> dict:
    return _load("supply_policy.json")


def facility() -> dict:
    p = CONFIG_DIR / "facility_demo.json"
    if not p.exists():
        raise ServiceError("POLICY_NOT_APPROVED", "configs/facility_demo.json 없음")
    return _load("facility_demo.json")


def products() -> dict[str, dict]:
    return {p["product_id"]: p for p in product_master()["products"]}


def get_product(product_id: str) -> dict:
    ps = products()
    if product_id not in ps:
        raise ServiceError("UNKNOWN_PRODUCT",
                           f"'{product_id}'는 7종 제조품목이 아님 (허용: {', '.join(ps)}). "
                           "KAMP ID40 Product_Number는 제조품목으로 사용할 수 없습니다.")
    return ps[product_id]


def kamp_baseline() -> dict | None:
    if not KAMP_BASELINE_FILE.exists():
        return None
    return json.loads(KAMP_BASELINE_FILE.read_text(encoding="utf-8"))


def defect_rate(product_id: str) -> dict:
    """품목의 생산 전 불량률과 근거. KAMP 품목은 학습구간 기준선 산출물이 있어야 한다."""
    p = get_product(product_id)
    if p["rate_source"] == "DEMO_ASSUMED_RATE":
        return {"rate": float(p["planned_defect_rate"]), "provenance": ["DEMO_ASSUMED_RATE"],
                "ci95": None, "sample": None, "warnings": []}
    base = kamp_baseline()
    if base is None or product_id not in base["products"]:
        raise ServiceError("MODEL_NOT_READY", "KAMP 학습구간 불량률 미산출: `python -m ml.kamp.baseline` 실행 필요")
    r = base["products"][product_id]
    return {"rate": float(r["estimated_defect_rate"]), "provenance": ["KAMP_TRAIN_BASELINE"],
            "ci95": [r["ci95_low"], r["ci95_high"]], "raw_rate": r["raw_rate"],
            "sample": {"train_n": r["train_n"], "train_defects": r["train_defects"]},
            "method": base["method"], "warnings": r["warnings"]}
