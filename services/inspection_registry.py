"""이미지 검사 등록·작업자 판정·개체 버킷 관리 (AI-Hub 도어/범퍼/휀더 계획 전용).

불변식
- 사진 1장 ≠ 제품 1개. 개체는 (plan_id, unit_id)로 식별, 같은 개체의 사진 여러 장은 1개로 집계.
- 같은 image_hash 재업로드는 기존 결과를 그대로 반환(결과 불변). 다른 개체에 같은 사진 재사용은 거부.
- AI 결함 의심(AI_SUSPECTED_DEFECT)만으로는 확인 폐기량이 늘지 않는다. 작업자 CONFIRMED_SCRAP만 반영.
- 조합 모델(ensemble)에서 판정이 엇갈리거나 애매하면 AI_DISAGREE로 기록한다(작업자 확인 필요, kg 불변).
- 이미지의 부품 유형/계획 품목이 다르면 거부(DOOR 이미지로 CN7·BUMPER 계획 갱신 불가).
- (2026-11-14) 등록 사진은 긴 변 1600px 축소본으로 DB 옆 photos/에 저장, AI 판정 근거는 ai_json에 저장.
- (2026-11-14) 기록 취소 = 소프트 삭제(누가·언제·왜 남김, 복원 가능). 취소된 기록은 집계·kg·중복 검사에서 빠진다.
  개체의 마지막 유효 사진이 취소되면 그 개체의 폐기 kg도 빠진다. 영구 삭제는 취소된 기록만 가능.
- 판정·취소·복원·영구 삭제는 decision_log에 남는다(영구 삭제로 개체 기록이 모두 없어지면 그 개체 이력도 삭제).
"""
from __future__ import annotations

import hashlib
import io
import json

from services import db
from services.config import ServiceError, get_product
from services.production_plan import get_plan

DECISIONS = ("CONFIRMED_SCRAP", "CONFIRMED_GOOD", "REWORK_PENDING")
BUCKETS = ("CONFIRMED_SCRAP", "CONFIRMED_GOOD", "REWORK_PENDING", "PRODUCED_UNINSPECTED", "UNPRODUCED")
IMAGE_PRODUCTS = ("DOOR", "BUMPER", "FENDER")
CHECK_STATUSES = ("AI_DISAGREE", "PART_MISMATCH_SUSPECTED")  # 작업자 확인 필요
PHOTO_LONG_SIDE = 1600
ACTIVE = "cancelled_at IS NULL"
# 시험셋 측정: 정상 부품 이미지 오차단율
#   v3(2026-11-10, 기존 분할 1,500장) conf>=0.5: 8.7%, 0.9: 3.5%, 0.95: 2.8%
#   split821(2026-11-11, 8:2:1 시험 1,080장) conf>=0.5: 10.0%, 0.9: 4.2%, 0.95: 3.3%  (reports/vision/effnet_split821_part_mismatch.json)
#   YOLO11m-cls 30ep(2026-11-11, 같은 시험셋) conf>=0.5: 9.5%, 0.9: 4.5%, 0.95: 3.3%, 0.99: 2.1%
#     다른 부품 사진을 막는 비율(0.95): YOLO11 78.1% / EfficientNet 75.7% → 두 모델 모두 0.95 유지
#   조합 모델(원본 전체 재학습 2종 평균, 2026-11-13, test+holdout 5,779장) 0.95: 정상 오차단 0.8%, 다른 부품 차단 76.7%
#     (reports/vision/ensemble_full_part_mismatch.json) → 0.95 유지
# 오차단 시 작업자는 사진 없이 record_operator_decision으로 판정을 기록할 수 있다.
PART_MISMATCH_BLOCK_CONF = 0.95
AI_KEYS = ("status", "verdict", "check_reasons", "ai_recommendation", "shot", "predicted_part", "part_confidence",
           "defect_prob", "threshold", "members", "model_version", "boxes")


def _verdict(ai: dict) -> str:
    """모델이 낸 판정(조합 모델은 AI_DISAGREE 포함)을 우선, 없으면 점수·임계값으로 계산."""
    if ai.get("verdict") in ("AI_SUSPECTED_DEFECT", "AI_PASS", "AI_DISAGREE"):
        return ai["verdict"]
    return "AI_SUSPECTED_DEFECT" if ai.get("defect_prob", 0) >= ai.get("threshold", 0.5) else "AI_PASS"


def image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _check_plan_for_images(plan: dict) -> None:
    if plan["product_id"] not in IMAGE_PRODUCTS:
        raise ServiceError("PRODUCT_MISMATCH",
                           f"{plan['product_id']} 계획은 AI-Hub 이미지 검사 대상이 아닙니다(도어/범퍼/휀더만 가능).")


def _ensure_produced(con, plan: dict) -> None:
    n_units = con.execute(f"SELECT COUNT(DISTINCT unit_id) FROM inspection WHERE plan_id=? AND {ACTIVE}",
                          (plan["plan_id"],)).fetchone()[0]
    if n_units > plan["planned_units"]:
        raise ServiceError("INVALID_INPUT", "검사된 고유 개체 수가 계획수량을 초과합니다.")
    if n_units > plan["produced_units"]:  # 검사됨 ⇒ 생산됨
        con.execute("UPDATE production_plan SET produced_units=? WHERE plan_id=?", (n_units, plan["plan_id"]))


def _log(con, plan_id: str, unit_id: str, action: str, decision: str | None = None, operator_id: str | None = None,
         note: str | None = None, detail: dict | None = None) -> None:
    con.execute("INSERT INTO decision_log (log_id, plan_id, unit_id, action, decision, operator_id, note, detail, at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (db.new_id("LOG"), plan_id, unit_id, action, decision, operator_id, note,
                 json.dumps(detail, ensure_ascii=False) if detail else None, db.now()))


def _save_photo(plan_id: str, h: str, image_bytes: bytes) -> str | None:
    """긴 변 1600px JPEG 축소본 저장 → DB 폴더 기준 상대 경로. 이미지가 아니면 저장하지 않는다."""
    try:
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    except Exception:  # noqa: BLE001 - 테스트용 가짜 바이트 등
        return None
    im.thumbnail((PHOTO_LONG_SIDE, PHOTO_LONG_SIDE))
    rel = f"photos/{plan_id}/{h[:24]}.jpg"
    dst = db.db_path().parent / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=90)
    return rel


def photo_file(rel: str | None):
    """저장 사진의 실제 경로(없으면 None)."""
    if not rel:
        return None
    p = db.db_path().parent / rel
    return p if p.exists() else None


def _ai_json(ai: dict) -> str:
    keep = {k: ai[k] for k in AI_KEYS if k in ai}
    if ai.get("cam") is not None:  # 결함 위치 히트맵(작은 격자) - 나중에 사진 위에 다시 그림
        keep["cam"] = [[round(float(v), 2) for v in row] for row in ai["cam"]]
    return json.dumps(keep, ensure_ascii=False)


def _unit_decision(con, plan_id: str, unit_id: str) -> tuple[str | None, str | None]:
    r = con.execute(f"SELECT scrap_decision, confirmed_at FROM inspection WHERE plan_id=? AND unit_id=? AND {ACTIVE} "
                    "AND scrap_decision IS NOT NULL LIMIT 1", (plan_id, unit_id)).fetchone()
    return (r[0], r[1]) if r else (None, None)


def _sync_ledger(con, plan: dict, unit_id: str) -> None:
    """유효 기록이 있고 판정이 CONFIRMED_SCRAP이면 폐기 kg 1건, 아니면 0건이 되게 맞춘다."""
    src = f"{plan['plan_id']}:{unit_id}"
    decision, confirmed_at = _unit_decision(con, plan["plan_id"], unit_id)
    has = con.execute("SELECT 1 FROM stock_ledger WHERE source_id=?", (src,)).fetchone()
    if decision == "CONFIRMED_SCRAP" and not has:
        kg = float(get_product(plan["product_id"])["unit_weight_kg"])
        con.execute("INSERT INTO stock_ledger VALUES (?,?,?,?,?,?,?,?)",
                    (db.new_id("EVT"), plan["plan_id"], plan["product_id"], kg, "CONFIRMED_SCRAP_IN",
                     (confirmed_at or db.now())[:10], 1, src))
    elif decision != "CONFIRMED_SCRAP" and has:
        con.execute("DELETE FROM stock_ledger WHERE source_id=?", (src,))


def register_inspection(plan_id: str, unit_id: str, image_bytes: bytes, ai_result: dict | None,
                        shot: str = "surface") -> dict:
    """AI 추론 결과(또는 None=MODEL_NOT_READY)를 개체에 연결해 기록한다."""
    plan = get_plan(plan_id)
    _check_plan_for_images(plan)
    h = image_hash(image_bytes)
    ai_result = ai_result or {"status": "MODEL_NOT_READY"}
    if ai_result.get("predicted_part") and ai_result["predicted_part"] != plan["product_id"] \
            and ai_result.get("part_confidence", 0) >= PART_MISMATCH_BLOCK_CONF:
        raise ServiceError("PRODUCT_MISMATCH",
                           f"이미지 부품 유형({ai_result['predicted_part']})이 계획 품목({plan['product_id']})과 다릅니다. "
                           "다른 제품군의 예측값은 갱신하지 않습니다.")
    with db.connect() as con:
        same = db.rows(con, f"SELECT * FROM inspection WHERE plan_id=? AND image_hash=? AND {ACTIVE}", (plan_id, h))
        if same:
            if same[0]["unit_id"] != unit_id:
                raise ServiceError("DUPLICATE_IMAGE", f"이 사진은 이미 개체 {same[0]['unit_id']}에 등록되어 있습니다.")
            return {**same[0], "duplicate": True}
        status = ai_result.get("status", "MODEL_NOT_READY")
        review = _verdict(ai_result) if status == "OK" else status
        if status == "OK" and ai_result.get("predicted_part") != plan["product_id"] \
                and ai_result.get("part_confidence", 0) >= 0.5:
            review = "PART_MISMATCH_SUSPECTED"  # 기록하되 작업자 확인 필요(kg 불변)
        decision, confirmed_at = _unit_decision(con, plan_id, unit_id)  # 개체 단위 판정 이어받기
        prev_op = con.execute(f"SELECT operator_id FROM inspection WHERE plan_id=? AND unit_id=? AND {ACTIVE} "
                              "AND scrap_decision IS NOT NULL LIMIT 1", (plan_id, unit_id)).fetchone()
        row = {"inspection_id": db.new_id("INSP"), "plan_id": plan_id, "product_id": plan["product_id"],
               "unit_id": unit_id, "image_hash": h, "model_version": ai_result.get("model_version"),
               "ai_label": ai_result.get("predicted_class"), "ai_defect_prob": ai_result.get("defect_prob"),
               "review_status": review, "scrap_decision": decision, "operator_id": prev_op[0] if prev_op else None,
               "confirmed_at": confirmed_at, "created_at": db.now(),
               "photo_path": _save_photo(plan_id, h, image_bytes), "ai_json": _ai_json(ai_result), "shot": shot,
               "cancelled_at": None, "cancelled_by": None, "cancel_reason": None}
        old = con.execute("SELECT inspection_id FROM inspection WHERE plan_id=? AND unit_id=? AND image_hash=?",
                          (plan_id, unit_id, h)).fetchone()
        if old:  # 같은 개체에 취소했던 같은 사진을 다시 등록 → 그 기록을 새 판정으로 되살림
            row["inspection_id"] = old[0]
            con.execute("DELETE FROM inspection WHERE inspection_id=?", (old[0],))
            _log(con, plan_id, unit_id, "RESTORE_BY_REUPLOAD", detail={"inspection_id": old[0]})
        cols = ", ".join(row)
        con.execute(f"INSERT INTO inspection ({cols}) VALUES ({', '.join(':' + k for k in row)})", row)
        _ensure_produced(con, plan)
        _sync_ledger(con, plan, unit_id)
    return {**row, "duplicate": False, "ai_result": ai_result}


def _check_shot(plan: dict, shot: str) -> None:
    if shot not in ("surface", "gap"):
        raise ServiceError("INVALID_INPUT", "shot은 surface 또는 gap")
    if shot == "gap" and plan["product_id"] != "FENDER":
        raise ServiceError("INVALID_INPUT", "틈(단차) 촬영 판정은 휀더 계획에서만 씁니다.")


def inspect_part_image(plan_id: str, unit_id: str, image_bytes: bytes, shot: str = "surface") -> dict:
    """이미지 추론(모델 없으면 MODEL_NOT_READY) 후 개체에 기록. kg는 바꾸지 않는다."""
    plan = get_plan(plan_id)
    _check_plan_for_images(plan)
    _check_shot(plan, shot)
    try:
        from ml.vision.infer import predict
        ai = predict(image_bytes, expected_part=plan["product_id"], shot=shot)
        if ai.get("status") == "OK":  # 결함 위치 박스(표시 전용, 판정·kg 영향 없음)
            from ml.vision.infer import detect_defects
            try:
                ai["boxes"] = detect_defects(image_bytes)
            except Exception:  # noqa: BLE001 - 위치 표시 실패는 판정에 영향 없음
                ai["boxes"] = None
    except ImportError:
        ai = {"status": "MODEL_NOT_READY", "message": "torch 미설치"}
    rec = register_inspection(plan_id, unit_id, image_bytes, ai, shot=shot)
    rec["ai_result"] = ai
    return rec


def preview_part_image(plan_id: str, image_bytes: bytes, shot: str = "surface") -> dict:
    """카메라 실시간 미리보기용 판정. 추론만 하고 DB에 아무것도 기록하지 않는다(kg 불변).

    영상 프레임마다 기록하면 한 개체가 수십 번 집계되므로, 기록은 작업자가 '이 장면 등록'을 눌러
    inspect_part_image()를 호출할 때만 한다.
    """
    plan = get_plan(plan_id)
    _check_plan_for_images(plan)
    if not image_bytes:
        raise ServiceError("INVALID_INPUT", "빈 이미지입니다.")
    _check_shot(plan, shot)
    try:
        from ml.vision.infer import predict, preview_backend
        # 미리보기는 단독 모델(EfficientNet+DINO 조합일 때 EfficientNet, 사용자 결정 2026-11-18) — 속도 유지
        ai = predict(image_bytes, expected_part=plan["product_id"], with_cam=False, shot=shot, backend_name=preview_backend())
    except ImportError:
        ai = {"status": "MODEL_NOT_READY", "message": "torch 미설치"}
    if ai.get("status") == "OK":
        ai["verdict"] = _verdict(ai)
        mismatch = ai["predicted_part"] != plan["product_id"]
        conf = ai.get("part_confidence", 0)
        ai["part_mismatch"] = ("BLOCK" if mismatch and conf >= PART_MISMATCH_BLOCK_CONF
                               else "SUSPECTED" if mismatch and conf >= 0.5 else None)
    ai.pop("cam", None)
    ai.pop("class_probs", None)
    return {**ai, "plan_id": plan_id, "plan_part": plan["product_id"], "recorded": False,
            "provenance": ["LIVE_PREVIEW_ONLY", "CAMERA_UNVALIDATED"]}


def record_operator_decision(plan_id: str, unit_id: str, decision: str, operator_id: str,
                             decision_date: str | None = None, note: str | None = None) -> dict:
    """작업자 최종 판정(개체 단위). CONFIRMED_SCRAP만 보관 ledger에 1회 입고. note: 선택 메모(예: 보류 사유)."""
    if decision not in DECISIONS:
        raise ServiceError("INVALID_INPUT", f"decision은 {DECISIONS} 중 하나")
    if not operator_id:
        raise ServiceError("INVALID_INPUT", "operator_id 필요")
    plan = get_plan(plan_id)
    product = get_product(plan["product_id"])
    ts = db.now()
    with db.connect() as con:
        prev, _ = _unit_decision(con, plan_id, unit_id)
        n = con.execute(f"SELECT COUNT(*) FROM inspection WHERE plan_id=? AND unit_id=? AND {ACTIVE}",
                        (plan_id, unit_id)).fetchone()[0]
        if n == 0:  # 사진 없이 육안 검사한 개체(모든 7종 계획에서 허용, 이미지 AI 결과 없음)
            con.execute("INSERT INTO inspection (inspection_id, plan_id, product_id, unit_id, image_hash, review_status, "
                        "created_at, shot) VALUES (?,?,?,?,?,?,?,?)",
                        (db.new_id("INSP"), plan_id, plan["product_id"], unit_id, f"MANUAL:{unit_id}:{ts}",
                         "OPERATOR_ONLY", ts, "manual"))
        con.execute(f"UPDATE inspection SET scrap_decision=?, operator_id=?, confirmed_at=? "
                    f"WHERE plan_id=? AND unit_id=? AND {ACTIVE}", (decision, operator_id, ts, plan_id, unit_id))
        _ensure_produced(con, plan)
        src = f"{plan_id}:{unit_id}"
        con.execute("DELETE FROM stock_ledger WHERE source_id=?", (src,))
        if decision == "CONFIRMED_SCRAP":
            con.execute("INSERT INTO stock_ledger VALUES (?,?,?,?,?,?,?,?)",
                        (db.new_id("EVT"), plan_id, plan["product_id"], float(product["unit_weight_kg"]),
                         "CONFIRMED_SCRAP_IN", decision_date or ts[:10], 1, src))
        _log(con, plan_id, unit_id, "DECIDE", decision, operator_id, note or None,
             {"previous": prev} if prev else None)
    return {"plan_id": plan_id, "unit_id": unit_id, "decision": decision, "operator_id": operator_id,
            "confirmed_at": ts, "provenance": ["DEMO_OPERATOR_CONFIRMATION", "DEMO_SYNTHETIC_BRIDGE"]}


# ── 기록 취소 · 복원 · 영구 삭제 ───────────────────────────────────────────
def _rows_by_ids(con, plan_id: str, ids: list[str]) -> list[dict]:
    if not ids:
        raise ServiceError("INVALID_INPUT", "선택한 기록이 없습니다.")
    q = ",".join("?" * len(ids))
    rows = db.rows(con, f"SELECT * FROM inspection WHERE plan_id=? AND inspection_id IN ({q})", (plan_id, *ids))
    if len(rows) != len(set(ids)):
        raise ServiceError("NOT_FOUND", "이 생산계획에 없는 기록이 포함되어 있습니다.")
    return rows


def cancel_preview(plan_id: str, ids: list[str]) -> list[dict]:
    """취소하면 개체별로 무엇이 바뀌는지(남는 유효 사진 수, 폐기 kg 변화)."""
    plan = get_plan(plan_id)
    kg = float(get_product(plan["product_id"])["unit_weight_kg"])
    with db.connect() as con:
        rows = [r for r in _rows_by_ids(con, plan_id, ids) if r["cancelled_at"] is None]
        out = []
        for unit in sorted({r["unit_id"] for r in rows}):
            n_sel = sum(r["unit_id"] == unit for r in rows)
            n_act = con.execute(f"SELECT COUNT(*) FROM inspection WHERE plan_id=? AND unit_id=? AND {ACTIVE}",
                                (plan_id, unit)).fetchone()[0]
            decision, _ = _unit_decision(con, plan_id, unit)
            left = n_act - n_sel
            out.append({"unit_id": unit, "cancel_records": n_sel, "records_left": left, "decision": decision,
                        "kg_change": -kg if (left == 0 and decision == "CONFIRMED_SCRAP") else 0.0})
    return out


def cancel_inspections(plan_id: str, ids: list[str], operator_id: str, reason: str) -> dict:
    """기록 취소(소프트 삭제). 마지막 유효 기록이 취소된 개체는 판정·폐기 kg에서 빠진다."""
    if not operator_id or not (reason or "").strip():
        raise ServiceError("INVALID_INPUT", "작업자 ID와 취소 사유가 필요합니다.")
    plan = get_plan(plan_id)
    preview = cancel_preview(plan_id, ids)
    ts = db.now()
    with db.connect() as con:
        rows = [r for r in _rows_by_ids(con, plan_id, ids) if r["cancelled_at"] is None]
        for r in rows:
            con.execute("UPDATE inspection SET cancelled_at=?, cancelled_by=?, cancel_reason=? WHERE inspection_id=?",
                        (ts, operator_id, reason.strip(), r["inspection_id"]))
        for unit in {r["unit_id"] for r in rows}:
            _sync_ledger(con, plan, unit)
            _log(con, plan_id, unit, "CANCEL_RECORD", operator_id=operator_id, note=reason.strip(),
                 detail={"inspection_ids": [r["inspection_id"] for r in rows if r["unit_id"] == unit]})
    return {"cancelled": len(rows), "units": preview}


def restore_inspections(plan_id: str, ids: list[str], operator_id: str) -> dict:
    """취소한 기록을 되살린다. 같은 사진이 다른 개체에 유효하게 등록돼 있으면 거부."""
    if not operator_id:
        raise ServiceError("INVALID_INPUT", "operator_id 필요")
    plan = get_plan(plan_id)
    with db.connect() as con:
        rows = [r for r in _rows_by_ids(con, plan_id, ids) if r["cancelled_at"] is not None]
        for r in rows:
            clash = con.execute(f"SELECT unit_id FROM inspection WHERE plan_id=? AND image_hash=? AND {ACTIVE} "
                                "AND inspection_id<>?", (plan_id, r["image_hash"], r["inspection_id"])).fetchone()
            if clash:
                raise ServiceError("DUPLICATE_IMAGE", f"같은 사진이 이미 개체 {clash[0]}에 등록되어 있어 복원할 수 없습니다.")
        for r in rows:
            con.execute("UPDATE inspection SET cancelled_at=NULL, cancelled_by=NULL, cancel_reason=NULL "
                        "WHERE inspection_id=?", (r["inspection_id"],))
        _ensure_produced(con, plan)
        for unit in {r["unit_id"] for r in rows}:
            _sync_ledger(con, plan, unit)
            _log(con, plan_id, unit, "RESTORE_RECORD", operator_id=operator_id,
                 detail={"inspection_ids": [r["inspection_id"] for r in rows if r["unit_id"] == unit]})
    return {"restored": len(rows)}


def purge_inspections(plan_id: str, ids: list[str], operator_id: str) -> dict:
    """영구 삭제: '취소된 기록'만 가능. 저장 사진도 지운다. 개체 기록이 모두 사라지면 그 개체 판정 이력도 지운다."""
    if not operator_id:
        raise ServiceError("INVALID_INPUT", "operator_id 필요")
    get_plan(plan_id)
    with db.connect() as con:
        rows = _rows_by_ids(con, plan_id, ids)
        active = [r["unit_id"] for r in rows if r["cancelled_at"] is None]
        if active:
            raise ServiceError("INVALID_INPUT", "영구 삭제는 먼저 '기록 취소'한 기록만 가능합니다: " + ", ".join(sorted(set(active))))
        for r in rows:
            con.execute("DELETE FROM inspection WHERE inspection_id=?", (r["inspection_id"],))
            p = photo_file(r.get("photo_path"))
            if p and not con.execute("SELECT 1 FROM inspection WHERE photo_path=?", (r["photo_path"],)).fetchone():
                p.unlink(missing_ok=True)
        gone = []
        for unit in {r["unit_id"] for r in rows}:
            if not con.execute("SELECT 1 FROM inspection WHERE plan_id=? AND unit_id=?", (plan_id, unit)).fetchone():
                con.execute("DELETE FROM decision_log WHERE plan_id=? AND unit_id=?", (plan_id, unit))
                gone.append(unit)
            else:
                _log(con, plan_id, unit, "PURGE_RECORD", operator_id=operator_id,
                     detail={"inspection_ids": [r["inspection_id"] for r in rows if r["unit_id"] == unit]})
    return {"purged": len(rows), "units_removed": sorted(gone)}


# ── 조회 ───────────────────────────────────────────────────────────────────
def _with_ai(r: dict) -> dict:
    r = dict(r)
    r["ai"] = json.loads(r["ai_json"]) if r.get("ai_json") else {}
    return r


def unit_records(plan_id: str, unit_id: str) -> list[dict]:
    """개체의 유효 기록(사진 경로·AI 판정 근거 포함), 오래된 순."""
    with db.connect() as con:
        return [_with_ai(r) for r in db.rows(con, f"SELECT * FROM inspection WHERE plan_id=? AND unit_id=? AND {ACTIVE} "
                                                  "ORDER BY created_at", (plan_id, unit_id))]


def decision_history(plan_id: str, unit_id: str) -> list[dict]:
    with db.connect() as con:
        return db.rows(con, "SELECT * FROM decision_log WHERE plan_id=? AND unit_id=? ORDER BY at", (plan_id, unit_id))


def pending_units(plan_id: str) -> dict:
    """처리 대기함: REWORK = 재작업 보류 개체, CHECK = 작업자 확인 필요(AI_DISAGREE·부품 확인)인데 아직 판정 없는 개체.
    각 개체에 그렇게 된 사진(유효 기록)과 최근 메모를 붙인다."""
    with db.connect() as con:
        rows = [_with_ai(r) for r in db.rows(con, f"SELECT * FROM inspection WHERE plan_id=? AND {ACTIVE} "
                                                  "ORDER BY created_at", (plan_id,))]
        notes = {r["unit_id"]: r["note"] for r in db.rows(
            con, "SELECT unit_id, note FROM decision_log WHERE plan_id=? AND action='DECIDE' ORDER BY at", (plan_id,))}
    units: dict[str, list[dict]] = {}
    for r in rows:
        units.setdefault(r["unit_id"], []).append(r)
    out = {"REWORK": [], "CHECK": []}
    for uid, rs in units.items():
        decision = next((r["scrap_decision"] for r in rs if r["scrap_decision"]), None)
        item = {"unit_id": uid, "decision": decision, "records": rs, "note": notes.get(uid),
                "last_at": max(r["created_at"] for r in rs), "photos": sum(1 for r in rs if r.get("photo_path")),
                "flagged": [r for r in rs if r["review_status"] in CHECK_STATUSES]}
        if decision == "REWORK_PENDING":
            out["REWORK"].append(item)
        elif decision is None and item["flagged"]:
            out["CHECK"].append(item)
    for k in out:
        out[k].sort(key=lambda u: u["last_at"], reverse=True)
    return out


def unit_buckets(plan_id: str) -> dict:
    plan = get_plan(plan_id)
    with db.connect() as con:
        units = db.rows(con, "SELECT unit_id, MAX(scrap_decision) AS decision, "
                             "GROUP_CONCAT(DISTINCT review_status) AS review FROM inspection "
                             f"WHERE plan_id=? AND {ACTIVE} GROUP BY unit_id", (plan_id,))
    b = dict.fromkeys(BUCKETS, 0)
    decided = 0
    ai_only = 0
    for u in units:
        if u["decision"] in DECISIONS:
            b[u["decision"]] += 1
            decided += 1
        else:
            ai_only += 1
    b["PRODUCED_UNINSPECTED"] = plan["produced_units"] - decided
    b["UNPRODUCED"] = plan["planned_units"] - plan["produced_units"]
    assert sum(b.values()) == plan["planned_units"], "버킷 합계 불일치"
    return {"plan_id": plan_id, "product_id": plan["product_id"], "planned_units": plan["planned_units"],
            "produced_units": plan["produced_units"], "buckets": b,
            "units_with_ai_result_awaiting_operator": ai_only, "sum_check": True}


def list_inspections(plan_id: str, include_cancelled: bool = False) -> list[dict]:
    with db.connect() as con:
        where = "" if include_cancelled else f" AND {ACTIVE}"
        return db.rows(con, f"SELECT * FROM inspection WHERE plan_id=?{where} ORDER BY created_at", (plan_id,))
