"""검사 기록 취소·복원·영구 삭제, 처리 대기함, 사진 저장, 공용 DB 초기화(2026-11-14 사용자 결정)."""
import io
import sqlite3
from datetime import date
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from services import db
from services import inspection_registry as ir
from services import production_plan as pp
from services.config import ServiceError

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()


def _jpg(seed: int) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (2000, 1500), (seed % 255, 80, 120)).save(b, "JPEG")
    return b.getvalue()


def _ai(verdict="AI_PASS", part="DOOR"):
    return {"status": "OK", "predicted_part": part, "part_confidence": 0.99, "defect_prob": 0.3, "threshold": 0.5,
            "verdict": verdict, "model_version": "t", "check_reasons": ["두 모델 판정이 엇갈림"] if verdict == "AI_DISAGREE" else [],
            "cam": [[0.0, 1.0], [0.5, 0.2]]}


def _ledger(src):
    with db.connect() as con:
        return con.execute("SELECT COUNT(*) FROM stock_ledger WHERE source_id=?", (src,)).fetchone()[0]


def test_photo_saved_downscaled_with_ai_reason():
    plan = pp.create_production_plan("DOOR", TODAY, 10)
    rec = ir.register_inspection(plan["plan_id"], "U1", _jpg(1), _ai("AI_DISAGREE"))
    f = ir.photo_file(rec["photo_path"])
    assert f is not None and max(Image.open(f).size) == ir.PHOTO_LONG_SIDE
    r = ir.unit_records(plan["plan_id"], "U1")[0]
    assert r["ai"]["check_reasons"] == ["두 모델 판정이 엇갈림"] and r["ai"]["cam"]


def test_cancel_removes_scrap_kg_and_restore_brings_it_back():
    plan = pp.create_production_plan("DOOR", TODAY, 10)
    pid, src = plan["plan_id"], f"{plan['plan_id']}:U1"
    a = ir.register_inspection(pid, "U1", _jpg(1), _ai())
    b = ir.register_inspection(pid, "U1", _jpg(2), _ai())
    ir.record_operator_decision(pid, "U1", "CONFIRMED_SCRAP", "op")
    assert _ledger(src) == 1
    prev = ir.cancel_preview(pid, [a["inspection_id"]])
    assert prev[0]["records_left"] == 1 and prev[0]["kg_change"] == 0  # 사진 1장 남음 → kg 유지
    ir.cancel_inspections(pid, [a["inspection_id"]], "op", "흔들린 사진")
    assert _ledger(src) == 1
    prev = ir.cancel_preview(pid, [b["inspection_id"]])
    assert prev[0]["records_left"] == 0 and prev[0]["kg_change"] < 0  # 마지막 사진 → kg 빠짐
    ir.cancel_inspections(pid, [b["inspection_id"]], "op", "잘못된 개체")
    assert _ledger(src) == 0
    assert ir.unit_buckets(pid)["buckets"]["CONFIRMED_SCRAP"] == 0
    ir.restore_inspections(pid, [a["inspection_id"], b["inspection_id"]], "op")
    assert _ledger(src) == 1 and ir.unit_buckets(pid)["buckets"]["CONFIRMED_SCRAP"] == 1
    actions = [h["action"] for h in ir.decision_history(pid, "U1")]
    assert actions.count("CANCEL_RECORD") == 2 and "RESTORE_RECORD" in actions and "DECIDE" in actions


def test_cancel_requires_reason_and_purge_only_cancelled():
    plan = pp.create_production_plan("DOOR", TODAY, 10)
    pid = plan["plan_id"]
    rec = ir.register_inspection(pid, "U1", _jpg(3), _ai())
    with pytest.raises(ServiceError):
        ir.cancel_inspections(pid, [rec["inspection_id"]], "op", "  ")
    with pytest.raises(ServiceError):
        ir.purge_inspections(pid, [rec["inspection_id"]], "op")  # 취소 전에는 영구 삭제 불가
    photo = ir.photo_file(rec["photo_path"])
    ir.cancel_inspections(pid, [rec["inspection_id"]], "op", "테스트")
    r = ir.purge_inspections(pid, [rec["inspection_id"]], "op")
    assert r["purged"] == 1 and r["units_removed"] == ["U1"]
    assert not photo.exists() and ir.list_inspections(pid, include_cancelled=True) == []
    assert ir.decision_history(pid, "U1") == []


def test_reupload_after_cancel_and_duplicate_rules():
    plan = pp.create_production_plan("DOOR", TODAY, 10)
    pid, img = plan["plan_id"], _jpg(4)
    rec = ir.register_inspection(pid, "U1", img, _ai())
    ir.cancel_inspections(pid, [rec["inspection_id"]], "op", "다른 개체 사진")
    other = ir.register_inspection(pid, "U2", img, _ai())  # 취소 후 다른 개체에 등록 가능
    assert other["duplicate"] is False
    with pytest.raises(ServiceError) as e:
        ir.restore_inspections(pid, [rec["inspection_id"]], "op")  # 같은 사진이 U2에 유효 → 복원 거부
    assert e.value.code == "DUPLICATE_IMAGE"


def test_pending_queue_rework_and_operator_check():
    plan = pp.create_production_plan("DOOR", TODAY, 10)
    pid = plan["plan_id"]
    ir.register_inspection(pid, "U1", _jpg(5), _ai())
    ir.record_operator_decision(pid, "U1", "REWORK_PENDING", "op", note="도장 재작업")
    ir.register_inspection(pid, "U2", _jpg(6), _ai("AI_DISAGREE"))
    ir.register_inspection(pid, "U3", _jpg(7), _ai("AI_DISAGREE"))
    ir.record_operator_decision(pid, "U3", "CONFIRMED_GOOD", "op")  # 확정되면 대기함에서 빠짐
    q = ir.pending_units(pid)
    assert [u["unit_id"] for u in q["REWORK"]] == ["U1"] and q["REWORK"][0]["note"] == "도장 재작업"
    assert [u["unit_id"] for u in q["CHECK"]] == ["U2"] and q["CHECK"][0]["flagged"][0]["photo_path"]
    ir.record_operator_decision(pid, "U1", "CONFIRMED_SCRAP", "op")
    assert ir.pending_units(pid)["REWORK"] == []


def test_reset_database_backs_up_and_empties():
    plan = pp.create_production_plan("DOOR", TODAY, 10)
    ir.register_inspection(plan["plan_id"], "U1", _jpg(8), _ai())
    r = db.reset_database()
    assert Path(r["backup_db"]).exists() and r["deleted_rows"]["inspection"] == 1
    assert Path(r["backup_photos"]).exists() and not db.photo_root().exists()
    assert pp.list_plans() == []
    with sqlite3.connect(r["backup_db"]) as b:
        assert b.execute("SELECT COUNT(*) FROM inspection").fetchone()[0] == 1


def test_old_database_is_migrated(tmp_path, monkeypatch):
    old = tmp_path / "old.db"
    with sqlite3.connect(old) as c:
        c.execute("CREATE TABLE inspection (inspection_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, product_id TEXT NOT NULL, "
                  "unit_id TEXT NOT NULL, image_hash TEXT NOT NULL, model_version TEXT, ai_label TEXT, ai_defect_prob REAL, "
                  "review_status TEXT NOT NULL, scrap_decision TEXT, operator_id TEXT, confirmed_at TEXT, created_at TEXT NOT NULL, "
                  "UNIQUE (plan_id, unit_id, image_hash))")
    monkeypatch.setenv("RESOURCE_DB", str(old))
    with db.connect() as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(inspection)")}
    assert {"photo_path", "ai_json", "cancelled_at", "cancel_reason"} <= cols


def test_records_section_renders_with_filters():
    plan = pp.create_production_plan("DOOR", TODAY, 10)
    rec = ir.register_inspection(plan["plan_id"], "U0001", _jpg(9), _ai("AI_DISAGREE"))
    ir.cancel_inspections(plan["plan_id"], [rec["inspection_id"]], "op", "테스트")
    ir.register_inspection(plan["plan_id"], "U0002", _jpg(10), _ai("AI_DISAGREE"))
    at = AppTest.from_file(str(ROOT / "ui/pages/4_inspection.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    at.segmented_control(key="queue_kind").set_value("CHECK").run()
    assert not at.exception, at.exception
    at.button(key="load_CHECK_U0002").click().run()
    assert at.text_input(key=f"insp_unit_{plan['plan_id']}").value == "U0002"
    at.segmented_control(key="rec_filter").set_value("취소됨").run()
    assert not at.exception, at.exception
