"""카메라 입력 불변식: 실시간 미리보기는 DB·kg를 바꾸지 않고, '이 장면 등록'한 한 장만 검사 기록이 된다."""
import base64
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import db
from services import production_plan as pp
from services.config import ServiceError
from services.inspection_registry import inspect_part_image, preview_part_image, unit_buckets

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "data/demo/photos/closeup/door_defect.jpg"


def _counts():
    with db.connect() as con:
        return (con.execute("SELECT COUNT(*) FROM inspection").fetchone()[0],
                con.execute("SELECT COUNT(*) FROM stock_ledger").fetchone()[0])


def _ready():
    from ml.vision.infer import status
    return status()["status"] == "OK" and PHOTO.exists()


def test_preview_rejects_non_image_plan_and_empty_image():
    p = pp.create_production_plan("CN7_LH", "2026-11-12", 100)
    with pytest.raises(ServiceError) as e:
        preview_part_image(p["plan_id"], b"x")
    assert e.value.code == "PRODUCT_MISMATCH"
    d = pp.create_production_plan("DOOR", "2026-11-12", 100)
    with pytest.raises(ServiceError):
        preview_part_image(d["plan_id"], b"")


def test_preview_writes_nothing_then_register_writes_one():
    if not _ready():
        pytest.skip("모델 또는 데모 사진 없음")
    plan = pp.create_production_plan("DOOR", "2026-11-12", 100)
    data = PHOTO.read_bytes()
    before = _counts()
    for _ in range(5):  # 영상 프레임 여러 장
        r = preview_part_image(plan["plan_id"], data)
        assert r["status"] == "OK" and r["recorded"] is False
        assert r["verdict"] in ("AI_SUSPECTED_DEFECT", "AI_PASS", "AI_DISAGREE")
        assert "LIVE_PREVIEW_ONLY" in r["provenance"] and "cam" not in r
    assert _counts() == before
    assert unit_buckets(plan["plan_id"])["produced_units"] == 0
    rec = inspect_part_image(plan["plan_id"], "U1", data)  # '이 장면 등록'
    assert _counts() == (before[0] + 1, before[1])  # 검사 1건, kg(ledger) 불변
    assert rec["review_status"] in ("AI_SUSPECTED_DEFECT", "AI_PASS", "AI_DISAGREE")


def test_decode_image_accepts_only_jpeg_data_url():
    from ui.live_camera import decode_image
    jpg = PHOTO.read_bytes() if PHOTO.exists() else b"\xff\xd8\xff"
    url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()
    assert decode_image({"image": url}) == jpg
    assert decode_image({"image": "data:text/html;base64,PGI+"}) is None
    assert decode_image(None) is None and decode_image({"seq": 1}) is None


@pytest.mark.parametrize("mode", ["파일 업로드", "카메라 촬영", "실시간 미리보기"])
def test_inspection_page_modes_render(mode):
    pp.create_production_plan("DOOR", "2026-11-12", 100)
    at = AppTest.from_file(str(ROOT / "ui/pages/4_inspection.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    at.segmented_control(key="photo_mode").set_value(mode).run()
    assert not at.exception, at.exception
    # 카메라 '성능 미검증' 안내는 2026-11-21 사용자 요청으로 화면에서 뺐다
    assert not any("성능 미검증" in str(w.value) or "성능 미검증" in str(getattr(w, "title", ""))
                   for w in at.warning)
