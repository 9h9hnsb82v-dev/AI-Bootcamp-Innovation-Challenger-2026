"""Streamlit 화면 스모크 테스트(헤드리스). Ollama 없이도 모든 화면이 예외 없이 렌더링되어야 한다."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import production_plan as pp

PAGES = sorted(p for p in (Path(__file__).resolve().parents[1] / "ui/pages").glob("*.py")
               if not p.stem.endswith(("_d", "_f")))  # 시안 대시보드는 메뉴 안에서만 돈다 → test_app_shell_per_design


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders(page):
    pp.create_production_plan("BUMPER", "2026-11-12", 1000)
    pp.create_production_plan("CN7_LH", "2026-11-12", 5000)
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("design", ["classic", "d", "f"])
def test_app_shell_per_design(design, monkeypatch):
    """시안 선택(RESOURCE_DESIGN)별 진입점: 메뉴·사이드바·대시보드가 예외 없이 그려진다. classic 은 기존 경로."""
    monkeypatch.setenv("RESOURCE_DESIGN", design)
    pp.create_production_plan("FENDER", "2026-11-12", 1000)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui/app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.session_state["horizon"] >= 1


@pytest.mark.parametrize("design,btn", [("d", "d_book_top"), ("f", "f_book")])
def test_design_booking_flow(design, btn, monkeypatch):
    """시안 대시보드의 '수거 예약하기' → 확인 창 '예약 확정' → 장부에 수거 예약이 들어간다."""
    from datetime import date

    from services.storage_rooms import list_collections
    monkeypatch.setenv("RESOURCE_DESIGN", design)
    pp.create_production_plan("RG3_RH", date.today().isoformat(), 20000)  # 용량 초과가 나도록
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui/app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert not at.button(key=btn).disabled
    at.button(key=btn).click().run()
    confirm = [b for b in at.button if b.label == "예약 확정"]
    assert confirm, "확인 창이 열리지 않음"
    confirm[0].click().run()
    assert not at.exception, at.exception
    assert any(str(c["source_id"]).startswith("COLLECT:") for c in list_collections())
