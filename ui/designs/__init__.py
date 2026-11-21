"""화면 디자인 선택 (2026-11-16 사용자 요청: 기존 화면은 그대로 두고 시안 두 개를 각각 적용).

RESOURCE_DESIGN 환경변수로 고른다.
  classic (기본값) — 기존 화면. 코드 경로를 바꾸지 않았다(롤백 = 이 값으로 다시 시작).
  d               — 추천안 · A 기본 + C 5단계 카드 (테마: ui/designs/d.toml)
  f               — 테이스트안 · design-taste-frontend (테마: ui/designs/f.toml)
테마(색·글꼴)는 Streamlit 설정이라 실행할 때 --theme.base 로 넘긴다(scripts/run_design.*).
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
NAMES = {"classic": "기존 화면", "d": "추천안 (A 기본 + C 5단계 카드)", "f": "테이스트안 (design-taste-frontend)"}


def current() -> str:
    v = os.environ.get("RESOURCE_DESIGN", "classic").strip().lower()
    return v if v in NAMES else "classic"


def inject_css(name: str) -> None:
    """시안 전용 CSS. 사용자가 시안 화면을 그대로 적용해 달라고 요청해서 쓴다(네이티브 테마로 안 되는 부분만)."""
    p = HERE / f"{name}.css"
    if p.exists():
        st.html(f"<style>{p.read_text(encoding='utf-8')}</style>")
