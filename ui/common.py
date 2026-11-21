"""Streamlit 공통 헬퍼."""
from __future__ import annotations

import streamlit as st

from services.config import ServiceError
from services.provenance import PROVENANCE

PRODUCT_LABELS = {"CN7_LH": "CN7 사이드 몰딩 LH", "CN7_RH": "CN7 사이드 몰딩 RH", "RG3_LH": "RG3 사이드 몰딩 LH",
                  "RG3_RH": "RG3 사이드 몰딩 RH", "DOOR": "가상 플라스틱 도어", "BUMPER": "가상 플라스틱 범퍼",
                  "FENDER": "가상 플라스틱 휀더"}


def tags(tag_list) -> None:
    if isinstance(tag_list, str):
        tag_list = tag_list.split(",")
    st.caption(" · ".join(f"`{t}`" for t in tag_list if t), help="\n\n".join(
        f"**{t}**: {PROVENANCE.get(t, '')}" for t in tag_list if t))


def run(fn, *a, **kw):
    """ServiceError를 화면 오류로 표시하고 None 반환."""
    try:
        return fn(*a, **kw)
    except ServiceError as e:
        st.error(f"**{e.code}** — {e.message}")
        return None


def product_label(pid: str) -> str:
    return f"{pid} · {PRODUCT_LABELS.get(pid, pid)}"


def kg(x: float) -> str:
    return f"{x:,.2f} kg"
