"""체험 모드 (2026-11-20, 대회 QR 체험용): 접속한 브라우저 세션마다 시연 DB 사본을 따로 준다.

환경변수 RESOURCE_SANDBOX=1 일 때만 켜진다(팀 VM 공개 앱 8603). 세션에서 등록·확정·삭제한 내용은 그 사본에만 남고
공용 DB(원본)와 다른 사람 화면은 바뀌지 않는다. 사본은 생산계획 0건의 빈 DB로 시작한다. 사본 경로는 세션 상태(st.session_state)에 두고(services.db.db_path 가 읽음),
파일은 data/app/sandbox/<무작위 이름>/ 에 둔다. 12시간 동안 쓰지 않은 사본은 지운다.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

import streamlit as st

from services import db

TTL_SECONDS = 12 * 3600


def enabled() -> bool:
    return os.environ.get("RESOURCE_SANDBOX") == "1"


def _root() -> Path:
    return db.seed_db_path().parent / "sandbox"


def _copy_seed(dst_dir: Path) -> Path:
    """세션용 빈 DB(생산계획 0건)를 만든다(2026-11-20 사용자 요청: 체험은 0건에서 시작). 표는 첫 연결 때 자동 생성."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "resource.db"
    con = sqlite3.connect(dst)
    con.executescript(db.SCHEMA)
    con.close()
    return dst


def _prune(keep: Path) -> None:
    root = _root()
    if not root.exists():
        return
    now = time.time()
    for d in root.iterdir():
        if d.is_dir() and d != keep and now - d.stat().st_mtime > TTL_SECONDS:
            shutil.rmtree(d, ignore_errors=True)


def enable() -> None:
    if not enabled():
        return
    ss = st.session_state
    from ui.access_gate import is_presenter
    if is_presenter():  # 발표자 비밀번호로 들어온 세션: 체험 모드 없이 팀 공용 데이터(시연 계획)
        ss.pop(db.SANDBOX_KEY, None)  # 사이드바 안내 없음(2026-11-21 사용자 요청)
        return
    cur = ss.get(db.SANDBOX_KEY)
    if not cur or not Path(cur).exists():
        d = _root() / uuid.uuid4().hex
        ss[db.SANDBOX_KEY] = str(_copy_seed(d))
        _prune(d)
    else:
        os.utime(Path(cur).parent)  # 사용 중인 사본은 정리 대상에서 빠지도록
    with st.sidebar:
        st.info("**체험 모드**")  # 문구·이모지 최소화(2026-11-21 사용자 요청)
        if st.button("처음(빈 상태)으로 되돌리기", key="_sandbox_reset", width="stretch"):
            # 열린 연결이 있어도 안전하도록 새 사본으로 바꾸고, 예전 사본은 시간이 지나면 정리된다
            ss[db.SANDBOX_KEY] = str(_copy_seed(_root() / uuid.uuid4().hex))
            st.cache_data.clear()
            st.rerun()
