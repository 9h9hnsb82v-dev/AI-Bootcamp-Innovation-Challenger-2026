"""SQLite 저장소. 경로는 환경변수 RESOURCE_DB (기본 data/app/resource.db)."""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from services.config import ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS production_plan (
  plan_id TEXT PRIMARY KEY, product_id TEXT NOT NULL, plan_date TEXT NOT NULL,
  planned_units INTEGER NOT NULL CHECK (planned_units > 0),
  produced_units INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL CHECK (source IN ('DIRECT_INPUT','DEMO_SUPPLY_BRIDGE')),
  source_forecast_id TEXT, approved_at TEXT NOT NULL, is_demo INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS supply_forecast (
  forecast_id TEXT PRIMARY KEY, product_number TEXT NOT NULL, forecast_as_of TEXT NOT NULL,
  target_date TEXT NOT NULL, horizon INTEGER NOT NULL, predicted_planned_order_units REAL NOT NULL,
  model_version TEXT NOT NULL, data_split TEXT, provenance TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS supply_plan_draft (
  draft_id TEXT PRIMARY KEY, supply_forecast_id TEXT NOT NULL REFERENCES supply_forecast(forecast_id),
  source_product_number TEXT NOT NULL, model_version TEXT NOT NULL,
  target_manufacturing_product_id TEXT NOT NULL, planned_date TEXT NOT NULL,
  quantity_proposed REAL NOT NULL, quantity_approved INTEGER,
  approval_status TEXT NOT NULL CHECK (approval_status IN ('DRAFT','CONFIRMED','REJECTED')),
  is_demo INTEGER NOT NULL DEFAULT 1, assumption_flags TEXT NOT NULL,
  warning_acknowledged_at TEXT NOT NULL, confirmed_at TEXT, plan_id TEXT);
CREATE TABLE IF NOT EXISTS inspection (
  inspection_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL REFERENCES production_plan(plan_id),
  product_id TEXT NOT NULL, unit_id TEXT NOT NULL, image_hash TEXT NOT NULL,
  model_version TEXT, ai_label TEXT, ai_defect_prob REAL, review_status TEXT NOT NULL,
  scrap_decision TEXT, operator_id TEXT, confirmed_at TEXT, created_at TEXT NOT NULL,
  UNIQUE (plan_id, unit_id, image_hash));
CREATE TABLE IF NOT EXISTS forecast_snapshot (
  snapshot_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, initial_expected_kg REAL NOT NULL,
  confirmed_scrap_kg REAL NOT NULL, remaining_expected_kg REAL NOT NULL,
  updated_expected_total_kg REAL NOT NULL, provenance TEXT NOT NULL, timestamp TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stock_ledger (
  event_id TEXT PRIMARY KEY, plan_id TEXT, product_id TEXT, kg REAL NOT NULL,
  event_type TEXT NOT NULL CHECK (event_type IN ('CONFIRMED_SCRAP_IN','COLLECTION_OUT','ADJUSTMENT')),
  date TEXT NOT NULL, is_demo INTEGER NOT NULL DEFAULT 1, source_id TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS factory_layout (
  owner TEXT PRIMARY KEY, layout_json TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decision_log (
  log_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, unit_id TEXT NOT NULL,
  action TEXT NOT NULL, decision TEXT, operator_id TEXT, note TEXT, detail TEXT, at TEXT NOT NULL);
"""
# 기존 DB(팀 VM team.db 등)에 없는 열을 추가한다(2026-11-14: 사진 저장·기록 취소)
INSPECTION_COLUMNS = {"photo_path": "TEXT", "ai_json": "TEXT", "shot": "TEXT", "cancelled_at": "TEXT",
                      "cancelled_by": "TEXT", "cancel_reason": "TEXT"}


def _migrate(con: sqlite3.Connection) -> None:
    have = {r[1] for r in con.execute("PRAGMA table_info(inspection)")}
    for col, typ in INSPECTION_COLUMNS.items():
        if col not in have:
            con.execute(f"ALTER TABLE inspection ADD COLUMN {col} {typ}")


SANDBOX_KEY = "_sandbox_db"  # 체험 모드(ui/sandbox.py): 세션 상태에 그 세션의 시연 DB 사본 경로


def _session_db() -> Path | None:
    if os.environ.get("RESOURCE_SANDBOX") != "1":
        return None
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx(suppress_warning=True) is None:
            return None  # Streamlit 밖(백그라운드 스레드·스크립트)에서는 공용 DB
        import streamlit as st
        p = st.session_state.get(SANDBOX_KEY)
        return Path(p) if p else None
    except Exception:  # noqa: BLE001
        return None


def seed_db_path() -> Path:
    """공용(원본) DB. 체험 모드에서는 세션 사본의 원본이 된다."""
    return Path(os.environ.get("RESOURCE_DB", ROOT / "data/app/resource.db"))


def db_path() -> Path:
    return _session_db() or seed_db_path()


def connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    _migrate(con)
    return con


def photo_root() -> Path:
    """등록 사진(축소본) 저장 폴더 = DB 파일 옆 photos/."""
    return db_path().parent / "photos"


def reset_database() -> dict:
    """공용 DB를 완전히 빈 상태로 되돌린다. 직전 DB와 사진은 backups/<DB이름>_<시각>.db / _photos 로 보관."""
    import shutil
    p = db_path()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = p.parent / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    backup = bdir / f"{p.stem}_{stamp}.db"
    with connect() as con:
        dst = sqlite3.connect(backup)
        con.backup(dst)  # 앱이 켜져 있어도 일관된 사본
        dst.close()
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
        con.execute("PRAGMA foreign_keys = OFF")
        for t in tables:
            con.execute(f"DELETE FROM {t}")
        con.execute("PRAGMA foreign_keys = ON")
    photos = photo_root()
    photo_backup = None
    if photos.exists():
        photo_backup = bdir / f"{p.stem}_{stamp}_photos"
        shutil.move(str(photos), str(photo_backup))
    return {"backup_db": str(backup), "backup_photos": str(photo_backup) if photo_backup else None,
            "deleted_rows": counts, "at": stamp}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rows(con: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    return [dict(r) for r in con.execute(sql, params).fetchall()]
