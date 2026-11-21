"""서버 시작 시 백그라운드 예열(2026-11-18 탭 속도 개선 B, 사용자 승인).

새로 켠 서버에서 탭에 처음 들어갈 때 이미지 검사 20.6초·처리업체 7.7초·챗봇 6.5초가 걸렸다(실측, 브라우저).
화면을 기다리게 하지 않고 별도 스레드에서 판정 모델·지도 라이브러리·문서 색인·처리업체 자료를 미리 불러 둔다.
- 프로세스당 한 번만(모듈 변수). 실패해도 화면에는 영향 없음(그 탭에서 평소처럼 불러온다).
- 끄기: RESOURCE_WARMUP=0 (테스트는 conftest 에서 끈다).
"""
from __future__ import annotations

import os
import threading
import time

STATUS: dict = {"state": "idle", "steps": {}}
_LOCK = threading.Lock()


def _step(name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        fn()
        STATUS["steps"][name] = round(time.perf_counter() - t0, 1)
    except Exception as e:  # noqa: BLE001 - 예열 실패는 무시(그 화면에서 다시 시도)
        STATUS["steps"][name] = f"실패: {type(e).__name__}"


def _vision() -> None:
    from ml.vision import infer
    b = infer.backend()
    if infer.status(b)["status"] != "OK":
        return
    members = infer.ENSEMBLES[b][1] if b in infer.ENSEMBLES else (b,)
    for m in members:
        infer._load(m)
    if infer.gap_ready(b):
        infer._load_gap(infer._gap_dir(b))
    if infer.det_ready():
        infer._load_det()


def _maps() -> None:
    import altair  # noqa: F401
    import pydeck  # noqa: F401
    from services import waste_processors
    waste_processors._load()
    waste_processors.waste_codes()


def _docs() -> None:
    from services import doc_search
    doc_search.build()


def _run() -> None:
    STATUS["state"] = "running"
    for name, fn in (("지도·처리업체", _maps), ("판정 모델", _vision), ("문서 색인", _docs)):
        _step(name, fn)
    STATUS["state"] = "done"


def start() -> None:
    if os.environ.get("RESOURCE_WARMUP", "1") == "0":
        return
    with _LOCK:
        if STATUS["state"] != "idle":
            return
        STATUS["state"] = "starting"
    threading.Thread(target=_run, name="resource-warmup", daemon=True).start()
