#!/usr/bin/env bash
# 시안별 실행 (VM/리눅스).  사용: scripts/run_design.sh d 8602   |   scripts/run_design.sh f 8603
# 기존 화면은 지금처럼 ./run_app.sh (RESOURCE_DESIGN 없음 = classic).
set -euo pipefail
cd "$(dirname "$0")/.."
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"  # 동시 접속 시 Too many open files 방지(기본 1024, 2026-11-20)
DESIGN="${1:-classic}"; PORT="${2:-8601}"
case "$DESIGN" in classic|d|f) ;; *) echo "design must be classic|d|f"; exit 2;; esac
export RESOURCE_DESIGN="$DESIGN"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # 하위 화면 주소로 바로 들어와도 services 를 찾게
EXTRA=()
[ "$DESIGN" != "classic" ] && EXTRA+=("--theme.base=ui/designs/$DESIGN.toml")
PY="${PYTHON:-python3}"
[ -x .venv/bin/python ] && PY=.venv/bin/python
exec "$PY" -m streamlit run ui/app.py --server.port="$PORT" --server.address=127.0.0.1 --server.headless=true "${EXTRA[@]}"
