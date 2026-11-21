"""수용 테스트 14: KECO CSV·DRM·지역/처리방법 분석 서비스·API·메뉴 없음."""
import re
from pathlib import Path

from agent.tools import TOOLS

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ["services", "agent", "ui", "ml"]
# 2026-11-19: '한국환경공단' 단어는 뺐다 — 사용자가 공단의 재활용가능자원 가격조사 간행물·공공데이터 사용을 승인(2026-11-18·19).
# 제외 범위는 그대로: 마감인계서(magam)·DRM·지역별 인계량·처리방법 분석과 KECO 모듈·도구 이름.
PATTERN = re.compile(r"keco|magam|마감인계|drm|인계량|처리방법", re.I)


def test_no_keco_paths():
    for p in ["data/raw/keco", "analytics/keco", "services/keco_analytics.py"]:
        assert not (ROOT / p).exists()


def test_no_keco_code_tools_or_menus():
    hits = []
    for d in CODE_DIRS:
        for f in (ROOT / d).rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            # 제외 선언 문구는 허용(에이전트 시스템 프롬프트의 '제공하지 않습니다')
            for line in text.splitlines():
                if PATTERN.search(line) and "제공하지 않" not in line and "존재하지 않는다" not in line \
                        and "범위 제외" not in line:
                    hits.append(f"{f.relative_to(ROOT)}: {line.strip()}")
    assert hits == [], hits
    names = {t["function"]["name"] for t in TOOLS}
    assert not any(PATTERN.search(n) for n in names)
