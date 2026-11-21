"""발표용 구조도(파이프라인 그림) 생성 — 전체 시스템 흐름 · 챗봇 ReAct 루프 (2026-11-19).

SVG 로 그린 뒤 Chrome 헤드리스로 PNG(2배 해상도)를 만든다. 숫자는 현재 코드·설정 값만 쓴다.
실행: .venv/Scripts/python -m scripts.draw_pipeline
출력: reports/figures/system_pipeline.{svg,png}, reports/figures/chatbot_react_loop.{svg,png}
"""
from __future__ import annotations

import html
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
FONT = "'Pretendard','Malgun Gothic','Apple SD Gothic Neo',sans-serif"
MONO = "'D2Coding','Consolas','Menlo',monospace"
GREEN_BG, GREEN_LINE, GREEN_TXT = "#EEF6EE", "#5DA36A", "#2E7D3A"
AMBER_BG, AMBER_LINE = "#FFF8EC", "#D9A441"


def recycle_icon(cx: float, cy: float, r: float, color: str, sw: float = 2.4) -> str:
    """자원순환(재활용) 선 아이콘: 삼각형 둘레를 도는 화살표 3개(120°씩 회전). 이모지 대신 직접 그린다(2026-11-19 사용자 결정)."""
    import math
    pts = [(cx + r * math.cos(math.radians(-90 + 120 * k)), cy + r * math.sin(math.radians(-90 + 120 * k))) for k in range(3)]
    out = []
    for k in range(3):
        (x1, y1), (x2, y2), (x3, y3) = pts[k], pts[(k + 1) % 3], pts[(k + 2) % 3]
        lerp = lambda a, b, t: (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)  # noqa: E731
        s = lerp((x1, y1), (x2, y2), 0.50)          # 변 가운데쯤에서 시작(앞 화살촉과 간격)
        c = (x2, y2)                                # 다음 꼭짓점에서 둥글게 꺾임
        e = lerp((x2, y2), (x3, y3), 0.26)          # 다음 변으로 조금 들어가 끝
        m = lerp((x1, y1), (x2, y2), 0.78)
        out.append(f'<path d="M{s[0]:.1f},{s[1]:.1f} L{m[0]:.1f},{m[1]:.1f} Q{c[0]:.1f},{c[1]:.1f} {e[0]:.1f},{e[1]:.1f}" '
                   f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')
        # 화살촉: 끝점에서 진행 방향
        dx, dy = x3 - x2, y3 - y2
        n = math.hypot(dx, dy)
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        a = 5.2
        tip = (e[0] + ux * a * 0.9, e[1] + uy * a * 0.9)
        b1 = (e[0] - ux * a * 0.4 + px * a * 0.75, e[1] - uy * a * 0.4 + py * a * 0.75)
        b2 = (e[0] - ux * a * 0.4 - px * a * 0.75, e[1] - uy * a * 0.4 - py * a * 0.75)
        out.append(f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{b1[0]:.1f},{b1[1]:.1f} L{b2[0]:.1f},{b2[1]:.1f} Z" fill="{color}" '
                   f'stroke="{color}" stroke-width="1" stroke-linejoin="round"/>')
    return "".join(out)


class Fig:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.el = w, h, []

    def header(self, tag: str, title: str, lines: list[str]):
        x, y, w, h = 30, 24, self.w - 60, 118
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{GREEN_BG}" stroke="{GREEN_LINE}" stroke-width="1.6"/>')
        cx, cy = x + 58, y + h / 2
        self.el.append(f'<circle cx="{cx}" cy="{cy}" r="27" fill="#fff" stroke="{GREEN_LINE}" stroke-width="2"/>')
        self.el.append(recycle_icon(cx, cy + 2, 16.5, GREEN_TXT))
        self.el.append(self._t(x + 108, y + 44, tag, 15, GREEN_TXT, 700))
        self.el.append(self._t(x + 108, y + 80, title, 25, "#111", 800))
        for i, ln in enumerate(lines):
            self.el.append(self._t(x + 470, y + 50 + i * 34, ln, 19, "#222", 500))

    @staticmethod
    def _t(x, y, s, size, color, weight=400, anchor="start", font=FONT):
        return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{color}" font-weight="{weight}" '
                f'text-anchor="{anchor}">{html.escape(s)}</text>')

    def box(self, x, y, w, h, title, sub="", kind="plain", sub2=""):
        fill, line, dash = {"plain": ("#fff", "#4a4a4a", ""), "green": (GREEN_BG, GREEN_LINE, ""),
                            "dashed": ("#fafafa", "#9a9a9a", ' stroke-dasharray="6 5"'),
                            "amber": (AMBER_BG, AMBER_LINE, ' stroke-dasharray="6 5"')}[kind]
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{line}" stroke-width="1.6"{dash}/>')
        cx = x + w / 2
        n = 1 + bool(sub) + bool(sub2)
        top = y + h / 2 - (n - 1) * 13 + 6
        self.el.append(self._t(cx, top, title, 20, "#111", 800, "middle"))
        if sub:
            self.el.append(self._t(cx, top + 27, sub, 15, "#555", 400, "middle"))
        if sub2:
            self.el.append(self._t(cx, top + 49, sub2, 14, "#777", 400, "middle"))
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (cx, y), "b": (cx, y + h)}

    def diamond(self, cx, cy, w, h, text):
        self.el.append(f'<polygon points="{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}" fill="#fff" '
                       f'stroke="#4a4a4a" stroke-width="1.6"/>')
        self.el.append(self._t(cx, cy + 7, text, 19, "#111", 800, "middle"))
        return {"l": (cx - w / 2, cy), "r": (cx + w / 2, cy), "t": (cx, cy - h / 2), "b": (cx, cy + h / 2)}

    def arrow(self, pts, label="", dashed=False, color="#6b6b6b", lpos=None, lanchor="middle"):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.el.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{dash} marker-end="url(#ah)"/>')
        if label:
            if lpos is None:
                (x1, y1), (x2, y2) = pts[0], pts[1]
                lpos = ((x1 + x2) / 2, (y1 + y2) / 2 - 10)
            self.el.append(self._t(lpos[0], lpos[1], label, 15, "#444", 500, lanchor))

    def note(self, x, y, s, size=14, color="#777", anchor="start"):
        self.el.append(self._t(x, y, s, size, color, 400, anchor))

    def svg(self) -> str:
        defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
                '<path d="M0,0 L10,5 L0,10 z" fill="#6b6b6b"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f'<rect width="100%" height="100%" fill="#fff"/>{defs}{"".join(self.el)}</svg>')


def system() -> Fig:
    f = Fig(1720, 1010)
    f.header("구조", "전체 시스템 흐름",
             ["확정 생산계획에서 폐플라스틱 발생량(kg)을 미리 예측하고, 이미지 검사와 작업자",
              "확정으로 갱신한 뒤 보관량 · 수거 일정 · 위탁 처리업체까지 이어서 추천합니다."])
    W, H = 240, 96
    a1 = f.box(30, 190, 220, H, "생산계획", "7종 품목 · 날짜 · 수량")
    a2 = f.box(310, 190, 270, H, "폐플라스틱 사전 예측", "수량 × 불량률 × 단중 = kg", sub2="KAMP ID4 불량률 · 가상 단중")
    a3 = f.box(640, 190, 270, H, "이미지 검사", "EfficientNet-B0 + DINOv2", "green", sub2="2차 판정 · 결함 위치 표시")
    d1 = f.diamond(1060, 238, 190, 104, "판정 일치?")
    a4 = f.box(1230, 190, W, H, "자동 판정", "결함 의심 · 양품 추정", sub2="작업자에게 추천")
    c1 = f.box(950, 380, 220, H, "작업자 확인 필요", "처리 대기함", "amber", sub2="엇갈리거나 애매한 사진")
    b1 = f.box(1230, 380, W, H, "작업자 확정", "폐기 · 양품 · 재작업 보류")
    r1 = f.box(1230, 580, W, H, "예측 갱신", "확정 폐기만 kg에 반영")
    r2 = f.box(910, 580, W, H, "보관량 예측", "경보일 · 용량 초과일")
    r3 = f.box(590, 580, W, H, "수거 추천", "권장 수거일 · 요청 마감", sub2="수거 예약")
    r4 = f.box(270, 580, W, H, "위탁 처리업체", "공공데이터 처리업소 1,726곳*", sub2="TMAP 화물차 도로거리")
    f.arrow([a1["r"], a2["l"]])
    f.arrow([a2["r"], a3["l"]])
    f.arrow([a3["r"], d1["l"]])
    f.arrow([d1["r"], a4["l"]], "일치", lpos=(1172, 228))
    f.arrow([d1["b"], c1["t"]], "엇갈림 · 애매", lpos=(1070, 340), lanchor="start")
    f.arrow([a4["b"], b1["t"]])
    f.arrow([c1["r"], b1["l"]])
    f.arrow([b1["b"], r1["t"]], "kg 재계산", lpos=(1360, 535), lanchor="start")
    f.arrow([r1["l"], r2["r"]])
    f.arrow([r2["l"], r3["r"]])
    f.arrow([r3["l"], r4["r"]])
    # 수주 예측(독립) → 초안 → 사용자 재확정 → 생산계획
    s1 = f.box(30, 800, 250, H, "미래 수주량 예측", "KAMP ID40 · T일 예정 수주량(개)", "dashed")
    s2 = f.box(340, 800, 270, H, "시뮬레이션 계획 초안", "경고 확인 → 품목 · 수량 직접 선택", "dashed")
    f.arrow([s1["r"], s2["l"]])
    f.arrow([(380, 800), (380, 760), (140, 760), (140, 286)], "사용자 재확정", dashed=True, lpos=(150, 742), lanchor="start")
    f.note(30, 925, "수주량(개)은 kg이 아니며 7종 품목과 자동 연결하지 않음")
    # 챗봇 · 문서 검색
    ch = f.box(700, 800, 470, 110, "AI 챗봇 (로컬 Ollama 8B)", "ReAct 루프 · 도구 15개 · 코드 가드레일 · 숫자 검증", "green",
               sub2="쓰기 작업(계획 등록 · 판정 · 확정)은 확인 버튼을 눌러야 실행")
    rag = f.box(1230, 800, 450, 110, "문서 검색 (RAG)", "법령 5종 · 폐기물 분류번호 · 재활용 가격조사", sub2="데이터 출처 · 이미지 모델 비교 · 앱 사용법")
    f.arrow([ch["r"], rag["l"]], "검색", lpos=(1200, 843))
    f.arrow([(935, 800), (935, 690)], "모든 단계를 도구로 조회", dashed=True, lpos=(948, 750), lanchor="start")
    f.note(30, 990, "* 위탁 처리업체: 처리폐기물 51-03-01(폐합성수지류) 재활용업 기준. 허가 정보이며 현재 수탁 가능 여부는 아님.  "
                    "점선 상자 = 독립·예외 경로, 초록 = AI 모델 단계.", 13)
    return f


def chatbot() -> Fig:
    f = Fig(1720, 690)
    f.header("정의", "챗봇 ReAct 루프",
             ["질문을 코드가 먼저 검사하고, 모델이 도구 · 문서를 골라 반복 호출합니다.",
              "최종 답의 숫자는 코드가 도구 결과 · 문서와 대조합니다(PEV 대신 ReAct)."])
    y, H = 210, 96
    q = f.box(30, y, 170, H, "질문", "대화 기록에 추가")
    g = f.box(240, y, 230, H, "코드 가드레일", "개인정보 · 계정 · 인젝션", sub2="인계 데이터 분석")
    d1 = f.diamond(600, y + H / 2, 170, 100, "범위 밖?")
    m = f.box(730, y, 220, H, "모델 호출", "기록 정리 · num_ctx 8192", "green", sub2="오래된 대화 · 도구 결과 축약")
    d2 = f.diamond(1085, y + H / 2, 170, 100, "도구 요청?")
    v = f.box(1225, y, 220, H, "숫자 검증", "도구 · 문서에 있는 숫자만")
    a = f.box(1490, y, 200, H, "답", "실행 흔적 카드")
    f.arrow([q["r"], g["l"]])
    f.arrow([g["r"], d1["l"]])
    f.arrow([d1["r"], m["l"]], "아니오", lpos=(700, y + 38))
    f.arrow([m["r"], d2["l"]])
    f.arrow([d2["r"], v["l"]], "없음", lpos=(1188, y + 38))
    f.arrow([v["r"], a["l"]], "통과", lpos=(1468, y + 38))
    ref = f.box(500, 420, 200, H, "거절", "모델 호출 없음", "amber")
    f.arrow([d1["b"], ref["t"]], "예", lpos=(612, 390), lanchor="start")
    t = f.box(975, 420, 220, H, "도구 실행", "기존 13 · 문서 · 처리업체", sub2="TOOLS[이름](**인자)")
    r = f.box(730, 560, 220, 90, "결과 기록", "tool 메시지 · 소요 시간")
    f.arrow([d2["b"], (1085, 420)], "있음", lpos=(1097, 390), lanchor="start")
    f.arrow([(1085, 516), (1085, 605), r["r"]])
    f.arrow([r["t"], m["b"]], "다시 호출 · 최대 5회", lpos=(852, 395), lanchor="start")  # 결과 기록 → 모델 호출(바로 위)
    lim = f.box(1265, 420, 190, H, "상한 도달", "도구 없이 답 마무리", "dashed")
    f.arrow([t["r"], lim["l"]], "6번째", dashed=True, lpos=(1230, 456))
    re_ = f.box(1480, 420, 220, H, "다시 답하기 (1회)", "그래도 남으면 배지", "dashed", sub2="'검증 안 된 숫자'")
    f.arrow([v["b"], (1335, 366), (1588, 366), re_["t"]], "근거 없는 숫자", dashed=True, lpos=(1350, 356), lanchor="start")
    return f


def render(fig: Fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{name}.svg"
    svg.write_text(fig.svg(), encoding="utf-8")
    page = Path(tempfile.mkdtemp()) / "p.html"
    page.write_text(f'<!doctype html><meta charset="utf-8"><style>html,body{{margin:0}}</style>{fig.svg()}', encoding="utf-8")
    png = OUT / f"{name}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={fig.w},{fig.h}",
                    "--force-device-scale-factor=2", f"--screenshot={png}", page.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    return png


if __name__ == "__main__":
    print(render(system(), "system_pipeline"))
    print(render(chatbot(), "chatbot_react_loop"))
