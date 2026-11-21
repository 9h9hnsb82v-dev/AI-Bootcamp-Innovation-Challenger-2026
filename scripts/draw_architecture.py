"""발표용 설계도 4종 (2026-11-19): 전체 시스템 구조도 · 핵심 업무 설계도 · 모델/데이터/DB 설계도 · 프로젝트 구조도.

draw_pipeline.py 와 같은 그림체(둥근 상자, 초록 = AI 단계, 점선 = 독립·예외). 숫자·이름은 현재 코드·설정에서 확인한 값만 쓴다.
실행: .venv/Scripts/python -m scripts.draw_architecture
출력: reports/figures/{arch_system, arch_business, arch_model_data_db, arch_project}.{svg,png}
"""
from __future__ import annotations

import html

from scripts.draw_pipeline import AMBER_BG, AMBER_LINE, FONT, GREEN_BG, GREEN_LINE, GREEN_TXT, MONO, Fig, render

GRAY_BG, GRAY_LINE = "#F6F7F8", "#C9CDD2"
BLUE_BG, BLUE_LINE, BLUE_TXT = "#EEF3FB", "#7A9CD3", "#2F5597"


class Fig2(Fig):
    def group(self, x, y, w, h, title, sub="", color="gray"):
        bg, line, txt = {"gray": (GRAY_BG, GRAY_LINE, "#444"), "green": (GREEN_BG, GREEN_LINE, GREEN_TXT),
                         "blue": (BLUE_BG, BLUE_LINE, BLUE_TXT), "amber": (AMBER_BG, AMBER_LINE, "#8a5a00")}[color]
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{bg}" stroke="{line}" stroke-width="1.4"/>')
        self.el.append(self._t(x + 18, y + 30, title, 17, txt, 800))
        if sub:
            self.el.append(self._t(x + 18, y + 52, sub, 13, "#777", 400))

    def chip(self, x, y, w, h, title, sub="", kind="plain", mono=False, sub2=""):
        fill, line, dash = {"plain": ("#fff", "#8a8a8a", ""), "green": (GREEN_BG, GREEN_LINE, ""),
                            "dashed": ("#fafafa", "#9a9a9a", ' stroke-dasharray="5 4"'),
                            "amber": (AMBER_BG, AMBER_LINE, ' stroke-dasharray="5 4"'), "blue": ("#fff", BLUE_LINE, "")}[kind]
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{line}" stroke-width="1.3"{dash}/>')
        cy = y + h / 2 + (6 if not sub else -3) - (10 if sub2 else 0)
        self.el.append(self._t(x + w / 2, cy, title, 15, "#111", 700, "middle", MONO if mono else FONT))
        if sub:
            self.el.append(self._t(x + w / 2, cy + 19, sub, 12, "#666", 400, "middle"))
        if sub2:
            self.el.append(self._t(x + w / 2, cy + 38, sub2, 12.5, "#2F5597", 700, "middle"))
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y), "b": (x + w / 2, y + h)}

    def table(self, x, y, w, name, cols, kind="plain", note=""):
        """DB 테이블 상자: 이름 + 열 목록(PK·FK 표시)."""
        rh, hh = 21, 34
        h = hh + rh * len(cols) + 10
        head_bg = {"plain": "#E9ECEF", "green": GREEN_BG, "blue": BLUE_BG, "dashed": "#F3F3F3"}[kind]
        line = {"plain": "#7b7b7b", "green": GREEN_LINE, "blue": BLUE_LINE, "dashed": "#9a9a9a"}[kind]
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#fff" stroke="{line}" stroke-width="1.4"/>')
        self.el.append(f'<path d="M{x},{y + hh} L{x},{y + 10} Q{x},{y} {x + 10},{y} L{x + w - 10},{y} Q{x + w},{y} {x + w},{y + 10} '
                       f'L{x + w},{y + hh} Z" fill="{head_bg}" stroke="{line}" stroke-width="1.4"/>')
        self.el.append(self._t(x + 12, y + 23, name, 15, "#111", 800, "start", MONO))
        if note:
            self.el.append(self._t(x + w - 10, y + 22, note, 11, "#777", 400, "end"))
        for i, c in enumerate(cols):
            color = "#B03A2E" if c.startswith("PK") else BLUE_TXT if c.startswith("FK") else "#333"
            self.el.append(self._t(x + 12, y + hh + 17 + i * rh, c, 12.5, color, 500 if c[:2] in ("PK", "FK") else 400, "start", MONO))
        return {"l": (x, y + hh / 2), "r": (x + w, y + hh / 2), "t": (x + w / 2, y), "b": (x + w / 2, y + h), "h": h}

    def rule(self, x, y, w, h, num, title, lines, kind="plain"):
        """설계 원칙·업무 규칙 카드: 번호 동그라미 + 제목 + 설명 줄."""
        fill, line = {"plain": ("#fff", "#5a5a5a"), "green": (GREEN_BG, GREEN_LINE), "blue": (BLUE_BG, BLUE_LINE),
                      "amber": (AMBER_BG, AMBER_LINE)}[kind]
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{line}" stroke-width="1.5"/>')
        self.el.append(f'<circle cx="{x + 30}" cy="{y + 30}" r="15" fill="#fff" stroke="{GREEN_LINE}" stroke-width="1.8"/>')
        self.el.append(self._t(x + 30, y + 36, str(num), 15, GREEN_TXT, 800, "middle"))
        self.el.append(self._t(x + 54, y + 36, title, 17, "#111", 800))
        for i, ln in enumerate(lines):
            self.el.append(self._t(x + 20, y + 68 + i * 22, ln, 13.5, "#333", 500))

    def card(self, x, y, w, h, title, lines, kind="plain", desc=""):
        fill, line = {"plain": ("#fff", "#5a5a5a"), "green": (GREEN_BG, GREEN_LINE), "gray": (GRAY_BG, GRAY_LINE),
                      "blue": (BLUE_BG, BLUE_LINE)}[kind]
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{line}" stroke-width="1.5"/>')
        self.el.append(self._t(x + 16, y + 30, title, 18, "#111", 800, "start", MONO))
        if desc:
            self.el.append(self._t(x + 16, y + 52, desc, 13.5, "#555", 500))
        top = y + (78 if desc else 58)
        for i, ln in enumerate(lines):
            name, _, d = ln.partition("|")
            self.el.append(self._t(x + 20, top + i * 23, name.strip(), 13, "#222", 600, "start", MONO))
            if d:
                self.el.append(self._t(x + w - 14, top + i * 23, d.strip(), 12.5, "#777", 400, "end"))


# ── 1. 전체 시스템 구조도 ────────────────────────────────────────────────
def system() -> Fig2:
    f = Fig2(1720, 1260)
    f.header("구조", "전체 시스템 구조도",
             ["브라우저 화면 → 서비스 → AI 모델 → 데이터 저장소의 계층 구조와 실행 환경.",
              "모든 숫자는 Python 서비스가 계산하고, 화면과 챗봇은 같은 서비스를 부릅니다."])
    X, W = 30, 1290
    # 사용자
    f.group(X, 165, W, 110, "사용자", "브라우저(localhost 또는 SSH 터널)")
    for i, (t, s) in enumerate([("생산 관리자", "계획 · 수거 · 처리업체"), ("검사 작업자", "사진 등록 · 최종 판정"),
                                ("카메라 · 사진 파일", "근접 · 차량 전체 사진")]):
        f.chip(X + 250 + i * 345, 185, 325, 70, t, s)
    # 화면
    f.group(X, 300, W, 165, "화면 계층 · Streamlit (ui/)", "디자인 3종 classic · D · F 전환(RESOURCE_DESIGN) · 시작 시 모델 예열")
    pages = ["대시보드", "생산계획", "사전 예측", "이미지 검사", "예측 갱신", "보관 · 수거", "공장 배치도", "처리업체 지도",
             "수주량 예측", "AI 챗봇", "근거 · 성능"]
    for i, p in enumerate(pages):
        f.chip(X + 20 + (i % 6) * 210, 365 + (i // 6) * 48, 196, 38, p, kind="green" if p in ("이미지 검사", "AI 챗봇") else "plain")
    # 서비스
    f.group(X, 490, W, 175, "서비스 계층 (services/ · agent/)", "업무 규칙 · 계산 · 저장. 쓰기 작업은 사용자 확인 후 실행")
    svcs = [("production_plan", "생산계획"), ("waste_forecast", "사전 예측 kg"), ("inspection_registry", "검사 기록 · 판정"),
            ("waste_forecast_update", "검사 후 갱신"), ("storage_projection", "보관량 예측"), ("collection_recomm.", "수거 추천"),
            ("waste_processors", "처리업체 · 분류번호"), ("truck_route", "TMAP 도로거리"), ("supply_forecast", "수주량 예측"),
            ("doc_search", "문서 검색 RAG"), ("ollama_client", "챗봇 ReAct 루프"), ("guard · tools", "가드레일 · 도구 15개")]
    for i, (n, d) in enumerate(svcs):
        f.chip(X + 20 + (i % 6) * 210, 552 + (i // 6) * 54, 196, 46, n, d, kind="green" if n in ("ollama_client", "doc_search") else "plain", mono=True)
    # 모델
    f.group(X, 690, W, 150, "AI · 모델 계층 (ml/ · models/ · Ollama)", "학습은 Azure ML H100·팀 VM, 앱에서는 CPU 추론")
    mods = [("EfficientNet-B0 + DINOv2", "근접 사진 양품/불량 · 2차 판정"), ("단차 · 결함 위치", "EfficientNet 단차 · YOLO11 검출(표시)"),
            ("Mask R-CNN", "차량 사진 부품 위치 안내"), ("KAMP ID4 불량률", "Jeffreys 사후평균 · XGB 참고"),
            ("KAMP ID40 수주 예측", "HGB · XGB 백테스트"), ("Ollama", "llama3.1 한국어 8B · bge-m3")]
    for i, (n, d) in enumerate(mods):
        f.chip(X + 20 + (i % 3) * 420, 748 + (i // 3) * 46, 405, 42, n, d, kind="green")
    # 데이터
    f.group(X, 865, W, 135, "데이터 계층", "")
    data = [("SQLite DB", "계획 · 검사 · 장부 · 설정 10개 표"), ("configs/*.json", "품목 · 창고 · 수주 정책(시연 가정)"),
            ("data/docs", "법령 · 공공데이터 · 출처 · 사용법"), ("원천 데이터", "AI-Hub · KAMP · 공공데이터 CSV"),
            ("검사 사진", "data/app/photos")]
    for i, (n, d) in enumerate(data):
        f.chip(X + 20 + i * 252, 905, 238, 70, n, d, kind="blue")
    # 계층 사이 화살표
    for y0, y1 in ((275, 300), (465, 490), (665, 690), (840, 865)):
        f.arrow([(X + W / 2, y0), (X + W / 2, y1)])
    # 오른쪽: 외부 · 실행 환경
    RX, RW = 1345, 345
    f.group(RX, 165, RW, 300, "외부 서비스", "앱에서 호출하는 것만")
    f.chip(RX + 20, 230, RW - 40, 60, "TMAP (SK open API)", "화물차 경로 · 장소 검색 · 좌표→주소")
    f.chip(RX + 20, 305, RW - 40, 60, "국가법령정보센터", "법령 원문 수집 스크립트(오프라인)", "dashed")
    f.chip(RX + 20, 380, RW - 40, 60, "공공데이터포털", "파일 내려받기(키 없음)", "dashed")
    f.group(RX, 490, RW, 510, "실행 환경", "")
    f.chip(RX + 20, 535, RW - 40, 120, "이 PC (Windows)", "localhost 8601 · 8602 · 8603", "blue")
    f.note(RX + RW / 2, 630, "RTX 4050 · Ollama 로컬", 12.5, "#666", "middle")
    f.chip(RX + 20, 675, RW - 40, 150, "팀 VM AICD (A10)", "tmux · 127.0.0.1:8601 / 8603", "blue")
    f.note(RX + RW / 2, 790, "팀 공용 DB team.db · Ollama GPU", 12.5, "#666", "middle")
    f.note(RX + RW / 2, 812, "외부 공개 없음 · SSH 터널로만 접속", 12.5, "#666", "middle")
    f.chip(RX + 20, 845, RW - 40, 130, "Azure ML gpu-h100", "학습 전용 · 유휴 시 0대", "dashed")
    f.note(RX + RW / 2, 958, "DINOv2 · YOLO · EfficientNet 학습", 12.5, "#666", "middle")
    # 학습 · 운영 (오프라인)
    f.group(30, 1025, 1660, 150, "학습 · 운영 (오프라인)", "앱 실행과 별도 · 결과 모델만 models/ 에 저장")
    steps = [("원천 데이터", "AI-Hub · HITL · carparts · KAMP"), ("데이터 감사 · 분할", "reports/audit · 촬영일 통째 분할"),
             ("Azure ML 학습", "gpu-h100 · resource-cpu"), ("MLflow 기록", "에폭 곡선 · 시험 지표 · 성능표"),
             ("모델 저장 · 지정", "models/ · serving.json")]
    prev = None
    for i, (t_, s_) in enumerate(steps):
        c = f.chip(260 + i * 288, 1085, 258, 64, t_, s_, kind="green" if i == 2 else "plain")
        if prev:
            f.arrow([prev["r"], c["l"]])
        prev = c
    f.note(30, 1205, "초록 = AI 모델·챗봇 관련, 파랑 = 데이터 저장소·실행 장비, 점선 = 앱 실행 중에는 쓰지 않는 것.", 13)
    f.note(30, 1230, "한국환경공단 마감인계서 데이터는 범위 제외(사용하지 않음). 인계서 제도 설명은 법령 문서로만 답합니다.", 13)
    return f


# ── 2. 핵심 업무 설계도 (스윔레인) ────────────────────────────────────────
def business() -> Fig2:
    f = Fig2(1720, 1390)
    f.header("업무", "핵심 업무 설계도",
             ["역할별로 누가 무엇을 하고 시스템이 무엇을 계산하는지 나눈 흐름입니다.",
              "AI 판정은 추천일 뿐이고, kg에는 작업자의 '폐기 확정'만 반영됩니다."])
    lanes = [("생산 관리자", 165), ("검사 작업자", 380), ("시스템 (계산 · AI)", 595), ("물류 · 환경 담당", 810)]
    for name, y in lanes:
        f.el.append(f'<rect x="30" y="{y}" width="1660" height="200" rx="14" fill="{GRAY_BG}" stroke="{GRAY_LINE}" stroke-width="1.2"/>')
        f.el.append(f'<rect x="30" y="{y}" width="150" height="200" rx="14" fill="#E9ECEF" stroke="{GRAY_LINE}" stroke-width="1.2"/>')
        f.el.append(f._t(105, y + 106, name, 16, "#222", 800, "middle"))
    W, H = 200, 84
    p1 = f.box(210, 223, W, H, "① 생산계획 등록", "품목 · 날짜 · 수량")
    s1 = f.box(210, 653, W, H, "② 사전 예측", "수량×불량률×단중", "green")
    w1 = f.box(470, 438, W, H, "③ 부품 사진 등록", "파일 · 카메라 · 샘플")
    s2 = f.box(470, 653, W, H, "④ AI 판정", "EfficientNet+DINOv2", "green")
    d = f.diamond(800, 695, 160, 96, "일치?")
    w2 = f.box(700, 438, W, H, "⑤ 실물 확인", "처리 대기함", "amber")
    w3 = f.box(950, 438, W, H, "⑥ 최종 판정", "폐기 · 양품 · 재작업")
    s3 = f.box(950, 653, W, H, "⑦ 예측 갱신", "확정 폐기 → kg · 장부 입고")
    s4 = f.box(1200, 653, W, H, "⑧ 보관량 예측", "경보 800 · 용량 1,000 kg")
    l1 = f.box(1200, 868, W, H, "⑨ 수거 예약", "권장일 · 요청 마감 2일 전")
    l2 = f.box(1460, 868, 210, H, "⑩ 위탁 처리업체", "가까운 순 · 도로거리")
    s5 = f.box(1460, 653, 210, H, "장부 출고", "수거 시 보관량 차감")
    p2 = f.box(470, 223, W, H, "수주량 예측 참고", "KAMP ID40 · 개수", "dashed")
    p3 = f.box(730, 223, W, H, "계획 초안 확정", "경고 확인 후 직접", "dashed")
    f.arrow([p1["b"], s1["t"]])
    f.arrow([s1["r"], (440, 695), (440, 480), w1["l"]])
    f.arrow([w1["b"], s2["t"]])
    f.arrow([s2["r"], d["l"]])
    f.arrow([d["t"], w2["b"]], "엇갈림", lpos=(812, 600), lanchor="start")
    f.arrow([d["r"], (915, 695), (915, 600), (1000, 600), (1000, 522)], "일치(AI 추천)", lpos=(1010, 590), lanchor="start")
    f.arrow([w2["r"], w3["l"]])
    f.arrow([(1110, 522), (1110, 653)], "확정", lpos=(1120, 600), lanchor="start")
    f.arrow([s3["r"], s4["l"]])
    f.arrow([s4["b"], l1["t"]], "경보 · 초과 전", lpos=(1312, 815), lanchor="start")
    f.arrow([l1["r"], l2["l"]])
    f.arrow([l2["t"], s5["b"]], "수거 완료", lpos=(1577, 815), lanchor="start")
    f.arrow([p2["r"], p3["l"]], dashed=True)
    f.arrow([(830, 223), (830, 200), (310, 200), (310, 223)], "사용자 재확정", dashed=True, lpos=(560, 192))
    # 계산 · 규칙 (현재 코드 값)
    f.el.append(f._t(30, 1052, "계산 · 판정 규칙 (현재 코드 값)", 17, "#222", 800))
    RW_, RH_ = 540, 128
    rules = [
        (2, "사전 예측 kg", ["계획수량 × 불량률 × 단위중량", "예) CN7_LH 5,000개 × 0.519% × 1.30 kg = 33.76 kg",
                           "불량률: KAMP ID4 학습 구간(KAMP_TRAIN_BASELINE)"], "green"),
        (4, "AI 판정 → 자동 / 작업자 확인", ["EfficientNet · DINOv2 · 2차 판정 셋이 같고", "2차 확률이 기준(도어 0.77 · 범퍼 0.89 · 휀더 0.78)",
                                     "±0.10 밖이면 자동 판정, 아니면 처리 대기함"], "green"),
        (3, "부품 오인식 차단", ["다른 부품 확신도 ≥ 0.95 → 등록 거부", "≥ 0.5 → '작업자 확인' 표시",
                           "같은 사진 두 번 등록은 막음(중복 집계 없음)"], "plain"),
        (7, "검사 후 kg 갱신", ["확정 폐기 kg + 남은 수량 × 남은 불량률 × 단중", "개체 상태 5종(폐기 · 양품 · 재작업 · 생산 미확정 · 미생산)",
                          "의 합 = 계획수량 · 재작업 대기는 kg 미합산"], "plain"),
        (8, "보관량 예측", ["S(t+1) = S(t) + 발생 G(t) − 수거 C(t)", "확정 폐기는 장부 입고(개체당 1회), 남은 예상분은 발생 예정",
                      "경보 80% = 800 kg · 용량 1,000 kg (DEMO_ASSUMED)"], "blue"),
        (9, "수거 권장일", ["경보 도달 전 권장일 · 요청 마감 = 리드타임 2일 전", "수거는 평일(월~금) · 수거 시 보관량 0으로 출고",
                       "configs/facility_demo.json (사용자 승인 가정)"], "blue"),
    ]
    for i, (n, t_, ls, k) in enumerate(rules):
        f.rule(30 + (i % 3) * (RW_ + 20), 1068 + (i // 3) * (RH_ + 16), RW_, RH_, n, t_, ls, k)
    f.note(30, 1372, "AI 결함 의심 ≠ 폐기 확정 · 계획 등록 · 판정 · 초안 확정은 확인 버튼 후 실행 · 챗봇은 같은 서비스를 읽기로 조회합니다.", 13)
    return f


# ── 3. 모델 / 데이터 / DB 설계도 ──────────────────────────────────────────
def model_data_db() -> Fig2:
    f = Fig2(1720, 1230)
    f.header("설계", "모델 · 데이터 · DB 설계도",
             ["원천 데이터 → 학습 모델 → 서비스 → DB 표의 연결. 데이터셋끼리는 자동으로 섞지 않고",
              "사용자가 연결한 가상 시나리오만 DEMO_SYNTHETIC_BRIDGE 로 표시합니다."])
    # 원천 데이터
    f.group(30, 165, 400, 960, "원천 데이터", "data/raw · data/docs")
    src = [("AI-Hub 부품 품질 검사(자동차)", "도어 · 범퍼 · 휀더 51,986장"),
           ("세션 비겹침 분할 (촬영일 통째)", "학습 37,087 · 검증 7,953 · 시험 6,946"),
           ("검증 사진 7,953장", "기준값 · 2차 판정 계수만 맞춤"),
           ("AI-Hub 단차 · 결함 박스 라벨", "휀더 틈 · 스크래치 · 외관 손상"),
           ("car-seg · HITL 차량 부품", "CC BY 4.0 · CC0"),
           ("KAMP ID4 사출성형", "공정 · 불량 라벨(CN7 · RG3)"),
           ("KAMP ID40 공급망", "원본 32일 스냅샷 · 수주량(개)"),
           ("법령 · 공공데이터 · 사용법 문서", "data/docs (RAG)")]
    sp = {}
    for i, (t, s_) in enumerate(src):
        sp[i] = f.chip(50, 230 + i * 108, 360, 86, t, s_, kind="blue" if i < 7 else "plain")
    f.note(50, 1110, "처리업체 CSV(9,728행)는 모델 없이 서비스가 바로 조회", 12.5)
    # 모델
    f.group(470, 165, 520, 960, "모델 · 계산 (시험셋 성능)", "models/ · ml/ · Ollama · 파란 글씨 = 시험 결과")
    mods = [("EfficientNet-B0 (세션 분할)", "6클래스 · 448px · 미리보기 단독", "green", "정확도 도어 96.3 · 범퍼 96.9 · 휀더 79.7%"),
            ("DINOv2 ViT-B/14", "6클래스 · 448px · 15에폭 · Apache-2.0", "green", "정확도 도어 98.1 · 범퍼 99.4 · 휀더 82.5%"),
            ("2차 판정 (로지스틱 회귀)", "부품별 · 애매 폭 ±0.10 → 작업자 확인", "green", "자동 판정 정확도 98.5 · 99.6 · 88.6%"),
            ("단차 EfficientNet · YOLO11 검출", "휀더 틈 판정 · 결함 위치(표시 전용)", "green", "단차 자동 판정 90.8% · 검출 mAP50 0.379"),
            ("Mask R-CNN 부품 위치 (v2′)", "21개 부위 · 근접 촬영 안내 · BSD-3", "green", "mask AP 0.600 · 외부 사진 0.641"),
            ("KAMP 불량률 (Jeffreys)", "학습 구간만 · XGB 센서는 참고", "plain", "CN7_LH 0.519% ~ RG3_RH 4.486%"),
            ("수주 예측 HGB · XGB · 최근 계획", "T+1~T+4 · 검증 WAPE로 모델 선택", "plain", "시험 WAPE T+1 0.218 ~ T+4 0.412"),
            ("Ollama llama3.1 8B · bge-m3", "ReAct 챗봇 · 도구 15개 · 숫자 검증", "green", "문서 조각 715개 · 평가 17문항")]
    mp = {}
    for i, (t, s_, k, s2) in enumerate(mods):
        mp[i] = f.chip(490, 230 + i * 108, 480, 86, t, s_, kind=k, sub2=s2)
    for i in range(8):  # 같은 줄끼리 연결(원천 → 모델)
        f.arrow([sp[i]["r"], mp[i]["l"]])
    # DB
    f.group(1030, 165, 660, 960, "SQLite DB 표 (services/db.py)", "data/app/resource.db · 팀 VM team.db")
    T = {}
    T["plan"] = f.table(1050, 235, 300, "production_plan", ["PK plan_id", "product_id · plan_date", "planned_units · produced_units",
                                                          "source  DIRECT_INPUT | DEMO_SUPPLY_BRIDGE", "FK source_forecast_id"], "blue")
    T["insp"] = f.table(1050, 430, 300, "inspection", ["PK inspection_id", "FK plan_id", "unit_id · image_hash · shot",
                                                       "ai_defect_prob · ai_json · photo_path", "review_status · scrap_decision",
                                                       "operator_id · cancelled_at"], "green")
    T["log"] = f.table(1050, 660, 300, "decision_log", ["PK log_id", "FK plan_id · unit_id", "action · decision · note"])
    T["snap"] = f.table(1050, 800, 300, "forecast_snapshot", ["PK snapshot_id", "FK plan_id", "initial · confirmed_scrap_kg",
                                                             "remaining · updated_total_kg"])
    T["ledger"] = f.table(1370, 235, 300, "stock_ledger", ["PK event_id", "FK plan_id · product_id", "kg · date",
                                                         "event_type  CONFIRMED_SCRAP_IN", "  | COLLECTION_OUT | ADJUSTMENT"])
    T["etc"] = f.table(1370, 410, 300, "설정 · 캐시", ["factory_layout  배치(작업자별)", "app_settings  공장 위치",
                                                      "route_cache  TMAP 도로거리"], "dashed")
    T["draft"] = f.table(1370, 560, 300, "supply_plan_draft", ["PK draft_id", "FK supply_forecast_id", "target_product_id · qty",
                                                              "approval_status · plan_id"], "dashed")
    T["sf"] = f.table(1370, 735, 300, "supply_forecast", ["PK forecast_id", "product_number · target_date", "horizon · predicted_units",
                                                         "model_version · provenance"])
    blue = "#7A9CD3"
    for y_ in (447, 677, 817):  # inspection · decision_log · forecast_snapshot 의 FK plan_id → production_plan (왼쪽 세로선)
        f.arrow([(1050, y_), (1040, y_), (1040, 290), (1050, 290)], color=blue)
    for p_, q_ in (((1370, 280), (1350, 280)), ((1520, 688), (1520, 735))):
        f.arrow([p_, q_], color=blue)  # FK → 가리키는 표
    f.arrow([(1670, 620), (1682, 620), (1682, 222), (1200, 222), (1200, 235)], "초안 확정 → 계획", dashed=True, lpos=(1560, 215))
    # 모델 → 표 (판정 저장 · kg 계산 · 예측 저장)
    f.arrow([mp[2]["r"], (1010, mp[2]["r"][1]), (1010, 520), (1050, 520)])
    f.arrow([mp[5]["r"], (1010, mp[5]["r"][1]), (1010, 868), (1050, 868)])
    f.arrow([mp[6]["r"], (1020, mp[6]["r"][1]), (1020, 1100), (1520, 1100), (1520, 863)])
    f.note(1065, 1092, "예측 저장", 13, "#444")
    f.note(30, 1160, "빨강 PK = 기본키, 파랑 FK = 다른 표를 가리키는 키(파랑 화살표). 회색 화살표: 2차 판정 → inspection(판정 저장), "
                     "KAMP 불량률 → forecast_snapshot(kg 계산), 수주 예측 → supply_forecast.", 13)
    f.note(30, 1184, "검사 기록은 취소해도 지우지 않고 cancelled_at 으로 남깁니다. 판정 모델 구성은 models/vision/serving.json(ensemble_dino)이 정합니다.", 13)
    return f


# ── 4. 프로젝트 구조도 (폴더) ─────────────────────────────────────────────
def project() -> Fig2:
    f = Fig2(1720, 1270)
    f.header("Re:Source Predict AI", "프로젝트 구조도",
             ["ReSource_Predict_AI/ 폴더별 역할. 화면(ui) → 업무(services · agent) → 모델(ml · models)",
              "→ 데이터(data · configs) 순서로 의존하며, 학습·배포·보고 폴더는 따로 둡니다."])
    CW, GAP = 400, 20
    cols = [30 + i * (CW + GAP) for i in range(4)]
    f.card(cols[0], 165, CW, 290, "ui/", ["app.py | 진입점 · 메뉴", "pages/ | 화면 11개 (+D · F 대시보드)",
                                         "designs/ | 시안 D · F 테마 · 메뉴", "live_camera.py | 카메라 컴포넌트",
                                         "geo_locate.py | 내 위치 가져오기", "warmup.py | 서버 시작 예열", "common.py | 공통 표시"],
           "green", "Streamlit 화면")
    f.card(cols[1], 165, CW, 290, "services/", ["production_plan | 생산계획", "waste_forecast(_update) | kg 예측 · 갱신",
                                               "inspection_registry | 검사 · 판정 기록", "storage_* · collection_* | 보관 · 수거",
                                               "waste_processors · truck_route | 처리업체", "supply_* | 수주 예측 · 초안",
                                               "doc_search · db · config | 검색 · 저장"], "plain", "업무 규칙 · 계산 (19개 모듈)")
    f.card(cols[2], 165, CW, 290, "agent/", ["ollama_client.py | ReAct 루프 · 기록 정리", "tools.py | 도구 15개 · 확인 대기",
                                            "guard.py | 코드 가드레일", "eval_tool_calls.py | 도구 · 거절 평가 17문항"],
           "green", "AI 챗봇")
    f.card(cols[3], 165, CW, 290, "ml/", ["vision/ | 이미지 학습 · 추론 · 평가", "  infer.py | 앱 판정(EnsembleDINO)",
                                         "  common.py | 가벼운 상수", "  fit_dino_ensemble.py | 조합 맞추기",
                                         "kamp/ | ID4 불량률 · XGB", "supply/ | ID40 수주 예측", "tracking.py | MLflow 기록"],
           "plain", "학습 · 추론 코드")
    f.card(cols[0], 480, CW, 250, "data/", ["raw/ | AI-Hub · KAMP · 공공데이터 원본", "processed/ | 분할 · 매니페스트",
                                           "app/ | SQLite DB · 검사 사진", "docs/ | RAG 문서 · .index 색인", "demo/ | 시연 사진 · 샘플"],
           "blue", "데이터 (원본은 수정하지 않음)")
    f.card(cols[1], 480, CW, 250, "models/", ["vision/ensemble_sd_dino | 현재 판정 조합", "vision/serving.json | 쓰는 조합 지정",
                                             "vision/yolo11m_det · partseg | 위치 표시", "kamp/ | 불량률 · 센서 모델",
                                             "supply/ | 수주 예측 h1~h4"], "blue", "학습된 모델 · 설정")
    f.card(cols[2], 480, CW, 250, "configs/", ["product_master.json | 7종 품목 · 단중 · 불량률", "facility_demo.json | 창고 용량 · 경보 · 리드타임",
                                              "supply_policy.json | 수주→kg 직접 변환 금지"], "blue", "시연 가정값 (사용자 승인)")
    f.card(cols[3], 480, CW, 250, "tests/", ["test_ui_smoke · test_acceptance | 화면 · 수용", "test_rag_guard | RAG · 가드레일",
                                            "test_ollama_context | 컨텍스트 관리", "test_processors 외 | 업무 규칙",
                                            "pytest 121개 통과 | 2026-11-18"], "plain", "자동 테스트 (Ollama 없이 동작)")
    f.card(cols[0], 755, CW, 230, "scripts/", ["run_design.cmd / .sh | 시안별 실행", "fetch_laws.py | 법령 수집",
                                              "draw_pipeline · draw_architecture | 그림", "demo_scenario.py | 시연 DB",
                                              "restore_design_backup.py | 디자인 롤백"], "gray", "실행 · 도구")
    f.card(cols[1], 755, CW, 230, "azureml/", ["job_sd_*.yml | 세션 분할 학습 작업", "dino/run_dino_cls.py | DINOv2 학습",
                                              "yolo/ · environment/ | 학습 환경", "README.md | 실행 기록 · 비용"], "gray", "Azure ML 학습 (H100)")
    f.card(cols[2], 755, CW, 230, "reports/", ["vision/ | 모델 비교 · DINO 결과 · 곡선", "audit/ | 데이터 감사",
                                              "agent/ | 챗봇 평가", "figures/ | 구조도 · 설계도 그림"], "gray", "평가 · 보고 결과")
    f.card(cols[3], 755, CW, 230, "기타", ["MASTER_PROMPT.md | 최상위 요구사항", "README.md · TEAM_ACCESS.md | 실행 · 접속",
                                          "references/ | 교재 · 공공데이터 원본", "backups/ | 디자인 · 배포 전 백업"], "gray", "문서 · 참고자료")
    # 의존 방향: 화면 → 업무 ← 챗봇
    f.arrow([(cols[0] + CW, 310), (cols[1], 310)])
    f.arrow([(cols[2], 400), (cols[1] + CW, 400)])
    f.el.append(f._t(30, 1020, "핵심 설계 원칙", 17, "#222", 800))
    PW = (1660 - 4 * 16) / 5
    principles = [("AI는 제안만", ["작업자가 확정한 것만", "폐기 kg에 반영"]), ("단위 분리", ["kg(폐기물)와 개(수주량)", "자동 합산 · 연결 금지"]),
                  ("계산은 Python", ["LLM은 도구 선택만", "쓰기 작업은 사용자 확인"]), ("출처 표시", ["모든 숫자에 출처 태그", "실측과 가상 값 구분"]),
                  ("쉬운 모델 전환", ["serving.json 하나로", "DINO 조합 ↔ YOLO 조합"])]
    for i, (t_, ls) in enumerate(principles):
        f.rule(30 + i * (PW + 16), 1036, PW, 118, i + 1, t_, ls, "green" if i in (0, 2) else "plain")
    f.note(30, 1190, "초록 = 화면 · 챗봇, 파랑 = 데이터 · 모델 · 설정, 회색 = 실행 · 학습 · 보고. 키 파일(.streamlit/secrets.toml)은 "
                     "어느 폴더에도 공유하지 않습니다.", 13)
    f.note(30, 1214, "실행: scripts\\run_design.cmd classic 8601 · d 8602 · f 8603  (또는 .venv\\Scripts\\streamlit run ui/app.py)", 13)
    return f


if __name__ == "__main__":
    for fn, name in ((system, "arch_system"), (business, "arch_business"), (model_data_db, "arch_model_data_db"),
                     (project, "arch_project")):
        print(render(fn(), name))
