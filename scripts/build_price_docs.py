"""재활용가능자원 가격조사 간행물(PDF, 2025-01 ~ 2026-10) → 챗봇 문서 검색(RAG)용 문서 (사용자 요청 2026-11-19).

- 원본: references/recycling_price/*.pdf (순환자원정보센터 www.re.or.kr 재활용가격 게시판에서 사용자가 내려받음, 한국환경공단 국가승인통계 제392001호)
- 출력 (data/docs/재활용가격/):
  1) 가격 시계열.md — 품목 26개 × (2021~2025 연평균, 2025-01 ~ 2026-10 월별) × (전국 + 8개 권역), 품목별 가격조사 단계 기준
  2) 월별 시장 동향/<YYYY-MM>.md — 각 호의 국내외 경제 동향 · 원자재 시장 동향 · 품목별 전월 대비 문장
  3) 부록.md — 최신호의 FAQ · 용어 안내
  + data/processed/recycling_price/price_long.csv (품목·기간·권역·가격·출처 호) — 같은 달 값이 호마다 다르면 최신 호 값을 쓰고 수정 건수를 남긴다
실행: .venv/Scripts/python -m scripts.build_price_docs
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "references/recycling_price"
OUT = ROOT / "data/docs/재활용가격"
CSV = ROOT / "data/processed/recycling_price/price_long.csv"
REGIONS = ["전국", "수도권", "강원", "충북", "충남", "전북", "전남", "경북", "경남"]
# 바깥 묶음 필수(없으면 '|-' 가 정규식 전체를 갈라 놓음). 소수부는 짧게 잡고 뒤에 빈칸·끝·'-'·다음 천 단위 수가 와야 끝낸다
# (PDF 추출이 '2,663.62,672.8'처럼 두 값을 붙여 내는 경우 → 2,663.6 / 2,672.8)
NUM = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+?(?=\s|$|-|\d{1,3},\d{3}))?|-)"


def edition(p: Path) -> str:
    m = re.search(r"(20\d\d)년\s*(\d{1,2})월", unicodedata.normalize("NFC", p.name))
    return f"{m.group(1)}-{int(m.group(2)):02d}"


def clean(t: str) -> str:
    """세로로 찍힌 오른쪽 탭 글자(한 글자 줄)와 쪽 머리를 지운다."""
    lines = []
    for ln in t.splitlines():
        s = ln.strip()
        if len(s) <= 1 or re.fullmatch(r"(한국환경공단|\d+ MARKET PRICE SURVEY ON RECYCLABLE MATERIALS|한국환경공단 \d+)", s):
            continue
        if re.match(r"^한국환경공단 \d+\s*$", s) or re.match(r"^\d+ MARKET PRICE SURVEY", s):
            continue
        lines.append(s)
    return "\n".join(lines)


LABEL_FIX: list = []

# 2026-11-19: 표가 줄글로 풀린 부분을 작은 모델이 읽기 쉽게(실측: 플레이크/펠렛 구분을 놓치고 유가 US$/배럴을 원으로 답함)
_SPACED = re.compile(r"^((?:[가-힣A-Z] )+[가-힣A-Z])(?=\s|$)")  # '압 축', '폐 신 문 지', 'E P S' → 붙여 쓰기


def _join(ln: str) -> str:
    return _SPACED.sub(lambda m: m.group(1).replace(" ", ""), ln.strip())


_HDR_TOK = re.compile(r"20\d\d년(?:\s*\d{1,2}월)?|전월비|전년\s*동월비")
_HDR: list = []  # 바로 앞 '구분 …' 머리글의 열 이름(섹션이 바뀌면 비움)


def _label_row(s: str) -> str:
    """'Dubai 75.7 … 113.7 28.0% 62.4%' → 'Dubai: 2024년 75.7, …, 2026년 9월 113.7, 전월비 28.0%, …' (열 수가 머리글과 같을 때만)."""
    m = re.match(r"^(\D+?)\s+(-?[\d,.]+%?(?:\s+-?[\d,.]+%?)+)\s*$", s)
    if not (_HDR and m):
        return s
    vals = m.group(2).split()
    if len(vals) != len(_HDR):
        return s
    return f"{m.group(1)}: " + ", ".join(f"{h} {v}" for h, v in zip(_HDR, vals))


def tidy_line(ln: str, cat: str | None, unit: str | None) -> str:
    s = _join(ln)
    if s.startswith("구분"):
        _HDR[:] = [re.sub(r"\s+", " ", t) for t in _HDR_TOK.findall(s)]
        return s
    if _ROW.match(s):
        s2 = _label_row(s)
        if s2 != s:
            s = s2 if not (unit and "," not in unit and not re.search(r"물\s*량|수입량", s)) else f"{s2} (단위: {unit})"
            return s
    if cat and "전월 대비" in s and not s.startswith(cat):
        s = f"{cat} {s}"  # 'PE 전월 대비 …' → '압축 PE 전월 대비 …'
    # 단위는 '이름 + 숫자 4개 이상'인 표 행에만, 단위가 하나일 때만 붙인다(물량 행·'US$/톤, 톤' 같은 혼합 단위는 건드리지 않음)
    if unit and "," not in unit and _ROW.match(s) and not re.search(r"물\s*량|수입량", s):
        s = f"{s} (단위: {unit})"
    return s


_ROW = re.compile(r"^[A-Za-z가-힣][A-Za-z가-힣()·\s]{0,15}\s+-?[\d,]+(\.\d+)?(\s+-?[\d,]+(\.\d+)?%?){3,}\s*$")


def next_state(ln: str, cat: str | None, unit: str | None) -> tuple[str | None, str | None]:
    s = ln.strip()
    mu = re.match(r"^\(단위\s*:\s*([^)]+)\)", s)
    if mu:
        return None, mu.group(1).strip()
    if s.startswith(("*", "•", "[", "·")):
        return None, None
    if _SPACED.fullmatch(s) or (_SPACED.match(s) and _SPACED.match(s).group(0) == s):
        return _join(s), None  # 품목 묶음 이름만 있는 줄
    return cat, unit


def parse_pdf(p: Path) -> tuple[list[dict], dict, list[str], list[str]]:
    """반환: (가격 행, 품목 정의, 시장 동향 쪽, 부록 쪽)."""
    r = PdfReader(str(p))
    ed = edition(p)
    texts = [pg.extract_text() or "" for pg in r.pages]
    rows, defs, trend, appx = [], {}, [], []
    first_item = None
    for i, t in enumerate(texts):
        head = re.search(r"^\s*(\d{1,2})\s{1,3}([^\n]+?)\s*$", t.split("구분 전국")[0].strip().splitlines()[-1] if "구분 전국" in t else "", re.M)
        if "구분 전국 수도권" in t:
            first_item = first_item if first_item is not None else i - 1
            name = None
            for ln in t.splitlines():
                m = re.match(r"^\s*(\d{1,2})\s+(.+?)\s*$", ln)
                if m and not re.search(NUM + r"\s+" + NUM, ln) and "MARKET" not in ln:
                    name = re.sub(r"\s+", " ", m.group(2)).strip()
                    break
            body = t.split("구분 전국 수도권 강원 충북 충남 전북 전남 경북 경남", 1)[1].split("(단위", 1)[0]
            body = re.sub(r"(\d)\s*,\s*(\d{3})", r"\1,\2", body)  # PDF 추출이 '2 ,688.3'처럼 천 단위 쉼표를 띄우는 경우
            body = re.sub(r"2\s*0\s*(\d)\s*(\d)\s*\(\s*평\s*균\s*\)", r"20\1\2(평균)", body)  # '2 0 2 2 (평 균)'
            # 줄 단위: 라벨(2021(평균) · 2025.10 · 11=같은 해 달) + 값 9개. 값이 다음 줄로 넘어가면 이어 붙인다.
            year, buf, monthly = None, "", []
            for ln in body.splitlines():
                buf = (buf + " " + ln).strip() if buf else ln.strip()
                m = re.match(r"^(20\d\d\(평균\)|20\d\d\.\d{1,2}|\d{1,2})\s+(.*)$", buf)
                if not m:
                    buf = ""
                    continue
                vals = re.findall(NUM, m.group(2))
                if len(vals) < 9:
                    continue  # 다음 줄과 합쳐 다시 본다
                lab, vals, buf = m.group(1), vals[:9], ""
                if lab.endswith("(평균)"):
                    period = lab[:4]
                elif "." in lab:
                    year, mon = lab.split(".")
                    period = f"{year}-{int(mon):02d}"
                elif year and 1 <= int(lab) <= 12:
                    period = f"{year}-{int(lab):02d}"
                else:
                    continue
                if len(period) == 7:
                    monthly.append((period, vals))  # 월 행은 아래에서 순서로 기간을 다시 매긴다
                    continue
                for reg, v in zip(REGIONS, vals):
                    rows.append({"item": name, "period": period, "region": reg,
                                 "price": None if v == "-" else float(v.replace(",", ""))})
            # 월 행 = 해당 호 월로 끝나는 연속 월(간행물 라벨 오타 대비: 2025-02호가 '2024.2'를 '2025.2'로 표기)
            ey, em = map(int, ed.split("-"))
            for j, (lab_period, vals) in enumerate(reversed(monthly)):
                yy, mm = ey, em - j
                while mm <= 0:
                    yy, mm = yy - 1, mm + 12
                period = f"{yy}-{mm:02d}"
                if period != lab_period:
                    LABEL_FIX.append((ed, name, lab_period, period))
                for reg, v in zip(REGIONS, vals):
                    rows.append({"item": name, "period": period, "region": reg,
                                 "price": None if v == "-" else float(v.replace(",", ""))})
            # 정의 쪽(바로 앞 쪽)
            prev = texts[i - 1]
            d = {}
            for key in ("품질 상태", "품질 등급", "가격조사단계 기준"):
                m = re.search(key + r"\s*(.+)", prev)
                if m:
                    val = m.group(1).split("품질 등급")[0].strip() if key == "품질 상태" else m.group(1).strip()
                    d[key] = re.sub(r"\s+", " ", val)
            if name:
                defs[name] = d
        _ = head
    # 시장 동향: 목차 다음(조사 소개) ~ 첫 품목 정의 쪽 전까지
    start = next((i for i, t in enumerate(texts) if "국내 경제" in t), 7)
    trend = [clean(texts[i]) for i in range(start, (first_item or start + 7))]
    ap = next((i for i, t in enumerate(texts) if t.strip().startswith("부록1") or "부록1. 재활용가능자원 가격조사 FAQ" in t), None)
    if ap is not None:
        appx = [clean(t) for t in texts[ap + 1:] if len(t.strip()) > 60]
    return rows, defs, trend, appx


def main() -> None:
    pdfs = sorted(SRC.glob("*.pdf"), key=edition)
    allrows, defs, trends, appx = [], {}, {}, []
    for p in pdfs:
        ed = edition(p)
        rows, d, tr, ax = parse_pdf(p)
        for r_ in rows:
            r_["edition"] = ed
        allrows += rows
        defs.update(d)
        trends[ed] = tr
        appx = ax or appx
        print(f"{ed}: 가격 행 {len(rows):,} · 품목 {len(d)} · 동향 {len(tr)}쪽")
    df = pd.DataFrame(allrows).dropna(subset=["item"])
    df = df.sort_values("edition")
    rev = (df.dropna(subset=["price"]).groupby(["item", "period", "region"])["price"].nunique() > 1).sum()
    latest = df.groupby(["item", "period", "region"], as_index=False).last()
    CSV.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(CSV, index=False, encoding="utf-8-sig")
    items = list(dict.fromkeys(df["item"]))
    print(f"품목 {len(items)} · 기간 {latest.period.nunique()} · 호마다 달라진 값(수정) {rev}건 → 최신 호 값 사용 · 라벨 오타 보정 {len({(a, c) for a, b, c, d in LABEL_FIX})}건 {sorted({a for a, *_ in LABEL_FIX})}")

    OUT.mkdir(parents=True, exist_ok=True)
    head = (f"- 출처: 한국환경공단 「재활용가능자원 가격조사」 간행물 {pdfs[0].name and edition(pdfs[0])} ~ {edition(pdfs[-1])}호(국가승인통계 제392001호), "
            "순환자원정보센터 www.re.or.kr 재활용가격 게시판 · 원본 references/recycling_price/\n"
            "- **단위: 원/kg, 부가가치세(VAT) 포함, 물류비 제외.** 매월 10일 조사 · 26일 공표. 각 권역 안 조사업체 가격의 평균(업체별 가격은 다를 수 있음).\n"
            "- '-' = 조사 불가 또는 업체 수 부족으로 미공표(전국 평균에는 미공표 권역 가격도 포함해 공표). 제주는 표본 부족으로 조사 제외.\n"
            "- **주의: 대부분 가공업체가 다음 단계 업체에 '판매'하는 가격이다(폐지·철스크랩만 중간가공업체 '매입' 가격).** "
            "공장이 불량 사출품을 처리업체에 넘길 때 받는 값이나 내는 처리비가 아니며, 실제 거래가는 업체 견적이 필요하다.\n"
            f"- 같은 달 값이 호마다 다르면 최신 호 값을 썼다(수정 {rev}건). 2025년 2월호 표의 월 라벨 오타('2024.2'를 '2025.2'로 표기)는 행 순서로 바로잡았다.\n"
            "- 공공데이터포털 CSV(2026-10)와 대조: 208칸 중 207칸 일치, 철캔 경남만 다름(간행물 377.3 · CSV 369.9) → 간행물 값을 씀.\n"
            f"- 생성일: {date.today()}\n")
    L = ["# 재활용가능자원 가격 시계열 (2021~2025 연평균 · 2025-01 ~ 2026-10 월별)", "", head]
    for it in items:
        d = defs.get(it, {})
        L += [f"## {it} 가격 (원/kg)", ""]
        if d:
            L.append(f"- {it} 품질 상태: {d.get('품질 상태', '—')} · 품질 등급: {d.get('품질 등급', '—')}")
            L.append(f"- {it} 가격조사 단계 기준: {d.get('가격조사단계 기준', '—')}")
            L.append("")
        g = latest[latest["item"] == it].pivot(index="period", columns="region", values="price")
        g = g.reindex(columns=REGIONS)
        g = g.loc[sorted(g.index, key=lambda s: (len(s) == 7, s))]
        # 한 줄 = 한 기간(작은 LLM 이 넓은 표의 칸을 잘못 읽어 줄 단위 '권역 값' 형식으로, 2026-11-19)
        for per, row in g.iterrows():
            lab = f"{per}년 연평균" if len(per) == 4 else f"{per[:4]}년 {int(per[5:])}월({per})"
            L.append(f"- {it} {lab} 원/kg: " + ", ".join(f"{reg} {'미공표' if pd.isna(v) else format(v, ',.1f')}"
                                                       for reg, v in row.items()))
        L.append("")
    (OUT / "재활용가능자원 가격 시계열.md").write_text("\n".join(L), encoding="utf-8")

    tdir = OUT / "월별 시장 동향"
    tdir.mkdir(exist_ok=True)
    for ed, pages in trends.items():
        y, m = ed.split("-")
        body = "\n\n".join(pages)
        heads = re.compile(r"^(국내 경제|미국 경제|일본 경제|유럽 경제|중국 경제|경제 지표|국내 폐지|수입 폐지|국내 폐플라스틱|수입 폐플라스틱|"
                           r"국내 폐유리병|국내 폐금속|국내 폐타이어|현물 유가 전망|국제 유가|LME가격|원자재 시장동향 및 전망)\s*$")
        out = [f"# 재활용가능자원 가격조사 {y}년 {int(m)}월호 시장 동향", "",
               f"- 출처: 한국환경공단 재활용가능자원 가격조사 {y}년 {int(m)}월호(매월 10일 조사). 원/kg은 VAT 포함·물류비 제외.", ""]
        cat, unit = None, None
        for ln in body.splitlines():
            if heads.match(ln.strip()):  # 쪽 안의 소제목마다 섹션(검색 조각 단위)
                out += ["", f"## {y}년 {int(m)}월호 {ln.strip()}", ""]
                cat, unit = None, None
                _HDR.clear()
            else:
                out.append(tidy_line(ln, cat, unit))
                cat, unit = next_state(ln, cat, unit)
        (tdir / f"{ed} 시장 동향.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    A = ["# 재활용가능자원 가격조사 부록 (FAQ · 품목별 세부 기준 · 용어)", "",
         f"- 출처: 한국환경공단 재활용가능자원 가격조사 {edition(pdfs[-1])}호 부록", ""]
    for i, t in enumerate(appx):
        first = t.splitlines()[0][:40] if t else f"{i + 1}"
        A += [f"## 부록 {i + 1}: {first}", "", t, ""]
    (OUT / "재활용가능자원 가격조사 부록.md").write_text("\n".join(A), encoding="utf-8")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
