"""챗봇 문서 검색(RAG)용 법령 원문 수집 (사용자 승인 2026-11-18: 국가법령정보센터 웹페이지에서 수집).

- 국가법령정보센터(law.go.kr) 공개 웹페이지의 현행 법령 본문을 받아 조문 단위 Markdown 으로 저장한다.
- 법령은 저작권법 제7조에 따라 보호받지 않는 저작물이다. 출처·시행일·수집일을 파일 머리에 남긴다.
- 키·개인정보를 보내지 않는다. 요청 사이 1초 쉬고, 실패하면 그 법령만 건너뛴다.
실행: .venv/Scripts/python -m scripts.fetch_laws
출력: data/docs/법령/<법령명>.md
"""
from __future__ import annotations

import html
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/docs/법령"
LAWS = ["폐기물관리법", "폐기물관리법 시행령", "폐기물관리법 시행규칙",
        "자원의 절약과 재활용촉진에 관한 법률", "순환경제사회 전환 촉진법"]
UA = {"User-Agent": "Mozilla/5.0 (ReSource Predict AI law fetch)"}
BLOCK = r"</?(p|div|br|li|tr|h\d|table|ul|ol)[^>]*>"


def _text(raw: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw, flags=re.S | re.I)
    t = re.sub(BLOCK, "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)  # 링크 등 인라인 태그는 붙여 쓴다
    t = html.unescape(t).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.splitlines()]
    junk = re.compile(r"href=|src=|return false|onclick|javascript:")  # 속성 안의 '>' 때문에 남는 태그 조각
    return "\n".join(ln for ln in lines if ln and not junk.search(ln))


def fetch(name: str) -> tuple[str, dict]:
    page = requests.get(f"https://www.law.go.kr/법령/{quote(name)}", headers=UA, timeout=30).text
    seq = re.search(r"lsiSeq\s*[:=]\s*'?(\d+)", page)
    ef = re.search(r"efYd\s*[:=]\s*'?(\d{8})", page)
    if not seq:
        raise RuntimeError("법령 일련번호를 찾지 못함")
    meta = {"lsiSeq": seq.group(1), "efYd": ef.group(1) if ef else ""}
    time.sleep(1)
    body = requests.get("https://www.law.go.kr/LSW/lsInfoR.do", headers=UA, timeout=60, params={
        "lsiSeq": meta["lsiSeq"], "efYd": meta["efYd"], "chrClsCd": "010202", "urlMode": "lsInfoP",
        "viewCls": "lsInfoP", "ancYnChk": "0"}).text
    t = _text(body)
    start = t.find("제1조(")
    if start < 0:
        raise RuntimeError("본문(제1조)을 찾지 못함")
    t = t[start:]
    end = re.search(r"\n부칙(\s|<|$)", t, flags=re.M)  # 부칙은 길고 검색에 방해 → 제외(본문만)
    if end:
        t = t[:end.start()]
    t = re.sub(r"\n(제\d+조(의\d+)?\()", r"\n\n## \1", "\n" + t).strip()  # 조문마다 머리글
    prev, last = 0, 0  # 조문 번호가 다시 작아지는 곳 = 부칙 시작(부칙 머리 표기가 법령마다 달라 번호로 판단)
    for m in re.finditer(r"^## 제(\d+)조", t, flags=re.M):
        n = int(m.group(1))
        if n < prev:
            cut = t.find("\n부칙", last)  # 마지막 본문 조문 뒤 첫 '부칙'(조문 머리 없는 옛 부칙 포함)
            t = t[:cut if 0 < cut < m.start() else m.start()].rstrip()
            break
        prev, last = n, m.start()
    tail = re.search(r"\n(부\s*칙|제\d+조\s+\(|조문목록)", t[last:])  # 옛 부칙('제4조 (경과조치)')·화면 잔여 문구
    if tail:
        t = t[:last + tail.start()].rstrip()
    return t, meta


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            body, meta = fetch(name)
        except Exception as e:  # noqa: BLE001
            print(f"[건너뜀] {name}: {e}")
            continue
        ef = meta["efYd"]
        head = (f"# {name}\n\n- 출처: 국가법령정보센터 https://www.law.go.kr/법령/{name.replace(' ', '%20')} "
                f"(법령일련번호 {meta['lsiSeq']})\n- 시행일: {ef[:4]}-{ef[4:6]}-{ef[6:]}\n- 수집일: {date.today()}\n"
                "- 범위: 본문 조문(부칙·별표 제외). 법적 판단은 원문과 담당 기관 확인이 필요하다.\n\n")
        p = OUT / f"{name}.md"
        p.write_text(head + body + "\n", encoding="utf-8")
        print(f"[저장] {p.relative_to(ROOT)}  조문 {body.count('## 제')}개, {len(body):,}자")
        time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1:] or LAWS)
