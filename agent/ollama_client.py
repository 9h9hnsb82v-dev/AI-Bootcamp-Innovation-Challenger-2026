"""Ollama /api/chat 도구 호출 루프. Ollama가 꺼져 있으면 OLLAMA_UNAVAILABLE을 반환하며,
나머지 서비스(Streamlit 폼)는 영향 없이 동작한다.

컨텍스트 관리(2026-11-18 사용자 승인):
- num_ctx 를 요청마다 지정(RESOURCE_OLLAMA_NUM_CTX, 기본 8192. 0 이면 보내지 않고 Ollama 기본값 4096 가정).
  공용 Ollama 설정은 건드리지 않는다 — 요청 옵션이라 이 앱의 요청에만 적용된다.
- 모델에 보내는 기록만 줄인다(화면의 대화 기록은 그대로): 이전 질문의 도구 결과 원문·도구 호출 메시지는 빼고
  (이전 답의 '근거' 메시지에 summary_ko 가 남아 있다), 예산을 넘으면 가장 오래된 질문부터 뺀다.
- 넘침 감지: Ollama 가 실제로 읽은 토큰 수(prompt_eval_count)가 한도에 닿으면 truncated=True.
- Ollama 오류·시간 초과는 예외 대신 OLLAMA_ERROR 로 돌려준다.

ReAct 루프 보강(2026-11-18 사용자 승인 ②~④, PEV 대신 ReAct 유지):
- 가드레일: 모델을 부르기 전에 agent.guard.precheck 로 범위·권한 밖 요청을 거절(REFUSED).
- 숫자 검증: 최종 답의 숫자가 이번 도구 결과·문서·질문에 있는지 코드로 대조. 없으면 한 번만 다시 답하게 하고,
  그래도 남으면 '검증되지 않은 숫자'로 표시한다(추가 LLM 검사 담당 없음 → 평소엔 호출 0회 추가).
- 반복 상한: max_steps 동안 도구만 부르면 도구 없이 한 번 더 불러 답을 마무리하고 limit_reached 를 표시.
- 실행 흔적: 모델 호출 수, 도구별 인자·결과 요약·걸린 시간을 trace 로 돌려준다(화면 카드용).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

import requests

from agent.guard import precheck
from agent.tools import TOOLS, dispatch, to_json

# Windows 에서 'localhost' 는 IPv6(::1)를 먼저 시도하다 IPv4 로 넘어가 요청마다 약 2.2초 늦었다(실측, 127.0.0.1 은 0.14초).
# 탭 속도 개선 C(2026-11-18): 기본 127.0.0.1, 환경변수에 localhost 가 있어도 127.0.0.1 로 바꿔 쓴다.
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").replace("//localhost", "//127.0.0.1")
MODEL = os.environ.get("OLLAMA_MODEL", "benedict/linkbricks-llama3.1-korean:8b")
NUM_CTX = int(os.environ.get("RESOURCE_OLLAMA_NUM_CTX", "8192"))
OLLAMA_DEFAULT_CTX = 4096          # Ollama 기본 컨텍스트(실측: /api/ps context_length)
REPLY_RESERVE = 1024               # 답변 생성 몫
CHARS_PER_TOKEN = 1.5              # 보수적 추정. 실측(2026-11-18, 이 모델): 한국어 질문 1.65, 답변 1.93, 도구 JSON 1.87~2.32 자/토큰
BASE_CHARS_PER_TOKEN = 3.0         # 시스템+도구 정의 실측 4,596자 = 1,399토큰(3.3자/토큰)
SYSTEM = """당신은 Re:Source Predict AI 도우미입니다. 규칙:
- 숫자는 반드시 도구 결과나 search_docs 문서에 있는 값만 인용하고 직접 계산·추정하지 마세요. 도구 결과에 summary_ko가 있으면 그 문장을 그대로 인용하세요. provenance 태그를 함께 밝히세요.
- 법·제도·인계서 절차·폐기물 분류번호·재활용 자원 가격·데이터 출처·모델 성능·앱 사용법 질문은 search_docs로 찾아 답하고, 문서명과 조문(섹션)을 밝히세요. 문서에 없으면 모른다고 하세요.
- 재활용자원 가격 숫자(원/kg)는 get_recycling_price로 조회하세요. 이 가격은 가공업체 간 판매가(폐지·철스크랩은 매입가)이며 공장이 받는 값이나 처리비가 아닙니다.
- 처리업체 찾기는 find_waste_processors를 쓰세요. 허가 정보는 현재 수탁 가능 여부가 아닙니다. 사업장폐기물 위탁처리 단가는 공개 자료가 없어 업체 견적이 필요하다고 안내하세요.
- KAMP ID40 수주량(개수)은 폐기물 kg가 아니며, 비식별 Product_Number를 CN7/RG3/DOOR/BUMPER/FENDER와 임의로 연결하지 마세요.
- 'T일 예정 수주량'은 예정(계획) 수주량이지 확정 주문 실적이 아닙니다.
- AI 결함 의심은 폐기 확정이 아닙니다. 작업자 확정만 kg에 반영됩니다.
- 생산계획 생성·판정·초안·확정은 사용자 확인이 필요합니다(도구가 PENDING_USER_CONFIRMATION을 반환).
- 폐기물 인계 데이터 분석·통계는 제공하지 않습니다(인계서 제도·절차 설명은 search_docs로 안내).
한국어로 간결하게 답하세요."""


def ctx_limit() -> int:
    return NUM_CTX if NUM_CTX > 0 else OLLAMA_DEFAULT_CTX


def _options() -> dict:
    o = {"temperature": 0, "num_predict": 768}  # 답 길이 상한: 같은 문장을 끝없이 반복해 시간 초과되는 경우 방지(실측 2026-11-19)
    if NUM_CTX > 0:
        o["num_ctx"] = NUM_CTX
    return o


def est_tokens(obj, cpt: float = CHARS_PER_TOKEN) -> int:
    if not obj:
        return 0
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return int(len(s) / cpt) + 4


BASE_TOKENS = est_tokens(SYSTEM, BASE_CHARS_PER_TOKEN) + est_tokens(TOOLS, BASE_CHARS_PER_TOKEN)


def _msg_tokens(m: dict) -> int:
    return est_tokens(m.get("content") or "") + est_tokens(m.get("tool_calls") or "")


def _shrink_tool(m: dict) -> dict:
    """도구 결과 원문 → 상태·요약만(지금 질문의 도구 결과가 예산을 넘을 때)."""
    try:
        d = json.loads(m.get("content") or "{}")
        keep = {k: d[k] for k in ("status", "summary_ko", "message", "code", "provenance") if k in d}
        content = json.dumps(keep or {"note": "결과가 길어 요약만 보냄"}, ensure_ascii=False)[:600]
    except ValueError:
        content = (m.get("content") or "")[:600]
    return {**m, "content": content}


def compact(messages: list[dict], limit: int | None = None) -> tuple[list[dict], dict]:
    """모델에 보낼 기록(시스템 제외)을 만든다. 반환: (보낼 메시지, 정보{dropped_turns, est_tokens, budget}).
    마지막 사용자 질문 이후(지금 질문과 이번 도구 루프)는 그대로 두고, 그 앞은 도구 메시지를 빼고 예산 안에서 최근 것부터 남긴다."""
    msgs = [m for m in messages if m.get("role") != "system"]
    last_user = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user"), len(msgs))
    current = msgs[last_user:]
    old = [m for m in msgs[:last_user]
           if m.get("role") in ("user", "assistant") and not m.get("tool_calls") and (m.get("content") or "").strip()]
    turns: list[list[dict]] = []  # 질문 하나 + 그 답들
    for m in old:
        if m["role"] == "user" or not turns:
            turns.append([])
        turns[-1].append(m)
    budget = (limit or ctx_limit()) - REPLY_RESERVE - BASE_TOKENS
    used = sum(_msg_tokens(m) for m in current)
    shrunk = 0
    for i, m in enumerate(current):  # 지금 질문만으로 넘치면 오래된 도구 결과부터 요약으로 줄임
        if used <= budget:
            break
        if m.get("role") == "tool":
            small = _shrink_tool(m)
            used += _msg_tokens(small) - _msg_tokens(m)
            current[i] = small
            shrunk += 1
    kept: list[list[dict]] = []
    for t in reversed(turns):
        cost = sum(_msg_tokens(m) for m in t)
        if used + cost > budget:
            break
        kept.insert(0, t)
        used += cost
    out = [m for t in kept for m in t] + current
    return out, {"dropped_turns": len(turns) - len(kept), "shrunk_tool_results": shrunk,
                 "est_tokens": used + BASE_TOKENS, "budget": budget, "over_budget": used > budget}


_AVAIL: dict = {}  # {BASE_URL: (확인 시각, 결과)} — 챗봇 화면은 다시 그릴 때마다 연결을 확인하므로 10초 재사용
AVAIL_TTL = 10.0


def available(timeout: float = 1.5) -> bool:
    hit = _AVAIL.get(BASE_URL)
    if hit and time.monotonic() - hit[0] < AVAIL_TTL:
        return hit[1]
    try:
        ok = requests.get(f"{BASE_URL}/api/tags", timeout=timeout).ok
    except requests.RequestException:
        ok = False
    _AVAIL[BASE_URL] = (time.monotonic(), ok)
    return ok


def _unavailable(messages: list[dict]) -> dict:
    return {"status": "OLLAMA_UNAVAILABLE", "messages": messages, "pending": [], "evidence": [],
            "message": f"Ollama({BASE_URL}) 연결 불가 — 각 화면의 폼에서 동일 기능을 사용할 수 있습니다."}


def _error(messages: list[dict], e: Exception) -> dict:
    kind = "시간 초과" if isinstance(e, requests.Timeout) else type(e).__name__
    return {"status": "OLLAMA_ERROR", "messages": messages, "pending": [], "evidence": [],
            "message": f"AI 응답을 받지 못했습니다({kind}). 잠시 뒤 '다시 답변'을 누르거나, 대화가 길면 새 대화로 물어보세요."}


def _context_info(info: dict, prompt_tokens: int | None) -> dict:
    lim = ctx_limit()
    return {**info, "num_ctx": lim, "prompt_tokens": prompt_tokens,
            "truncated": bool(prompt_tokens and prompt_tokens >= lim - REPLY_RESERVE // 4)}


# ── 숫자 검증(코드) ─────────────────────────────────────────────────────
_NUM = re.compile(r"(?<![\d.,A-Za-z])\d(?:[\d,]*\d)?(?:\.\d+)?")  # 뒤에 단위(kg·원·톤)가 붙어도 잡는다
_SKIP_AFTER = re.compile(r"^\s*(년|월|일|조|항|호|위|번째|단계|차|종|개월|시|분|초|층)")


def _num_tokens(text: str) -> list[tuple[str, float, int]]:
    """(원문, 값, 소수 자릿수). 날짜(2026-11-18)·분류번호·조문 번호·순위·한 자리 정수는 뺀다."""
    t = re.sub(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{2}-\d{2}-\d{2}|\d{2}-\d{2}", " ", text or "")
    out = []
    for m in _NUM.finditer(t):
        raw = m.group(0).rstrip(",")
        if _SKIP_AFTER.match(t[m.end():m.end() + 4]):
            continue
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        if dec == 0 and v < 10:
            continue
        out.append((raw, v, dec))
    return out


def unverified_numbers(answer: str, sources: str) -> list[str]:
    """답변 숫자 중 근거 글(도구 결과·문서·질문)에 없는 것. 반올림·%(0.025 ↔ 2.5%)·천 단위 쉼표 차이는 허용."""
    src = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", sources or "") if x.replace(",", "")]
    bad = []
    for raw, v, dec in _num_tokens(answer):
        ok = any(round(s, dec) == round(v, dec) or round(s * 100, dec) == round(v, dec)
                 or (dec == 0 and abs(s - v) < 0.5) for s in src)
        if not ok:
            bad.append(raw)
    return list(dict.fromkeys(bad))


# ── ReAct 루프 ───────────────────────────────────────────────────────────
_CITY = {"서울": "수도권", "경기": "수도권", "인천": "수도권", "대구": "경북", "부산": "경남", "울산": "경남", "광주": "전남",
         "대전": "충남", "세종": "충남"}
_REG = ["전국", "수도권", "강원", "충북", "충남", "전북", "전남", "경북", "경남"]
PRICE_NOTE = ("※ 재활용가능자원 가격조사 값은 원/kg(VAT 포함·물류비 제외)이며 가공업체가 다음 단계 업체에 판매하는 가격입니다"
              "(폐지·철스크랩은 중간가공업체 매입가). 공장이 불량품을 넘길 때 받는 값이나 처리비가 아니며 실제 거래가는 업체 견적이 필요합니다.")


# 사업장폐기물 위탁처리 단가: 공개 자료가 없어 모델에 맡기지 않고 코드가 고정 안내(2026-11-19 사용자 요청, 실측: 작은 모델이 무관한 법령 조문을 붙임)
_DISPOSAL_COST = re.compile(r"(위탁\s*처리|위탁|처리)\s*(단가|비용|요금|수수료)|처리비")
DISPOSAL_COST_NOTE = ("사업장폐기물 위탁처리 단가(처리비)는 공개된 공식 자료가 없어 챗봇이 숫자로 답할 수 없습니다. "
                      "단가는 폐기물 종류·양·운반 조건과 업체에 따라 달라지므로, 위탁할 처리업체에 직접 견적을 받아야 합니다.\n\n"
                      "챗봇에서 도와드릴 수 있는 것:\n"
                      "- 처리업체 찾기: 예) '대구 51-03-01 재활용업체 찾아줘' (허가 정보이며 현재 수탁 가능 여부는 업체에 확인 필요)\n"
                      "- 재활용가능자원 가격조사 조회: 예) '압축 PP 2026년 10월 전국 가격은?' "
                      "(가공업체가 다음 단계 업체에 판매하는 가격이며, 공장이 내는 처리비나 받는 값이 아님)")


def disposal_cost_notice(question: str) -> str | None:
    q = question or ""
    return DISPOSAL_COST_NOTE if _DISPOSAL_COST.search(q) else None


def price_args(args: dict, question: str) -> dict:
    """가격 조회 인자를 사용자 질문의 날짜·지역으로 바로잡는다(작은 모델이 '2026년 10월'을 '2025'로 넘기는 일이 있어, 실측)."""
    a = dict(args)
    months, year = set(), None  # '2026년 1월부터 10월까지'처럼 연도 없는 달은 앞에 나온 연도를 쓴다
    for y, m, m_only in re.findall(r"(20\d\d)\s*[년.\-/]\s*(\d{1,2})\s*월?|(?<![\d.])(\d{1,2})\s*월", question or ""):
        if y:
            year = y
            months.add(f"{y}-{int(m):02d}")
        elif year and 1 <= int(m_only) <= 12:
            months.add(f"{year}-{int(m_only):02d}")
    months = sorted(months)
    if len(months) >= 2:
        a.update(start=months[0], end=months[-1])
        a.pop("period", None)
    elif len(months) == 1:
        if re.search(r"부터|이후|동안|추이|변화", question):
            a.update(start=months[0])
            a.pop("period", None)
        else:
            a["period"] = months[0]
            a.pop("start", None)
            a.pop("end", None)
    else:
        yrs = re.findall(r"(20\d\d)\s*년", question or "")
        if len(yrs) == 1 and re.search(r"평균|연간|한 해", question):
            a["period"] = yrs[0]
    for k, v in _CITY.items():
        if k in (question or ""):
            a["region"] = v
    for r_ in _REG:
        if r_ in (question or ""):
            a["region"] = r_
    if a.get("start") or a.get("end"):
        a.pop("period", None)
    if not any(k in (question or "") for k in list(_CITY) + _REG + ["지역", "권역"]):
        a["region"] = "전국"  # 지역을 말하지 않으면 전국만(9개 권역을 다 나열해 답이 길고 느려짐, 실측)
    return a


def text_tool_call(content: str, question: str) -> list[dict]:
    """답 글자가 '{"name": "도구", "parameters": {...}}' 꼴이면 도구 호출로 바꾼다. 인자가 깨졌으면 질문에서 다시 채운다.
    가격 조회는 질문 속 품목 이름으로 item 을 찾는다. 알 수 없으면 빈 목록(원래 답 유지)."""
    m = re.search(r'"name"\s*:\s*"([a-z_]+)"', content or "")
    names = {t["function"]["name"] for t in TOOLS}
    if not m or m.group(1) not in names or len(content.strip()) and not content.strip().startswith("{"):
        return []
    name = m.group(1)
    args: dict = {}
    pm = re.search(r'"(?:parameters|arguments)"\s*:\s*(\{.*?\})', content, flags=re.S)
    if pm:
        try:
            args = {k: v for k, v in json.loads(pm.group(1)).items() if isinstance(v, (str, int, float)) and len(str(v)) < 40}
        except ValueError:
            args = {}
    if name == "get_recycling_price":
        from services.recycling_price import items
        q = re.sub(r"\s", "", question or "").upper()
        hit = sorted((it for it in items() if re.sub(r"[\s()]", "", it).upper() in q), key=len, reverse=True)
        if hit:
            args["item"] = hit[0]
        if not args.get("item"):
            return []
    return [{"function": {"name": name, "arguments": args}}]


def _run_tool(c: dict, attachments, pending: list, evidence: list, question: str = "") -> dict:
    fn = c["function"]
    args = dict(fn.get("arguments") or {})
    if fn["name"] == "get_recycling_price" and question:
        args = price_args(args, question)
    if fn["name"] == "search_docs" and question and question not in str(args.get("query", "")):
        # 작은 모델은 검색어를 짧게 바꿔 핵심어(예: '입력 기한')를 빠뜨린다(실측) → 사용자 질문을 함께 검색
        args["query"] = f"{args.get('query', '')} {question[:200]}".strip()
    t0 = time.perf_counter()
    out = dispatch(fn["name"], args, attachments)
    ms = round((time.perf_counter() - t0) * 1000)
    if out.get("status") == "PENDING_USER_CONFIRMATION":
        pending.append(out)
    evidence.append({"tool": fn["name"], "status": out.get("status"), "args": args, "ms": ms,
                     "summary_ko": out.get("summary_ko") or out.get("message") or out.get("code", "")})
    return {"role": "tool", "content": to_json(out), "tool_name": fn["name"]}


def _post(model, msgs: list[dict], tools: bool, stream: bool):
    body = {"model": model or MODEL, "stream": stream, "options": _options(),
            "messages": [{"role": "system", "content": SYSTEM}] + msgs}
    if tools:
        body["tools"] = TOOLS
    r = requests.post(f"{BASE_URL}/api/chat", json=body, timeout=300, stream=stream)
    r.raise_for_status()
    return r


def _clean(msgs: list[dict]) -> list[dict]:
    """Ollama 에 보낼 필드만(화면용 trace 등 제거)."""
    keep = ("role", "content", "tool_calls", "tool_name")
    return [{k: m[k] for k in keep if k in m} for m in msgs]


# ── 동시 접속 대비(2026-11-20, 대회 QR 체험) ─────────────────────────────────────
# ① 답 저장: 대화의 첫 질문이고, 쓴 도구가 모두 '사람마다 같은 자료'(문서·가격·업체·수주 이력)일 때만 저장해
#    같은 질문에 모델을 다시 부르지 않는다. 생산계획·보관량처럼 사람(체험 세션)마다 다른 데이터를 쓴 답은 저장하지 않는다.
# ② 대기 안내: 이 프로세스에서 동시에 처리 중인 질문 수가 Ollama 동시 처리 칸 수(RESOURCE_OLLAMA_SLOTS) 이상이면 안내.
STATIC_TOOLS = {"search_docs", "get_recycling_price", "find_waste_processors", "get_provenance",
                "forecast_order_quantity", "get_supply_backtest"}
ANSWER_TTL = 6 * 3600
ANSWER_MAX = 300
_ANSWERS: dict[str, tuple[float, str, list]] = {}
_INFLIGHT = {"n": 0}
_LOCK = threading.Lock()


def _slots() -> int:
    try:
        return max(1, int(os.environ.get("RESOURCE_OLLAMA_SLOTS", "1")))
    except ValueError:
        return 1


def _answer_key(full: list[dict], attachments, model: str | None) -> str | None:
    users = [m for m in full if m.get("role") == "user"]
    if len(users) != 1 or attachments or any(m.get("role") == "tool" for m in full):
        return None  # 대화 중간·첨부 파일이 있으면 앞 맥락에 따라 답이 달라짐
    q = " ".join(str(users[0].get("content", "")).split())
    return f"{model or MODEL}\x00{q}" if q else None


def _cached(key: str | None):
    if not key:
        return None
    with _LOCK:
        hit = _ANSWERS.get(key)
        if hit and time.time() - hit[0] < ANSWER_TTL:
            return hit
        _ANSWERS.pop(key, None)
    return None


def _store(key: str | None, content: str, evidence: list) -> None:
    if not key or not content.strip() or any(e.get("tool") not in STATIC_TOOLS for e in evidence):
        return
    with _LOCK:
        if len(_ANSWERS) >= ANSWER_MAX:
            _ANSWERS.pop(min(_ANSWERS, key=lambda k: _ANSWERS[k][0]))
        _ANSWERS[key] = (time.time(), content, evidence)


def chat_stream(messages: list[dict], attachments: dict | None = None, max_steps: int = 5, model: str | None = None):
    """스트리밍 대화. 이벤트: ("thinking", 문구) / ("token", 조각) / ("done", 결과).
    결과: {"status": OK|REFUSED|OLLAMA_ERROR|OLLAMA_UNAVAILABLE, "messages", "pending", "evidence", "context", "trace"}.
    최종 답 메시지에 "trace"(실행 흔적)가 붙는다."""
    full = [m for m in messages if m.get("role") != "system"]
    key = _answer_key(full, attachments, model)
    hit = _cached(key)
    if hit:  # 같은 질문의 저장된 답(사람마다 같은 자료만 쓴 답)
        _, content, ev = hit
        trace = {"model_calls": 0, "tools": ev, "cached": True, "limit_reached": False, "unverified": [], "seconds": 0.0}
        yield ("token", content)
        yield ("done", {"status": "OK", "pending": [], "evidence": ev, "context": {}, "trace": trace,
                        "messages": full + [{"role": "assistant", "content": content, "trace": trace}]})
        return
    with _LOCK:
        ahead = _INFLIGHT["n"]
        _INFLIGHT["n"] += 1
    try:
        if ahead >= _slots():
            yield ("thinking", f"다른 사람의 질문 {ahead}개를 처리하고 있어요. 차례가 되면 바로 답할게요(동시 {_slots()}개씩 처리)…")
        done = None
        for ev, val in _chat_stream_inner(full, attachments, max_steps, model):
            if ev == "done":
                done = val
            yield ev, val
        if done and done.get("status") == "OK" and not done.get("pending") and not done["trace"].get("unverified") \
                and not done["trace"].get("guard") and not done["trace"].get("notice"):
            _store(key, done["messages"][-1].get("content", ""), done.get("evidence", []))
    finally:
        with _LOCK:
            _INFLIGHT["n"] -= 1


def _chat_stream_inner(full: list[dict], attachments, max_steps: int, model: str | None):
    messages = full
    last_q = next((m.get("content", "") for m in reversed(full) if m.get("role") == "user"), "")
    g = precheck(last_q)
    if g:  # 코드 가드레일: 모델을 부르지 않는다
        trace = {"model_calls": 0, "tools": [], "guard": g["category"], "limit_reached": False, "unverified": []}
        yield ("token", g["message"])
        yield ("done", {"status": "REFUSED", "pending": [], "evidence": [], "context": {}, "trace": trace,
                        "messages": full + [{"role": "assistant", "content": g["message"], "trace": trace}]})
        return
    note = disposal_cost_notice(last_q)
    if note:  # 공개 자료 없는 질문: 모델을 부르지 않고 고정 안내
        trace = {"model_calls": 0, "tools": [], "notice": "DISPOSAL_COST", "limit_reached": False, "unverified": []}
        yield ("token", note)
        yield ("done", {"status": "OK", "pending": [], "evidence": [], "context": {}, "trace": trace,
                        "messages": full + [{"role": "assistant", "content": note, "trace": trace}]})
        return
    if not available():
        yield ("done", _unavailable(messages))
        return
    pending, evidence, info, prompt_tokens, calls_n, limit = [], [], {}, None, 0, False
    t_start = time.perf_counter()
    try:
        for step in range(max_steps + 1):
            final_only = step == max_steps  # 상한: 도구 없이 답만 받는다
            if final_only:
                limit = True
                yield ("thinking", f"도구 호출 상한({max_steps}회)에 도달 — 지금까지 결과로 답을 마무리하는 중…")
            else:
                yield ("thinking", "질문을 이해하고 필요한 계산·문서를 고르는 중…" if step == 0
                       else "도구 결과를 바탕으로 답변을 정리하는 중…")
            sent, info = compact(full)
            r = _post(model, _clean(sent), tools=not final_only, stream=True)
            calls_n += 1
            content, calls = "", []
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                m = chunk.get("message") or {}
                if m.get("tool_calls"):
                    calls += m["tool_calls"]
                piece = m.get("content") or ""
                if piece:
                    content += piece
                    if not calls:  # 도구 호출이 아닌 답변만 화면에 흘려보냄
                        yield ("token", piece)
                if chunk.get("done"):
                    prompt_tokens = max(prompt_tokens or 0, chunk.get("prompt_eval_count") or 0) or None
                    break
            if not calls and not final_only:
                calls = text_tool_call(content, last_q)  # 모델이 도구 호출을 글자로 써 버린 경우(실측) 코드가 대신 부른다
                if calls:
                    content = ""
                    yield ("thinking", "도구 호출 형식을 바로잡아 다시 실행하는 중…")
            msg = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls
            full.append(msg)
            if not calls:
                break
            for c in calls:
                yield ("thinking", f"도구 실행: {c['function']['name']}")
                full.append(_run_tool(c, attachments, pending, evidence, last_q))
        # 숫자 검증(코드): 이번 질문 이후의 도구 결과·질문 글이 근거
        turn_start = max(i for i, m in enumerate(full) if m.get("role") == "user")
        sources = "\n".join(m.get("content") or "" for m in full[turn_start:-1] if m.get("role") in ("tool", "user"))
        bad = unverified_numbers(full[-1].get("content", ""), sources)
        retried = False
        if bad:
            retried = True
            yield ("thinking", f"숫자 검증: 근거에 없는 숫자 {', '.join(bad[:5])} → 다시 답하는 중…")
            nudge = {"role": "user", "content": f"[검증] 방금 답의 숫자 {', '.join(bad[:8])}는 도구 결과·문서에 없습니다. "
                                                "도구 결과·문서에 있는 숫자만 써서 같은 질문에 다시 답하세요. 근거가 없으면 모른다고 하세요."}
            sent, info = compact(full + [nudge])
            data = _post(model, _clean(sent), tools=False, stream=False).json()
            calls_n += 1
            if data.get("error"):
                raise RuntimeError(data["error"])
            again = (data.get("message") or {}).get("content", "").strip()
            if again:
                full[-1] = {"role": "assistant", "content": again}
                bad = unverified_numbers(again, sources)
    except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
        yield ("done", _error(messages, e))
        return
    price_used = any(e["tool"] == "get_recycling_price" or (e["tool"] == "search_docs" and "재활용가능자원 가격" in str(e.get("summary_ko")))
                     for e in evidence)
    if price_used and PRICE_NOTE not in full[-1].get("content", ""):  # 가격 해석 주의를 모델에 맡기지 않고 코드로 붙인다
        full[-1] = {**full[-1], "content": full[-1].get("content", "").rstrip() + "\n\n" + PRICE_NOTE}
    trace = {"model_calls": calls_n, "tools": evidence, "limit_reached": limit, "max_steps": max_steps,
             "unverified": bad, "retried_for_numbers": retried, "seconds": round(time.perf_counter() - t_start, 1)}
    full[-1] = {**full[-1], "trace": trace}
    yield ("done", {"status": "OK", "messages": full, "pending": pending, "evidence": evidence,
                    "context": _context_info(info, prompt_tokens), "trace": trace})


def chat(messages: list[dict], attachments: dict | None = None, max_steps: int = 5, model: str | None = None) -> dict:
    """chat_stream 과 같은 루프(평가 스크립트용). 반환: chat_stream 의 결과."""
    done: dict = {}
    for ev, val in chat_stream(messages, attachments, max_steps, model):
        if ev == "done":
            done = val
    return done
