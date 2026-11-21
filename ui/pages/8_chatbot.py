"""AI 챗봇 — ChatGPT식 입력(2026-11-14 사용자 요청).

- 입력창 왼쪽 첨부 버튼(st.chat_input accept_file): 사진(검사 도구용)·텍스트 파일(내용을 질문에 덧붙임)
- 입력창이 비어 있을 때 Tab = 예시 질문 넣기(누를 때마다 다음 예시)
- 답변 중 '생각 중' 상태(모델 호출·도구 실행 단계) + 최종 답변 스트리밍
- 마지막 답변 '다시 답변' · 마지막 질문 '고쳐묻기'
- 대화 목록·예시 질문 버튼(팀원 앱 반영). 숫자는 Python 도구 결과만, 쓰기 작업은 확인 버튼 후 실행.
"""
import json
import uuid

import streamlit as st

from agent import ollama_client as oc
from agent.tools import execute_confirmed

ss = st.session_state
EXAMPLES = ["📈 다음 주 범퍼 1,000개 생산하면 폐플라스틱이 얼마나 나와?", "🚚 언제 수거 요청해야 해?",
            "🏭 창고가 언제 가득 차?", "♻️ 우리 공장 폐플라스틱 어디로 보내?"]
TAB_EXAMPLES = ["범퍼 1000개 생산 시 예상 폐기물은?", "Product_c0의 2022-03-12 기준 T+1 예정 수주량은?"] + \
               [e.split(" ", 1)[1] for e in EXAMPLES] + [
    # 2026-11-19 사용자 요청: 법령 5종 · 폐기물 분류번호 표 · 위탁처리 단가(견적 안내) · 재활용가능자원 가격조사
    "폐기물관리법에서 사업장폐기물배출자의 의무는?",
    "폐기물관리법 시행령에서 사업장의 범위는?",
    "폐기물 인계서는 언제까지 전자정보처리프로그램에 입력해야 해?",
    "자원재활용법에서 재활용지정사업자는 무엇이야?",
    "순환경제사회 전환 촉진법에서 순환자원 인정은 어떻게 받아?",
    "처리업체 목록에 나오는 51-03-01 코드는 어떤 폐기물이야?",
    "사업장폐기물 위탁처리 단가는 kg당 얼마야?",
    "압축 PP 2026년 10월 전국 가격은?",
    "폐골판지 2026년 1월부터 10월까지 가격 추이 알려줘",
    "재활용가능자원 가격조사 가격이 시중 거래가격과 다른 이유는?",
    "2026년 10월 국내 폐플라스틱 시장 동향 알려줘"]
TEXT_TYPES = ("txt", "csv", "md")

# 입력창이 비어 있으면 Tab으로 예시 질문을 넣는 작은 스크립트(누를 때마다 다음 예시, Shift+Tab은 원래 동작)
_TAB_JS = """
export default function (component) {
  const doc = component.parentElement.ownerDocument
  const win = doc.defaultView
  win.__rsTabExamples = (component.data && component.data.examples) || []
  if (win.__rsTabInstalled) return
  win.__rsTabInstalled = true
  doc.addEventListener("keydown", (e) => {
    const ta = e.target
    if (e.key !== "Tab" || e.shiftKey || !ta || !ta.matches || !ta.matches('[data-testid="stChatInputTextArea"]')) return
    const ex = win.__rsTabExamples
    if (!ex.length) return
    const i = ex.indexOf(ta.value.trim())
    if (ta.value.trim() && i < 0) return  // 직접 쓴 글이 있으면 그대로 둠
    e.preventDefault()
    const next = ex[(i + 1) % ex.length]
    Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, "value").set.call(ta, next)
    ta.dispatchEvent(new win.Event("input", { bubbles: true }))
  }, true)
}
"""
_TAB = st.components.v2.component("resource_chat_tab_examples", html="<span></span>", js=_TAB_JS)

st.title("AI 챗봇")
ok = oc.available()
st.caption(f"{'🟢' if ok else '🔴'} Ollama `{oc.MODEL}` @ {oc.BASE_URL} — {'연결됨' if ok else '연결 안 됨 (각 화면의 폼은 정상 동작)'}. "
           "숫자는 Python 도구 결과만 사용하며, 계획 생성·판정·초안·확정은 아래 확인 버튼을 눌러야 실행됩니다.")

if "convs" not in ss:
    first = str(uuid.uuid4())[:8]
    ss.convs = {first: {"title": "새 대화", "chat": ss.pop("chat", []), "pending": ss.pop("pending", [])}}
    ss.conv_id = first


def new_conv() -> None:
    cid = str(uuid.uuid4())[:8]
    ss.convs[cid] = {"title": "새 대화", "chat": [], "pending": []}
    ss.conv_id = cid
    ss.pop("editing", None)


left, main = st.columns([1, 3], gap="large")
with left:
    st.button("＋ 새 대화", on_click=new_conv, width="stretch")
    st.caption("대화 목록")
    for cid, c in reversed(list(ss.convs.items())):
        st.button(("💬 " if cid == ss.conv_id else "") + c["title"][:18], key=f"conv_{cid}", width="stretch",
                  type="primary" if cid == ss.conv_id else "secondary", on_click=lambda cid=cid: ss.update(conv_id=cid))
conv = ss.convs[ss.conv_id]


def shown_msgs() -> list[tuple[int, dict]]:
    return [(i, m) for i, m in enumerate(conv["chat"]) if m["role"] in ("user", "assistant") and m.get("content")]


def last_user_index() -> int | None:
    return next((i for i in range(len(conv["chat"]) - 1, -1, -1) if conv["chat"][i]["role"] == "user"), None)


TOOL_KO = {"search_docs": "문서 검색", "find_waste_processors": "처리업체 검색", "predict_preproduction_waste": "폐기물 예측",
           "get_storage_projection": "보관량 예측", "recommend_collection": "수거 추천", "get_provenance": "출처 태그",
           "list_production_plans": "생산계획 목록", "forecast_order_quantity": "수주량 예측", "get_supply_backtest": "백테스트"}


def trace_card(tr: dict, idx: int) -> None:
    """실행 흔적(시연용, 2026-11-18): 배지 줄 + 펼치면 도구별 인자·결과 요약·시간."""
    badges = []
    if tr.get("guard"):
        badges.append(":red-badge[:material/block: 거절 · 코드 가드레일]")
    if tr.get("limit_reached"):
        badges.append(f":orange-badge[:material/warning: 상한 도달 ({tr.get('max_steps', 5)}회)]")
    if tr.get("unverified"):
        badges.append(f":orange-badge[:material/error: 검증 안 된 숫자 {', '.join(tr['unverified'][:4])}]")
    elif tr.get("model_calls"):
        badges.append(":green-badge[:material/check: 숫자 검증 통과]" + (" (한 번 다시 답함)" if tr.get("retried_for_numbers") else ""))
    tools = tr.get("tools") or []
    head = f"실행 흔적 · 모델 호출 {tr.get('model_calls', 0)}회 · 도구 {len(tools)}회" + (f" · {tr['seconds']}초" if tr.get("seconds") else "")
    if badges:
        st.markdown(" ".join(badges))
    if tr.get("cached"):
        st.markdown(":blue-badge[:material/bolt: 저장된 답 · 같은 질문을 먼저 한 사람의 답을 바로 보여 줌]")
    if tr.get("notice"):
        st.markdown(":blue-badge[:material/info: 고정 안내 · 공개 자료 없음]")
    if not tools and not tr.get("guard"):
        st.caption(head)
        return
    with st.expander(head, icon=":material/route:"):
        if tr.get("guard"):
            st.caption(f"모델을 부르기 전에 코드 규칙({tr['guard']})으로 거절했습니다.")
        for n, e in enumerate(tools, 1):
            with st.container(border=True):
                st.markdown(f"**{n}. {TOOL_KO.get(e['tool'], e['tool'])}** `{e['tool']}` · {e.get('status')} · {e.get('ms', 0)}ms")
                if e.get("args"):
                    st.caption("인자: " + ", ".join(f"{k}={v}" for k, v in e["args"].items()))
                if e.get("summary_ko"):
                    st.caption("결과: " + str(e["summary_ko"])[:400])


def run_answer(attachments: dict) -> None:
    """현재 대화(마지막이 사용자 질문)에 대한 답을 '생각 중' 표시와 스트리밍으로 그린 뒤 저장한다."""
    result = {}
    with st.chat_message("assistant"):
        status = st.status("생각 중…", expanded=False)

        def tokens():
            for ev, val in oc.chat_stream(conv["chat"], attachments=attachments):
                if ev == "thinking":
                    status.update(label=f"생각 중… {val}")
                    status.write(val)
                elif ev == "token":
                    yield val
                else:
                    result.update(val)

        st.write_stream(tokens())
        label = {"OK": "답변 완료", "REFUSED": "거절 (범위·권한 밖 요청)", "OLLAMA_ERROR": "응답 실패"}.get(result.get("status"), "연결 안 됨")
        status.update(label=label, state="complete" if result.get("status") in ("OK", "REFUSED") else "error")
    if result.get("status") == "REFUSED":
        conv["chat"] = result["messages"]
        return
    if result.get("status") != "OK":
        conv["chat"].append({"role": "assistant", "content": result.get("message", "응답 없음")})
        return
    conv["chat"] = result["messages"]
    conv["ctx"] = result.get("context") or {}
    conv["pending"] += result["pending"]
    if result["evidence"]:
        lines = [f"- `{e['tool']}` [{e['status']}] {e['summary_ko']}" for e in result["evidence"]]
        if result["pending"]:
            lines.append("- ⚠️ 위 요청은 **아직 실행되지 않았습니다**. 아래 '확인 후 실행' 버튼을 눌러야 반영됩니다.")
        conv["chat"].append({"role": "assistant", "content": "**근거(Python 도구 계산 결과)**\n\n" + "\n".join(lines)})


def submit(text: str, files: list | None = None) -> None:
    att, extra, names = {}, [], []
    for f in files or []:
        names.append(f.name)
        if f.name.lower().rsplit(".", 1)[-1] in TEXT_TYPES:
            extra.append(f"\n\n[첨부 파일 {f.name}]\n" + f.getvalue().decode("utf-8", errors="replace")[:4000])
        else:
            att[f.name] = f.getvalue()
    ss["att"] = att
    shown = text.strip() + (f"\n\n📎 {', '.join(names)}" if names else "")
    if conv["title"] == "새 대화":
        conv["title"] = (text.strip() or names[0] if names else "새 대화")[:30]
    conv["chat"].append({"role": "user", "content": shown + "".join(extra)})
    ss["to_answer"] = True


with main:
    msgs = shown_msgs()
    if not msgs and not ss.get("to_answer"):
        st.markdown("### ♻️ 무엇을 도와드릴까요?")
        st.caption("생산계획 → 폐플라스틱 예측 → 보관 → 수거 → 처리업체까지 물어보세요. 입력창에서 **Tab**을 누르면 예시 질문이 들어갑니다.")
        ex = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if ex[i % 2].button(q, key=f"ex{i}", width="stretch"):
                submit(q.split(" ", 1)[1])
                st.rerun()
    lu = last_user_index()
    for i, m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"].split("\n\n[첨부 파일 ")[0])
            if m.get("trace"):
                trace_card(m["trace"], i)
    cx = conv.get("ctx") or {}
    if msgs and (cx.get("truncated") or cx.get("over_budget")):
        st.warning("대화가 AI가 한 번에 읽을 수 있는 길이를 넘어 앞부분이 잘렸을 수 있습니다. 정확한 답을 위해 **＋ 새 대화**로 물어보세요.",
                   icon=":material/warning:")
    elif msgs and (cx.get("dropped_turns") or cx.get("shrunk_tool_results")):
        st.caption(f"대화가 길어 AI에는 최근 내용 위주로 보냈습니다(앞선 질문 {cx.get('dropped_turns', 0)}개 제외, "
                   f"도구 결과 {cx.get('shrunk_tool_results', 0)}개는 요약만). 화면의 기록은 그대로입니다.")
    # 마지막 질문·답변 도구: 다시 답변 / 고쳐묻기
    if msgs and not ss.get("to_answer") and lu is not None and ok:
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("↻ 다시 답변", key="regen", help="마지막 질문에 답을 새로 받습니다"):
            conv["chat"] = conv["chat"][:lu + 1]
            conv["pending"] = []
            conv.pop("ctx", None)
            ss["to_answer"] = True
            st.rerun()
        if b2.button("✎ 고쳐묻기", key="edit", help="마지막 질문을 고쳐 다시 묻습니다"):
            ss["editing"] = conv["chat"][lu]["content"].split("\n\n📎 ")[0]
        if ss.get("editing") is not None:
            with st.form("edit_form"):
                new_q = st.text_area("질문 고치기", ss["editing"])
                c1, c2 = st.columns(2)
                send, cancel = c1.form_submit_button("보내기", type="primary"), c2.form_submit_button("취소")
            if send and new_q.strip():
                conv["chat"] = conv["chat"][:lu]
                conv["pending"] = []
                ss.pop("editing", None)
                submit(new_q)
                st.rerun()
            if cancel:
                ss.pop("editing", None)
                st.rerun()
    if ss.pop("to_answer", False):
        run_answer(ss.get("att", {}))
        st.rerun()
    for i, p in enumerate(list(conv["pending"])):
        with st.container(border=True):
            st.write(f"**확인 대기:** `{p['tool']}`")
            st.code(json.dumps(p["args"], ensure_ascii=False, indent=1))
            if p.get("warning"):
                st.warning(p["warning"])
            c1, c2 = st.columns(2)
            if c1.button("확인 후 실행", key=f"ok{ss.conv_id}{i}", type="primary"):
                r = execute_confirmed(p["tool"], p["args"], ss.get("att", {}))
                conv["chat"].append({"role": "assistant", "content": f"`{p['tool']}` 실행 결과: "
                                     + json.dumps(r, ensure_ascii=False)[:1500]})
                conv["pending"].pop(i)
                st.rerun()
            if c2.button("취소", key=f"no{ss.conv_id}{i}"):
                conv["pending"].pop(i)
                st.rerun()

_TAB(key="tab_examples", data={"examples": TAB_EXAMPLES})
q = st.chat_input("예: 범퍼 1000개 생산 시 예상 폐기물은? / Product_c0의 2022-03-12 기준 T+1 예정 수주량은?  (Tab: 예시 넣기)",
                  accept_file="multiple", file_type=["jpg", "jpeg", "png", *TEXT_TYPES], key="chat_box")
if q:
    text = q.text if hasattr(q, "text") else str(q)
    files = list(getattr(q, "files", []) or [])
    if text.strip() or files:
        submit(text, files)
        st.rerun()
