"""이미지 검사 작업대 (C안, 2026-11-14 사용자 선택).

왼쪽: 촬영 영역(근접 표면 판정 / 차량 위치 찾기) · 오른쪽: 현재 개체 · 마지막 판정 · 작업자 확정 · 최근 개체.
생산계획은 부품 버튼으로 고르면 오늘(없으면 가장 가까운 날짜) 계획이 자동 연결된다. 같은 부품 계획이 여럿이면 '계획 변경'.
계획 부품은 판정 기준이자 다른 부품 사진 차단의 기준이므로 AI 추정으로 계획을 고르지 않는다.
"""
import io
import re
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from services import db
from services.config import ROOT, ServiceError
from services.inspection_registry import (IMAGE_PRODUCTS, cancel_inspections, cancel_preview, decision_history,
                                          inspect_part_image, list_inspections, pending_units, photo_file,
                                          purge_inspections, record_operator_decision, restore_inspections,
                                          unit_buckets, unit_records, CHECK_STATUSES)
from services.production_plan import list_plans
from ui.common import run, tags

PART_KO = {"DOOR": "도어", "BUMPER": "범퍼", "FENDER": "휀더"}
VERDICT_KO = {"AI_SUSPECTED_DEFECT": "결함 의심", "AI_PASS": "양품 추정", "AI_DISAGREE": "작업자 확인 필요",
              "PART_MISMATCH_SUSPECTED": "부품 확인 필요", "MODEL_NOT_READY": "AI 판정 없음"}
DECISION_KO = {"CONFIRMED_SCRAP": "폐기 확정", "CONFIRMED_GOOD": "양품 확정", "REWORK_PENDING": "재작업 보류"}
ss = st.session_state

# ── 머리말 + 작업 대상(부품 → 계획 자동 연결) ─────────────────────────────
head_l, head_r = st.columns([1, 1], vertical_alignment="bottom")
head_l.title("이미지 검사 작업대")
plans = [p for p in list_plans() if p["product_id"] in IMAGE_PRODUCTS]
if not plans:
    st.info("도어/범퍼/휀더 확정 생산계획이 없습니다. 왼쪽 메뉴의 '생산계획'에서 먼저 등록하세요.")
    st.stop()
parts = [p for p in IMAGE_PRODUCTS if any(x["product_id"] == p for x in plans)]
part = head_r.segmented_control("검사 부품", parts, default=parts[0], required=True, key="insp_part",
                                format_func=lambda p: PART_KO[p])
today = date.today().isoformat()
cands = sorted([p for p in plans if p["product_id"] == part],
               key=lambda p: (p["plan_date"] != today, abs((date.fromisoformat(p["plan_date"]) - date.today()).days)))
plan_key = f"insp_plan_{part}"
if ss.get(plan_key) not in [p["plan_id"] for p in cands]:
    ss[plan_key] = cands[0]["plan_id"]
plan = next(p for p in cands if p["plan_id"] == ss[plan_key])
info_l, info_r = st.columns([3, 1], vertical_alignment="center")
info_l.caption(f"연결된 생산계획 **{plan['plan_date']}** · {PART_KO[part]} {plan['planned_units']:,}개 · "
               f"생산 {plan['produced_units']:,}개 · `{plan['plan_id']}`"
               + ("" if plan["plan_date"] == today else " · 오늘 계획이 없어 가장 가까운 날짜를 연결"))
if len(cands) > 1:
    with info_r.popover("계획 변경", width="stretch"):
        st.radio("같은 부품의 생산계획", [p["plan_id"] for p in cands], key=plan_key,
                 format_func=lambda x: next(f"{p['plan_date']} · {p['planned_units']:,}개 · {x}" for p in cands if p["plan_id"] == x))
plan_id = plan["plan_id"]

try:
    from ml.vision.infer import status as vstatus
    vs = vstatus()
except ImportError:
    vs = {"status": "MODEL_NOT_READY", "message": "torch 미설치"}
if vs["status"] != "OK":
    st.warning(f"**{vs['status']}** — {vs.get('message', '')}. 사진은 기록되지만 AI 판정 없이 작업자 판정만 가능합니다.")


# ── 개체 ID: 다음 개체 자동 번호 ───────────────────────────────────────────
def next_unit_id(pid: str) -> str:
    nums = [int(m.group(1)) for r in list_inspections(pid) if (m := re.fullmatch(r"U(\d+)", str(r["unit_id"])))]
    return f"U{(max(nums) + 1 if nums else 1):04d}"


unit_key = f"insp_unit_{plan_id}"
if unit_key not in ss:
    ss[unit_key] = next_unit_id(plan_id)


def go_next_unit() -> None:
    ss[unit_key] = next_unit_id(plan_id)
    ss.pop("last", None)


def overlay(img_bytes, cam, boxes=None):
    """cam = Grad-CAM 히트맵(빨간 음영), boxes = YOLO11 결함 위치 박스(빨강 = 스크래치·외관 손상)."""
    from PIL import ImageDraw, ImageOps
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    im.thumbnail((720, 720))
    if cam:
        c = Image.fromarray((np.array(cam) * 255).astype("uint8")).resize(im.size, Image.BILINEAR)
        heat = Image.merge("RGB", (c, Image.new("L", im.size, 0), Image.new("L", im.size, 0)))
        im = Image.blend(im, heat, 0.4)
    if boxes:
        d = ImageDraw.Draw(im)
        W, H = im.size
        for bx in boxes:
            x1, y1, x2, y2 = bx["xyxyn"]
            d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(255, 40, 40), width=3)
            d.text((x1 * W + 3, max(0, y1 * H - 14)), f"{bx['name']} {bx['conf']:.2f}", fill=(255, 40, 40))
    return im


def remember(name: str, b: bytes, rec: dict | None, camera: bool) -> None:
    if rec:
        ss["last"] = {"name": name, "bytes": b, "rec": rec, "camera": camera, "plan_id": plan_id}


def live_badge(r: dict) -> dict:
    """미리보기 결과 → 영상 위 배지 문구."""
    if r.get("status") != "OK":
        return {"text": f"{r.get('status')}\n{r.get('message', '')}", "tone": "idle"}
    defect = r["verdict"] == "AI_SUSPECTED_DEFECT"
    if r.get("members"):
        head = VERDICT_KO.get(r["verdict"], r["verdict"])
        if r["verdict"] == "AI_DISAGREE" and r.get("check_reasons"):
            head += f" ({r['check_reasons'][0]})"
        lines = [head] + [f"{m['name']} {m['defect_prob']:.2f} (기준 {m['threshold']:.2f})" for m in r["members"]]
    else:
        lines = [f"{'결함 의심' if defect else '양품 추정'} · 불량 점수 {r['defect_prob']:.2f} (기준 {r['threshold']:.2f})"]
    lines.append(f"부품 추정: {PART_KO.get(r['predicted_part'], r['predicted_part'])} {r['part_confidence']:.0%}")
    tone = "defect" if defect else "mismatch" if r["verdict"] == "AI_DISAGREE" else "pass"
    if r.get("part_mismatch") == "BLOCK":
        lines.append(f"계획 부품({PART_KO.get(r['plan_part'])})이 아님 → 등록하면 차단됨")
        tone = "mismatch"
    elif r.get("part_mismatch") == "SUSPECTED":
        lines.append("부품 종류 확인 필요")
        tone = "mismatch"
    lines.append("미리보기 · 기록 안 됨")
    return {"text": "\n".join(lines), "tone": tone}


@st.fragment
def live_section(plan_id: str, unit_id: str, shot: str = "surface", guide: float | None = None) -> None:
    """실시간 미리보기는 이 구역만 다시 실행된다(페이지 전체 재실행 없음)."""
    from services.config import ROOT, ServiceError
    from services.inspection_registry import preview_part_image
    from ui.live_camera import decode_image, live_camera

    def on_frame():
        ev = ss["live_cam"].frame
        b = decode_image(ev)
        if b is None:
            return
        try:
            ss["live_result"] = {"seq": ev.get("seq"), **live_badge(preview_part_image(plan_id, b, shot=shot))}
        except ServiceError as e:
            ss["live_result"] = {"seq": ev.get("seq"), "text": f"{e.code}\n{e.message}", "tone": "mismatch"}

    def on_register():
        b = decode_image(ss["live_cam"].register)
        if b is None:
            ss["live_notice"] = "등록 실패: 영상 프레임을 받지 못했습니다."
            return
        try:
            rec = inspect_part_image(plan_id, unit_id, b, shot=shot)
            remember("등록 장면", b, rec, camera=True)
            ss["live_notice"] = f"{unit_id} 등록 완료 ({VERDICT_KO.get(rec['review_status'], rec['review_status'])})"
        except ServiceError as e:
            ss["live_notice"] = f"등록 거부 — {e.code}: {e.message}"
        ss["live_app_rerun"] = True

    live_camera(key="live_cam", live=ss.get("live_result"), notice=ss.get("live_notice"),
                on_frame=on_frame, on_register=on_register, guide=guide)
    if ss.pop("live_app_rerun", False):
        st.rerun()  # 오른쪽 판정 패널·최근 개체 새로 고침


# ── 2단 작업대 ─────────────────────────────────────────────────────────────
left, right = st.columns([2, 1], gap="large")
unit_id = ss[unit_key]

with left:
    task = st.segmented_control("검사 종류", ["근접 표면 판정", "차량 위치 찾기 (선택)"], default="근접 표면 판정",
                                required=True, key="insp_task", label_visibility="collapsed")
    if task == "차량 위치 찾기 (선택)":
        with st.container(border=True):
            st.caption("차량 전체 사진에서 도어(초록)·범퍼(노랑)·휀더(빨강) 영역을 표시합니다.  \n"
                       "근접 촬영 위치 안내용이며 결함 판정·폐기 kg에는 반영되지 않습니다.")
            car_mode = st.segmented_control("사진 입력 방식", ["카메라 촬영", "파일 업로드"], default="카메라 촬영",
                                            required=True, key="car_mode")
            if car_mode == "카메라 촬영":
                from ui.live_camera import camera_snapshot
                st.caption("차량 전체 촬영 — '카메라 켜기' → 차량 전체가 들어오게 맞추고 '촬영'")
                car = camera_snapshot(key="car_snap")
                if car:
                    st.image(car, caption="촬영한 차량 사진", width=360)
            else:
                up = st.file_uploader("차량 전체 사진(JPG/PNG)", type=["jpg", "jpeg", "png"], key="car_photo")
                car = up.getvalue() if up else None
            if car and st.button("부품 위치 확인", type="primary"):
                from services.part_locator import locate_plan_part
                loc = run(locate_plan_part, plan_id, car)
                if loc and loc["status"] != "OK":
                    st.warning(f"**{loc['status']}** — {loc.get('message', '')}")
                    st.info(loc["guidance"])
                elif loc:
                    st.image(loc["overlay"], caption="초록 = 도어, 노랑 = 범퍼, 빨강 = 휀더")
                    (st.success if loc["plan_part_found"] else st.warning)(loc["guidance"])
                    tags(loc["provenance"])
    else:
        with st.container(border=True):
            shot, guide = "surface", None
            if part == "FENDER":  # 2안(사용자 결정 2026-11-14): 같은 화면에서 촬영 부위 전환, 휀더에서만 표시
                kind = st.segmented_control("촬영 부위", ["표면", "틈(단차)"], default="표면", required=True,
                                            key="shot_kind",
                                            help="틈(단차)을 고르면 가이드 박스가 나타납니다. 틈이 박스 가운데에서 박스의 "
                                                 "약 2/3를 채우도록 찍으면 단차 전용 모델이 함께 판정합니다.")
                if kind == "틈(단차)":
                    from ml.vision.infer import GAP_META, gap_ready
                    shot = "gap"
                    guide = __import__("json").loads(GAP_META.read_text(encoding="utf-8"))["guide_frac"] if gap_ready() else 0.5
                    st.info("틈이 **노란 가이드 박스 가운데**에서 박스의 **약 2/3**를 채우게 맞춰 찍으세요. "
                            "카메라 촬영·파일 업로드는 사진 가운데(짧은 변의 절반 크기)를 틈 영역으로 봅니다.")
            mode = st.segmented_control("사진 입력 방식", ["실시간 미리보기", "카메라 촬영", "파일 업로드", "AI-Hub 샘플"],
                                        default="실시간 미리보기", required=True, key="photo_mode",
                                        help="AI-Hub 샘플: 학습에 쓰지 않은 시험셋 사진으로 카메라 없이 시연하고 정답과 비교합니다.")
            if mode == "실시간 미리보기":
                st.caption("영상 위 판정은 미리보기이며 기록되지 않습니다. '이 장면 등록'을 누른 1장만 "
                           f"현재 개체(**{unit_id}**)의 검사 기록이 됩니다. 카메라는 localhost 또는 https 접속에서만 열립니다.")
                live_section(plan_id, unit_id, shot, guide)
            elif mode == "카메라 촬영":
                from ui.live_camera import camera_snapshot, clear_snapshot
                st.caption(f"{PART_KO[part]} {'틈' if shot == 'gap' else '표면'} 촬영 — '카메라 켜기' → 맞추고 '촬영' → "
                           "사진을 확인한 뒤 '이 사진 등록'. 카메라가 여러 대면 목록에서 고르세요(적외선·가상 카메라는 자동 제외).")
                b = camera_snapshot(key="cam_snap", guide=guide)
                if b:
                    st.image(b, caption="촬영한 사진 (다시 찍으려면 '촬영'을 한 번 더)", width=360)
                    if st.button("이 사진 등록", type="primary"):
                        remember("카메라 촬영", b, run(inspect_part_image, plan_id, unit_id, b, shot), camera=True)
                        clear_snapshot("cam_snap")
            elif mode == "AI-Hub 샘플":
                sdir = ROOT / "data/demo/aihub_samples"
                if not (sdir / "index.csv").exists():
                    st.info("샘플 세트가 없습니다. 원천 데이터가 있는 PC에서 `python -m ml.vision.build_demo_samples`로 만드세요.")
                else:
                    idx = pd.read_csv(sdir / "index.csv", encoding="utf-8-sig")
                    idx = idx[(idx.part == part) & (idx.shot == shot)]
                    truth = st.segmented_control("정답 라벨(시연용)", ["전체", "양품", "불량"], default="전체", required=True,
                                                 key="sample_truth_filter")
                    if truth != "전체":
                        idx = idx[idx.quality == ("GOOD" if truth == "양품" else "DEFECT")]
                    pick = st.selectbox(f"{PART_KO[part]} {'틈 ' if shot == 'gap' else ''}샘플 ({len(idx)}장)", list(idx.file),
                                        key="sample_pick",
                                        format_func=lambda f: next(f"{'불량' if r.quality == 'DEFECT' else '양품'} · {r.subtype} · {r.stem[:12]}"
                                                                   for r in idx.itertuples() if r.file == f))
                    if pick:
                        row = idx[idx.file == pick].iloc[0]
                        st.image(str(sdir / pick), caption=f"입력 사진 · AI-Hub 정답: {'불량' if row.quality == 'DEFECT' else '양품'}"
                                 f"({row.subtype}) · 공식 Validation(학습 미사용)", width=480)
                        if st.button("이 샘플로 검사", type="primary"):
                            b = (sdir / pick).read_bytes()
                            rec = run(inspect_part_image, plan_id, unit_id, b, shot)
                            remember(f"샘플 {pick}", b, rec, camera=False)
                            if rec:
                                ss.setdefault("sample_truth", {})[rec["image_hash"]] = (row.quality, row.subtype)
            else:
                files = st.file_uploader("부품 근접 사진(JPG/PNG, 같은 개체의 여러 각도 가능)", type=["jpg", "jpeg", "png"],
                                         accept_multiple_files=True, key="part_files")
                if files and st.button("사진 등록", type="primary"):
                    for f in files:
                        b = f.getvalue()
                        remember(f.name, b, run(inspect_part_image, plan_id, unit_id, b, shot), camera=False)
            tags(["AIHUB_IMAGE_AI", "DEMO_SYNTHETIC_BRIDGE"] + (["CAMERA_UNVALIDATED"] if mode in ("실시간 미리보기", "카메라 촬영") else [])
                 + (["GAP_GUIDE_UNVALIDATED"] if shot == "gap" else []))

    # 현재 개체 기록: 어떤 사진에서 어떤 판정이 나왔는지(처리 대기함에서 불러온 개체도 여기서 확인)
    unit_recs = run(unit_records, plan_id, unit_id) or []
    with st.container(border=True):
        n_photo = sum(1 for r in unit_recs if r.get("photo_path"))
        st.markdown(f"**{unit_id} 기록** · 사진 {n_photo}장 · 기록 {len(unit_recs)}건")
        if not unit_recs:
            st.caption("이 개체의 기록이 아직 없습니다.")
        # 처리 대기함 '작업자 확인'의 원인 사진을 맨 위에 따로 보여 준다(2026-11-18: 원인 사진이 최신이 아니면 격자에 묻혀 찾기 어려웠음)
        decided = any(r.get("scrap_decision") for r in unit_recs)
        flagged = [r for r in unit_recs if r["review_status"] in CHECK_STATUSES]
        if flagged and not decided:
            with st.container(border=True, key="flagged_box"):
                st.markdown(f":orange-badge[:material/warning: 작업자 확인 필요] **확인이 필요한 사진 {len(flagged)}장** — "
                            "실물과 비교해 오른쪽 '작업자 확정'에서 폐기·양품·재작업을 고르세요.")
                fcols = st.columns(2)
                for i, r in enumerate(reversed(flagged)):
                    ai = r["ai"]
                    with fcols[i % 2]:
                        pf = photo_file(r.get("photo_path"))
                        if pf:
                            st.image(overlay(pf.read_bytes(), ai.get("cam"), ai.get("boxes")),
                                     caption=f"{r['created_at'][5:16].replace('T', ' ')} 등록 · 빨간 영역 = 결함 의심 위치")
                        else:
                            st.caption("저장된 사진 없음")
                        st.markdown(f"**{VERDICT_KO.get(r['review_status'], r['review_status'])}**"
                                    + (" · 틈 촬영" if r.get("shot") == "gap" else ""))
                        if ai.get("check_reasons"):
                            st.caption("이유: " + " · ".join(ai["check_reasons"]))
                        if ai.get("members"):
                            st.caption(" / ".join(f"{m['name']} {m['defect_prob']:.2f}(기준 {m['threshold']:.2f}, "
                                                  f"{'불량' if m['verdict'] == 'AI_SUSPECTED_DEFECT' else '양품'})"
                                                  for m in ai["members"]))
        newest = next((r for r in reversed(unit_recs) if photo_file(r.get("photo_path"))), None)
        if newest:  # 최신 사진: 입력 사진과 결과(결함 의심 위치)를 나란히
            pf = photo_file(newest["photo_path"])
            raw = pf.read_bytes()
            c_in, c_out = st.columns(2)
            c_in.image(overlay(raw, None), caption="입력 사진")
            bx = newest["ai"].get("boxes")
            c_out.image(overlay(raw, newest["ai"].get("cam"), bx),
                        caption="AI 결과 · 빨간 음영 = 결함 의심 위치(Grad-CAM)"
                                + (f" · 빨간 박스 = YOLO11 결함 위치 {len(bx)}곳(스크래치·외관 손상, 표시 전용)" if bx else ""))
        grid = st.columns(3)
        older = [r for r in reversed(unit_recs) if r is not newest]
        if older:
            st.caption("이 개체의 다른 기록")
        for i, r in enumerate(older):  # 최신 사진부터
            ai = r["ai"]
            with grid[i % 3]:
                pf = photo_file(r.get("photo_path"))
                if pf:
                    st.image(overlay(pf.read_bytes(), ai.get("cam")),
                             caption=f"{r['created_at'][5:16].replace('T', ' ')} · 빨간 영역 = 결함 의심 위치")
                else:
                    st.caption("저장된 사진 없음 (육안 판정 또는 사진 저장 기능 이전 기록)")
                head = VERDICT_KO.get(r["review_status"], r["review_status"])
                st.markdown(f"**{head}**" + (" · 틈 촬영" if r.get("shot") == "gap" else ""))
                if ai.get("check_reasons"):
                    st.caption(" · ".join(ai["check_reasons"]))
                if ai.get("members"):
                    st.caption(" / ".join(f"{m['name']} {m['defect_prob']:.2f}(기준 {m['threshold']:.2f})"
                                          for m in ai["members"]))
        hist = decision_history(plan_id, unit_id)
        if hist:
            with st.expander(f"판정 이력 {len(hist)}건"):
                st.dataframe(pd.DataFrame(hist)[["at", "action", "decision", "operator_id", "note"]].rename(
                    columns={"at": "시각", "action": "작업", "decision": "판정", "operator_id": "작업자", "note": "메모"}),
                    hide_index=True)


def load_unit(uid: str) -> None:
    ss[unit_key] = uid
    ss.pop("last", None)
    ss["opened_msg"] = f"{uid}를 현재 개체로 불러왔습니다. 확인이 필요한 사진은 왼쪽 '{uid} 기록' 맨 위 주황색 상자에 있습니다."


with right:
    with st.container(border=True):
        st.caption("현재 개체")
        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        c1.text_input("개체 ID", key=unit_key, label_visibility="collapsed")
        c2.button("다음 개체", on_click=go_next_unit, width="stretch")
        decision = next((r["scrap_decision"] for r in unit_recs if r.get("scrap_decision")), None)
        st.caption(f"사진 {sum(1 for r in unit_recs if r.get('photo_path'))}장 · "
                   + (DECISION_KO.get(decision, "확정 대기") if unit_recs else "아직 기록 없음"))

    with st.container(border=True):
        st.caption("현재 개체 최근 판정")
        latest = unit_recs[-1] if unit_recs else None
        last = ss.get("last")
        if last and last.get("plan_id") == plan_id and last["rec"].get("duplicate"):
            st.info("방금 올린 사진은 이미 등록된 같은 사진입니다. 결과는 그대로이며 중복 집계하지 않습니다.")
        if msg := ss.pop("opened_msg", None):
            st.toast(msg, icon=":material/folder_open:")
            st.info(msg, icon=":material/arrow_back:")
        if not latest:
            st.write("아직 등록한 사진이 없습니다.")
        else:
            ai, status = latest["ai"], latest["review_status"]
            n_flag = sum(1 for r in unit_recs if r["review_status"] in CHECK_STATUSES)
            if n_flag and status not in CHECK_STATUSES and not any(r.get("scrap_decision") for r in unit_recs):
                st.warning(f"아래는 가장 최근 사진의 판정입니다. 이 개체에는 확인이 필요한 사진이 {n_flag}장 더 있습니다 "
                           "(왼쪽 주황색 상자).", icon=":material/warning:")
            st.subheader(VERDICT_KO.get(status, status))
            if status == "AI_DISAGREE":
                rec_txt = {"AI_SUSPECTED_DEFECT": "결함 의심", "AI_PASS": "양품"}.get(ai.get("ai_recommendation"), "")
                st.warning(" · ".join(ai.get("check_reasons", [])) or "AI 판정이 엇갈림",
                           title="작업자가 실물을 확인하세요" + (f" (2차 판정 추천: {rec_txt})" if rec_txt else ""))
            if status == "PART_MISMATCH_SUSPECTED":
                st.error(f"AI가 이 사진을 {PART_KO.get(ai.get('predicted_part'), ai.get('predicted_part'))}로 추정합니다. "
                         "계획 부품과 다를 수 있으니 실물을 확인하세요.")
            if status in ("AI_SUSPECTED_DEFECT", "AI_PASS") and ai.get("members"):
                from ml.vision.infer import auto_accuracy
                acc = auto_accuracy(part, latest.get("shot") or "surface")
                if acc:
                    st.success(f"판정 일치 → 자동 판정. 이 규칙으로 자동 판정된 사진의 정확도 **{acc['accuracy']:.1%}** "
                               f"(자동 판정 비율 {acc['auto_rate']:.0%}, {acc['set']}). "
                               + ("학습에 쓰지 않은 촬영일 사진으로 잰 수치(근거·성능 화면)" if acc.get("honest") else
                                  "⚠️ 평가 사진 약 98%가 학습 사진과 같은 촬영 세션이라 실제보다 높을 수 있음(근거·성능 화면)"))
            truth = ss.get("sample_truth", {}).get(latest["image_hash"])
            if truth:
                t_ko = "불량" if truth[0] == "DEFECT" else "양품"
                ok = {"AI_SUSPECTED_DEFECT": "DEFECT", "AI_PASS": "GOOD"}.get(status)
                mark = "일치" if ok == truth[0] else ("작업자 확인 대상" if ok is None else "불일치")
                st.info(f"AI-Hub 정답: **{t_ko}**({truth[1]}) → AI 결과와 {mark}")
            if ai.get("members"):
                st.dataframe(pd.DataFrame([{"모델": m["name"], "점수": round(m["defect_prob"], 3),
                                            "기준": round(m["threshold"], 2),
                                            "판정": "불량" if m["verdict"] == "AI_SUSPECTED_DEFECT" else "양품"}
                                           for m in ai["members"]]), hide_index=True)
            elif latest.get("ai_defect_prob") is not None:
                st.write(f"불량 점수 {latest['ai_defect_prob']:.3f}")

    with st.container(border=True):
        st.markdown(f"**작업자 확정** · {unit_id}")
        st.text_input("작업자 ID", "operator01", key="operator_id")
        st.text_input("메모 (선택)", key="decision_note", placeholder="예: 도장 재작업, 재촬영 필요")

        def decide(decision: str) -> None:
            try:
                record_operator_decision(plan_id, ss[unit_key], decision, ss.get("operator_id", ""),
                                         note=ss.get("decision_note") or None)
                ss["decided_msg"] = f"{ss[unit_key]}: {DECISION_KO[decision]} 저장 → 다음 개체로 이동"
                ss["decision_note"] = ""
                go_next_unit()
            except ServiceError as e:
                ss["decided_msg"] = f"저장 실패 — {e.code}: {e.message}"

        st.button("폐기 확정", type="primary", width="stretch", on_click=decide, args=("CONFIRMED_SCRAP",))
        b1, b2 = st.columns(2)
        b1.button("양품 확정", width="stretch", on_click=decide, args=("CONFIRMED_GOOD",))
        b2.button("재작업 보류", width="stretch", on_click=decide, args=("REWORK_PENDING",))
        if msg := ss.pop("decided_msg", None):
            (st.error if msg.startswith("저장 실패") else st.success)(msg)
        st.caption("폐기 kg에는 '폐기 확정'만 반영됩니다. 사진 없이 육안 판정한 개체도 여기서 확정할 수 있습니다.")

    # 폐기물 예측 변경 전·후 (팀원 앱 반영 2026-11-14): 확정하면 바로 이 계획의 kg 변화가 보인다
    from services.waste_forecast_update import compute_update
    upd = run(compute_update, plan_id)
    if upd:
        with st.container(border=True):
            st.caption(f"{plan_id} 폐기물 예측 변경 전·후")
            e1, e2 = st.columns(2)
            e1.metric("최초 예상", f"{upd['initial_expected_kg']:,.2f} kg")
            e2.metric("갱신 예상", f"{upd['updated_expected_total_kg']:,.2f} kg",
                      delta=f"{upd['updated_expected_total_kg'] - upd['initial_expected_kg']:+,.2f} kg", delta_color="inverse")
            e3, e4 = st.columns(2)
            e3.metric("확인 폐기", f"{upd['confirmed_scrap_kg']:,.2f} kg")
            e4.metric("잔여 예상", f"{upd['remaining_expected_kg']:,.2f} kg")
            from services.config import get_product
            w = float(get_product(plan["product_id"])["unit_weight_kg"])
            bk = upd["buckets"]
            st.caption(f"산식: 확정 폐기 {bk['CONFIRMED_SCRAP']}개×{w}kg + (미확정 생산 {bk['PRODUCED_UNINSPECTED']} + 미생산 "
                       f"{bk['UNPRODUCED']})개×남은 불량률 {upd['remaining_defect_rate']:.3f}×{w}kg · 재작업 대기 "
                       f"{upd['rework_pending_units_not_in_kg']}개(kg 미합산)")

    with st.container(border=True):
        pend = pending_units(plan_id)
        q = st.segmented_control("처리 대기함", ["REWORK", "CHECK"], default="REWORK", required=True, key="queue_kind",
                                 format_func=lambda k: (f"재작업 보류 {len(pend['REWORK'])}" if k == "REWORK"
                                                        else f"작업자 확인 {len(pend['CHECK'])}"))
        items = pend[q]
        if not items:
            st.caption("대기 중인 개체가 없습니다.")
        for it in items[:8]:
            if q == "REWORK":
                why = it["note"] or "재작업 보류"
            else:
                f = it["flagged"][-1]
                why = (f["ai"].get("check_reasons") or [VERDICT_KO.get(f["review_status"], f["review_status"])])[0]
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown(f"`{it['unit_id']}` · {why} · 사진 {it['photos']}장")
            c2.button("열기", key=f"load_{q}_{it['unit_id']}", on_click=load_unit, args=(it["unit_id"],),
                      width="stretch", help="이 개체를 현재 개체로 불러와 사진·판정 근거를 보고 확정합니다")
        if len(items) > 8:
            st.caption(f"외 {len(items) - 8}개 · 아래 검사 기록의 필터로 모두 볼 수 있습니다.")

# ── 하단: 개체 현황 ────────────────────────────────────────────────────────
b = run(unit_buckets, plan_id)
if b:
    st.subheader("개체 현황 (합계 = 계획수량)")
    kpi = st.columns(5)
    for col, (k, label) in zip(kpi, [("CONFIRMED_SCRAP", "폐기 확정"), ("CONFIRMED_GOOD", "양품 확정"),
                                     ("REWORK_PENDING", "재작업 보류"), ("PRODUCED_UNINSPECTED", "생산·미확정"),
                                     ("UNPRODUCED", "미생산")]):
        col.metric(label, f"{b['buckets'][k]:,}")
    st.caption(f"AI 결과만 있고 작업자 판정 대기: {b['units_with_ai_result_awaiting_operator']}개 (생산·미확정으로 계산)")


# ── 하단: 검사 기록(필터 · 선택 취소 · 복원 · 영구 삭제) ───────────────────
@st.dialog("선택 기록 취소")
def cancel_dialog(ids: list[str]) -> None:
    st.write(f"{len(ids)}건을 취소합니다. 취소한 기록은 집계·폐기 kg에서 빠지며 '취소됨'에서 복원할 수 있습니다.")
    impact = cancel_preview(plan_id, ids)
    st.dataframe(pd.DataFrame([{"개체": u["unit_id"], "취소 기록": u["cancel_records"], "남는 기록": u["records_left"],
                                "현재 판정": DECISION_KO.get(u["decision"], "확정 대기"),
                                "폐기 kg 변화": u["kg_change"]} for u in impact]), hide_index=True)
    reason = st.text_input("취소 사유 (필수)", placeholder="예: 잘못된 개체에 등록, 흔들린 사진")
    op = st.text_input("작업자 ID", ss.get("operator_id", "operator01"))
    if st.button("취소 실행", type="primary", disabled=not reason.strip()):
        try:
            cancel_inspections(plan_id, ids, op, reason)
            ss["records_msg"] = f"{len(ids)}건 취소했습니다."
            st.rerun()
        except ServiceError as e:
            st.error(f"{e.code} — {e.message}")


@st.dialog("선택 기록 영구 삭제")
def purge_dialog(ids: list[str]) -> None:
    st.error(f"{len(ids)}건을 영구 삭제합니다. 기록·저장 사진이 지워지고 되돌릴 수 없습니다. "
             "개체의 기록이 모두 사라지면 그 개체의 판정 이력도 지워집니다.")
    op = st.text_input("작업자 ID", ss.get("operator_id", "operator01"))
    if st.button("영구 삭제", type="primary"):
        try:
            r = purge_inspections(plan_id, ids, op)
            ss["records_msg"] = f"{r['purged']}건 영구 삭제했습니다."
            st.rerun()
        except ServiceError as e:
            st.error(f"{e.code} — {e.message}")


@st.dialog("공용 DB 초기화")
def reset_dialog() -> None:
    st.error("DB의 모든 데이터(생산계획·검사 기록·사진·보관 장부·수주 예측)를 지우고 완전히 빈 상태로 만듭니다. "
             "팀 VM에서는 모든 팀원의 기록이 함께 지워집니다.")
    st.write(f"초기화 직전 상태는 `{db.db_path().parent / 'backups'}` 폴더에 날짜별로 자동 백업됩니다.")
    if st.button("초기화 실행", type="primary"):
        r = db.reset_database()
        for k in [k for k in ss.keys() if str(k).startswith(("insp_", "last", "live_"))]:
            del ss[k]
        ss["records_msg"] = f"초기화했습니다. 백업: {r['backup_db']}"
        st.rerun()


st.subheader("검사 기록")
if msg := ss.pop("records_msg", None):
    st.success(msg)
FILTERS = {"전체": lambda r: r["cancelled_at"] is None,
           "확정 대기": lambda r: r["cancelled_at"] is None and not r["scrap_decision"],
           "재작업 보류": lambda r: r["cancelled_at"] is None and r["scrap_decision"] == "REWORK_PENDING",
           "작업자 확인": lambda r: r["cancelled_at"] is None and not r["scrap_decision"] and r["review_status"] in ("AI_DISAGREE", "PART_MISMATCH_SUSPECTED"),
           "폐기 확정": lambda r: r["cancelled_at"] is None and r["scrap_decision"] == "CONFIRMED_SCRAP",
           "양품 확정": lambda r: r["cancelled_at"] is None and r["scrap_decision"] == "CONFIRMED_GOOD",
           "취소됨": lambda r: r["cancelled_at"] is not None}
flt = st.segmented_control("보기", list(FILTERS), default="전체", required=True, key="rec_filter")
recs = [r for r in list_inspections(plan_id, include_cancelled=True) if FILTERS[flt](r)]
if not recs:
    st.caption("해당 기록이 없습니다.")
else:
    table = pd.DataFrame([{"개체": r["unit_id"], "사진": "있음" if r.get("photo_path") else "-",
                           "AI 판정": VERDICT_KO.get(r["review_status"], r["review_status"]),
                           "불량 점수": r["ai_defect_prob"], "작업자 판정": DECISION_KO.get(r["scrap_decision"], ""),
                           "작업자": r["operator_id"] or "", "등록 시각": r["created_at"],
                           "취소": (f"{r['cancelled_at']} · {r['cancelled_by']} · {r['cancel_reason']}" if r["cancelled_at"] else "")}
                          for r in recs])
    event = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="multi-row", key=f"rec_table_{flt}")
    sel = [recs[i]["inspection_id"] for i in event.selection.rows]
    a1, a2, _ = st.columns([1, 1, 2])
    if flt != "취소됨":
        if a1.button(f"선택 기록 취소 ({len(sel)}건)", disabled=not sel, width="stretch"):
            cancel_dialog(sel)
    else:
        if a1.button(f"선택 복원 ({len(sel)}건)", disabled=not sel, width="stretch"):
            r = run(restore_inspections, plan_id, sel, ss.get("operator_id", "operator01"))
            if r:
                ss["records_msg"] = f"{r['restored']}건 복원했습니다."
                st.rerun()
        if a2.button(f"선택 영구 삭제 ({len(sel)}건)", disabled=not sel, width="stretch"):
            purge_dialog(sel)
    st.caption("표에서 줄을 체크해 고릅니다. 영구 삭제는 '취소됨'에서만 할 수 있습니다.")

with st.expander("관리 도구"):
    st.caption("공용 DB를 완전히 빈 상태로 되돌립니다(직전 상태 자동 백업).")
    if st.button("공용 DB 초기화"):
        reset_dialog()
