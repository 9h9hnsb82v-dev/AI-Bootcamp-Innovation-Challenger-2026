"""폐플라스틱 위탁 처리업체 추천 지도 — 공공데이터 전국폐기물처리업소표준데이터, 공장에서 가까운 순.

공장 위치: 기본 계명대 성서캠퍼스(시연용). 2026-11-17부터 내 위치·장소 검색·좌표 입력으로 바꿀 수 있다(DB 저장).

2026-11-14 팀원 앱 반영: 수거 권장 상태·위탁 예정량, 화물차 제원 직접 수정, TMAP 켜기/끄기, 경로선 라벨, 자료 누락 지역 경고.
"""
from datetime import date

import pandas as pd
import pydeck as pdk
import streamlit as st

from services.collection_recommendation import recommend_collection
from services.config import ServiceError
from services.truck_route import DEFAULT_TRUCK, LIMITS, app_key, route
from services.waste_processors import RECYCLE_TYPES, industries, missing_regions, recommend
from ui.common import run, tags

ss = st.session_state
st.title("폐기물 순환 위탁 처리업체 추천 — 공장에서 가까운 순")

# ── 수거 요청 권장 · 위탁 예정량 (보관량 예측과 같은 계산) ──
rc = run(recommend_collection, (ss.get("as_of") or date.today()).isoformat(), int(ss.get("horizon") or 14))
planned_kg = None
if rc:
    label = {"NO_COLLECTION_NEEDED": "🟢 수거 불필요", "RECOMMENDED": "🟠 수거 요청 권장", "URGENT": "🔴 즉시 요청"}[rc["status"]]
    with st.container(border=True):
        c = st.columns(4)
        c[0].markdown(f"**수거 상태**\n\n### {label}")
        if rc["status"] != "NO_COLLECTION_NEEDED":
            planned_kg = rc["expected_collection_kg"]
            c[1].metric("수거 요청 마감일", rc["request_by_date"])
            c[2].metric("수거 예정일", rc["recommended_collection_date"])
            c[3].metric("위탁 예정량", f"{planned_kg:,.1f} kg")
        else:
            c[1].write(rc["message"])
        st.caption("보관량·수거 추천 화면과 같은 계산(사이드바 기준일·예측 기간). 이미지 검사로 폐기 확정이 바뀌면 자동으로 다시 계산됩니다.")

from services import factory_location as floc  # noqa: E402
from ui.geo_locate import geo_button  # noqa: E402


def _on_pos():
    v = ss["geo"].pos
    if v and "lat" in v:
        ss["geo_pos"] = v
        ss.pop("geo_addr", None)
        ss.pop("geo_err", None)


def _on_fail():
    ss["geo_err"] = (ss["geo"].fail or {}).get("msg", "위치를 받지 못했습니다.")


def _apply(**kw):
    if run(floc.set_location, **kw):
        for k in ("geo_pos", "geo_addr", "loc_results"):
            ss.pop(k, None)
        ss["loc_saved"] = kw.get("name") or "새 위치"
        st.rerun()


# ── 공장 위치 (2026-11-17: 고정 → 내 위치·검색·직접 입력으로 변경 가능, DB에 저장) ──
fl = floc.get()
if ss.pop("loc_saved", None):
    st.toast("공장 위치를 바꿨습니다. 추천 순위를 새 위치 기준으로 다시 계산합니다.", icon=":material/location_on:")
where = fl["address"] or f"{fl['lat']:.5f}, {fl['lon']:.5f}"
with st.expander(f"공장 위치: {fl['name']} · {where} ({floc.SOURCES.get(fl.get('source'), '사용자 지정')})",
                 icon=":material/location_on:", expanded=False):
    how = st.segmented_control("위치 정하는 방법", ["내 위치", "장소·주소 검색", "위도·경도 직접 입력"], default="장소·주소 검색",
                               required=True, key="loc_how")
    if how == "내 위치":
        st.caption("브라우저가 알려 주는 현재 위치를 씁니다. PC는 GPS가 없어 Wi-Fi·IP로 잡기 때문에 수십 m~수 km 오차가 "
                   "날 수 있습니다. 설정 후 아래 지도의 빨간 점을 확인하세요. 접속 주소가 localhost 또는 https 여야 합니다.")
        geo_button(key="geo", on_pos=_on_pos, on_fail=_on_fail)
        if ss.get("geo_err"):
            st.warning(ss["geo_err"])
        gp = ss.get("geo_pos")
        if gp:
            if "geo_addr" not in ss:
                ss["geo_addr"] = floc.reverse_geocode(gp["lat"], gp["lon"]) or ""
            st.write(f"찾은 위치: **{gp['lat']:.5f}, {gp['lon']:.5f}** · 오차 약 {gp.get('acc') or 0:,.0f} m"
                     + (f" · {ss['geo_addr']}" if ss["geo_addr"] else ""))
            nm = st.text_input("공장 이름", value="내 위치", key="geo_name", max_chars=60)
            if st.button("이 위치를 공장으로 설정", type="primary", key="geo_set"):
                _apply(lat=gp["lat"], lon=gp["lon"], name=nm, address=ss["geo_addr"], source="BROWSER_GPS",
                       accuracy_m=gp.get("acc"))
    elif how == "장소·주소 검색":
        with st.form("loc_search", border=False, enter_to_submit=True):
            c = st.columns([5, 1], vertical_alignment="bottom")
            q = c[0].text_input("장소 이름 또는 주소", placeholder="예: 성서산업단지 / 대구 달서구 달구벌대로 1095", key="loc_q")
            if c[1].form_submit_button("검색", width="stretch", disabled=not app_key()):
                ss["loc_results"] = run(floc.search_place, q)
        if not app_key():
            st.caption("TMAP 키가 없어 검색을 쓸 수 없습니다. '위도·경도 직접 입력'을 쓰세요.")
        res = ss.get("loc_results")
        if res:
            i = st.radio("검색 결과", range(len(res)), key="loc_pick",
                         format_func=lambda k: f"{res[k]['name']} · {res[k]['address']}")
            if st.button("이 장소를 공장으로 설정", type="primary", key="loc_set"):
                _apply(lat=res[i]["lat"], lon=res[i]["lon"], name=res[i]["name"], address=res[i]["address"],
                       source="TMAP_SEARCH")
        elif res == []:
            st.info("검색 결과가 없습니다. 다른 이름이나 도로명 주소로 찾아 보세요.")
    else:
        c = st.columns([2, 1, 1], vertical_alignment="bottom")
        nm = c[0].text_input("공장 이름", value="사용자 지정 공장", key="man_name", max_chars=60)
        lat = c[1].number_input("위도", floc.LAT[0], floc.LAT[1], float(fl["lat"]), step=0.0001, format="%.6f", key="man_lat")
        lon = c[2].number_input("경도", floc.LON[0], floc.LON[1], float(fl["lon"]), step=0.0001, format="%.6f", key="man_lon")
        st.caption("지도 앱(카카오맵·네이버지도 등)에서 공장 위치를 눌러 나온 위도·경도를 넣으세요. 국내 좌표만 받습니다.")
        if st.button("이 좌표를 공장으로 설정", type="primary", key="man_set"):
            _apply(lat=lat, lon=lon, name=nm, address=floc.reverse_geocode(lat, lon) or "", source="MANUAL")
    if fl.get("source") != "DEFAULT" and st.button("기본 위치(계명대학교 성서캠퍼스)로 되돌리기", key="loc_reset",
                                                    icon=":material/restart_alt:"):
        floc.reset()
        ss["loc_saved"] = "기본"
        st.rerun()
all_types = run(industries) or []
c1, c2 = st.columns([4, 1])
types = c1.multiselect("업종", all_types, default=[t for t in RECYCLE_TYPES if t in all_types], key="proc_types",
                       help="폐합성수지류(51-03-01) 처리 허가가 있는 업체 중 고른 업종만 추천합니다.")
n = c2.number_input("추천 수", 1, 50, 10, key="proc_n")

t1, t2 = st.columns([1, 3])
use_tmap = t1.toggle("화물차 도로거리 (TMAP)", value=bool(app_key()), key="use_tmap", disabled=not app_key(),
                     help="켜면 TMAP 화물차 경로안내로 도로거리·시간·경로선을 받아 도로거리 순으로 정렬합니다(결과는 저장해 재사용).")
truck = dict(DEFAULT_TRUCK)
with t2.expander(f"화물차 제원: 5톤 카고트럭(가정) — 높이 {ss.get('tr_height_cm', 340) / 100:.1f} m · 총중량 {ss.get('tr_total_kg', 11500) / 1000:.1f} t"):
    labels = {"height_cm": "높이(cm)", "width_cm": "폭(cm)", "length_cm": "길이(cm)", "load_kg": "적재중량(kg)", "total_kg": "총중량(kg)"}
    cols = st.columns(3)
    for i, k in enumerate(labels):
        lo, hi = LIMITS[k]
        truck[k] = cols[i % 3].number_input(labels[k], lo, hi, DEFAULT_TRUCK[k], step=10 if k.endswith("cm") else 100, key=f"tr_{k}")
    changed = truck != DEFAULT_TRUCK
    st.caption("폐플라스틱 운반 화물차 가정값(5톤 카고트럭). TMAP 화물차 경로안내의 통행 제한(높이·폭·중량·길이) 판단에 씁니다. "
               f"상태: {'USER_EDITED' if changed else 'ASSUMED_DEFAULT'} · 제원이 바뀌면 경로를 새로 조회합니다.")

ROUTE_POOL = 20  # 직선거리 상위 20곳만 도로 경로 조회(무료 사용량 절약, 결과 저장)
r = run(recommend, tuple(types), max(int(n), ROUTE_POOL), origin=fl) if types else None
if not r:
    st.info("업종을 하나 이상 고르세요.")
    st.stop()
f = r["factory"]
df = pd.DataFrame(r["items"])
routes = {}
if use_tmap and app_key():
    # TMAP 무료 사용량 절약: 화면을 열 때는 저장된 경로만, '도로거리 조회'를 누를 때만 상위 N곳 새로 조회
    fetch = st.button(f"도로거리 조회 (직선거리 상위 {ROUTE_POOL}곳 중 저장 안 된 곳만 TMAP 호출)",
                      help="이미 조회한 구간은 저장값을 씁니다. 화물차 제원을 바꾸면 새로 조회해야 합니다.")
    try:
        with st.spinner("TMAP 화물차 경로 조회 중…" if fetch else "저장된 경로 불러오는 중…"):
            for it in r["items"][:ROUTE_POOL]:
                routes[it["순위"]] = route((f["lat"], f["lon"]), (float(it["위도"]), float(it["경도"])), "공장", it["시설명"],
                                          truck=truck, cache_only=not fetch)
    except ServiceError as e:
        st.error(f"{e.code} — {e.message}")
    errs = [v["error"] for v in routes.values() if v and "error" in v]
    if errs:
        st.warning(f"TMAP 경로 조회 실패 {len(errs)}건(해당 업체는 직선거리만 표시): {errs[0]}")
    elif not any(routes.values()):
        st.info("이 화물차 제원으로 저장된 경로가 없습니다. '도로거리 조회'를 누르거나 표에서 업체를 고르면 그 업체만 조회합니다.")
elif not app_key():
    st.info("TMAP 앱 키가 없어 **직선거리** 순위만 표시합니다. 설정: `.streamlit/secrets.toml`에 `TMAP_APP_KEY = \"발급키\"`")
ok = {k: v for k, v in routes.items() if v and "error" not in v}
df["도로거리_km"] = [round(ok[k]["distance_km"], 1) if k in ok else None for k in df["순위"]]
df["예상시간_분"] = [round(ok[k]["minutes"]) if k in ok else None for k in df["순위"]]
if ok:
    df = df.sort_values(["도로거리_km", "직선거리_km"], na_position="last").reset_index(drop=True)
df = df.head(int(n)).copy()
df["직선순위"] = df["순위"]
df["순위"] = range(1, len(df) + 1)

st.caption(f"공장 위치: {f['name']} · {f['address'] or '주소 없음'} ({f['lat']:.5f}, {f['lon']:.5f}) · 폐기물: 폐합성수지류(폐염화비닐수지류 제외) · "
           + (f"위탁 예정량: {planned_kg:,.1f} kg · " if planned_kg else "")
           + f"허가 재활용업체 {r['candidates']:,}곳 중 {len(df)}곳 · 거리 기준: {'TMAP 화물차 도로거리' if ok else '직선거리'}")
radius = float(max(df["직선거리_km"].max(), 5.0))
gaps = missing_regions(radius, f)
if gaps:
    st.warning("추천 거리 안에 업체 자료가 빠졌을 수 있는 지역: " + ", ".join(f"{g['name']}({g['km']:.0f} km)" for g in gaps[:5])
               + (f" 외 {len(gaps) - 5}곳" if len(gaps) > 5 else "")
               + ". 이 지역의 더 가까운 업체가 목록에 없을 수 있습니다(해당 지자체 자료 미제출 추정, 지도의 회색 원).")

show = df[["순위", "시설명", "업종명", "소재지도로명주소", "도로거리_km", "예상시간_분", "직선거리_km", "시설·장비명", "전화번호",
           "영업구역", "데이터기준일자"]].copy()
show["직선거리_km"] = show["직선거리_km"].round(2)
event = st.dataframe(show.fillna("—"), hide_index=True, on_select="rerun", selection_mode="single-row", key="proc_table")
sel = event.selection.rows[0] if event.selection.rows else 0
pick = df.iloc[sel]
sel_route = ok.get(int(pick["직선순위"]))
if sel_route is None and use_tmap and app_key() and event.selection.rows:  # 고른 업체 1곳만 새로 조회
    one = run(route, (f["lat"], f["lon"]), (float(pick["위도"]), float(pick["경도"])), "공장", pick["시설명"], truck=truck)
    if one and "error" not in one:
        sel_route = one
    elif one:
        st.warning(one["error"])

m1, m2, m3 = st.columns(3)
m1.metric("사용자 공장", f["name"].replace(" (시연용 공장 위치)", ""), delta=f"{f['lat']:.5f}, {f['lon']:.5f}", delta_color="off")
m2.metric("선택한 업체", pick["시설명"], delta=pick["소재지도로명주소"] or pick["소재지지번주소"], delta_color="off")
if sel_route:
    m3.metric("화물차 도로거리", f"{sel_route['distance_km']:.1f} km",
              delta=f"약 {sel_route['minutes']:.0f}분 · 직선 {pick['직선거리_km']:.2f} km · {pick['순위']}위", delta_color="off")
else:
    m3.metric("직선거리", f"{pick['직선거리_km']:.2f} km", delta=f"{pick['순위']}위", delta_color="off")

# 점은 각 업체의 정확한 좌표에 작게 찍는다. 좌표가 완전히 같은 업체(같은 건물, 예: 2·3위 성서로72길 82)만
# 점 하나로 묶고 안에 "2, 3"처럼 번호를 함께 쓴다(2026-11-17 사용자 결정: 가까운 업체는 묶지 않음).
_k = df["위도"].round(6).astype(str) + "," + df["경도"].round(6).astype(str)
_groups = [list(ix) for ix in df.groupby(_k, sort=False).indices.values()]
pts = pd.DataFrame([{
    "위도": float(df["위도"].iloc[gr[0]]), "경도": float(df["경도"].iloc[gr[0]]),
    "label": ", ".join(str(df["순위"].iloc[k]) for k in gr),
    "순위": ", ".join(str(df["순위"].iloc[k]) for k in gr),
    "시설명": " / ".join(df["시설명"].iloc[k] for k in gr),
    "업종명": " / ".join(dict.fromkeys(df["업종명"].iloc[k] for k in gr)),
    "소재지도로명주소": str(df["소재지도로명주소"].iloc[gr[0]] or ""),
    "color": [234, 88, 12] if sel in gr else [37, 99, 235],
    "r": 110 + 25 * (len(", ".join(str(df["순위"].iloc[k]) for k in gr)) - 1),  # 반지름(m) — 확대하면 점도 커진다
} for gr in _groups])
# 한글 라벨: pydeck 은 문자열 인자를 계산식(@@=)으로 바꿔 보내서 character_set="auto" 가 함수가 되어
# TextLayer 초기화가 실패했다(라벨 안 보임, 2026-11-17 브라우저 콘솔 확인). 따옴표로 감싸 문자열 'auto' 로 넘긴다.
CHARS = '"auto"'
layers = []
if gaps:
    layers.append(pdk.Layer("ScatterplotLayer", data=gaps, get_position="[lon, lat]", get_radius=3000,
                            get_fill_color=[128, 128, 128, 70], get_line_color=[128, 128, 128, 160], stroked=True, line_width_min_pixels=1))
if sel_route:
    mid = sel_route["coords"][len(sel_route["coords"]) // 2]
    layers += [pdk.Layer("PathLayer", data=[{"path": sel_route["coords"]}], get_path="path", get_color=[234, 88, 12], width_min_pixels=4),
               pdk.Layer("TextLayer", data=[{"pos": mid, "t": f"도로 {sel_route['distance_km']:.1f} km · {sel_route['minutes']:.0f}분"}],
                         get_position="pos", get_text="t", get_size=15, get_color=[234, 88, 12], background=True,
                         get_background_color=[255, 255, 255, 230], character_set=CHARS)]
else:
    layers.append(pdk.Layer("LineLayer", data=[{"s": [f["lon"], f["lat"]], "t": [float(pick["경도"]), float(pick["위도"])]}],
                            get_source_position="s", get_target_position="t", get_color=[234, 88, 12], get_width=3))
layers += [
    pdk.Layer("ScatterplotLayer", data=pts, get_position="[경도, 위도]", get_fill_color="color", get_radius="r",
              radius_min_pixels=10, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True),
    pdk.Layer("TextLayer", data=pts, get_position="[경도, 위도]", get_text="label", get_size=120, size_units='"meters"',
              size_min_pixels=9, size_max_pixels=40, get_color=[255, 255, 255], font_weight=700, character_set='"auto"'),
    pdk.Layer("ScatterplotLayer", data=[{"lon": f["lon"], "lat": f["lat"]}], get_position="[lon, lat]",
              get_fill_color=[220, 38, 38], get_radius=200, radius_min_pixels=10),
    pdk.Layer("TextLayer", data=[{"pos": [f["lon"], f["lat"]], "t": f"공장: {f['name']}"}],
              get_position="pos", get_text="t", get_size=14, get_color=[220, 38, 38], get_pixel_offset=[0, -22],
              background=True, get_background_color=[255, 255, 255, 230], character_set=CHARS),
]
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=f["lat"], longitude=f["lon"], zoom=11),
                         map_provider="carto", map_style="light", tooltip={"text": "{순위}. {시설명}\n{업종명}\n{소재지도로명주소}"}),
                height=540)
st.caption("빨강 = 사용자 공장, 파랑 = 추천 업체(숫자 = 순위, 점 하나에 '2, 3'이면 같은 건물의 두 업체), 주황 = 선택한 업체까지 "
           + ("TMAP 화물차 경로" if sel_route else "직선") + ", 회색 원 = 자료 미제출 추정 지역(시군구 중심점).")
st.caption("폐합성수지류(51-03-01) 처리 허가 재활용업체(공공데이터). 허가 ≠ 현재 수탁 가능 — 위탁 전 업체 확인 필요. "
           f"공공데이터 기준일 {r['data_dates'][0]} ~ {r['data_dates'][1]}(지자체별 상이). 처리업체: 공공데이터포털 「전국폐기물처리업소표준데이터」. "
           "시군구 중심점: 통계청 SGIS 행정경계(공공누리 1유형) 가공본 vuski/admdongkor (CC BY 4.0). 화물차 경로: TMAP(SK open API, 티맵모빌리티). "
           "지도 배경 © OpenStreetMap contributors © CARTO")
tags(r["provenance"])
