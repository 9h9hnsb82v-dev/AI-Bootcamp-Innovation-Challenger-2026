"""결과 출처 태그 설명 (MASTER_PROMPT §8)."""

PROVENANCE = {
    "KAMP_TRAIN_BASELINE": "KAMP ID4 실제 불량 라벨의 학습 구간(제품별 층화 8:2:1 중 학습 샷)에서만 산출한 제품별 불량률(Jeffreys 사후평균)",
    "XGBOOST_SENSOR": "KAMP ID4 공정 센서 기반 샷 단위 불량확률(보조 참고, kg 아님)",
    "AIHUB_IMAGE_AI": "AI-Hub 도어/범퍼/휀더 이미지 분류 모델의 결함 의심 확률(폐기 확정 아님)",
    "DEMO_ASSUMED_RATE": "사용자 승인 시연용 가상 불량률(도어 1.5%, 범퍼 2.0%, 휀더 2.5%)",
    "DEMO_SYNTHETIC_WEIGHT": "사용자 승인 가상 단중(실측 아님)",
    "DEMO_OPERATOR_CONFIRMATION": "시연 작업자의 최종 폐기/양품 판정",
    "DEMO_SYNTHETIC_BRIDGE": "서로 다른 데이터셋을 사용자가 명시적으로 연결한 가상 시나리오(실제 연계 아님)",
    "KAMP_ID40_PLANNED_ORDER_FORECAST": "KAMP ID40 비식별 제품의 'T일 예정 수주량' 예측(개수, 확정 주문 아님)",
    "KAMP_ID40_BACKTEST": "2022년 KAMP ID40 원본 스냅샷 기반 시간순 홀드아웃 백테스트",
    "UNVERIFIED_FUTURE_INPUT": "사용자가 직접 입력한, 데이터셋으로 검증되지 않은 미래 입력값",
    "DEMO_ASSUMED": "사용자 승인 가상 창고·수거 정책(법령/실사업장 기준 아님)",
    "LIVE_PREVIEW_ONLY": "카메라 실시간 미리보기 판정 - DB 기록·kg 변경 없음. '이 장면 등록'을 눌러야 검사 기록이 생성됨",
    "PUBLIC_DATA_WASTE_PROCESSORS": "공공데이터포털 전국폐기물처리업소표준데이터(지자체 제출 자료, 기준일 지자체별 상이) - 허가 정보이며 현재 수탁 가능 여부는 아님",
    "USER_SET_FACTORY_LOCATION": "사용자가 화면에서 정한 공장 위치(브라우저 위치·TMAP 검색·직접 입력). 브라우저 위치는 PC에서 수십~수천 m 오차가 날 수 있음",
    "USER_SPECIFIED_DEMO_LOCATION": "사용자가 정한 시연용 공장 위치(계명대학교 성서캠퍼스 대표 좌표, 오차 약 1km) - 실제 공장 아님",
    "GAP_GUIDE_UNVALIDATED": "휀더 틈(단차) 가이드 촬영 판정 - 단차 모델은 AI-Hub 사진의 단차 부위로 학습·평가했고, 실제 카메라 가이드 촬영 성능은 검증되지 않음",
    "CAMERA_UNVALIDATED": "카메라로 찍은 사진 - 학습 사진(AI-Hub 근접 정지 사진)과 촬영 조건이 달라 판정 성능이 검증되지 않음",
    "PUBLIC_RECYCLING_PRICE_SURVEY": "재활용가능자원 가격조사 간행물(국가승인통계 제392001호, 2025-01~2026-10호) - 원/kg, VAT 포함·물류비 제외, 가공업체 간 판매가(폐지·철스크랩은 매입가). 공장이 받는 값·처리비 아님",
    "PARTSEG_GUIDE_ONLY": "차량 전체 사진의 부품 영역 분할(Mask R-CNN, HITL CC0 데이터) - 근접 촬영 안내용, kg·판정에 미반영",
}


def get_provenance(tag: str | None = None) -> dict:
    if tag:
        return {tag: PROVENANCE.get(tag, "UNKNOWN_TAG")}
    return PROVENANCE
