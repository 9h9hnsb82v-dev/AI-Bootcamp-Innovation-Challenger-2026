# Loop:in

> **대회 제출용 공개 저장소** — AI Bootcamp Innovation Challenger 2026 
>
> - **라이선스:** [GNU AGPL-3.0](LICENSE). 결함 위치 박스 표시에 쓰는 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)(AGPL-3.0)에 맞춰 이 저장소 전체를 AGPL-3.0으로 공개합니다.
> - **포함한 모델:** `models/kamp`(KAMP 학습 구간 불량률 집계·센서 모델), `models/supply`(수주량 모델·성능 지표). 그래서 내려받은 그대로도 생산계획 등록 → 폐기물 kg 예측 → 보관량 예측·수거 추천이 동작합니다.
> - **포함하지 않은 것:** 학습·평가 데이터(AI-Hub 자동차 부품 품질검사 영상, KAMP 원본 — 각 기관 이용약관상 재배포 불가), AI-Hub로 학습한 이미지 모델 가중치(약관 확인 중), 실제 수주량이 담긴 백테스트 결과, 챗봇 문서 원문, 비밀 설정(`secrets.toml`).
>   - 이미지 판정·부품 위치 찾기 → `MODEL_NOT_READY` 표시
>   - 미래 수주량 예측 → KAMP ID40 원본(`data/raw`)이 있어야 동작 (KAMP에서 직접 내려받기)
>   - 처리업체 추천 → 공공데이터포털 '전국폐기물처리업소표준데이터' CSV를 `data/raw/waste_processors/`에 넣어야 동작
>   - 이미지 판정까지 포함한 완성본은 발표 자료의 **공개 주소(QR)**에서 바로 체험할 수 있습니다.

확정 생산계획 기반 폐플라스틱 발생량(kg) 예측, 이미지 검사 후 예측 갱신, 보관·수거 일정 추천을 한 서비스로 묶었다. 여기에 독립 기능으로 KAMP ID 40 미래 수주량 예측을 붙였다.
최상위 사양은 팀 내부 문서(`MASTER_PROMPT.md`, 공개본 미포함)로 관리했다. 한국환경공단 인계서 분석은 범위에서 제외했다.

## 빠른 시작 (Windows, Python 3.11)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # torch는 requirements 주석의 CUDA 인덱스 사용
.venv\Scripts\python -m pip install --no-deps -r requirements-yolo11.txt   # 근접 사진 판정 YOLO11(AGPL-3.0, 내부 전용). 없으면 MODEL_NOT_READY
# 1) 데이터 감사
.venv\Scripts\python -m ml.supply.audit          # reports/audit/kamp_id40_audit.json
.venv\Scripts\python -m ml.kamp.audit            # reports/audit/kamp_id4_audit.json
.venv\Scripts\python -m ml.vision.audit_aihub_labels
# 2) 학습 (모두 로컬, 유료 리소스 없음)
.venv\Scripts\python -m ml.kamp.baseline         # KAMP 학습구간 불량률
.venv\Scripts\python -m ml.kamp.train_xgb        # 보조 센서 모델(참고용)
.venv\Scripts\python -m ml.supply.train          # ID40 수주량 baseline/HGB/XGB 백테스트
.venv\Scripts\python -m ml.vision.prepare_subset # AI-Hub tar → 축소 서브셋
.venv\Scripts\python -m ml.vision.train --epochs 8
# 3) 테스트 / 실행
.venv\Scripts\python -m pytest -q
.venv\Scripts\streamlit run ui/app.py
# 4) 발표 시연용 데이터(별도 DB)
.venv\Scripts\python -m scripts.demo_scenario
$env:RESOURCE_DB="data/app/demo.db"; .venv\Scripts\streamlit run ui/app.py
```
Ollama(선택): 기본 모델 `benedict/linkbricks-llama3.1-korean:8b`(평가 8문항 도구·인자·답변 충실도 100%)(도구 호출 평가: `python -m agent.eval_tool_calls <모델>`). 설치 후 `.env`에 `OLLAMA_BASE_URL`, `OLLAMA_MODEL`을 설정한다. Ollama 없이도 모든 폼은 동작한다.

## 데이터·모델 준비 (직접 받아 재현)

이 저장소에는 데이터와 이미지 모델 가중치를 넣지 않았습니다(각 제공처 약관·용량). 아래 경로에 받아 두면 전체 기능이 동작합니다.

| 데이터 | 받는 곳 | 넣을 위치 |
|---|---|---|
| KAMP ID 4 사출성형 공정·불량 데이터 | [KAMP 인공지능 중소벤처 제조 플랫폼](https://www.kamp-ai.kr) | `data/raw/kamp/04. Dataset_Molding.zip` |
| KAMP ID 40 사출성형 공급망최적화 AI 데이터셋 | KAMP | `data/raw/kamp_supply/Dataset_사출성형 공급망최적화 AI 데이터셋.zip` |
| 부품 품질 검사 영상 데이터(자동차) | [AI-Hub](https://www.aihub.or.kr) (데이터셋 162) | `data/raw/aihub/162.부품 품질 검사 영상 데이터(자동차)/` |
| Car parts segmentation (car-seg, CC BY 4.0) | Roboflow Universe · Gianmarco Russo | `data/raw/carparts_seg/carparts-seg.zip` |
| Car parts and car damages dataset (CC0 1.0) | Humans in the Loop | `data/raw/hitl_carparts/` |
| 전국폐기물처리업소표준데이터 | [공공데이터포털](https://www.data.go.kr) | `data/raw/waste_processors/전국폐기물처리업소표준데이터.csv` |
| 행정경계 중심점 (CC BY 4.0) | vuski/admdongkor (통계청 SGIS 가공본) | `data/raw/admin_centroids/coordinate_UTMK_names.tsv` |
| 챗봇 문서(법령) | 국가법령정보센터 | `python -m scripts.fetch_laws` |
| 챗봇 문서(재활용가능자원 가격조사) | 한국환경공단 순환자원정보센터 | `python -m scripts.build_price_docs` |
| TMAP 앱 키(처리업체 도로거리) | SK open API | `.streamlit/secrets.toml` (`secrets.toml.example` 참고) |

**모델 다시 만들기**
- KAMP 불량률·수주량: 저장소에 포함(`models/kamp`, `models/supply`). 다시 만들려면 `python -m ml.kamp.baseline`, `python -m ml.supply.train`.
- 이미지 판정(EfficientNet-B0 + DINOv2 + 2차 판정): 세션 비겹침 분할 `python -m ml.vision.resplit_session` → 학습 작업 `azureml/job_sd_effnet.yml`, `azureml/job_sd_dinov2_cls.yml`, `azureml/job_sd_gap.yml`, `azureml/job_sd_gap_dinov2.yml` → 조합 `python -m ml.vision.fit_dino_ensemble`, `python -m ml.vision.fit_gap_dino` → `models/vision/ensemble_sd_dino/`.
- 결함 위치 박스(YOLO11 검출): `azureml/job_sd_det.yml` → `models/vision/yolo11m_det/`.
- 부품 위치(Mask R-CNN): `python -m ml.vision.prepare_partseg` → `azureml/job_partseg_h100_v2.yml` → `models/vision/partseg/`.
- Azure ML 작업 파일의 `<workspace>`, `<resource-group>`은 각자의 작업 공간으로 바꿔 실행합니다.

## 데이터 경계 (코드로 강제)
| 트랙 | 모듈 | 산출 단위 | 연결 규칙 |
|---|---|---|---|
| K: KAMP ID 4 | `ml/kamp`, `services/waste_forecast.py` | kg | CN7/RG3 4종 |
| V: AI-Hub | `ml/vision`, `services/inspection_registry.py` | 판정 → kg | DOOR/BUMPER/FENDER 계획만 갱신. 판정 모델: `models/vision/serving.json` = `ensemble`(원본 전체 51,986장으로 재학습한 EfficientNet-B0 + YOLO11m-cls 조합, 엇갈리면 `AI_DISAGREE` 작업자 확인, YOLO11은 AGPL 내부 전용). `RESOURCE_VISION_BACKEND`로 `effnet_full`·`yolo11m_full`·`yolo11m_cls`·`effnet` 전환. 사진 입력: 파일 업로드 · 카메라 촬영 · 실시간 미리보기(`ui/live_camera.py`, 미리보기는 기록 없음 `LIVE_PREVIEW_ONLY`, '이 장면 등록' 한 장만 기록) |
| P: 부품 위치(선택) | `ml/vision/partseg_infer.py`, `services/part_locator.py` | 안내만 | 차량 전체 사진 → 부품 영역 표시. DB·kg 불변(`PARTSEG_GUIDE_ONLY`) |
| S: KAMP ID 40 | `ml/supply`, `services/supply_forecast.py` | **개(units)** | 7종과 자동 연결 금지. `supply_to_plan_draft.py`: 경고 확인 → 사용자가 품목·수량·날짜 선택 → 초안 → 재확정 |

- `create_production_plan()`은 7종이 아닌 ID(`Product_xx`)를 거부한다. `DEMO_SUPPLY_BRIDGE` 계획은 초안 확정 경로로만 만들 수 있다.
- 에이전트의 상태 변경 도구는 `PENDING_USER_CONFIRMATION`만 반환한다. 실제 실행은 UI 버튼을 눌러야 한다.
- 수주예측 결과는 `stock_ledger`나 kg 계산에 들어가지 않는다.

## 주요 산출물
- `reports/data_audit_summary.md`: 세 데이터셋 감사 결과와 확정 결정사항
- `models/kamp/baseline_rates.json`, `models/supply/metrics.json`, `models/vision/meta.json`: 모델 지표
- `mlflow.db`: 로컬 MLflow(SQLite). `mlflow ui --backend-store-uri sqlite:///mlflow.db`로 연다
- `reports/vision/part_segmentation.md`: ① 차량 전체 사진 부품 위치 → ② 근접 사진 결함 판정 흐름, 부품 분할 모델 지표·한계
- `reports/vision/model_comparison.md`: 근접 사진 판정 모델 비교(EfficientNet / YOLOv7 / YOLO11 분류)
- 시연 사진: `python -m scripts.demo_scenario` 실행 시 `data/demo/photos/{car,closeup}/` 생성

## 부품 위치 모델 재현(선택)
```powershell
.venv\Scripts\python -m ml.vision.prepare_partseg          # HITL(CC0) + carparts-seg(CC BY 4.0) → data/processed/partseg
.venv\Scripts\python -m ml.vision.pseudo_label_carparts    # v1 모델로 carparts 절반에 의사 라벨(휀더 등) 추가
az ml job create -f azureml/job_partseg_h100_v2.yml -w <workspace> -g <resource-group>
```

## 데이터 출처 표기
- Car parts segmentation: Gianmarco Russo, "car-seg", Roboflow Universe (2023), **CC BY 4.0**, Ultralytics "carparts-seg"로 배포
  (https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm). 리사이즈·클래스 매핑·의사 라벨을 추가해 사용했다.
- Humans in the Loop "Car parts and car damages dataset", CC0 1.0
- AI-Hub "부품 품질 검사 영상 데이터(자동차)", KAMP ID 4·ID 40: 각 제공처 이용약관에 따름
