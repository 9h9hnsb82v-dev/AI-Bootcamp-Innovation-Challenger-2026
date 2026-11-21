# Re:Source Predict AI v3.0

확정 생산계획 기반 폐플라스틱 발생량(kg) 예측, 이미지 검사 후 예측 갱신, 보관·수거 일정 추천을 한 서비스로 묶었다. 여기에 독립 기능으로 KAMP ID 40 미래 수주량 예측을 붙였다.
최상위 사양은 [MASTER_PROMPT.md](MASTER_PROMPT.md)다. 한국환경공단 인계서 분석은 범위에서 제외했다([reports/migration_review.md](reports/migration_review.md)).

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
az ml job create -f azureml/job_partseg_h100_v2.yml -w AIBIC_10_24_workspace -g stu_AIBIC2026_15_RG_EUS2
```

## 데이터 출처 표기
- Car parts segmentation: Gianmarco Russo, "car-seg", Roboflow Universe (2023), **CC BY 4.0**, Ultralytics "carparts-seg"로 배포
  (https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm). 리사이즈·클래스 매핑·의사 라벨을 추가해 사용했다.
- Humans in the Loop "Car parts and car damages dataset", CC0 1.0
- AI-Hub "부품 품질 검사 영상 데이터(자동차)", KAMP ID 4·ID 40: 각 제공처 이용약관에 따름
