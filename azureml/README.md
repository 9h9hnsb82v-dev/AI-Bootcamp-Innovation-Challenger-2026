# Azure ML 실행 기록

사용자 승인: 2026-11-10. 워크스페이스 `<workspace>`(RG `<resource-group>`, eastus2, 구독 <subscription>)

| 대상 | 컴퓨트 | 실행 | 등록 모델 |
|---|---|---|---|
| KAMP ID4 감사·불량률 기준선·센서 XGBoost / KAMP ID40 감사·수주량 학습·백테스트 | `resource-cpu` (E4ds_v4, 0~1대, 자동 축소) | 작업 `gentle_napkin_1zr71t8xd4` (Completed) | `resource-kamp4-defect-baseline:1`, `resource-kamp40-order-forecast:1` |
| AI-Hub 이미지 EfficientNet-B0 6클래스(448px) | 팀 VM **AICD** (Standard_NV18ads_A10_v5, East US, 팀 RG) — Azure ML 작업 아님 | `~/resource_predict_v3_aml` (azureuser) | `resource-aihub-vision-effb0` (워크스페이스 MLflow run + 레지스트리) |
| 근접 사진 판정 비교 실험: YOLOv7(30/100에폭, 3클래스+IW), YOLO11 분류 | `gpu-h100` (NC40ads_H100_v5, 0~1대) | `jovial_rhythm_vqdl97xg48`, `loving_battery_yz0wnt613x`, `orange_dog_wlp03pwlg0`, `upbeat_hamster_gfyfk0s891` | 등록 안 함(비교용, YOLO11은 AGPL - 내부 전용) |
| 부품 위치 Mask R-CNN v1 / v2 (이전 8:1:1 분할) | `gpu-h100` | `gifted_car_k7wvdmbd22`, `silly_crayon_8gv9w6n8rf` | 로컬 백업 `models/vision/partseg_v2_split811` |
| **8:2:1 재분할(2026-11-11)** EfficientNet-B0 / YOLO11-cls / YOLOv7 100에폭 | `gpu-h100` | `wheat_worm_c84pn8qwvr`, `dreamy_brain_29f7wfqkb5`, `teal_avocado_jn1z0gjfq5`, YOLO11m 50에폭 `dynamic_melon_32s4hz0mfc` | EfficientNet만 로컬 서비스 `models/vision/efficientnet_b0_6cls.pt`(`aihub_effb0_6cls_img448_exif_split821`). YOLO 계열은 비교용으로만 쓰고 등록하지 않음 |
| **8:2:1** 부품 위치 v1' / v2' | `gpu-h100` | `maroon_shoe_73kxh8m8ln`, `lucid_train_n7h9lz5w1z` | 로컬 서비스 `models/vision/partseg` (`partseg_maskrcnn_v2_hitl_cp_split821`) |
| **8:2:1** 모델 성능표(사후 기록, 학습 없음) | 로컬 → MLflow | run `resource-vision-scoreboard-split821` + 각 작업 `*/test_*` 지표 (`azureml/log_model_scoreboard.py`), 에폭별 곡선 (`azureml/backfill_epoch_metrics.py`) | — |
| **8:2:1** KAMP ID4 층화 / ID40 22·6·3일 재현 | `resource-cpu` | `keen_ball_j087lsxk04` | `resource-kamp4-defect-baseline:2`, `resource-kamp40-order-forecast:2` |

| **세션 비겹침** DINOv2 ViT-B/14 미세조정 (2026-11-18 사용자 승인, YOLO11m-cls 와 같은 조건 448px·15에폭) | `gpu-h100` | `good_bean_wcbrhhynb9` (46분, 2시간 상한) | 등록 안 함. 로컬 `models/vision/dinov2_vitb14_sd_e15` (Apache-2.0). 비교 `reports/vision/dino_sd_eval.md` |

| **세션 비겹침** DINOv2 재학습 v2 (2026-11-19 사용자 승인, 학습 곡선 지표 전부 기록) | `gpu-h100` | `frank_drain_wj7y3szqbm` | 등록 안 함. 로컬 `models/vision/dinov2_vitb14_sd_e15_v2` (그래프용, 앱은 v1 유지) |
| **세션 비겹침** 휀더 단차 DINOv2 (2026-11-19 사용자 승인, 단차 EfficientNet과 같은 조건) | `gpu-h100` | `crimson_hamster_g7c4qxkk13` (약 11분) | 등록 안 함. 로컬 `models/vision/gap_dinov2_sd`. 비교 `reports/vision/gap_dino_eval.md` |
| **세션 비겹침** ConvNeXt-Small (2026-11-19 사용자 승인, DINOv2와 같은 조건) | `gpu-h100` | `keen_rhubarb_zg7x46gs5g` (40분) | 등록 안 함. 로컬 `models/vision/convnext_small_sd_e15` (Apache-2.0 가중치). 비교 `reports/vision/convnext_compare.md` |

## 재현
```powershell
# 환경(1회): Python 3.11 + 로컬과 같은 버전의 패키지
az ml environment create -f azureml/environment/environment.yml -w <workspace> -g <resource-group>
# 표 형식 학습 작업(코드는 .amlignore를 적용해 업로드, KAMP ZIP 2개는 job inputs로 업로드)
az ml job create -f azureml/job_tabular.yml -w <workspace> -g <resource-group>
# 이미지(팀 VM): 번들 생성 → scp → VM에서 학습 → 결과를 워크스페이스에 기록
.venv\Scripts\python -m scripts.bundle_vision_for_vm <bundle.tar>
.venv-aml\Scripts\python azureml/log_external_vision_run.py <모델 폴더>
```

## 검증 결과
- 작업 `gentle_napkin_1zr71t8xd4`: 원본 해시와 분할이 로컬과 같다. HGB와 기준선 지표는 로컬과 완전히 같고, horizon별 모델 선택도 같다.
  XGB의 WAPE는 ±0.001~0.009 범위에서 다르다(Linux와 Windows 간 멀티스레드 비결정성). KAMP 불량률은 완전히 같다.
- `azureml-mlflow`는 `mlflow-skinny<=3.15`가 필요해 앱 venv(mlflow 3.18)와 충돌한다. 그래서 워크스페이스 기록 전용 `.venv-aml`을 따로 둔다.

## 비용·정리 메모
- `resource-cpu`는 유휴 시 0대로 축소된다.
- `gpu-h100` (Standard_NC40ads_H100_v5, 0~1대, 유휴 300초 후 0대): 위 표의 실험에 사용했다. 유휴 시 비용 없음. 필요 없으면 삭제한다:
  `az ml compute delete -n gpu-h100 -w <workspace> -g <resource-group>`
- 팀 VM AICD는 팀 리소스다. 이 프로젝트는 `~/resource_predict_v3_aml` 폴더만 사용하고, 접속에는 이 PC의 전용 키 `~/.ssh/aicd_ed25519`를 쓴다.
