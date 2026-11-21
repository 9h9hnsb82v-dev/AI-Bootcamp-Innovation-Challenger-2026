#!/usr/bin/env bash
# 대회 체험용 우리 전용 Ollama (2026-11-20). 팀 공용 Ollama(11434, systemd) 설정은 건드리지 않는다.
#   시작: tmux new -d -s resource-ollama "SLOTS=2 ~/resource_predict_v3_aml/app/scripts/run_event_ollama.sh"
#   중지: tmux kill-session -t resource-ollama
# 공개 앱(8603)만 OLLAMA_BASE_URL=http://127.0.0.1:11435 로 여기에 연결한다.
# 모델 파일은 공용 폴더를 읽기만 한다(OLLAMA_NOPRUNE). 동시 처리 칸 수 = SLOTS(기본 2, 2026-11-21 실측: 10명 동시 시 가장 늦은 사람 2칸 61초·3칸 57초로 차이 작음 → GPU 여유가 큰 2칸, 9.4/12.3GB).
set -euo pipefail
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
export OLLAMA_NOPRUNE=1
export OLLAMA_NUM_PARALLEL="${SLOTS:-2}"
export OLLAMA_FLASH_ATTENTION=1        # 메모리 절약 계산
export OLLAMA_KV_CACHE_TYPE=q8_0       # 대화 기억(KV) 메모리 절반
export OLLAMA_MAX_LOADED_MODELS=2      # 챗봇(GPU) + 임베딩(CPU, num_gpu=0)
export OLLAMA_KEEP_ALIVE=60m
exec /usr/local/bin/ollama serve
