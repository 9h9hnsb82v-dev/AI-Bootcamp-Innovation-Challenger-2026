#!/usr/bin/env bash
# Azure ML 작업 본문: YOLOv7 학습 → 시험셋 mAP → 이미지별 점수 → MLflow 기록 → 출력 폴더 복사
# 사용: bash azureml/yolo/run_yolov7.sh <dataset_dir> <output_dir> <epochs> <batch>
# 선택 환경변수: YOLO_DROP_GAP_NORMAL=1 (gap_normal 박스 제거 → 3클래스, 양품=배경)
#               YOLO_IMAGE_WEIGHTS=1   (--image-weights, 박스 없는 이미지는 평균 가중치로 보정)
set -euo pipefail
DATA_IN="$1"; OUT="$2"; EPOCHS="${3:-30}"; BATCH="${4:-32}"
CODE="$(pwd)"
WORK=/tmp/yolo_work
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # YOLOv7 체크포인트(pickle) 로딩 — torch 2.6 기본값 대응
mkdir -p "$WORK" "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cd "$WORK"
git clone -q https://github.com/WongKinYiu/yolov7.git && cd yolov7 && git checkout -q a207844
wget -q https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
# 라벨 캐시(.cache)를 쓰므로 읽기 전용 입력을 로컬 디스크로 복사
cp -r "$DATA_IN" data/aihub_yolo
DROP="${YOLO_DROP_GAP_NORMAL:-0}"; IW="${YOLO_IMAGE_WEIGHTS:-0}"; NC=4; EXTRA=""
sed -i 's/\r$//' data/aihub_yolo/data.yaml   # Windows CRLF 제거
if [ "$DROP" = "1" ]; then
  find data/aihub_yolo/labels -name "*.txt" -exec sed -i '/^3 /d' {} +
  sed -i -e "s/^nc: 4/nc: 3/" -e "s/^names:.*/names: ['scratch', 'exterior_damage', 'gap_defect']/" data/aihub_yolo/data.yaml
  NC=3
fi
if [ "$IW" = "1" ]; then
  # 원본은 박스 없는 이미지 가중치가 0 → 배경(양품)이 샘플링되지 않음. 박스 있는 이미지 평균 가중치를 부여
  python - <<'PY'
from pathlib import Path
p = Path("train.py"); s = p.read_text(encoding="utf-8")
old = "iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights"
assert old in s
p.write_text(s.replace(old, old + "\n" + " " * 16 + "iw = np.where(iw > 0, iw, iw[iw > 0].mean())  # background images keep mean weight"), encoding="utf-8")
PY
  EXTRA="--image-weights"
fi
cp data/aihub_yolo/data.yaml data/aihub_yolo.yaml
cat data/aihub_yolo.yaml
sed "s/^nc: 80/nc: $NC/" cfg/training/yolov7.yaml > cfg/training/yolov7-aihub.yaml
echo "labels: $(cat data/aihub_yolo/labels/train/*.txt | cut -d' ' -f1 | sort | uniq -c | tr '
' ' ')"
echo "images: train $(ls data/aihub_yolo/images/train | wc -l) val $(ls data/aihub_yolo/images/val | wc -l) test $(ls data/aihub_yolo/images/test | wc -l)"

T0=$(date +%s)
python train.py --workers 16 --device 0 --batch-size "$BATCH" --data data/aihub_yolo.yaml --img 640 640 \
  --cfg cfg/training/yolov7-aihub.yaml --weights yolov7_training.pt --name aihub_yolov7 \
  --hyp data/hyp.scratch.custom.yaml --epochs "$EPOCHS" --cache-images --exist-ok $EXTRA 2>&1 | tr '\r' '\n' | grep -vE "it/s\]$|s/it\]$" || true
T1=$(date +%s)
W=runs/train/aihub_yolov7/weights/best.pt
test -f "$W"

python test.py --task test --data data/aihub_yolo.yaml --weights "$W" --img 640 --batch 32 --conf 0.001 --iou 0.65 \
  --name aihub_test --exist-ok --verbose 2>&1 | tee "$OUT/test_map.txt"
python "$CODE/ml/vision/yolo_image_scores.py" --weights "$W" --data-root ./data/aihub_yolo --img 640 --out "$OUT/yolo_scores.csv"

mkdir -p "$OUT/run"
cp -r runs/train/aihub_yolov7/* "$OUT/run/"
find "$OUT/run/weights" -name "*.pt" ! -name "best.pt" ! -name "last.pt" -delete   # 에폭별 체크포인트 제외(출력 용량 절감)
echo "{\"train_seconds\": $((T1 - T0)), \"epochs\": $EPOCHS, \"batch\": $BATCH, \"img\": 640, \"commit\": \"a207844\", \"weights_init\": \"yolov7_training.pt\", \"nc\": $NC, \"drop_gap_normal\": $DROP, \"image_weights\": $IW}" > "$OUT/yolo_meta.json"
python "$CODE/ml/vision/log_yolo_results.py" "$OUT"
echo "DONE"
