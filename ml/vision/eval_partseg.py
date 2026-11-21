"""학습된 부품 분할 모델을 추가 평가셋으로 재평가하고 models/vision/partseg/metrics.json 에 병합.

실행: .venv/Scripts/python -m ml.vision.eval_partseg ext_carparts ext_carparts_any
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ml.vision.partseg_infer import META, MODEL_DIR, WEIGHTS
from ml.vision.train_partseg import PartSegDataset, build_model, evaluate

DATA = Path(__file__).resolve().parents[2] / "data/processed/partseg"


def main(splits: list[str]) -> dict:
    meta = json.loads(META.read_text(encoding="utf-8"))
    cm = json.loads((DATA / "classes.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(len(meta["classes"]), meta["min_size"], meta["max_size"])
    m.load_state_dict(torch.load(WEIGHTS, map_location=dev, weights_only=True))
    m.to(dev).eval()
    path = MODEL_DIR / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    for sp in splits:
        ds = PartSegDataset(DATA, sp, meta["classes"], False)
        eval_cls = meta["classes"] if sp in ("val", "test") else cm["external_eval_classes"]
        r = evaluate(m, ds, dev, meta["classes"], eval_cls, meta["service_parts"])
        metrics[f"local_eval_{sp}"] = r
        print(sp, "n", r["n_images"], "mask AP", round(r["segm"]["AP"], 3), "AP50", round(r["segm"]["AP50"], 3),
              {k: round(v, 3) for k, v in r["segm"]["per_class_AP"].items() if v is not None and k in
               ("Front-door", "Back-door", "Front-bumper", "Back-bumper", "Fender")})
        print("  presence", json.dumps(r["service_presence@0.5"]))
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
