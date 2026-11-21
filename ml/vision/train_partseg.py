"""1단계 부품 영역 분할: torchvision Mask R-CNN (ResNet50-FPN v2, COCO 사전학습, BSD) 미세조정.

데이터: ml.vision.prepare_partseg 산출물 (HITL CC0 학습/검증/시험 + carparts-seg 외부 검증)
평가: COCO mask/box AP(전체·클래스별), 서비스 품목(DOOR/BUMPER/FENDER) 사진 단위 존재 판정 정밀도·재현율.
      최적 에폭은 검증 mask AP로 고르고, 시험셋·외부셋은 최종 1회만 평가한다.
실행(로컬 스모크): python -m ml.vision.train_partseg --data data/processed/partseg --out models/vision/partseg_smoke --epochs 1 --limit 6 --device cpu --min-size 320
실행(Azure): azureml/job_partseg_h100.yml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as TF

PRESENCE_SCORE = 0.5


def rasterize(obj: dict, w: int, h: int) -> np.ndarray:
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.polygon([tuple(p) for p in obj["exterior"]], fill=1)
    for hole in obj.get("interiors", []):
        if len(hole) >= 3:
            d.polygon([tuple(p) for p in hole], fill=0)
    return np.array(im, dtype=np.uint8)


class PartSegDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, classes: list[str], train: bool, limit: int | None = None):
        self.root, self.train, self.classes = root, train, classes
        self.recs = []
        for s in split.split(","):  # 쉼표로 여러 분할 결합(예: "train,cp_train")
            self.recs += json.loads((root / f"ann_{s}.json").read_text(encoding="utf-8"))
        if limit:
            self.recs = self.recs[:limit]

    def __len__(self):
        return len(self.recs)

    def target(self, i: int):
        r = self.recs[i]
        masks, labels = [], []
        for o in r["objects"]:
            if o["label"] not in self.classes:
                continue
            m = rasterize(o, r["width"], r["height"])
            if m.sum() < 4:
                continue
            masks.append(m)
            labels.append(self.classes.index(o["label"]) + 1)
        if masks:
            mk = torch.from_numpy(np.stack(masks))
            boxes = []
            for m in mk:
                ys, xs = torch.where(m > 0)
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            bx = torch.tensor(boxes, dtype=torch.float32)
        else:
            mk = torch.zeros((0, r["height"], r["width"]), dtype=torch.uint8)
            bx = torch.zeros((0, 4), dtype=torch.float32)
        return {"boxes": bx, "labels": torch.tensor(labels, dtype=torch.int64), "masks": mk}

    def __getitem__(self, i: int):
        r = self.recs[i]
        img = TF.to_tensor(Image.open(self.root / r["file"]).convert("RGB"))
        t = self.target(i)
        if self.train:
            if random.random() < 0.5:  # 좌우 반전: HITL 클래스에 좌/우 구분이 없어 라벨 변경 불필요
                img = img.flip(-1)
                t["masks"] = t["masks"].flip(-1)
                w = img.shape[-1]
                if len(t["boxes"]):
                    t["boxes"] = torch.stack([w - t["boxes"][:, 2], t["boxes"][:, 1], w - t["boxes"][:, 0], t["boxes"][:, 3]], 1)
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        t["image_id"] = i
        return img, t


def collate(b):
    return tuple(zip(*b))


def build_model(n_classes: int, min_size: int, max_size: int):
    m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1, min_size=min_size, max_size=max_size)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, n_classes + 1)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, n_classes + 1)
    return m


def gt_coco(ds: PartSegDataset, cat_ids: list[int]):
    from pycocotools import mask as mask_utils  # 평가 전용 의존성(추론 서비스에는 불필요)
    from pycocotools.coco import COCO
    images, anns, k = [], [], 1
    for i, r in enumerate(ds.recs):
        images.append({"id": i, "width": r["width"], "height": r["height"]})
        t = ds.target(i)
        for m, l in zip(t["masks"].numpy(), t["labels"].tolist()):
            if l not in cat_ids:
                continue
            rle = mask_utils.encode(np.asfortranarray(m))
            area = float(mask_utils.area(rle))
            bb = mask_utils.toBbox(rle).tolist()
            rle["counts"] = rle["counts"].decode()
            anns.append({"id": k, "image_id": i, "category_id": l, "segmentation": rle, "area": area, "bbox": bb, "iscrowd": 0})
            k += 1
    c = COCO()
    c.dataset = {"images": images, "annotations": anns, "categories": [{"id": j} for j in cat_ids]}
    c.createIndex()
    return c


@torch.no_grad()
def predict_all(model, ds, device, bs=4):
    model.eval()
    out = []
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=2)
    for imgs, tg in dl:
        preds = model([x.to(device) for x in imgs])
        for p, t in zip(preds, tg):
            out.append((t["image_id"], {k: v.cpu() for k, v in p.items()}))
    return out


def evaluate(model, ds: PartSegDataset, device, classes: list[str], eval_classes: list[str], service_parts: dict) -> dict:
    from pycocotools import mask as mask_utils
    from pycocotools.cocoeval import COCOeval
    cat_ids = [classes.index(c) + 1 for c in eval_classes]
    gt = gt_coco(ds, cat_ids)
    preds = predict_all(model, ds, device)
    dets = []
    for iid, p in preds:
        for m, s, l in zip(p["masks"], p["scores"].tolist(), p["labels"].tolist()):
            if l not in cat_ids:
                continue
            rle = mask_utils.encode(np.asfortranarray((m[0] > 0.5).numpy().astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            dets.append({"image_id": iid, "category_id": l, "segmentation": rle, "score": s})
    res = {"n_images": len(ds)}
    for iou_type in ("segm", "bbox"):
        if not dets:
            res[iou_type] = {"AP": 0.0, "AP50": 0.0, "per_class_AP": {}}
            continue
        if iou_type == "segm":
            cur = [dict(d) for d in dets]  # loadRes가 입력을 변경하므로 복사본 사용
        else:
            cur = [{"image_id": d["image_id"], "category_id": d["category_id"], "score": d["score"],
                    "bbox": mask_utils.toBbox(d["segmentation"]).tolist()} for d in dets]
        dt = gt.loadRes(cur)
        E = COCOeval(gt, dt, iou_type)
        E.params.catIds = cat_ids
        E.evaluate(); E.accumulate(); E.summarize()
        prec = E.eval["precision"]  # [T,R,K,A,M]
        per = {}
        for ki, cid in enumerate(cat_ids):
            pk = prec[:, :, ki, 0, -1]
            per[classes[cid - 1]] = float(pk[pk > -1].mean()) if (pk > -1).any() else None
        res[iou_type] = {"AP": float(E.stats[0]), "AP50": float(E.stats[1]), "per_class_AP": per}
    # 서비스 품목 사진 단위 존재 판정
    pmap = {iid: p for iid, p in preds}
    sp = {}
    for part, cls_names in service_parts.items():
        ids = [classes.index(c) + 1 for c in cls_names if c in eval_classes]
        if not ids:
            continue
        tp = fp = fn = tn = 0
        for i in range(len(ds)):
            g = any(l in ids for l in ds.target(i)["labels"].tolist())
            p = pmap[i]
            d = any(l in ids and s >= PRESENCE_SCORE for l, s in zip(p["labels"].tolist(), p["scores"].tolist()))
            tp += g and d; fp += (not g) and d; fn += g and (not d); tn += (not g) and (not d)
        sp[part] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    res["service_presence@0.5"] = sp
    return res


def main(a) -> dict:
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "classes.json").read_text(encoding="utf-8"))
    classes, service_parts = meta["classes"], meta["service_parts"]
    device = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tr = PartSegDataset(root, a.train_splits, classes, True, a.limit)
    va = PartSegDataset(root, "val", classes, False, a.limit)
    model = build_model(len(classes), a.min_size, a.max_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, collate_fn=collate, num_workers=a.workers, drop_last=True)
    total = a.epochs * max(len(dl), 1)
    warm = min(300, total // 5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    hist, best, best_ep = [], -1.0, -1
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        losses = []
        for imgs, tg in dl:
            imgs = [x.to(device) for x in imgs]
            tg = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tg]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ld = model(imgs, tg)
                loss = sum(ld.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss))
        h = {"epoch": ep, "train_loss": float(np.mean(losses)) if losses else None, "sec": round(time.time() - t0, 1)}
        if ep % a.eval_every == 0 or ep == a.epochs:
            v = evaluate(model, va, device, classes, classes, service_parts)
            h["val_mask_AP"], h["val_mask_AP50"] = v["segm"]["AP"], v["segm"]["AP50"]
            h["val_fender_mask_AP"] = v["segm"]["per_class_AP"].get("Fender")
            if v["segm"]["AP"] > best:
                best, best_ep = v["segm"]["AP"], ep
                torch.save(model.state_dict(), out / "maskrcnn_partseg.pt")
        hist.append(h)
        print(json.dumps(h), flush=True)
    train_s = time.time() - t0
    model.load_state_dict(torch.load(out / "maskrcnn_partseg.pt", map_location=device, weights_only=True))
    te = PartSegDataset(root, "test", classes, False, a.limit)
    ex = PartSegDataset(root, a.ext_split, classes, False, a.limit)
    res = {"model": "torchvision maskrcnn_resnet50_fpn_v2 (COCO_V1) fine-tuned", "license": "torchvision BSD-3; data HITL CC0 1.0",
           "train_splits": a.train_splits, "ext_split": a.ext_split, "n_train_images": len(tr),
           "classes": classes, "service_parts": service_parts, "min_size": a.min_size, "max_size": a.max_size,
           "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "best_epoch": best_ep, "train_seconds": round(train_s, 1),
           "history": hist,
           "test": evaluate(model, te, device, classes, classes, service_parts),
           "external_carparts": evaluate(model, ex, device, classes, meta["external_eval_classes"], service_parts)}
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({k: res[k] for k in ("model", "license", "classes", "service_parts", "min_size", "max_size",
                                                                   "best_epoch", "train_splits", "n_train_images")},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("TEST", json.dumps({k: res["test"][k] for k in ("segm", "service_presence@0.5")}, ensure_ascii=False))
    print("EXT", json.dumps({k: res["external_carparts"][k] for k in ("segm", "service_presence@0.5")}, ensure_ascii=False))
    try:
        import mlflow
        with mlflow.start_run(nested=bool(mlflow.active_run())):
            mlflow.log_params({"epochs": a.epochs, "batch": a.batch, "lr": a.lr, "min_size": a.min_size, "best_epoch": best_ep})
            mlflow.log_metrics({"test_mask_AP": res["test"]["segm"]["AP"], "test_mask_AP50": res["test"]["segm"]["AP50"],
                                "ext_mask_AP": res["external_carparts"]["segm"]["AP"], "train_seconds": train_s})
    except Exception as e:
        print("mlflow skip:", e)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/partseg")
    ap.add_argument("--out", default="models/vision/partseg")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--max-size", type=int, default=1024)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--train-splits", default="train")
    ap.add_argument("--ext-split", default="ext_carparts")
    ap.add_argument("--device", default="auto")
    main(ap.parse_args())
