"""휀더 단차 전용 이진 분류기(양품/불량) - 원본 해상도에서 자른 단차 부위로 학습 (2026-11-13, 사용자 요청).

- 백본: torchvision EfficientNet-B0 (ImageNet 사전학습, BSD). 입력 IMG px.
- 학습 입력: crops/ 의 단차 박스 주변(한 변 2L, L=박스 긴 변)에서 매번 중심을 ±0.25L, 한 변을 1.2~1.8L로 흔들어 자른다
  (작업자가 틈을 화면 가운데에 맞춰 찍는 오차를 흉내).
- 평가 3가지(val/test/holdout, 임계값은 val에서 방식별로 결정 - 평가 스크립트 ml/vision/eval_gap.py):
  gt     : 박스 중심, 한 변 1.5L (위치를 정확히 알 때 = 상한)
  jitter : 중심 ±0.25L, 한 변 1.2~1.8L 무작위 1회(seed 고정) (촬영 안내로 틈을 가운데 맞춘 경우)
  tiles  : full/ 원본 전체를 조각으로 나눠 모두 판정, 최대 불량 점수 (위치를 모를 때). 최대 조각이 박스와 겹쳤는지도 기록
- 에폭별 학습 곡선을 Azure ML Metrics에 기록. 최적 에폭 = 검증 gt AUC 최대.
실행: RESOURCE_GAP_DATA=<aihub_gap 폴더> RESOURCE_GAP_OUT=<출력> python -m ml.vision.train_gap --epochs 12
- --backbone dino (2026-11-19 사용자 승인): DINOv2 ViT-B/14(timm vit_base_patch14_dinov2.lvd142m, Apache-2.0)로 같은 데이터·자르기·평가.
  층별 학습률(뒤 블록 크게, 감쇠 0.75) · 머리 1e-3 · 첫 에폭 워밍업 → 코사인. 가중치 파일 gap_dinov2.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import zlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DATA = Path(os.environ.get("RESOURCE_GAP_DATA", "data/processed/aihub_gap"))
OUT = Path(os.environ.get("RESOURCE_GAP_OUT", "models/vision/gap"))
IMG = int(os.environ.get("RESOURCE_GAP_IMG", 448))
SEED = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
JIT_SHIFT, JIT_SIDE = 0.25, (1.2, 1.8)
GT_SIDE = 1.5


def crop_square(im: Image.Image, cx: float, cy: float, side: float) -> Image.Image:
    h = side / 2
    return im.crop((round(cx - h), round(cy - h), round(cx + h), round(cy + h))).resize((IMG, IMG), Image.BICUBIC)


class CropDS(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str):
        self.r = df.reset_index(drop=True)
        self.mode = mode
        self.aug = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                       transforms.ColorJitter(0.25, 0.25, 0.15, 0.02)])

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        r = self.r.iloc[i]
        im = Image.open(DATA / r.crop_path.split("aihub_gap/", 1)[1]).convert("RGB")
        L = float(r.crop_L)
        if self.mode == "gt":
            x = crop_square(im, r.crop_cx, r.crop_cy, GT_SIDE * L)
        else:
            rng = random if self.mode == "train" else random.Random(zlib.crc32(f"{SEED}:{r.stem}".encode()))
            dx, dy = (rng.uniform(-JIT_SHIFT, JIT_SHIFT) * L for _ in range(2))
            x = crop_square(im, r.crop_cx + dx, r.crop_cy + dy, rng.uniform(*JIT_SIDE) * L)
            if self.mode == "train":
                x = self.aug(x)
        return NORM(x), int(r.quality == "DEFECT")


BACKBONE = "effnet"
LIMIT = 0


def build(pretrained: bool = True) -> nn.Module:
    if BACKBONE == "dino":
        import timm
        return timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=pretrained, img_size=IMG, num_classes=2,
                                 drop_path_rate=0.1)
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def dino_groups(m: nn.Module, lr: float = 5e-5, head_lr: float = 1e-3, decay: float = 0.75, wd: float = 0.05) -> list[dict]:
    """층별 학습률 감쇠(azureml/dino/run_dino_cls.py 와 같은 규칙)."""
    n, groups = len(m.blocks), {}
    for name, p in m.named_parameters():
        if name.startswith("head"):
            depth, glr = n + 1, head_lr
        elif name.startswith("blocks."):
            depth = int(name.split(".")[1]) + 1
            glr = lr * decay ** (n - depth)
        elif name.startswith(("norm", "fc_norm")):
            depth, glr = n, lr
        else:
            depth, glr = 0, lr * decay ** n
        no_wd = p.ndim == 1 or name.endswith(".bias") or "token" in name or "pos_embed" in name
        g = groups.setdefault((depth, no_wd), {"params": [], "lr": glr, "weight_decay": 0.0 if no_wd else wd})
        g["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def predict(m, loader, dev) -> tuple[np.ndarray, np.ndarray]:
    m.eval()
    ps, ys = [], []
    for x, y in loader:
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            ps.append(torch.softmax(m(x.to(dev)).float(), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


@torch.no_grad()
def tile_scores(m, df: pd.DataFrame, dev, tile: float) -> pd.DataFrame:
    """원본 전체(긴 변 1536)를 한 변 tile 조각(보폭 tile/2)으로 나눠 최대 불량 점수."""
    m.eval()
    rows = []
    for _, r in df.iterrows():
        im = Image.open(DATA / r.full_path.split("aihub_gap/", 1)[1]).convert("RGB")
        W, H = im.size
        xs = list(np.arange(0, max(W - tile, 0) + 1, tile / 2)) + ([W - tile] if W > tile else [])
        ys = list(np.arange(0, max(H - tile, 0) + 1, tile / 2)) + ([H - tile] if H > tile else [])
        boxes = sorted({(round(x), round(y)) for x in xs for y in ys})
        batch = torch.stack([NORM(im.crop((x, y, x + round(tile), y + round(tile))).resize((IMG, IMG), Image.BICUBIC))
                             for x, y in boxes]).to(dev)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            p = torch.softmax(m(batch).float(), 1)[:, 1].cpu().numpy()
        k = int(p.argmax())
        bx, by = boxes[k]
        hit = not (bx + tile < r.full_x or r.full_x + r.full_w < bx or by + tile < r.full_y or r.full_y + r.full_h < by)
        rows.append({"stem": r.stem, "score_tiles": float(p.max()), "tiles_n": len(boxes), "tiles_hit_box": bool(hit)})
    return pd.DataFrame(rows)


def log_epoch(h: dict) -> None:
    if not os.environ.get("AZUREML_RUN_ID"):
        return
    try:
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in h.items() if k not in ("epoch", "sec")}, step=int(h["epoch"]))
    except Exception as e:  # noqa: BLE001
        print("mlflow skip:", repr(e)[:120])


def main(epochs: int, batch: int, workers: int) -> dict:
    random.seed(SEED)
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = pd.read_csv(os.environ.get("RESOURCE_GAP_MANIFEST", DATA / "manifest.csv"), encoding="utf-8-sig")
    if LIMIT:
        man = pd.concat([g.sample(min(LIMIT, len(g)), random_state=0) for _, g in man.groupby("subset")])
    tr, va = man[man.subset == "train"], man[man.subset == "val"]
    ltr = DataLoader(CropDS(tr, "train"), batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True,
                     persistent_workers=workers > 0, drop_last=True)
    lva = DataLoader(CropDS(va, "gt"), batch_size=batch, num_workers=workers)
    m = build(pretrained=not os.environ.get("RESOURCE_GAP_NO_PRETRAINED")).to(dev)
    wname = "gap_dinov2.pt" if BACKBONE == "dino" else "gap_effb0.pt"
    n = tr.quality.value_counts()
    w = torch.tensor([len(tr) / (2 * n["GOOD"]), len(tr) / (2 * n["DEFECT"])], dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    if BACKBONE == "dino":
        opt = torch.optim.AdamW(dino_groups(m))
        warm = 1  # 첫 에폭 워밍업 후 코사인(에폭 단위)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: (e + 1) / (warm + 1) if e < warm else 0.5 * (1 + np.cos(np.pi * (e - warm) / max(1, epochs - warm))))
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    best, hist = -1.0, []
    for ep in range(1, epochs + 1):
        m.train()
        t0, tot, hit, seen = time.time(), 0.0, 0, 0
        for x, y in ltr:
            opt.zero_grad()
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                out = m(x.to(dev))
                loss = crit(out, y.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            hit += int((out.argmax(1).cpu() == y).sum())
            seen += len(y)
        sched.step()
        p, y = predict(m, lva, dev)
        vloss = float(nn.functional.binary_cross_entropy(torch.tensor(p).clamp(1e-7, 1 - 1e-7), torch.tensor(y).float()))
        h = {"epoch": ep, "train_loss": tot / seen, "train_acc": hit / seen, "val_loss": vloss,
             "val_acc_gt": float(((p >= 0.5) == y).mean()), "val_auc_gt": float(roc_auc_score(y, p)), "sec": time.time() - t0}
        hist.append(h)
        print(h, flush=True)
        log_epoch(h)
        if h["val_auc_gt"] > best:
            best = h["val_auc_gt"]
            torch.save(m.state_dict(), OUT / wname)
    m.load_state_dict(torch.load(OUT / wname, map_location=dev))
    ev = man[man.subset.isin(["val", "test", "holdout"])].reset_index(drop=True)
    out = ev[["subset", "stem", "quality"]].copy()
    for mode in ("gt", "jitter"):
        out[f"score_{mode}"], _ = predict(m, DataLoader(CropDS(ev, mode), batch_size=batch, num_workers=workers), dev)
    fw = np.maximum(va.full_w, va.full_h).dropna()
    tile = 1.5 * float(np.median(fw)) if len(fw) else 426.0  # 원본 전체 축척에서 단차 박스 긴 변 중앙값 × 1.5
    has_full = ev.full_path.fillna("").astype(str).str.len() > 0  # 재분할 시 공식 Training 출신 사진은 full 없음
    if has_full.any():
        out = out.merge(tile_scores(m, ev[has_full], dev, tile), on="stem", how="left", validate="1:1")
    out.to_csv(OUT / "gap_scores.csv", index=False, encoding="utf-8")
    meta = {"model_version": f"aihub_gap_{'dinov2_vitb14' if BACKBONE == 'dino' else 'effb0'}_img{IMG}", "backbone": BACKBONE,
            "weights": wname, "img": IMG, "epochs": epochs,
            "best_epoch": int(max(hist, key=lambda h: h["val_auc_gt"])["epoch"]), "tile_px_at_1536": tile,
            "train_counts": tr.quality.value_counts().to_dict(), "history": hist,
            "crop_rule": {"gt_side": GT_SIDE, "jitter_shift": JIT_SHIFT, "jitter_side": JIT_SIDE}}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("best_epoch", "tile_px_at_1536", "train_counts")}))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", default=None, help="aihub_gap 폴더(환경변수 RESOURCE_GAP_DATA 대신)")
    ap.add_argument("--out", default=None, help="출력 폴더(환경변수 RESOURCE_GAP_OUT 대신)")
    ap.add_argument("--backbone", default="effnet", choices=["effnet", "dino"])
    ap.add_argument("--limit", type=int, default=0, help="스모크 테스트: 분할별 최대 장수")
    a = ap.parse_args()
    BACKBONE = a.backbone
    LIMIT = a.limit
    if a.data:
        DATA = Path(a.data)
    if a.out:
        OUT = Path(a.out)
    main(a.epochs, a.batch, a.workers)
