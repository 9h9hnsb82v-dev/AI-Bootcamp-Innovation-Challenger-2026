"""휀더 전용 고해상도 이진(양품/불량) 분류기 — 2단계 검사의 2단계.

- 입력: manifest_fender_hr1024.csv (기존 v2와 같은 이미지·분할, 긴 변 1024px)
- 백본: EfficientNet-B0, 입력 576×768(H×W), 클래스 가중치
- 임계값: 검증셋에서 Youden J(재현율+특이도-1) 최대값으로 결정 → 시험셋은 1회만 평가
- 보고: 전체/하위유형(단차·외관 손상)별 재현율·특이도·정밀도, v2 6클래스 모델의 같은 지표와 비교

실행(GPU 권장): RESOURCE_VISION_OUT=models/vision/fender_hr python -m ml.vision.train_fender --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from ml.tracking import log_run
from ml.vision.train import MEAN, ROOT, STD

MANIFEST = ROOT / "data/processed/aihub/manifest_fender_hr1024.csv"
OUT_DIR = Path(os.environ.get("RESOURCE_VISION_OUT", ROOT / "models/vision/fender_hr"))
H, W = 576, 768
SEED = 42
EVAL_TF = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
TRAIN_TF = transforms.Compose([transforms.RandomResizedCrop((H, W), scale=(0.85, 1.0), ratio=(1.25, 1.42)),
                               transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.1),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(self, df, tf):
        self.p, self.y, self.tf = df.path.tolist(), (df.quality == "DEFECT").astype(int).tolist(), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(ROOT / self.p[i]).convert("RGB")), self.y[i]


def build_model() -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def scores(model, loader, dev) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps, ys = [], []
    with torch.no_grad(), torch.autocast("cuda", enabled=dev.type == "cuda"):
        for x, y in loader:
            ps.append(torch.softmax(model(x.to(dev)).float(), 1)[:, 1].cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def binary_metrics(p, y, th) -> dict:
    yh = p >= th
    tp, tn = int((yh & (y == 1)).sum()), int((~yh & (y == 0)).sum())
    return {"n": int(len(y)), "recall": tp / max((y == 1).sum(), 1), "specificity": tn / max((y == 0).sum(), 1),
            "precision": tp / max(yh.sum(), 1), "accuracy": (tp + tn) / max(len(y), 1), "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None}


def main(epochs: int, batch: int) -> dict:
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    tr, va, te = (man[man.subset == s] for s in ("train", "val", "test"))
    dl = lambda df, tf, sh: DataLoader(DS(df, tf), batch_size=batch, shuffle=sh, num_workers=6, pin_memory=True,
                                       persistent_workers=True)
    ltr, lva, lte = dl(tr, TRAIN_TF, True), dl(va, EVAL_TF, False), dl(te, EVAL_TF, False)
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m = m.to(dev)
    pos = (tr.quality == "DEFECT").mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor([0.5 / (1 - pos), 0.5 / pos], dtype=torch.float32).to(dev))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best, hist = -1.0, []
    for ep in range(epochs):
        m.train()
        t0, tot = time.time(), 0.0
        for x, y in ltr:
            opt.zero_grad()
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = crit(m(x.to(dev)), y.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        p, y = scores(m, lva, dev)
        auc = roc_auc_score(y, p)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_auc": auc, "sec": time.time() - t0})
        print(hist[-1], flush=True)
        if auc > best:
            best = auc
            torch.save(m.state_dict(), OUT_DIR / "fender_binary.pt")
    m.load_state_dict(torch.load(OUT_DIR / "fender_binary.pt", map_location=dev))
    pv, yv = scores(m, lva, dev)
    grid = np.linspace(0.05, 0.95, 91)
    th = float(grid[np.argmax([binary_metrics(pv, yv, t)["recall"] + binary_metrics(pv, yv, t)["specificity"] for t in grid])])
    pt, yt = scores(m, lte, dev)
    te = te.assign(p=pt)
    meta = {"model_version": f"aihub_fender_effb0_bin_{H}x{W}", "provenance": "AIHUB_IMAGE_AI", "img_hw": [H, W],
            "seed": SEED, "epochs": epochs, "threshold": th, "threshold_rule": "val Youden J",
            "val": binary_metrics(pv, yv, th), "test": binary_metrics(pt, yt, th),
            "test_by_subtype": {s: binary_metrics(g.p.to_numpy(), (g.quality == "DEFECT").astype(int).to_numpy(), th)
                                for s, g in te.groupby("subtype")},
            "history": hist, "train_n": int(len(tr)), "val_n": int(len(va)), "test_n": int(len(te)),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    te[["path", "subtype", "quality", "p"]].to_csv(OUT_DIR / "test_scores.csv", index=False, encoding="utf-8-sig")
    meta["mlflow_run_id"] = log_run("aihub_vision_fender", meta["model_version"], {"h": H, "w": W, "epochs": epochs},
                                    {f"test_{k}": v for k, v in meta["test"].items() if isinstance(v, float)})
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    out = main(a.epochs, a.batch)
    print(json.dumps({k: out[k] for k in ("threshold", "val", "test", "test_by_subtype")}, ensure_ascii=False, indent=1))
