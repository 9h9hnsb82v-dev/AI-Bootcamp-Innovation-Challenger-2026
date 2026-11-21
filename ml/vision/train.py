"""AI-Hub 도어/범퍼/휀더 6클래스(부품×양품/불량) 이미지 분류기 학습 (로컬 GPU, 유료 리소스 미사용).

- 백본: torchvision EfficientNet-B0 (ImageNet 사전학습, BSD 라이선스)
- 분할: prepare_subset.py 의 train/val(촬영일 그룹) / test(AI-Hub 공식 Validation)
- 보고: 클래스별 precision/recall/F1, 부품별 결함 recall, 혼동행렬
- 이미지 학습셋의 불량 비율은 생산현장 불량률이 아니다(불량률은 승인된 DEMO_ASSUMED_RATE 사용).

실행: .venv/Scripts/python -m ml.vision.train --epochs 8
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from pathlib import Path

from ml.tracking import log_run
from ml.vision.prepare_subset import MANIFEST, ROOT

MODEL_DIR = ROOT / "models/vision"
MAN_PATH = Path(__import__("os").environ.get("RESOURCE_VISION_MANIFEST", MANIFEST))  # 전체 데이터 학습 시 교체
OUT_DIR = Path(__import__("os").environ.get("RESOURCE_VISION_OUT", MODEL_DIR))  # 후보 학습은 별도 폴더
from ml.vision.common import CLASSES, MEAN, STD  # noqa: E402 - 판정 코드와 같은 상수(한 곳에서 관리)
IMG = int(__import__("os").environ.get("RESOURCE_VISION_IMG", 448))
SEED = 42
EVAL_TF = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
TRAIN_TF = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(0.2, 0.2, 0.1), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.p, self.y, self.tf = [p.replace("\\", "/") for p in df.path], df["class"].map(CLASSES.index).tolist(), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(ROOT / self.p[i]).convert("RGB")), self.y[i]


def build_model() -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m


def evaluate(model, loader, dev):
    model.eval()
    ps, ys = [], []
    with torch.no_grad(), torch.autocast("cuda", enabled=dev.type == "cuda"):
        for x, y in loader:
            ps.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def log_epoch(h: dict) -> None:
    """Azure ML 작업일 때 에폭별 학습 곡선을 작업 Metrics에 기록(로컬 실행은 건너뜀)."""
    if not __import__("os").environ.get("AZUREML_RUN_ID"):
        return
    try:
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in h.items() if k not in ("epoch", "sec")}, step=int(h["epoch"]))
    except Exception as e:  # noqa: BLE001 - 기록 실패는 학습에 영향 없음
        print("mlflow epoch skip:", repr(e)[:120])


def write_scores(model, man: pd.DataFrame, dev, batch: int) -> None:
    """val/test/holdout 이미지별 6클래스 확률을 저장(모델 조합 검증용, YOLO 점수 파일과 같은 형식)."""
    rows = []
    for s in ("val", "test", "holdout"):
        df = man[man.subset == s]
        if df.empty:
            continue
        p, y = evaluate(model, DataLoader(DS(df, EVAL_TF), batch_size=batch, num_workers=4), dev)
        for path, pi, yi in zip(df.path, p, y):
            r = {"subset": s, "stem": path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0],
                 "true": CLASSES[yi], "pred": CLASSES[int(pi.argmax())]}
            r.update({f"p_{c}": float(pi[i]) for i, c in enumerate(CLASSES)})
            rows.append(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "effnet_scores.csv", index=False, encoding="utf-8")


def main(epochs: int, batch: int) -> dict:
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = pd.read_csv(MAN_PATH, encoding="utf-8-sig")
    tr, va, te = (man[man.subset == s] for s in ("train", "val", "test"))
    dl = lambda df, tf, sh: DataLoader(DS(df, tf), batch_size=batch, shuffle=sh, num_workers=int(__import__("os").environ.get("RESOURCE_VISION_WORKERS", 4)), pin_memory=True,
                                       persistent_workers=True)
    ltr, lva, lte = dl(tr, TRAIN_TF, True), dl(va, EVAL_TF, False), dl(te, EVAL_TF, False)
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    m = m.to(dev)
    counts = tr["class"].value_counts().reindex(CLASSES).to_numpy(float)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32).to(dev))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, hist = -1.0, []
    for ep in range(epochs):
        m.train()
        t0, tot, hit = time.time(), 0.0, 0
        for x, y in ltr:
            opt.zero_grad()
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                out = m(x.to(dev))
                loss = crit(out, y.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            hit += int((out.argmax(1).cpu() == y).sum())  # 증강·학습 모드 기준 정확도
        sched.step()
        p, y = evaluate(m, lva, dev)
        rep = classification_report(y, p.argmax(1), labels=range(6), target_names=CLASSES, output_dict=True, zero_division=0)
        f1 = rep["macro avg"]["f1-score"]
        val_loss = float(nn.NLLLoss(weight=crit.weight.cpu())(torch.log(torch.from_numpy(p).clamp_min(1e-8)),
                                                              torch.from_numpy(y).long()))  # 학습 손실과 같은 클래스 가중치
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), "train_acc": hit / len(tr), "val_loss": val_loss,
                     "val_acc": float((p.argmax(1) == y).mean()), "val_macro_f1": f1, "sec": time.time() - t0})
        print(hist[-1], flush=True)
        log_epoch(hist[-1])
        if f1 > best:
            best = f1
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), OUT_DIR / "efficientnet_b0_6cls.pt")
    m.load_state_dict(torch.load(OUT_DIR / "efficientnet_b0_6cls.pt", map_location=dev))
    write_scores(m, man, dev, batch)
    p, y = evaluate(m, lte, dev)
    pred = p.argmax(1)
    rep = classification_report(y, pred, labels=range(6), target_names=CLASSES, output_dict=True, zero_division=0)
    part_metrics = {}
    for part in ("DOOR", "BUMPER", "FENDER"):
        gi, di = CLASSES.index(f"{part}_GOOD"), CLASSES.index(f"{part}_DEFECT")
        msk = np.isin(y, [gi, di])
        cond = p[msk, di] / (p[msk, gi] + p[msk, di])  # 부품을 안다고 할 때 결함 확률
        yt = (y[msk] == di).astype(int)
        yh = (cond >= 0.5).astype(int)
        tp = int(((yh == 1) & (yt == 1)).sum())
        part_metrics[part] = {"n": int(msk.sum()), "defect_recall": tp / max(yt.sum(), 1),
                              "defect_precision": tp / max(yh.sum(), 1),
                              "part_type_accuracy": float(np.mean(np.isin(pred[msk], [gi, di])))}
    tag = __import__("os").environ.get("RESOURCE_VISION_TAG", "")  # 예: "_exif_split821"
    meta = {"model_version": f"aihub_effb0_6cls_img{IMG}{tag}", "provenance": "AIHUB_IMAGE_AI", "classes": CLASSES,
            "split": {s: int((man.subset == s).sum()) for s in ("train", "val", "test")},
            "img_size": IMG, "seed": SEED, "epochs": epochs, "threshold": 0.5,
            "train_counts": tr["class"].value_counts().to_dict(), "val_n": int(len(va)), "test_n": int(len(te)),
            "test_source": "AI-Hub 공식 Validation 서브셋", "history": hist,
            "test_report": rep, "test_part_metrics": part_metrics,
            "test_confusion_matrix": confusion_matrix(y, pred, labels=range(6)).tolist(),
            "note": "결함 의심 확률은 폐기 확정이 아님. 작업자 CONFIRMED_SCRAP만 kg에 반영.",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    meta["mlflow_run_id"] = log_run("aihub_vision", meta["model_version"], {"epochs": epochs, "img": IMG, "seed": SEED},
                                    {"test_macro_f1": rep["macro avg"]["f1-score"],
                                     **{f"{k}_defect_recall": v["defect_recall"] for k, v in part_metrics.items()}})
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    out = main(a.epochs, a.batch)
    print(json.dumps({"macro_f1": out["test_report"]["macro avg"]["f1-score"], "parts": out["test_part_metrics"]}, indent=1))
