"""Azure ML 작업 본문: DINOv2 ViT-B/14 미세조정 6클래스(부품 x 양품/불량) 분류 (사용자 요청 2026-11-18: "YOLO11m처럼 학습").

YOLO11m-cls(`job_sd_yolo11m_cls.yml`)와 맞춘 조건:
- 같은 이미지(aihub-full-640:1)·같은 세션 비겹침 분할(manifest_sd.csv)·같은 6클래스·448px·15에폭·seed 42
- 검증 top-1 이 가장 좋은 에폭의 가중치로 val/test 이미지별 확률을 저장 → YOLO 점수 파일과 같은 형식(`dinov2_vitb14_scores.csv`)
  → ml/vision/eval_dino_sd.py 가 같은 서비스 규칙(부품별 Youden 임계값·2차 판정)으로 비교한다.
DINOv2: Meta, Apache-2.0 (코드·가중치). timm 의 `vit_base_patch14_dinov2.lvd142m` 사용(위치 임베딩을 448px 로 보간).
사용: python azureml/dino/run_dino_cls.py <data_dir> <out_dir> --manifest <csv> [--epochs 15] [--img 448] [--batch 32]
2026-11-19 사용자 승인: --model convnext_small.fb_in22k_ft_in1k (ConvNeXt v1, MIT 코드·Apache-2.0 가중치)도 같은 조건으로 학습.
  ConvNeXt 는 timm 의 층별 학습률 감쇠 묶음(param_groups_layer_decay)을 쓰고, 분류층만 head_lr.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["BUMPER_DEFECT", "BUMPER_GOOD", "DOOR_DEFECT", "DOOR_GOOD", "FENDER_DEFECT", "FENDER_GOOD"]  # YOLO 점수 열 순서와 같게
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MODEL = "vit_base_patch14_dinov2.lvd142m"
TAGS = {"vit_base_patch14_dinov2.lvd142m": ("dinov2_vitb14", "Apache-2.0 (DINOv2)"),
        "convnext_small.fb_in22k_ft_in1k": ("convnext_small", "MIT 코드 · Apache-2.0 가중치 (ConvNeXt v1)")}


class DS(Dataset):
    def __init__(self, rows: list[dict], tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(r["file"]).convert("RGB")), CLASSES.index(r["class"])


def read_manifest(manifest: str, data_in: Path, limit: int | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(manifest, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rel = r["path"].replace("\\", "/").split("aihub_full/", 1)[1]
            out.setdefault(r["subset"], []).append({"file": str(data_in / rel), "class": r["class"], "stem": r["stem"]})
    if limit:  # 스모크 테스트용
        rng = np.random.default_rng(0)
        out = {k: [v[i] for i in rng.choice(len(v), min(limit, len(v)), replace=False)] for k, v in out.items()}
    return out


def param_groups(m: nn.Module, lr: float, head_lr: float, decay: float, wd: float) -> list[dict]:
    if not hasattr(m, "blocks"):  # ConvNeXt 등: timm 층 묶음 기준 감쇠, 분류층(head.fc)만 head_lr
        from timm.optim._param_groups import param_groups_layer_decay
        head = {id(p) for n, p in m.named_parameters() if n.startswith("head.fc")}
        out = []
        for g in param_groups_layer_decay(m, weight_decay=wd, layer_decay=decay):
            for is_head in (False, True):
                ps = [p for p in g["params"] if (id(p) in head) == is_head]
                if ps:
                    blr = head_lr if is_head else lr * g.get("lr_scale", 1.0)
                    out.append({"params": ps, "weight_decay": g["weight_decay"], "lr": blr, "base_lr": blr})
        return out
    """층별 학습률 감쇠: 뒤쪽 블록일수록 크게, 패치 임베딩은 가장 작게. 머리(분류층)는 head_lr."""
    n = len(m.blocks)
    groups: dict[tuple, dict] = {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head"):
            depth, glr = n + 1, head_lr
        elif name.startswith("blocks."):
            depth = int(name.split(".")[1]) + 1
            glr = lr * decay ** (n - depth)
        elif name.startswith("norm") or name.startswith("fc_norm"):
            depth, glr = n, lr
        else:  # cls_token, pos_embed, patch_embed, reg_token
            depth, glr = 0, lr * decay ** n
        no_wd = p.ndim == 1 or name.endswith(".bias") or "token" in name or "pos_embed" in name
        k = (depth, no_wd)
        g = groups.setdefault(k, {"params": [], "lr": glr, "weight_decay": 0.0 if no_wd else wd})
        g["params"].append(p)
    for g in groups.values():
        g["base_lr"] = g["lr"]
    return list(groups.values())


@torch.no_grad()
def predict(m: nn.Module, rows: list[dict], tf, dev, batch: int, workers: int) -> tuple[np.ndarray, np.ndarray, float]:
    m.eval()
    ps, ys = [], []
    t0 = time.time()
    for x, y in DataLoader(DS(rows, tf), batch_size=batch, num_workers=workers, pin_memory=True):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            ps.append(torch.softmax(m(x.to(dev, non_blocking=True)).float(), 1).cpu().numpy())
        ys.append(y.numpy())
    ms = (time.time() - t0) * 1000 / max(len(rows), 1)
    return np.concatenate(ps), np.concatenate(ys), ms


def log_metrics(d: dict, step: int | None = None) -> None:
    if not os.environ.get("AZUREML_RUN_ID"):
        return
    try:
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in d.items()}, step=step)
    except Exception as e:  # noqa: BLE001 - 기록 실패는 학습에 영향 없음
        print("mlflow skip:", repr(e)[:120])


def main(a) -> None:
    import timm
    torch.manual_seed(42)
    np.random.seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = read_manifest(a.manifest, Path(a.data_in), a.limit)
    counts = {f"{s}/{c}": sum(r["class"] == c for r in rows) for s, rows in data.items() for c in CLASSES}
    print(json.dumps(counts, indent=1), flush=True)

    kw = {"img_size": a.img} if "vit" in a.model else {}
    m = timm.create_model(a.model, pretrained=not a.no_pretrained, num_classes=len(CLASSES),
                          drop_path_rate=a.drop_path, **kw).to(dev)
    tag, lic = TAGS.get(a.model, (a.model.split(".")[0], "?"))
    train_tf = transforms.Compose([transforms.RandomResizedCrop(a.img, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
                                   transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.1),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.Resize((a.img, a.img)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dl = DataLoader(DS(data["train"], train_tf), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(param_groups(m, a.lr, a.head_lr, a.layer_decay, 0.05), betas=(0.9, 0.999))
    total, warm = a.epochs * len(dl), max(1, len(dl))  # 첫 1에폭 선형 워밍업 → 코사인
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_ep, hist, step, t_start = -1.0, -1, [], 0, time.time()
    for ep in range(1, a.epochs + 1):
        m.train()
        t0, tot, n, correct = time.time(), 0.0, 0, 0
        for x, y in dl:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = m(x)
                loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            correct += int((logits.argmax(1) == y).sum())  # 학습 정확도(증강 사진 기준, 에폭 평균)
            tot, n, step = tot + loss.item() * len(y), n + len(y), step + 1
        p, yv, _ = predict(m, data["val"], eval_tf, dev, a.batch * 2, a.workers)
        pred = p.argmax(1)
        acc = float((pred == yv).mean())
        # 2026-11-19 사용자 요청(학습 곡선 빈칸 채우기): 검증 손실(라벨 스무딩 없는 CE) · 학습 정확도 · macro F1 · 부품별 양품/불량 정확도
        val_loss = float(-np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, 1)).mean())
        f1 = [2 * ((pred == k) & (yv == k)).sum() / max((pred == k).sum() + (yv == k).sum(), 1) for k in range(len(CLASSES))]
        part_acc = {}
        for part in ("DOOR", "BUMPER", "FENDER"):
            ids = [CLASSES.index(f"{part}_DEFECT"), CLASSES.index(f"{part}_GOOD")]
            mk = np.isin(yv, ids)
            dprob = p[mk, ids[0]] / np.maximum(p[mk, ids[0]] + p[mk, ids[1]], 1e-12)
            part_acc[part] = float(((dprob >= 0.5) == (yv[mk] == ids[0])).mean()) if mk.any() else None
        h = {"epoch": ep, "train_loss": tot / max(n, 1), "train_acc": correct / max(n, 1), "val_loss": val_loss,
             "val_accuracy_top1": acc, "val_macro_f1": float(np.mean(f1)), "val_part_acc": part_acc,
             "sec": round(time.time() - t0, 1)}
        hist.append(h)
        print(json.dumps(h), flush=True)
        log_metrics({"train_loss": h["train_loss"], "train_acc": h["train_acc"], "val_loss": val_loss,
                     "val_accuracy_top1": acc, "val_macro_f1": h["val_macro_f1"]}, step=ep)
        if acc > best:
            best, best_ep = acc, ep
            torch.save(m.state_dict(), out / "best.pt")
    train_s = time.time() - t_start

    m.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    rows_out = []
    for s in ("val", "test", "holdout"):
        if s not in data:
            continue
        p, y, ms = predict(m, data[s], eval_tf, dev, a.batch * 2, a.workers)
        for r, pi, yi in zip(data[s], p, y):
            row = {"subset": s, "stem": r["stem"], "true": CLASSES[yi], "pred": CLASSES[int(pi.argmax())],
                   "infer_ms": round(ms, 2)}
            row.update({f"p_{c}": float(pi[i]) for i, c in enumerate(CLASSES)})
            rows_out.append(row)
    with open(out / f"{tag}_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    test = [r for r in rows_out if r["subset"] == "test"]
    test_top1 = float(np.mean([r["true"] == r["pred"] for r in test])) if test else None
    summary = {"model": a.model, "license": lic, "drop_path": a.drop_path, "img": a.img, "epochs": a.epochs, "batch": a.batch,
               "lr": a.lr, "head_lr": a.head_lr, "layer_decay": a.layer_decay, "best_epoch": best_ep,
               "best_val_top1": best, "test_top1_6cls": test_top1, "train_seconds": round(train_s, 1),
               "timm": timm.__version__, "torch": torch.__version__, "counts": counts, "history": hist}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log_metrics({"best_val_top1": best, "test_top1_6cls": test_top1 or 0.0, "train_seconds": train_s})
    print("DONE", json.dumps({k: summary[k] for k in ("best_epoch", "best_val_top1", "test_top1_6cls", "train_seconds")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_in")
    ap.add_argument("out")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img", type=int, default=448)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="스모크 테스트: 분할별 최대 장수")
    ap.add_argument("--no-pretrained", action="store_true", help="스모크 테스트: 가중치 내려받지 않음")
    main(ap.parse_args())
