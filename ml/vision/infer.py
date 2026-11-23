"""근접 사진 결함 추론 + Grad-CAM 위치 표시. 모델이 없으면 MODEL_NOT_READY.

백엔드(모델) 선택: 환경변수 RESOURCE_VISION_BACKEND > models/vision/serving.json > "effnet"
- "ensemble"(2026-11-13 사용자 결정): 원본 전체로 재학습한 EfficientNet-B0 + YOLO11m-cls + 2차 판정 모델(stacking).
  세 판정이 모두 같고 2차 판정 확률이 기준값 ±0.10 밖이면 자동 판정, 아니면 verdict="AI_DISAGREE"(작업자 확인).
  부품 유형·6클래스 확률은 두 모델 평균, Grad-CAM은 EfficientNet. (reports/vision/full_data_retrain.md)
- "yolo11m_cls": YOLO11m-cls 30에폭(8:2:1) - 성능표 1위. Ultralytics AGPL-3.0 → 내부 사용 전용(외부 공개·배포 금지).
- "effnet": EfficientNet-B0(8:2:1) - 이전 서비스 모델, 라이선스 제약 없음(BSD). 대체(fallback)용.
두 백엔드 모두 출력 형식이 같다: defect_prob = P(계획부품_DEFECT) / P(계획부품), 6클래스 확률, 부품 오인식 정보, Grad-CAM.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from ml.vision.imageio import open_oriented
from ml.vision.common import CLASSES, MEAN, MODEL_DIR, STD  # 학습 코드(train.py)를 불러오지 않는다 — 탭 속도(2026-11-18 A)

PARTS = ("DOOR", "BUMPER", "FENDER")
WEIGHTS = MODEL_DIR / "efficientnet_b0_6cls.pt"
META = MODEL_DIR / "meta.json"
YOLO_DIR = MODEL_DIR / "yolo11m_cls"
YOLO_WEIGHTS = YOLO_DIR / "best.pt"
YOLO_META = YOLO_DIR / "meta.json"
ENS_DIR = MODEL_DIR / "ensemble_full"
ENS_META = ENS_DIR / "meta.json"
ENS_EFF_WEIGHTS = ENS_DIR / "effnet" / "efficientnet_b0_6cls.pt"
ENS_YOLO_WEIGHTS = ENS_DIR / "yolo11m_cls" / "best.pt"
ENSEMBLE_MEMBERS = ("effnet_full", "yolo11m_full")
ENSD_DIR = MODEL_DIR / "ensemble_sd_dino"  # 2026-11-18 사용자 결정: EfficientNet + DINOv2(세션 비겹침 분할), YOLO 대체
SERVING = MODEL_DIR / "serving.json"
BACKENDS = ("effnet", "yolo11m_cls", "effnet_full", "yolo11m_full", "ensemble", "effnet_sd", "dino_sd", "ensemble_dino")
# 조합 백엔드: (설정 폴더, 구성원 백엔드, 화면 이름, 단차 모델 폴더)
ENSEMBLES = {"ensemble": (ENS_DIR, ENSEMBLE_MEMBERS, ("EfficientNet-B0", "YOLO11m-cls"), MODEL_DIR / "gap"),
             "ensemble_dino": (ENSD_DIR, ("effnet_sd", "dino_sd"), ("EfficientNet-B0", "DINOv2 ViT-B/14"), ENSD_DIR / "gap_dino")}  # 단차: DINOv2(2026-11-19 사용자 결정), 되돌리기는 ENSD_DIR / "gap"


def _kind(b: str) -> str:
    return "yolo" if b.startswith("yolo") else "dino" if b.startswith("dino") else "effnet"


def _has(mod: str) -> bool:
    """패키지가 설치돼 있는지만 본다(불러오지 않음 — ultralytics 를 불러오면 약 3초)."""
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def preview_backend() -> str | None:
    """실시간 미리보기용 단독 모델(사용자 결정 2026-11-18: EfficientNet+DINO 조합일 때 미리보기는 EfficientNet 단독)."""
    return "effnet_sd" if backend() == "ensemble_dino" else None


def backend() -> str:
    b = os.environ.get("RESOURCE_VISION_BACKEND")
    if not b and SERVING.exists():
        b = json.loads(SERVING.read_text(encoding="utf-8")).get("backend")
    return b if b in BACKENDS else "effnet"


def _files(b: str):
    """모듈 변수를 호출 시점에 읽는다(테스트에서 경로 교체 가능)."""
    return {"yolo11m_cls": (YOLO_WEIGHTS, YOLO_META), "effnet": (WEIGHTS, META),
            "effnet_full": (ENS_EFF_WEIGHTS, ENS_DIR / "effnet" / "meta.json"),
            "yolo11m_full": (ENS_YOLO_WEIGHTS, ENS_DIR / "yolo11m_cls" / "meta.json"),
            "effnet_sd": (ENSD_DIR / "effnet" / "efficientnet_b0_6cls.pt", ENSD_DIR / "effnet" / "meta.json"),
            "dino_sd": (ENSD_DIR / "dinov2" / "best.pt", ENSD_DIR / "dinov2" / "meta.json")}[b]


def status(b: str | None = None) -> dict:
    b = b or backend()
    if b in ENSEMBLES:
        edir, members = ENSEMBLES[b][0], ENSEMBLES[b][1]
        for mb in members:
            s = status(mb)
            if s["status"] != "OK":
                return {**s, "backend": b, "message": f"조합 모델 구성원 {mb}: {s.get('message', '')}"}
        if not (edir / "meta.json").exists():
            return {"status": "MODEL_NOT_READY", "backend": b, "message": f"조합 설정 파일 없음: {edir.name}/meta.json"}
        meta = json.loads((edir / "meta.json").read_text(encoding="utf-8"))
        return {"status": "OK", "backend": b, "model_version": meta["model_version"],
                "test_part_metrics": meta["test_part_metrics"], "license_note": meta.get("license_note"),
                "rule": meta.get("rule")}
    w, m = _files(b)
    if not (w.exists() and m.exists()):
        return {"status": "MODEL_NOT_READY", "backend": b, "message": f"이미지 모델 파일 없음: {w.name}"}
    meta = json.loads(m.read_text(encoding="utf-8"))
    if _kind(b) == "yolo":  # 공개본: YOLO11(Ultralytics) 코드는 포함하지 않음
        return {"status": "MODEL_NOT_READY", "backend": b, "message": "YOLO11 백엔드는 공개본에 포함하지 않았습니다"}
    if _kind(b) == "dino" and not _has("timm"):
        return {"status": "MODEL_NOT_READY", "backend": b, "message": "timm 미설치: pip install timm==1.0.15 (Apache-2.0)"}
    return {"status": "OK", "backend": b, "model_version": meta["model_version"],
            "test_part_metrics": meta["test_part_metrics"], "license_note": meta.get("license_note")}


def _device():
    import torch
    want = os.environ.get("RESOURCE_VISION_DEVICE", "cpu")  # 시연 안정성: 기본 CPU(단일 이미지 추론)
    return torch.device("cuda" if want == "cuda" and torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=4)
def _load(b: str):
    """반환: (모델, 장치, meta, 전처리, Grad-CAM 대상 층, 모델 출력 클래스 순서)"""
    import torch
    dev = _device()
    w_path, m_path = _files(b)
    if _kind(b) == "yolo":
        raise RuntimeError("YOLO11 백엔드는 공개본에 포함하지 않았습니다")
    elif _kind(b) == "dino":
        import timm
        from torchvision import transforms
        meta = json.loads(m_path.read_text(encoding="utf-8"))
        size = int(meta.get("img_size", 448))
        m = timm.create_model(meta.get("timm_name", "vit_base_patch14_dinov2"), pretrained=False, img_size=size,
                              num_classes=len(meta["classes"]))
        m.load_state_dict(torch.load(w_path, map_location=dev))
        m.eval().to(dev)
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        names, layer = list(meta["classes"]), m.norm  # Grad-CAM 은 EfficientNet 것을 쓴다(DINO 는 with_cam=False)
    else:
        from torchvision import transforms
        from ml.vision.common import build_effnet as build_model
        m = build_model()
        m.load_state_dict(torch.load(w_path, map_location=dev))
        m.eval().to(dev)
        meta = json.loads(m_path.read_text(encoding="utf-8"))
        size = int(meta.get("img_size", 384))  # 학습 해상도와 동일하게 추론
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        names, layer = list(CLASSES), m.features[-1]
    for p in m.parameters():  # Grad-CAM은 활성값 기울기만 필요 — 파라미터 기울기 누적 방지
        p.requires_grad_(False)
    return m, dev, meta, tf, layer, names


def _uncrop(c, size):
    """YOLO11 전처리는 짧은 변 기준 가운데 정사각형만 본다. 히트맵을 원본 비율 격자에 놓고 잘린 바깥은 0으로 채운다.
    (그대로 원본 크기로 늘리면 가로·세로가 어긋남)"""
    import numpy as np
    w, h = size
    n = c.shape[0]
    if w == h:
        return c
    rows, cols = (n, round(n * w / h)) if w > h else (round(n * h / w), n)
    full = np.zeros((rows, cols), dtype=c.dtype)
    r0, c0 = (rows - n) // 2, (cols - n) // 2
    full[r0:r0 + n, c0:c0 + n] = c
    return full


def predict(image_bytes: bytes, expected_part: str | None = None, with_cam: bool = True,
            backend_name: str | None = None, shot: str = "surface") -> dict:
    """shot: "surface"(표면 근접 사진) | "gap"(휀더 틈을 가이드 박스에 맞춰 찍은 사진, 조합 모델에서만 단차 모델 사용)."""
    b = backend_name or backend()
    st = status(b)
    if st["status"] != "OK":
        return st
    if b in ENSEMBLES:
        return _predict_ensemble(image_bytes, expected_part, with_cam, st, shot, b)
    if _kind(b) == "dino":
        with_cam = False
    import torch
    m, dev, meta, tf, layer, names = _load(b)
    img = open_oriented(image_bytes, 512)  # EXIF 회전 적용 + 대형 JPEG 디코딩 메모리 절감
    x = tf(img).unsqueeze(0).to(dev)
    feats = {}

    def keep(_m, _i, o):
        feats["a"] = o
        if with_cam:
            o.register_hook(lambda g: feats.__setitem__("g", g))

    hook_f = layer.register_forward_hook(keep)
    try:
        with torch.set_grad_enabled(with_cam):
            x.requires_grad_(with_cam)  # 파라미터는 고정, 입력→활성값 경로로만 기울기 계산
            logits = m(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
        prob_raw = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        prob = {n: float(p) for n, p in zip(names, prob_raw)}
        part_p = {p: prob[f"{p}_GOOD"] + prob[f"{p}_DEFECT"] for p in PARTS}
        pred_part = max(part_p, key=part_p.get)
        part = expected_part if expected_part in part_p else pred_part
        di = names.index(f"{part}_DEFECT")
        defect_prob = float(prob[f"{part}_DEFECT"] / max(part_p[part], 1e-9))
        cam = None
        if with_cam:
            logits[0, di].backward()
            a, g = feats["a"][0].detach(), feats["g"][0]
            w = g.mean(dim=(1, 2), keepdim=True)
            c = torch.relu((w * a).sum(0)).cpu().numpy()
            c = c / c.max() if c.max() > 0 else c
            if _kind(b) == "yolo":
                c = _uncrop(c, img.size)
            cam = c.tolist()
    finally:
        hook_f.remove()
        feats.clear()
    th = meta.get("thresholds", {}).get(part, meta.get("threshold", 0.5))  # YOLO11: 부품별 검증 임계값
    return {"status": "OK", "backend": b, "model_version": meta["model_version"], "provenance": "AIHUB_IMAGE_AI",
            "class_probs": {c: prob[c] for c in CLASSES}, "predicted_class": max(prob, key=prob.get),
            "predicted_part": pred_part, "part_confidence": part_p[pred_part], "evaluated_part": part,
            "defect_prob": defect_prob, "threshold": float(th), "cam": cam,
            "verdict": "AI_SUSPECTED_DEFECT" if defect_prob >= th else "AI_PASS",
            "note": "결함 의심 확률이며 폐기 확정이 아님"}


def _stack(meta: dict, part: str, pe: float, py: float) -> tuple[float, float, float]:
    """2차 판정 모델(부품별 로지스틱 회귀, ml/vision/fit_stacking.py) → (불량 확률, 기준값, 애매 폭)."""
    import math
    st = meta["stacking"]
    c = st["parts"][part]
    lg = lambda p: math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6)))
    z = c["coef"][0] * lg(pe) + c["coef"][1] * lg(py) + c["intercept"]
    return 1 / (1 + math.exp(-z)), float(c["threshold"]), float(st["uncertain_margin"])


GAP_DIR = MODEL_DIR / "gap"
GAP_WEIGHTS = GAP_DIR / "gap_effb0.pt"
GAP_META = GAP_DIR / "meta.json"


def _gap_dir(b: str | None = None):
    b = b or backend()
    return ENSEMBLES[b][3] if b in ENSEMBLES else GAP_DIR


def _gap_weights(g) -> str:
    try:
        return json.loads((g / "meta.json").read_text(encoding="utf-8")).get("weights", "gap_effb0.pt")
    except (OSError, ValueError):
        return "gap_effb0.pt"


def gap_ready(b: str | None = None) -> bool:
    g = _gap_dir(b)
    return (g / "meta.json").exists() and (g / _gap_weights(g)).exists()


@lru_cache(maxsize=2)
def _load_gap(gdir=None):
    import torch
    from torchvision import models, transforms
    dev = _device()
    gdir = gdir or GAP_DIR
    meta = json.loads((gdir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("backbone") == "dino":  # DINOv2 ViT-B/14 단차 모델(timm, Apache-2.0)
        import timm
        m = timm.create_model(meta.get("timm_name", "vit_base_patch14_dinov2"), pretrained=False, img_size=meta["img"], num_classes=2)
    else:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load(gdir / meta.get("weights", "gap_effb0.pt"), map_location=dev))
    m.eval().to(dev)
    tf = transforms.Compose([transforms.Resize((meta["img"], meta["img"])), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    return m, dev, meta, tf


def gap_score(image_bytes: bytes, b: str | None = None) -> tuple[float, dict]:
    """사진 가운데 가이드 박스(짧은 변 x guide_frac) 영역을 잘라 단차 불량 확률. 작업자가 틈을 박스에 맞춰 찍었다고 가정."""
    import torch
    m, dev, meta, tf = _load_gap(_gap_dir(b))
    img = open_oriented(image_bytes, 2048)
    w, h = img.size
    side = meta["guide_frac"] * min(w, h)
    crop = img.crop((round((w - side) / 2), round((h - side) / 2), round((w + side) / 2), round((h + side) / 2)))
    with torch.no_grad():
        p = float(torch.softmax(m(tf(crop).unsqueeze(0).to(dev)), 1)[0, 1])
    return p, meta


def _predict_ensemble(image_bytes: bytes, expected_part: str | None, with_cam: bool, st: dict,
                      shot: str = "surface", b: str = "ensemble") -> dict:
    """세 판정(EfficientNet, YOLO11m, 2차 판정 모델)이 모두 같고 2차 확률이 애매하지 않으면 자동 판정,
    아니면 AI_DISAGREE(작업자 확인). 부품 유형은 두 모델 6클래스 확률 평균.
    휀더 틈(gap) 촬영: 단차 모델 판정 = 2차 판정(세 점수)이고 애매하지 않으면 자동 판정(ml/vision/fit_gap_service.py)."""
    edir, members_b, names, _ = ENSEMBLES[b]
    eff = predict(image_bytes, expected_part, with_cam=with_cam, backend_name=members_b[0])
    yol = predict(image_bytes, expected_part, with_cam=False, backend_name=members_b[1])  # 두 번째 구성원(YOLO 또는 DINO)
    meta = json.loads((edir / "meta.json").read_text(encoding="utf-8"))
    part = eff["evaluated_part"]
    if shot == "gap" and part == "FENDER":
        return _predict_gap_shot(image_bytes, eff, yol, st, b)
    sp, sth, margin = _stack(meta, part, eff["defect_prob"], yol["defect_prob"])
    sv = "AI_SUSPECTED_DEFECT" if sp >= sth else "AI_PASS"
    prob = {c: (eff["class_probs"][c] + yol["class_probs"][c]) / 2 for c in CLASSES}
    part_p = {p: prob[f"{p}_GOOD"] + prob[f"{p}_DEFECT"] for p in PARTS}
    pred_part = max(part_p, key=part_p.get)
    reasons = []
    if eff["verdict"] != yol["verdict"]:
        reasons.append("두 모델 판정이 엇갈림")
    elif sv != eff["verdict"]:
        reasons.append("2차 판정 모델이 두 모델과 다르게 판정")
    if abs(sp - sth) < margin:
        reasons.append(f"2차 판정 확률 {sp:.2f}가 기준 {sth:.2f}에 가까워 애매함(±{margin:.2f})")
    members = [{"name": n, "model_version": r["model_version"], "defect_prob": r["defect_prob"],
                "threshold": r["threshold"], "verdict": r["verdict"]}
               for n, r in ((names[0], eff), (names[1], yol))]
    members.append({"name": "2차 판정 모델", "model_version": "stacking_logreg_val", "defect_prob": sp,
                    "threshold": sth, "verdict": sv})
    return {"status": "OK", "backend": b, "model_version": st["model_version"], "provenance": "AIHUB_IMAGE_AI",
            "class_probs": prob, "predicted_class": max(prob, key=prob.get), "predicted_part": pred_part,
            "part_confidence": part_p[pred_part], "evaluated_part": part,
            "defect_prob": sp, "threshold": sth,  # 기록·표시 값 = 2차 판정 모델 확률
            "verdict": "AI_DISAGREE" if reasons else sv, "agree": not reasons, "check_reasons": reasons,
            "ai_recommendation": sv, "members": members, "cam": eff["cam"],
            "note": "결함 의심 판정이며 폐기 확정이 아님. 세 판정이 다르거나 애매하면 작업자 확인"}


def _predict_gap_shot(image_bytes: bytes, eff: dict, yol: dict, st: dict, b: str = "ensemble") -> dict:
    import math
    names = ENSEMBLES[b][2]
    prob = {c: (eff["class_probs"][c] + yol["class_probs"][c]) / 2 for c in CLASSES}
    part_p = {p: prob[f"{p}_GOOD"] + prob[f"{p}_DEFECT"] for p in PARTS}
    pred_part = max(part_p, key=part_p.get)
    base = {"status": "OK", "backend": b, "provenance": "AIHUB_IMAGE_AI", "shot": "gap",
            "class_probs": prob, "predicted_class": max(prob, key=prob.get), "predicted_part": pred_part,
            "part_confidence": part_p[pred_part], "evaluated_part": "FENDER", "cam": eff["cam"]}
    members = [{"name": n, "model_version": r["model_version"], "defect_prob": r["defect_prob"],
                "threshold": r["threshold"], "verdict": r["verdict"]} for n, r in ((names[0], eff), (names[1], yol))]
    if not gap_ready(b):
        return {**base, "model_version": st["model_version"], "defect_prob": eff["defect_prob"], "threshold": eff["threshold"],
                "verdict": "AI_DISAGREE", "agree": False, "check_reasons": ["단차 모델 파일 없음 - 작업자 확인"],
                "ai_recommendation": None, "members": members, "note": "단차 모델 없음"}
    gp, gm = gap_score(image_bytes, b)
    gv = "AI_SUSPECTED_DEFECT" if gp >= gm["gap_threshold"] else "AI_PASS"
    sk = gm["stacking"]
    lg = lambda p: math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6)))
    z = sk["coef"][0] * lg(eff["defect_prob"]) + sk["coef"][1] * lg(yol["defect_prob"]) + sk["coef"][2] * lg(gp) + sk["intercept"]
    sp, sth, margin = 1 / (1 + math.exp(-z)), float(sk["threshold"]), float(sk["uncertain_margin"])
    sv = "AI_SUSPECTED_DEFECT" if sp >= sth else "AI_PASS"
    reasons = []
    if gv != sv:
        reasons.append("단차 모델과 2차 판정이 엇갈림")
    if abs(sp - sth) < margin:
        reasons.append(f"2차 판정 확률 {sp:.2f}가 기준 {sth:.2f}에 가까워 애매함(±{margin:.2f})")
    members += [{"name": "단차 모델(가이드 영역)", "model_version": gm["model_version"], "defect_prob": gp,
                 "threshold": gm["gap_threshold"], "verdict": gv},
                {"name": "2차 판정 모델(단차 포함)", "model_version": "gap_stacking_logreg_val", "defect_prob": sp,
                 "threshold": sth, "verdict": sv}]
    return {**base, "model_version": f"{st['model_version']}+{gm['model_version']}", "defect_prob": sp, "threshold": sth,
            "verdict": "AI_DISAGREE" if reasons else sv, "agree": not reasons, "check_reasons": reasons,
            "ai_recommendation": sv, "members": members,
            "note": "휀더 틈 가이드 촬영 판정(실제 카메라 성능 미검증). 결함 의심 판정이며 폐기 확정이 아님"}


def auto_accuracy(part: str, shot: str = "surface") -> dict | None:
    """조합 모델 자동 판정 규칙의 평가셋 성적(일치해 자동 판정된 사진의 정확도·자동 판정 비율). 화면 안내용.

    세션 비겹침 재평가(reports/vision/sd_eval.json, 2026-11-14 사용자 결정)가 있으면 그 정직한 수치를 우선 쓴다."""
    try:
        if backend() == "ensemble_dino":  # 현재 조합의 세션 비겹침 시험 성적(fit_dino_ensemble.py)
            em = json.loads((ENSD_DIR / "meta.json").read_text(encoding="utf-8"))
            ev = em["evaluation"]
            days = re.search(r"촬영일 (\d+)일", ev["set"])
            days = f"처음 보는 촬영일 {days.group(1)}일" if days else "처음 보는 촬영일"
            if shot == "gap":
                g = ev["gap"]
                return {"accuracy": g["자동판정_정확도"], "auto_rate": g["자동판정_비율"], "honest": True,
                        "set": f"{days} · 휀더 틈 {g['n']:,}장(가이드 촬영 가정)"}
            r = ev["by_part"][part]
            return {"accuracy": r["auto_accuracy"], "auto_rate": r["auto_rate"], "honest": True,
                    "set": f"{days} · {part} {em['test_part_metrics'][part]['n']:,}장"}
        sd_path = MODEL_DIR.parent.parent / "reports" / "vision" / "sd_eval.json"
        if sd_path.exists():
            sd = json.loads(sd_path.read_text(encoding="utf-8"))
            days = sd["split"]["days"]["test"]
            if shot == "gap":
                g = sd["gap"]
                return {"accuracy": g["gap_rule_auto_acc"], "auto_rate": g["gap_rule_auto_rate"], "honest": True,
                        "set": f"처음 보는 촬영일 {days}일 · 휀더 틈 {g['n_test']:,}장(가이드 촬영 가정)"}
            r = sd["surface"][part]
            return {"accuracy": r["auto_acc"], "auto_rate": r["auto_rate"], "honest": True,
                    "set": f"처음 보는 촬영일 {days}일 · {part} {r['n_test']:,}장"}
        if shot == "gap":
            g = json.loads(GAP_META.read_text(encoding="utf-8"))
            r = g["rule_candidates"]["test+holdout"][g["chosen_rule"]]
            return {"accuracy": r["auto_accuracy"], "auto_rate": r["auto_rate"], "set": "휀더 틈 평가 1,663장(가이드 촬영 가정)"}
        e = json.loads(ENS_META.read_text(encoding="utf-8"))["evaluation"]
        r = e["by_part"][part]
        return {"accuracy": r["auto_accuracy"], "auto_rate": r["auto_rate"], "set": e["set"]}
    except Exception:  # noqa: BLE001 - 안내 문구용, 없으면 표시 안 함
        return None


# ── 결함 위치 박스(YOLO11m 검출, 2026-11-14 사용자 결정 (b)) — 표시 전용, 판정에는 쓰지 않음 ──
DET_DIR = MODEL_DIR / "yolo11m_det"
DET_WEIGHTS = DET_DIR / "best.pt"
DET_META = DET_DIR / "meta.json"
DET_NAMES_KO = {"scratch": "스크래치", "exterior_damage": "외관 손상", "gap_defect": "단차 불량", "gap_normal": "단차 정상"}


def det_ready() -> bool:
    return False  # 공개본: YOLO11 결함 위치 검출은 포함하지 않음


def detect_defects(image_bytes: bytes) -> list[dict] | None:
    """결함 위치 박스 [{name, name_ko, conf, xyxyn}] (meta의 표시 클래스·신뢰도 기준 이상만). 모델 없으면 None."""
    return None
