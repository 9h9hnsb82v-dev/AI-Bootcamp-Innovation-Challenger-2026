"""이미지 판정 공통 상수(가벼운 모듈, 2026-11-18 탭 속도 개선 A).

판정 코드(infer.py)가 학습 코드(train.py)를 불러오면 torch·torchvision·scikit-learn 이 화면을 열 때마다 함께 올라와
'이미지 검사' 탭 첫 진입이 약 16초 걸렸다(실측). 상수만 여기 두고, torch 는 실제 판정할 때 불러온다.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/vision"
CLASSES = ["DOOR_GOOD", "DOOR_DEFECT", "BUMPER_GOOD", "BUMPER_DEFECT", "FENDER_GOOD", "FENDER_DEFECT"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_effnet(n_classes: int = len(CLASSES)):
    """EfficientNet-B0(가중치 없이 구조만). torchvision 은 이 함수를 부를 때 불러온다."""
    from torch import nn
    from torchvision import models
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m
