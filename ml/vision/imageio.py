"""이미지 로딩 공통 규칙: EXIF Orientation을 적용한다.

AI-Hub 라벨 bbox는 EXIF 회전이 적용된 화면 기준 좌표이며(2026-11-10 시각 검증),
현장 사진 업로드도 브라우저 표시 방향과 같아야 하므로 학습·추론 모두 같은 규칙을 쓴다.
"""
from __future__ import annotations

import io

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def open_oriented(data: bytes, draft_long_side: int | None = None) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    if draft_long_side:
        im.draft("RGB", (draft_long_side * 2, draft_long_side * 2))
    return ImageOps.exif_transpose(im).convert("RGB")
