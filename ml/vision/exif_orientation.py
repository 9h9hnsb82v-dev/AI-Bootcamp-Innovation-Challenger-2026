"""매니페스트 이미지의 EXIF Orientation 감사.

발견(2026-11-10): AI-Hub 라벨 bbox는 EXIF 회전을 적용한 화면 기준 좌표다. EXIF orientation≠1 이미지를
원시 픽셀로 저장하면 bbox가 결함과 어긋난다 → 추출 시 ImageOps.exif_transpose 적용 필요.
헤더(앞 256KB)만 읽어 orientation과 JSON 크기 일치 여부를 기록한다.
출력: data/processed/aihub/exif_orientation.csv
실행: .venv/Scripts/python -m ml.vision.exif_orientation
"""
from __future__ import annotations

import io
import tarfile

import pandas as pd
from PIL import Image

from ml.vision.audit_aihub_labels import AIHUB_DIR
from ml.vision.prepare_subset import MANIFEST, ROOT

Image.MAX_IMAGE_PIXELS = None
OUT = ROOT / "data/processed/aihub/exif_orientation.csv"


def main() -> pd.DataFrame:
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    rows = []
    for tar_rel, g in m.groupby("source_tar"):
        want = set(g.member)
        with tarfile.open(AIHUB_DIR / tar_rel, "r:") as t:
            for mem in t:
                if mem.name not in want:
                    continue
                head = t.extractfile(mem).read(262144)
                try:
                    im = Image.open(io.BytesIO(head))
                    o, size = im.getexif().get(274, 1), im.size
                except Exception:  # noqa: BLE001
                    o, size = None, (None, None)
                rows.append({"member": mem.name, "orientation": o, "raw_w": size[0], "raw_h": size[1]})
                want.discard(mem.name)
                if not want:
                    break
        print(tar_rel, "done", flush=True)
    out = m[["subset", "part", "quality", "member", "path"]].merge(pd.DataFrame(rows), on="member", how="left")
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(out.groupby(["part", "orientation"], dropna=False).size().to_string())
    return out


if __name__ == "__main__":
    main()
