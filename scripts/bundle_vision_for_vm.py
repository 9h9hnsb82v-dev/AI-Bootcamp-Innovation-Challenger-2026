"""원격 GPU VM 학습용 번들(코드 + 매니페스트 + 서브셋 이미지)을 tar로 만든다.

매니페스트 경로를 POSIX 형식(/)으로 정규화한다(Windows 백슬래시 → Linux 호환).
실행: .venv/Scripts/python -m scripts.bundle_vision_for_vm <출력 tar 경로>
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/aihub/manifest.csv"


def main(out: Path) -> None:
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    m["path"] = m["path"].str.replace("\\", "/", regex=False)
    m.to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    missing = [p for p in m.path if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"누락 이미지 {len(missing)}개: {missing[:3]}")
    skip = lambda ti: None if "__pycache__" in ti.name else ti  # noqa: E731
    with tarfile.open(out, "w") as t:
        for d in ("ml", "configs"):
            t.add(ROOT / d, arcname=d, filter=skip)
        t.add(MANIFEST, arcname="data/processed/aihub/manifest.csv")
        for p in m.path:
            t.add(ROOT / p, arcname=p)
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB, images {len(m)}, {m.subset.value_counts().to_dict()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
