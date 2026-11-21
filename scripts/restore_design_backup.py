"""디자인 롤백: backups/design_classic_20261116/ui 를 ui/ 로 되돌린다.

    .venv/Scripts/python.exe scripts/restore_design_backup.py          # 확인만 (무엇이 바뀌는지 출력)
    .venv/Scripts/python.exe scripts/restore_design_backup.py --apply  # 실제 복원

복원 전에 지금 ui/ 를 backups/design_before_restore_<시각>/ui 에 먼저 복사해 둔다(아무것도 지우지 않음).
더 간단한 롤백: 환경변수 RESOURCE_DESIGN 을 비우거나 classic 으로 두고 앱을 다시 시작하면 기존 화면 그대로 나온다.
"""
from __future__ import annotations

import filecmp
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "backups/design_classic_20261116/ui"
DST = ROOT / "ui"


def _files(base: Path) -> set[str]:
    return {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def main(apply: bool) -> None:
    if not SRC.exists():
        sys.exit(f"백업이 없습니다: {SRC}")
    old, now = _files(SRC), _files(DST)
    changed = sorted(f for f in old & now if not filecmp.cmp(SRC / f, DST / f, shallow=False))
    print("되돌릴 파일:", changed or "없음")
    print("백업에 없는 새 파일(그대로 둠, 기존 화면에는 영향 없음):", sorted(now - old) or "없음")
    print("백업에만 있는 파일(복원):", sorted(old - now) or "없음")
    if not apply:
        print("\n확인만 했습니다. 실제로 되돌리려면 --apply 를 붙이세요.")
        return
    keep = ROOT / f"backups/design_before_restore_{datetime.now():%Y%m%d_%H%M%S}/ui"
    shutil.copytree(DST, keep, ignore=shutil.ignore_patterns("__pycache__"))
    for f in sorted(old):
        (DST / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / f, DST / f)
    print(f"복원 완료. 복원 직전 상태는 {keep} 에 있습니다. 앱을 다시 시작하세요.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
