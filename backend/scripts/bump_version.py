"""Bump backend patch version and refresh its CST version timestamp."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sys


BACKEND_DIR = Path(__file__).resolve().parent.parent
VERSION_PATH = BACKEND_DIR / "src" / "version.py"
VERSION_FILE_DISPLAY = "backend/src/version.py"


def main() -> None:
    source = VERSION_PATH.read_text(encoding="utf-8")
    current_version = _read_constant(source, "BACKEND_VERSION")
    next_version = sys.argv[1] if len(sys.argv) > 1 else bump_patch_version(current_version)
    if not re.match(r"^\d+\.\d+\.\d+$", next_version):
        raise SystemExit(f"Backend version must be semver like 0.1.1, got {next_version!r}")

    version_time = format_china_standard_time(datetime.now(timezone.utc))
    VERSION_PATH.write_text(
        "\n".join(
            [
                '"""Backend release version metadata."""',
                "",
                f'BACKEND_VERSION = "{next_version}"',
                f'BACKEND_VERSION_TIME = "{version_time}"',
                'BACKEND_VERSION_LABEL = f"v{BACKEND_VERSION} · {BACKEND_VERSION_TIME}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Backend version bumped in {VERSION_FILE_DISPLAY} to v{next_version} · {version_time}")


def _read_constant(source: str, name: str) -> str:
    match = re.search(rf'^{name}\s*=\s*"([^"]+)"', source, re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not find {name} in {VERSION_FILE_DISPLAY}")
    return match.group(1)


def bump_patch_version(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise SystemExit(f"Backend version must be semver like 0.1.0, got {version!r}")
    major, minor, patch = match.groups()
    return f"{major}.{minor}.{int(patch) + 1}"


def format_china_standard_time(date: datetime) -> str:
    cst_date = date.astimezone(timezone(timedelta(hours=8)))
    return cst_date.strftime("%Y-%m-%d %H:%M CST")


if __name__ == "__main__":
    main()
