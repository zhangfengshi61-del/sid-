#!/usr/bin/env python3
"""Build a deterministic, whitelist-only anonymous AAAI Code & Data ZIP."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

from package_spec import ROOT, package_files


ARCHIVE_ROOT = "ServeSID-AAAI27-Code-Data"
FIXED_TIME = (2026, 7, 27, 0, 0, 0)


def zip_info(name: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o755 if executable else 0o644
    info.external_attr = mode << 16
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ServeSID-AAAI27-Code-Data.zip",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "aaai_artifact/check_artifact.py"],
        cwd=ROOT,
        check=True,
    )

    files = package_files()
    if output in files:
        raise SystemExit("Output ZIP overlaps the package whitelist")

    hashes: list[str] = []
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            hashes.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
            executable = path.suffix in {".py", ".sh"}
            archive.writestr(
                zip_info(f"{ARCHIVE_ROOT}/{rel}", executable),
                data,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
        manifest = ("\n".join(hashes) + "\n").encode()
        archive.writestr(
            zip_info(f"{ARCHIVE_ROOT}/SHA256SUMS"),
            manifest,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"archive={output}")
    print(f"files={len(files)}")
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
