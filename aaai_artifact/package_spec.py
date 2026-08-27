"""Single whitelist shared by the checker and deterministic ZIP builder."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INCLUDE = (
    Path("README.md"),
    Path("code"),
    Path("data"),
    Path("delivery/2026-07-22"),
    Path("delivery/2026-07-27"),
    Path("aaai_artifact"),
)

EXCLUDED_NAMES = {
    ".DS_Store",
    "__pycache__",
}


def package_files() -> list[Path]:
    files: list[Path] = []
    for relative in INCLUDE:
        target = ROOT / relative
        if target.is_file():
            files.append(target)
            continue
        if not target.is_dir():
            raise FileNotFoundError(f"Missing whitelisted path: {relative}")
        for path in target.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDED_NAMES for part in rel.parts):
                continue
            if path.suffix == ".zip":
                continue
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())
