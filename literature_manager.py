"""Small CLI companion for the SpaceFL web literature library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_DIR / ".literature_config.json"
SUPPORTED_SUFFIXES = {".md", ".markdown", ".pdf", ".txt"}


def load_config() -> dict[str, str]:
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    return {"default_path": str(raw.get("default_path") or PROJECT_DIR / "文献")}


def save_default_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"Literature directory does not exist: {path}")
    CONFIG_FILE.write_text(
        json.dumps({"default_path": str(path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def scan_literature() -> list[Path]:
    root = Path(load_config()["default_path"]).expanduser().resolve()
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the SpaceFL literature directory")
    parser.add_argument("--set-default", metavar="PATH", help="Set an existing default literature directory")
    parser.add_argument("--list", action="store_true", help="List indexed literature files")
    args = parser.parse_args()

    if args.set_default:
        print(f"Default literature directory: {save_default_path(args.set_default)}")
    if args.list or not args.set_default:
        files = scan_literature()
        print(f"Found {len(files)} literature files")
        for path in files:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
