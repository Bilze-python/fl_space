"""Validate the local assets used by the SpaceFL 2D orbit viewer."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REQUIRED_ASSETS = {
    "viewer": PROJECT_DIR / "web" / "orbit_2d_viewer.html",
    "d3": PROJECT_DIR / "node_modules" / "d3" / "dist" / "d3.min.js",
    "topojson": PROJECT_DIR
    / "node_modules"
    / "topojson-client"
    / "dist"
    / "topojson-client.min.js",
    "world_map": PROJECT_DIR
    / "node_modules"
    / "world-atlas"
    / "countries-110m.json",
}


def validate_assets() -> list[str]:
    errors = [f"Missing {name}: {path}" for name, path in REQUIRED_ASSETS.items() if not path.is_file()]
    world_path = REQUIRED_ASSETS["world_map"]
    if world_path.is_file():
        try:
            world = json.loads(world_path.read_text(encoding="utf-8"))
            if "countries" not in world.get("objects", {}):
                errors.append("world-atlas data does not contain objects.countries")
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid world-atlas JSON: {exc}")
    return errors


def main() -> int:
    errors = validate_assets()
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        print("Run: npm install")
        return 1
    print("SpaceFL 2D orbit viewer assets are ready.")
    print("Open: http://127.0.0.1:8700/orbit_2d_viewer.html")
    print("Map source: https://github.com/topojson/world-atlas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
