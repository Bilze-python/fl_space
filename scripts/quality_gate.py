"""Run the repository's isolated formatting, import, and test gate."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], env: dict[str, str]) -> None:
    """Run one quality-gate command from the repository root."""
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    """Validate package provenance, Ruff, and the offline pytest suite."""
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import fl_space; "
                "root = Path(fl_space.__file__).resolve().parent; "
                "expected = Path.cwd() / 'fl_space'; "
                "assert root == expected, f'wrong fl_space root: {root}'"
            ),
        ],
        env,
    )
    run([sys.executable, "-m", "ruff", "check", "."], env)
    run([sys.executable, "-m", "ruff", "format", "--check", "."], env)
    run([sys.executable, "-m", "pytest", "-q"], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
