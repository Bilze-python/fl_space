"""Pytest isolation for the repository test suite."""

import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PACKAGE = (PROJECT_ROOT / "fl_space").resolve()


def _is_external_project_path(entry: str) -> bool:
    """Return whether a sys.path entry exposes another fl_space checkout."""
    if not entry:
        return False
    try:
        entry_path = Path(entry).resolve()
    except OSError:
        return False

    candidates = [entry_path / "fl_space", entry_path]
    return any(
        candidate.name == PROJECT_PACKAGE.name
        and (candidate / "__init__.py").is_file()
        and candidate != PROJECT_PACKAGE
        for candidate in candidates
    )


sys.path[:] = [entry for entry in sys.path if not _is_external_project_path(entry)]
project_root_string = str(PROJECT_ROOT)
if project_root_string in sys.path:
    sys.path.remove(project_root_string)
sys.path.insert(0, project_root_string)


def _purge_external_package() -> None:
    """Remove a stale fl_space package imported from another checkout."""
    for module_name, module in list(sys.modules.items()):
        if not (module_name == "fl_space" or module_name.startswith("fl_space.")):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            package_file = Path(module_file).resolve()
        except OSError:
            continue
        if PROJECT_PACKAGE not in package_file.parents:
            del sys.modules[module_name]


_purge_external_package()
loaded_package = importlib.import_module("fl_space")
if Path(loaded_package.__file__).resolve().parent != PROJECT_PACKAGE:
    raise RuntimeError(f"pytest imported the wrong fl_space package: {loaded_package.__file__}")


# This file is a legacy executable test script, not a pytest module. Its checks
# remain available through ``python tests/test_time_model.py``.
collect_ignore = ["test_time_model.py"]
