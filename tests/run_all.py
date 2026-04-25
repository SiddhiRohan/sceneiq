"""
SceneIQ — unified test runner.

Discovers every ``test_*.py`` module in this directory, runs every
``test_*`` function inside it, and prints a summary. Exit code 0 on pass,
1 if anything failed.

Usage (any of these work):
    python tests/run_all.py
    python -m tests.run_all
    pytest tests/                 # also works; this runner is a pytest-free fallback
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(HERE))   # so conftest.py is importable as a plain module


def _discover_modules() -> List[str]:
    """Return sorted list of ``test_*`` module names in this directory."""
    names = []
    for p in sorted(HERE.glob("test_*.py")):
        names.append(p.stem)
    return names


def _discover_tests(module_name: str) -> List[Tuple[str, Callable]]:
    """Import ``module_name`` and return (qualname, fn) for each ``test_*`` fn."""
    mod = importlib.import_module(module_name)
    items = []
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        obj = getattr(mod, name)
        if callable(obj):
            items.append((f"{module_name}.{name}", obj))
    return items


def run(verbose: bool = True, match: str | None = None) -> int:
    modules = _discover_modules()
    if verbose:
        print(f"Discovered {len(modules)} test modules: {modules}\n")

    all_tests: List[Tuple[str, Callable]] = []
    for m in modules:
        try:
            all_tests.extend(_discover_tests(m))
        except Exception as exc:   # noqa: BLE001
            print(f"IMPORT FAIL {m}: {type(exc).__name__}: {exc}")
            if verbose:
                traceback.print_exc()
            return 1

    if match:
        all_tests = [(n, f) for n, f in all_tests if match in n]

    if verbose:
        print(f"Running {len(all_tests)} tests\n")

    passed = 0
    failures = []
    for qualname, fn in all_tests:
        try:
            fn()
        except AssertionError as e:
            failures.append((qualname, f"assertion: {e}"))
            if verbose:
                print(f"FAIL  {qualname}: {e}")
        except Exception as e:    # noqa: BLE001
            failures.append((qualname, f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"ERROR {qualname}: {type(e).__name__}: {e}")
                traceback.print_exc()
        else:
            passed += 1
            if verbose:
                print(f"PASS  {qualname}")

    total = len(all_tests)
    print()
    print("=" * 60)
    print(f"  SceneIQ tests — {passed}/{total} passed")
    print("=" * 60)
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every SceneIQ test without needing pytest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print the summary line.")
    parser.add_argument("-k", "--match", type=str, default=None,
                        help="Run only tests whose qualname contains this substring.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(verbose=not args.quiet, match=args.match))
