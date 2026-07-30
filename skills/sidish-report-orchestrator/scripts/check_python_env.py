#!/usr/bin/env python3
"""Environment gate for the SIDISH skill: confirm an isolated env with the SIDISH stack.
Exit 0 if usable; exit 1 (with guidance) if not."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sidish-matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/sidish-numba-cache")

REQUIRED = ["torch", "scanpy", "anndata", "pandas", "numpy", "pyro", "lifelines",
            "torch_geometric", "sklearn", "yaml", "jinja2"]
OPTIONAL = ["openai", "playwright", "weasyprint", "streamlit"]


def _in_isolated_env() -> bool:
    return bool(os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
                or sys.prefix != getattr(sys, "base_prefix", sys.prefix))


def main() -> int:
    print(f"python: {sys.version.split()[0]}  ({sys.executable})")
    env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV") \
        or (Path(sys.prefix).name if _in_isolated_env() else "(base/none)")
    print(f"env: {env}")

    # The real gate is: can we import the SIDISH stack? (conda envs don't always expose a marker)
    missing = []
    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    for mod in OPTIONAL:
        try:
            importlib.import_module(mod)
            print(f"  optional {mod}: ok")
        except Exception:
            print(f"  optional {mod}: missing (the related chat, PDF, or web-app feature is unavailable)")

    if missing:
        print("MISSING required modules:", ", ".join(missing))
        print("Activate/point at a Python env that has the SIDISH stack, then re-run.")
        return 1
    if not _in_isolated_env() and env in ("(base/none)",):
        print("Note: no isolated env marker detected, but the SIDISH stack imports — proceeding.")
    print("Environment OK — SIDISH stack importable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
