#!/usr/bin/env python
"""Developer task runner for the FPM Teaching App.

One command per common action. Run `python dev.py` to see all of them.

The fast iteration loop
-----------------------
    python dev.py edit       # live reactive notebook (edit code + text, hot)
    python dev.py preview     # the *deployed* app experience, served locally, fast
    python dev.py deploy      # push to `web` -> GitHub Actions -> Pages

`edit` and `preview` both run a real Python kernel locally, so changes show up
instantly -- no Pyodide/WASM cold-load. You almost never need to touch the WASM
build by hand while iterating; `build`/`serve` are only for sanity-checking the
exact static artifact that GitHub Pages serves.
"""
from __future__ import annotations

import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
DIST = ROOT / "dist"
DEPLOY_BRANCH = "web"


def _run(cmd: list[str], **kw) -> int:
    print(">", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT, **kw)


def edit(_args: list[str]) -> int:
    """Open the live reactive notebook editor (primary dev loop)."""
    return _run(["marimo", "edit", str(APP)])


def preview(_args: list[str]) -> int:
    """Serve the app in run mode locally (what users will see), reactive + fast."""
    return _run(["marimo", "run", str(APP)])


def build(_args: list[str]) -> int:
    """Export the static WASM build to dist/ (what GitHub Pages hosts)."""
    return _run(
        ["marimo", "export", "html-wasm", str(APP), "-o", str(DIST), "--mode", "run"]
    )


def serve(_args: list[str]) -> int:
    """Build, then serve dist/ over http to sanity-check the real static artifact."""
    rc = build(_args)
    if rc != 0:
        return rc
    url = "http://localhost:8000"
    print(f"\nServing the static WASM build at {url} (Ctrl-C to stop).")
    print("Note: this is the slow Pyodide cold-load path -- for fast iteration use "
          "`preview` instead.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    return _run([sys.executable, "-m", "http.server", "8000", "--directory", str(DIST)])


def check(_args: list[str]) -> int:
    """Fast sanity check: compile + marimo dependency-graph load (no unresolved refs)."""
    rc = _run([sys.executable, "-m", "py_compile", str(APP)])
    if rc != 0:
        return rc
    code = (
        "from marimo._ast.load import load_app;"
        "app = load_app(r'" + str(APP) + "');"
        "cells = list(app._cell_manager.valid_cells());"
        "defs = set().union(*[set(c.defs) for _, c in cells]);"
        "refs = set().union(*[set(c.refs) for _, c in cells]);"
        "missing = sorted(r for r in refs if r not in defs and r not in dir(__builtins__));"
        "print('cells:', len(cells));"
        "print('unresolved refs:', missing);"
        "raise SystemExit(1 if missing else 0)"
    )
    return _run([sys.executable, "-c", code])


def deploy(_args: list[str]) -> int:
    """Commit any changes and push to the deploy branch (triggers CI -> Pages)."""
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if branch != DEPLOY_BRANCH:
        print(f"You are on '{branch}', not '{DEPLOY_BRANCH}'. Switch first:")
        print(f"    git checkout {DEPLOY_BRANCH}")
        return 1
    msg = " ".join(_args) or "Update FPM teaching app"
    _run(["git", "add", "-A"])
    # commit may be a no-op if nothing changed; don't treat that as failure.
    _run(["git", "commit", "-m", msg])
    return _run(["git", "push", "origin", DEPLOY_BRANCH])


COMMANDS = {
    "edit": edit,
    "preview": preview,
    "build": build,
    "serve": serve,
    "check": check,
    "deploy": deploy,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Commands:")
        for name, fn in COMMANDS.items():
            print(f"  {name:9s} {fn.__doc__.splitlines()[0]}")
        return 0 if len(sys.argv) < 2 else 2
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
