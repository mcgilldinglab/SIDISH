#!/usr/bin/env python3
"""Run one persisted SIDISH job without invoking a shell."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write(path: Path, value: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main(job_json: str) -> int:
    jf = Path(job_json).resolve()
    rec = json.loads(jf.read_text(encoding="utf-8"))
    for _ in range(100):
        if rec.get("pid"):
            break
        time.sleep(0.02)
        rec = json.loads(jf.read_text(encoding="utf-8"))
    log = Path(rec["log"])
    root = Path(rec["repo_root"])
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLCONFIGDIR", str(rec["job_dir"] + "/mpl-cache"))
    env.setdefault("NUMBA_CACHE_DIR", str(rec["job_dir"] + "/numba-cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    rec["status"] = "running"; rec["updated_at"] = now(); write(jf, rec)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(rec["argv"], cwd=root, stdout=handle,
                                   stderr=subprocess.STDOUT, env=env, shell=False)
    code = int(completed.returncode)
    (jf.parent / "exit_code").write_text(str(code), encoding="utf-8")
    rec = json.loads(jf.read_text(encoding="utf-8"))
    if rec.get("status") != "cancelled":
        rec["status"] = "succeeded" if code == 0 else "failed"
    rec["exit_code"] = code; rec["updated_at"] = now(); write(jf, rec)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
