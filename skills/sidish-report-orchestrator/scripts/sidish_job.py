#!/usr/bin/env python3
"""Durable job orchestration for SIDISH workflows — mirrors UNAGI's unagi_job.py.

Long SIDISH work (training on a new dataset, or a full genome-wide perturbation) can run
for hours. This runs it as a durable, detached job with persistent state under
outputs/skill_jobs/<job_id>/, surviving the launcher exiting.

    submit        --config <yaml> [--job-name N]   # launch a workflow (analysis or training)
    submit-analysis --config <yaml> --patient P     # convenience: build a report for a patient
    status <job_id>                                  # one job's state (+ liveness check)
    list                                             # all jobs
    tail <job_id> [--lines N]                        # last N log lines
    cancel <job_id>                                  # terminate a running job
    resume <job_id>                                  # re-launch a failed/cancelled job

State model: queued -> running -> succeeded | failed | cancelled.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path

JOB_ROOT_REL = Path("outputs/skill_jobs")
ACTIVE = {"queued", "running"}
TERMINAL = {"succeeded", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_root() -> Path:
    """The SIDISH repo root (contains sidish_cli.py)."""
    here = Path(__file__).resolve()
    for cand in [here.parents[3] if len(here.parents) > 3 else None, Path.cwd().resolve()]:
        if cand and (cand / "sidish_cli.py").exists():
            return cand
    return here.parents[3]


def job_root() -> Path:
    root = repo_root() / JOB_ROOT_REL
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_job_id(prefix: str = "job") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def job_dir(job_id: str) -> Path:
    value = str(job_id or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("invalid SIDISH job identifier")
    root = job_root().resolve()
    path = (root / value).resolve()
    if root not in path.parents:
        raise ValueError("job path escapes the SIDISH job root")
    return path


def _job_file(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_job(job_id: str) -> dict:
    f = _job_file(job_id)
    if not f.exists():
        raise FileNotFoundError(f"unknown job_id: {job_id}")
    return json.loads(f.read_text(encoding="utf-8"))


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _refresh(rec: dict) -> dict:
    """Reconcile a job's recorded status with the OS + exit-code file."""
    if rec.get("status") in ACTIVE:
        alive = _pid_alive(rec.get("pid"))
        rc_file = job_dir(rec["job_id"]) / "exit_code"
        if rc_file.exists():
            code = int(rc_file.read_text().strip() or "1")
            rec["status"] = "succeeded" if code == 0 else "failed"
            rec["exit_code"] = code
            rec["updated_at"] = utc_now()
            atomic_write_json(_job_file(rec["job_id"]), rec)
        elif not alive:                      # died without writing exit code
            rec["status"] = "failed"
            rec["exit_code"] = rec.get("exit_code", -1)
            rec["updated_at"] = utc_now()
            atomic_write_json(_job_file(rec["job_id"]), rec)
    return rec


def _launch(job_id: str, argv: list[str], meta: dict) -> dict:
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    log = d / "job.log"
    rc = d / "exit_code"
    if rc.exists():
        rc.unlink()
    root = repo_root()
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    runner = Path(__file__).with_name("sidish_job_runner.py")
    rec = {"job_id": job_id, "status": "queued", "pid": None, "argv": argv,
           "created_at": utc_now(), "updated_at": utc_now(), "log": str(log),
           "repo_root": str(root), "job_dir": str(d), **meta}
    atomic_write_json(_job_file(job_id), rec)
    proc = subprocess.Popen([sys.executable, str(runner), str(_job_file(job_id))],
                            cwd=str(root), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True, env=env)
    rec["pid"] = proc.pid
    rec["status"] = "running"
    rec["updated_at"] = utc_now()
    atomic_write_json(_job_file(job_id), rec)
    return rec


def cmd_submit(args) -> dict:
    jid = new_job_id(args.prefix or "run")
    py = sys.executable
    argv = [py, str(repo_root() / "sidish_cli.py"), "run", "-c", str(args.config)]
    return _launch(jid, argv, {"job_name": args.job_name or jid, "config": str(args.config),
                               "kind": "workflow"})


def cmd_submit_analysis(args) -> dict:
    jid = new_job_id("report")
    py = sys.executable
    argv = [py, str(repo_root() / "sidish_cli.py"), "report"]
    if args.config:
        argv += ["-c", str(args.config)]
    if args.patient:
        argv += ["--patient", args.patient]
    if args.pdf:
        argv += ["--pdf"]
    if args.use_llm:
        argv += ["--use-llm"]
    return _launch(jid, argv, {"job_name": args.job_name or jid, "kind": "report",
                               "patient": args.patient})


def cmd_status(args) -> dict:
    return _refresh(read_job(args.job_id))


def cmd_list(args) -> dict:
    jobs = []
    for jf in sorted(job_root().glob("*/job.json")):
        try:
            jobs.append(_refresh(json.loads(jf.read_text())))
        except Exception:
            continue
    return {"n": len(jobs),
            "jobs": [{k: j.get(k) for k in ("job_id", "status", "kind", "job_name", "created_at")}
                     for j in jobs]}


def cmd_tail(args) -> dict:
    log = job_dir(args.job_id) / "job.log"
    if not log.exists():
        return {"job_id": args.job_id, "log": None, "lines": []}
    lines = log.read_text(errors="replace").splitlines()[-int(args.lines):]
    return {"job_id": args.job_id, "log": str(log), "lines": lines}


def cmd_cancel(args) -> dict:
    rec = _refresh(read_job(args.job_id))
    if rec.get("status") in ACTIVE and _pid_alive(rec.get("pid")):
        try:
            os.killpg(os.getpgid(rec["pid"]), signal.SIGTERM)
        except Exception:
            try:
                os.kill(rec["pid"], signal.SIGTERM)
            except Exception:
                pass
    rec["status"] = "cancelled"
    rec["updated_at"] = utc_now()
    atomic_write_json(_job_file(args.job_id), rec)
    return rec


def cmd_resume(args) -> dict:
    rec = read_job(args.job_id)
    if rec.get("status") not in TERMINAL:
        return {"error": f"job {args.job_id} is {rec.get('status')}; only terminal jobs can resume"}
    jid = new_job_id(str(rec.get("kind", "run")))
    return _launch(jid, rec["argv"], {"job_name": rec.get("job_name"), "kind": rec.get("kind"),
                                      "config": rec.get("config"), "resumed_from": args.job_id})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sidish_job", description="Durable SIDISH job orchestration")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("submit"); s.add_argument("--config", required=True)
    s.add_argument("--job-name", default=None); s.add_argument("--prefix", default="run")
    s.set_defaults(fn=cmd_submit)

    sa = sub.add_parser("submit-analysis"); sa.add_argument("--config", default=None)
    sa.add_argument("--patient", default=None); sa.add_argument("--job-name", default=None)
    sa.add_argument("--pdf", action="store_true"); sa.add_argument("--use-llm", action="store_true")
    sa.set_defaults(fn=cmd_submit_analysis)

    st = sub.add_parser("status"); st.add_argument("job_id"); st.set_defaults(fn=cmd_status)
    ls = sub.add_parser("list"); ls.set_defaults(fn=cmd_list)
    tl = sub.add_parser("tail"); tl.add_argument("job_id"); tl.add_argument("--lines", default=80)
    tl.set_defaults(fn=cmd_tail)
    cn = sub.add_parser("cancel"); cn.add_argument("job_id"); cn.set_defaults(fn=cmd_cancel)
    rs = sub.add_parser("resume"); rs.add_argument("job_id"); rs.set_defaults(fn=cmd_resume)

    args = p.parse_args(argv)
    print(json.dumps(args.fn(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
