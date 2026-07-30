#!/usr/bin/env python3
"""Local web console for SIDISH — jobs dashboard, live log tail, and report viewer.
Mirrors UNAGI's monitor web console but dependency-free (Python stdlib http.server).

    python skills/sidish-report-orchestrator/scripts/sidish_monitor_web.py serve --port 36872
    # open http://127.0.0.1:36872
"""
from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parents[3] if len(here.parents) > 3 else None, Path.cwd().resolve()]:
        if cand and (cand / "sidish_cli.py").exists():
            return cand
    return here.parents[3]


ROOT = repo_root()
JOBS = ROOT / "outputs" / "skill_jobs"
REPORTS = ROOT / "reports"

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#eef1f5;color:#1a2330}
header{background:#1f4e79;color:#fff;padding:14px 24px;font-weight:700}
.wrap{max-width:1000px;margin:20px auto;padding:0 16px}
.card{background:#fff;border:1px solid #d9dee6;border-radius:6px;margin:14px 0;padding:14px 18px}
h2{font-size:16px;margin:.2em 0 .6em;color:#1f4e79}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #e3e7ee;padding:6px 9px;text-align:left}
th{background:#f4f6f9}
a{color:#1f4e79;text-decoration:none} a:hover{text-decoration:underline}
.badge{padding:1px 8px;border-radius:10px;font-size:11px;font-weight:700;color:#fff}
.running{background:#2d7dd2}.succeeded{background:#2e9e5b}.failed{background:#c0392b}
.cancelled{background:#888}.queued{background:#e0a800}
pre{background:#0d1117;color:#c9d1d9;padding:12px;border-radius:6px;overflow:auto;font-size:12px;max-height:460px}
"""


def _jobs():
    out = []
    if JOBS.exists():
        for jf in sorted(JOBS.glob("*/job.json"), reverse=True):
            try:
                out.append(json.loads(jf.read_text()))
            except Exception:
                continue
    return out


def _reports():
    if not REPORTS.exists():
        return []
    return sorted([p.name for p in REPORTS.glob("*.html")])


def _page(body):
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<meta http-equiv='refresh' content='10'><style>{CSS}</style>"
            f"<title>SIDISH monitor</title></head><body>"
            f"<header>SIDISH · Report Orchestrator — monitor</header>"
            f"<div class='wrap'>{body}</div></body></html>")


def _dashboard():
    jrows = ""
    for j in _jobs():
        st = html.escape(str(j.get("status", "?")))
        jid = html.escape(str(j.get("job_id", "")))
        jrows += (f"<tr><td><a href='/job/{jid}'>{jid}</a></td>"
                  f"<td><span class='badge {st}'>{st}</span></td>"
                  f"<td>{html.escape(str(j.get('kind','')))}</td>"
                  f"<td>{html.escape(str(j.get('job_name','')))}</td>"
                  f"<td>{html.escape(str(j.get('created_at','')))}</td></tr>")
    jtbl = (f"<table><tr><th>job</th><th>status</th><th>kind</th><th>name</th><th>created</th></tr>"
            f"{jrows or '<tr><td colspan=5>no jobs yet</td></tr>'}</table>")
    rrows = "".join(f"<li><a href='/report/{html.escape(n)}' target='_blank'>{html.escape(n)}</a></li>"
                    for n in _reports())
    return _page(f"<div class='card'><h2>Jobs</h2>{jtbl}</div>"
                 f"<div class='card'><h2>Reports</h2><ul>{rrows or '<li>none yet</li>'}</ul></div>"
                 f"<div class='card' style='color:#5b6675;font-size:12px'>Auto-refreshes every 10s · "
                 f"repo: {html.escape(str(ROOT))}</div>")


def _job_detail(job_id):
    d = JOBS / job_id
    jf = d / "job.json"
    if not jf.exists():
        return _page(f"<div class='card'>Unknown job: {html.escape(job_id)} · "
                     f"<a href='/'>back</a></div>")
    rec = json.loads(jf.read_text())
    log = (d / "job.log")
    tail = "\n".join(log.read_text(errors="replace").splitlines()[-200:]) if log.exists() else "(no log yet)"
    meta = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
                   for k, v in rec.items() if k != "argv")
    return _page(f"<div class='card'><a href='/'>&larr; back</a><h2>Job {html.escape(job_id)}</h2>"
                 f"<table>{meta}</table></div>"
                 f"<div class='card'><h2>Log (tail 200)</h2><pre>{html.escape(tail)}</pre></div>")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/" or path == "":
            self._send(_dashboard())
        elif path.startswith("/job/"):
            self._send(_job_detail(path[len("/job/"):]))
        elif path.startswith("/report/"):
            name = os.path.basename(path[len("/report/"):])
            f = REPORTS / name
            if f.exists() and f.suffix in (".html", ".pdf"):
                ctype = "application/pdf" if f.suffix == ".pdf" else "text/html; charset=utf-8"
                self._send(f.read_bytes(), ctype=ctype)
            else:
                self._send("not found", code=404)
        elif path == "/api/jobs":
            self._send(json.dumps(_jobs(), default=str), ctype="application/json")
        else:
            self._send("not found", code=404)


def serve(host, port):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"SIDISH monitor on http://{host}:{port}  (repo {ROOT})  Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=36872)
    args = ap.parse_args()
    serve(args.host, args.port)
