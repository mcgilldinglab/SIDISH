"""Filesystem-backed, case-isolated storage for the SIDISH application."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
import hashlib
import json
import os
import re
import secrets

from sidish_contracts import Artifact, CaseMetadata, SIDISHCaseResult, utc_now


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "outputs" / "cases"
MAX_SINGLE_CELL_BYTES = int(os.environ.get("SIDISH_MAX_H5AD_BYTES", str(20 * 1024**3)))
MAX_BULK_BYTES = int(os.environ.get("SIDISH_MAX_BULK_BYTES", str(2 * 1024**3)))


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-.")
    if not cleaned:
        cleaned = "case-" + secrets.token_hex(6)
    if len(cleaned) > 128:
        cleaned = cleaned[:112] + "-" + hashlib.sha256(cleaned.encode()).hexdigest()[:12]
    return cleaned


class CaseStore:
    def __init__(self, root: str | Path | None = None):
        configured = root or os.environ.get("SIDISH_CASE_ROOT") or DEFAULT_ROOT
        self.root = Path(configured).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def case_dir(self, case_id: str) -> Path:
        cid = safe_id(case_id)
        path = (self.root / cid).resolve()
        if self.root not in path.parents:
            raise ValueError("case path escapes the configured case root")
        return path

    def create(self, metadata: CaseMetadata) -> SIDISHCaseResult:
        metadata.case_id = safe_id(metadata.case_id)
        result = SIDISHCaseResult(schema_version="2.0", metadata=metadata)
        result.record("case_created")
        d = self.case_dir(metadata.case_id)
        (d / "inputs").mkdir(parents=True, exist_ok=False)
        (d / "artifacts").mkdir()
        (d / "jobs").mkdir()
        result.save(d / "result.json")
        return result

    def get(self, case_id: str) -> SIDISHCaseResult:
        return SIDISHCaseResult.load(self.case_dir(case_id) / "result.json")

    def save(self, result: SIDISHCaseResult) -> Path:
        return result.save(self.case_dir(result.metadata.case_id) / "result.json")

    def list_cases(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*/result.json"), reverse=True):
            try:
                r = SIDISHCaseResult.load(p)
                out.append({"case_id": r.metadata.case_id, "cancer_type": r.metadata.cancer_type,
                            "status": r.status, "updated_at": r.updated_at,
                            "demo": r.metadata.demo})
            except Exception:
                continue
        return out

    def save_upload(self, case_id: str, filename: str, source: bytes | BinaryIO,
                    kind: str) -> Path:
        allowed = {"single_cell": {".h5ad"}, "bulk_survival": {".csv"}}
        if kind not in allowed:
            raise ValueError(f"unsupported upload kind: {kind}")
        name = Path(filename).name
        if Path(name).suffix.lower() not in allowed[kind]:
            raise ValueError(f"{kind} upload must be one of {sorted(allowed[kind])}")
        limit = MAX_SINGLE_CELL_BYTES if kind == "single_cell" else MAX_BULK_BYTES
        target = self.case_dir(case_id) / "inputs" / f"{kind}{Path(name).suffix.lower()}"
        tmp = target.with_suffix(target.suffix + ".upload")
        total = 0
        digest = hashlib.sha256()
        with tmp.open("wb") as handle:
            if isinstance(source, (bytes, bytearray)):
                if len(source) > limit:
                    raise ValueError(f"upload exceeds {limit} bytes")
                handle.write(source)
                digest.update(source)
                total = len(source)
            else:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > limit:
                        handle.close()
                        tmp.unlink(missing_ok=True)
                        raise ValueError(f"upload exceeds {limit} bytes")
                    handle.write(block)
                    digest.update(block)
        tmp.replace(target)
        result = self.get(case_id)
        artifact_kind = f"input_{kind}"
        result.artifacts = [a for a in result.artifacts if a.kind != artifact_kind]
        result.artifacts.append(Artifact(
            kind=artifact_kind, path=str(target), sha256=digest.hexdigest(),
            description=f"Validated uploaded {kind.replace('_', ' ')} input"))
        result.record("file_uploaded", detail={"kind": kind, "filename": name,
                                                "stored_as": target.name, "bytes": total,
                                                "sha256": digest.hexdigest()})
        self.save(result)
        return target

    def append_chat(self, case_id: str, role: str, content: str,
                    metadata: dict | None = None, figures: list[str] | None = None,
                    table: list[dict] | dict | None = None) -> None:
        if role not in {"user", "assistant", "tool"}:
            raise ValueError("invalid chat role")
        path = self.case_dir(case_id) / "chat.jsonl"
        event = {"at": utc_now(), "role": role, "content": content,
                 "metadata": metadata or {}, "figures": list(figures or []),
                 "table": table}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def read_chat(self, case_id: str, limit: int = 200) -> list[dict]:
        path = self.case_dir(case_id) / "chat.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def assert_case_path(self, case_id: str, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        case = self.case_dir(case_id)
        if case != p and case not in p.parents:
            raise PermissionError("agent tools may only access files inside the active case")
        return p
