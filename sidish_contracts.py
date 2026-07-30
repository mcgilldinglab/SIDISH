"""Structured contracts for clinician-facing SIDISH case analysis.

The language model is never the source of record.  Analysis code writes a
``SIDISHCaseResult`` and both chat and report layers read from that bundle.
Every result is labelled with its scope and provenance so cohort results cannot
silently be presented as patient-specific predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import hashlib
import json
import re


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class EvidenceScope(str, Enum):
    SAMPLE = "sample-specific observation"
    PATIENT_MODEL = "patient-specific model perturbation"
    DATASET = "dataset-wide observation"
    DATASET_MODEL = "dataset-wide model perturbation"
    COHORT = "cohort-level association"
    EXTERNAL = "external reference"
    HYPOTHESIS = "computational hypothesis"


class QualityStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


@dataclass
class Provenance:
    method: str
    source_dataset: str
    model_version: str
    software_version: str = "SIDISH-Agent 2.4.0"
    generated_at: str = field(default_factory=utc_now)
    parameters: dict[str, Any] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    key: str
    title: str
    value: Any
    scope: EvidenceScope
    provenance: Provenance
    unit: str | None = None
    quality: QualityStatus = QualityStatus.NOT_ASSESSED
    limitations: list[str] = field(default_factory=list)
    references: list[dict[str, str]] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.key or not re.fullmatch(r"[a-z0-9_.-]+", self.key):
            errors.append(f"invalid evidence key: {self.key!r}")
        if not self.provenance.method:
            errors.append(f"{self.key}: missing method")
        if not self.provenance.source_dataset:
            errors.append(f"{self.key}: missing source dataset")
        if not self.provenance.model_version:
            errors.append(f"{self.key}: missing model version")
        return errors


@dataclass
class CaseMetadata:
    case_id: str
    cancer_type: str
    disease_label: str
    specimen_id: str | None = None
    patient_id: str | None = None
    patient_column: str | None = None
    analysis_scope: str = "patient"
    training_iterations: int = 10
    cancer_subtype: str | None = None
    stage: str | None = None
    prior_treatment: str | None = None
    data_source: str | None = None
    research_use_only: bool = True
    demo: bool = False


@dataclass
class QualityCheck:
    name: str
    status: QualityStatus
    detail: str
    value: Any = None
    threshold: Any = None


@dataclass
class Artifact:
    kind: str
    path: str
    sha256: str | None = None
    description: str = ""

    @classmethod
    def from_path(cls, kind: str, path: str | Path, description: str = "") -> "Artifact":
        p = Path(path)
        digest = None
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
            digest = h.hexdigest()
        return cls(kind=kind, path=str(p), sha256=digest, description=description)


@dataclass
class ReviewSignoff:
    scientific_reviewer: str | None = None
    clinical_reviewer: str | None = None
    reviewed_at: str | None = None
    status: str = "not_reviewed"
    comments: str = ""


@dataclass
class SIDISHCaseResult:
    schema_version: str
    metadata: CaseMetadata
    status: str = "created"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    qc: list[QualityCheck] = field(default_factory=list)
    evidence: dict[str, EvidenceItem] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    signoff: ReviewSignoff = field(default_factory=ReviewSignoff)
    audit: list[dict[str, Any]] = field(default_factory=list)

    def add_evidence(self, item: EvidenceItem) -> None:
        problems = item.validate()
        if problems:
            raise ValueError("; ".join(problems))
        self.evidence[item.key] = item
        self.updated_at = utc_now()

    def record(self, action: str, actor: str = "system", detail: dict[str, Any] | None = None) -> None:
        self.audit.append({"at": utc_now(), "actor": actor, "action": action,
                           "detail": detail or {}})
        self.updated_at = utc_now()

    def validate(self, for_report: bool = False) -> list[str]:
        errors: list[str] = []
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.metadata.case_id):
            errors.append("case_id must be a safe pseudonymous identifier")
        if not self.metadata.cancer_type:
            errors.append("cancer_type is required")
        for item in self.evidence.values():
            errors.extend(item.validate())
        if self.metadata.demo and not self.metadata.research_use_only:
            errors.append("demo cases must be research-use only")
        if for_report:
            if self.status != "analysis_complete":
                errors.append("analysis must be complete before report generation")
            if not self.evidence:
                errors.append("report requires at least one validated evidence item")
            failed = [q.name for q in self.qc if q.status == QualityStatus.FAIL]
            if failed:
                errors.append("failed QC checks: " + ", ".join(failed))
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(p)
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SIDISHCaseResult":
        md = CaseMetadata(**data["metadata"])
        qc = [QualityCheck(name=q["name"], status=QualityStatus(q["status"]),
                           detail=q["detail"], value=q.get("value"),
                           threshold=q.get("threshold")) for q in data.get("qc", [])]
        evidence: dict[str, EvidenceItem] = {}
        for key, raw in data.get("evidence", {}).items():
            prov = Provenance(**raw["provenance"])
            evidence[key] = EvidenceItem(
                key=raw["key"], title=raw["title"], value=raw.get("value"),
                unit=raw.get("unit"), scope=EvidenceScope(raw["scope"]),
                provenance=prov, quality=QualityStatus(raw.get("quality", "not_assessed")),
                limitations=list(raw.get("limitations", [])),
                references=list(raw.get("references", [])),
                supports=list(raw.get("supports", [])),
            )
        return cls(
            schema_version=data["schema_version"], metadata=md,
            status=data.get("status", "created"), created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()), qc=qc, evidence=evidence,
            artifacts=[Artifact(**a) for a in data.get("artifacts", [])],
            warnings=list(data.get("warnings", [])), errors=list(data.get("errors", [])),
            signoff=ReviewSignoff(**data.get("signoff", {})),
            audit=list(data.get("audit", [])),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SIDISHCaseResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def scope_label(value: str | EvidenceScope) -> str:
    return value.value if isinstance(value, EvidenceScope) else EvidenceScope(value).value
