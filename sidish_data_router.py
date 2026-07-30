"""Cancer-aware data routing and strict SIDISH bulk-survival validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import csv
import os


HERE = Path(__file__).resolve().parent

CANCER_ALIASES = {
    "breast": "breast", "brca": "breast", "tnbc": "breast", "breast cancer": "breast",
    "lung": "lung", "nsclc": "lung", "luad": "lung", "lusc": "lung", "lung cancer": "lung",
    "pancreatic": "pancreatic", "pancreas": "pancreatic", "pdac": "pancreatic",
    "pancreatic cancer": "pancreatic",
}

MANUSCRIPT_ASSETS = {
    "breast": {
        "bulk_env": "SIDISH_BREAST_BULK",
        "run_env": "SIDISH_BREAST_RUN",
        "bulk_default": HERE / "BREAST_CANCER" / "bulk_result.csv",
        "run_default": HERE / "BREAST_CANCER",
        "reference": "SIDISH manuscript breast bulk-survival cohort",
    },
    "lung": {
        "bulk_env": "SIDISH_LUNG_BULK",
        "run_env": "SIDISH_LUNG_RUN",
        "bulk_default": HERE / "LUNG_CANCER" / "bulk_result.csv",
        "run_default": HERE / "LUNG_CANCER",
        "reference": "SIDISH manuscript lung bulk-survival cohort",
    },
    "pancreatic": {
        "bulk_env": "SIDISH_PANCREATIC_BULK",
        "run_env": "SIDISH_PANCREATIC_RUN",
        "bulk_default": HERE / "PANCREAS_CANCER" / "bulk_result.csv",
        "run_default": HERE / "PANCREAS_CANCER",
        "reference": "SIDISH manuscript pancreatic bulk-survival cohort",
    },
}


@dataclass
class BulkValidation:
    ok: bool
    path: str
    n_samples: int = 0
    n_genes: int = 0
    errors: list[str] = None
    warnings: list[str] = None

    def __post_init__(self):
        self.errors = list(self.errors or [])
        self.warnings = list(self.warnings or [])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataRoute:
    cancer_type: str
    known_cancer: bool
    bulk_path: str | None
    run_dir: str | None
    bulk_source: str
    needs_user_bulk: bool
    ready: bool
    message: str
    validation: BulkValidation | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out


def normalize_cancer_type(value: str) -> str:
    raw = " ".join(str(value or "").strip().lower().replace("_", " ").split())
    return CANCER_ALIASES.get(raw, raw)


def _numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def validate_bulk_survival(path: str | Path, min_samples: int = 10,
                           min_genes: int = 50) -> BulkValidation:
    """Validate the complete table, not just its header.

    Required layout: ``duration,event,<gene1>,...`` with numeric non-negative
    duration, binary event, unique gene columns, and numeric expression values.
    """
    p = Path(path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not p.is_file():
        return BulkValidation(False, str(p), errors=["bulk-survival CSV was not found"])
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return BulkValidation(False, str(p), errors=["CSV is empty"])
            header = [h.strip() for h in header]
            if len(header) < 3:
                errors.append("CSV requires duration, event, and at least one gene column")
            if header[:2] != ["duration", "event"]:
                errors.append("first two columns must be exactly duration,event")
            genes = header[2:]
            if len(set(genes)) != len(genes):
                errors.append("gene column names must be unique")
            if any(not g for g in genes):
                errors.append("gene column names cannot be blank")
            rows = 0
            for line_no, row in enumerate(reader, start=2):
                if not row or not any(str(x).strip() for x in row):
                    continue
                rows += 1
                if len(row) != len(header):
                    errors.append(f"row {line_no} has {len(row)} values; expected {len(header)}")
                    if len(errors) >= 20:
                        break
                    continue
                if not _numeric(row[0]) or float(row[0]) < 0:
                    errors.append(f"row {line_no}: duration must be numeric and non-negative")
                if str(row[1]).strip() not in {"0", "1", "0.0", "1.0"}:
                    errors.append(f"row {line_no}: event must be 0 or 1")
                bad_gene = next((header[i] for i, x in enumerate(row[2:], start=2)
                                 if str(x).strip() == "" or not _numeric(x)), None)
                if bad_gene:
                    errors.append(f"row {line_no}: gene {bad_gene} is blank or non-numeric")
                if len(errors) >= 20:
                    break
    except (OSError, UnicodeError, csv.Error) as exc:
        return BulkValidation(False, str(p), errors=[f"could not read CSV: {exc}"])
    if rows < min_samples:
        errors.append(f"at least {min_samples} bulk samples are required; found {rows}")
    if len(header[2:]) < min_genes:
        errors.append(f"at least {min_genes} gene columns are required; found {len(header[2:])}")
    if rows < 50:
        warnings.append("small survival cohort; estimates may be unstable")
    return BulkValidation(not errors, str(p), rows, len(header[2:]), errors, warnings)


def _asset_path(spec: dict[str, Any], env_key: str, default_key: str) -> Path:
    configured = os.environ.get(spec[env_key])
    return Path(configured).expanduser().resolve() if configured else Path(spec[default_key]).resolve()


def route_case_data(cancer_type: str, user_bulk_path: str | None = None) -> DataRoute:
    """Apply the SIDISH manuscript-data policy requested by the project owner."""
    cancer = normalize_cancer_type(cancer_type)
    if cancer in MANUSCRIPT_ASSETS:
        spec = MANUSCRIPT_ASSETS[cancer]
        bulk = _asset_path(spec, "bulk_env", "bulk_default")
        run_dir = _asset_path(spec, "run_env", "run_default")
        if not bulk.is_file():
            return DataRoute(
                cancer, True, str(bulk), str(run_dir), spec["reference"], False, False,
                f"The {cancer} manuscript bulk-survival asset is configured but unavailable. "
                f"Set {spec['bulk_env']} to its location; do not substitute an unrelated user cohort.",
            )
        valid = validate_bulk_survival(bulk)
        return DataRoute(
            cancer, True, str(bulk), str(run_dir), spec["reference"], False, valid.ok,
            (f"Using the locked {cancer} manuscript bulk-survival reference."
             if valid.ok else "The configured manuscript bulk-survival table failed validation."),
            valid,
        )
    if not user_bulk_path:
        return DataRoute(
            cancer, False, None, None, "user supplied", True, False,
            "This disease is outside breast, lung, and pancreatic SIDISH references. "
            "Upload a matched bulk RNA-seq survival CSV with columns duration,event,<genes...>.",
        )
    valid = validate_bulk_survival(user_bulk_path)
    return DataRoute(
        cancer, False, str(Path(user_bulk_path).expanduser().resolve()), None,
        "user-supplied matched bulk RNA-seq and survival", True, valid.ok,
        "User bulk-survival table validated." if valid.ok else "User bulk-survival table is invalid.",
        valid,
    )
