"""Case-oriented SIDISH analysis that produces the authoritative result bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib
import json

from sidish_contracts import (
    Artifact, EvidenceItem, EvidenceScope, Provenance, QualityCheck,
    QualityStatus, SIDISHCaseResult,
)
from sidish_data_router import route_case_data


HERE = Path(__file__).resolve().parent
MIN_TRAINING_ITERATIONS = 1
MAX_TRAINING_ITERATIONS = 100


def validate_training_iterations(iterations: int | str) -> int:
    """Return a safe integer iteration count for a SIDISH training run."""
    raw = str(iterations).strip()
    if not raw.isdigit():
        raise ValueError("SIDISH training iterations must be a whole number")
    value = int(raw)
    if not MIN_TRAINING_ITERATIONS <= value <= MAX_TRAINING_ITERATIONS:
        raise ValueError(
            f"SIDISH training iterations must be between {MIN_TRAINING_ITERATIONS} "
            f"and {MAX_TRAINING_ITERATIONS}")
    return value


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _prov(method: str, source: str, model_version: str,
          parameters: dict[str, Any] | None = None,
          hashes: dict[str, str] | None = None) -> Provenance:
    return Provenance(method=method, source_dataset=source, model_version=model_version,
                      parameters=parameters or {}, source_hashes=hashes or {})


def inspect_patient_columns(single_cell_path: str | Path,
                            max_patients: int = 1000) -> dict[str, Any]:
    """Read only AnnData metadata and return usable patient/sample columns.

    Columns must contain between 1 and ``max_patients`` non-empty categories.
    Values are returned as strings so the UI can safely present an exact choice.
    """
    import scanpy as sc

    path = Path(single_cell_path).resolve()
    ad = sc.read_h5ad(path, backed="r")
    try:
        candidates: dict[str, list[str]] = {}
        for column in map(str, ad.obs.columns):
            series = ad.obs[column]
            unique = series.dropna().astype(str).unique()
            values = sorted({value.strip() for value in unique if value.strip()})
            if 1 <= len(values) <= max_patients:
                candidates[column] = values
        index_prefix = sorted({str(value).split("_")[0] for value in ad.obs_names
                               if str(value).strip()})
        if 1 <= len(index_prefix) <= max_patients:
            candidates["__index_prefix__"] = index_prefix
        return {"n_cells": int(ad.n_obs), "n_genes": int(ad.n_vars),
                "columns": candidates}
    finally:
        ad.file.close()


def _write_training_config(result: SIDISHCaseResult, single_cell_path: Path,
                           route, device: str) -> Path:
    import yaml

    case_dir = single_cell_path.parent.parent
    run_dir = case_dir / "run"
    config = {
        "mode": "training", "random_seed": 0,
        "case": {"result_path": str(case_dir / "result.json")},
        "data": {
            "adata_path": str(single_cell_path), "bulk_csv": route.bulk_path,
            "cancer_type": route.cancer_type, "bulk_source": route.bulk_source,
            "align_genes": True, "patient_column": result.metadata.patient_column,
            "analysis_scope": result.metadata.analysis_scope,
            "selected_patient": result.metadata.patient_id,
        },
        "training": {
            "device": device, "path": str(run_dir),
            "phase1": [225, 20, 16, [512, 256, 64], 256, "Adam", 0.001, 0.0001, 0, "Dense"],
            "phase2": [1000, 64, 1.0e-6, 0, 0.2, 512],
            "iterations": validate_training_iterations(result.metadata.training_iterations),
            "percentile": 0.90, "steepness": 1.0,
        },
    }
    config_path = case_dir / "jobs" / "training.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def configure_patient_selection(result: SIDISHCaseResult, single_cell_path: str | Path,
                                patient_column: str, patient_id: str | None,
                                user_bulk_path: str | None = None,
                                device: str = "cuda:1",
                                analysis_scope: str = "patient",
                                training_iterations: int = 10) -> tuple[SIDISHCaseResult, dict]:
    """Persist a patient-level or entire-dataset analysis selection."""
    route = route_case_data(result.metadata.cancer_type, user_bulk_path)
    if not route.ready:
        raise ValueError(route.message)
    schema = inspect_patient_columns(single_cell_path)
    choices = schema["columns"].get(patient_column)
    if choices is None:
        raise ValueError("selected patient column is not available in this AnnData file")
    if analysis_scope not in {"patient", "cohort"}:
        raise ValueError("analysis_scope must be 'patient' or 'cohort'")
    if analysis_scope == "patient" and str(patient_id) not in choices:
        raise ValueError("selected patient is not present in the chosen column")
    result.metadata.patient_column = patient_column
    result.metadata.analysis_scope = analysis_scope
    result.metadata.patient_id = str(patient_id) if analysis_scope == "patient" else None
    result.metadata.training_iterations = validate_training_iterations(training_iterations)
    config_path = _write_training_config(result, Path(single_cell_path).resolve(), route, device)
    result.status = "training_required"
    result.errors = []
    result.artifacts = [a for a in result.artifacts if a.kind != "training_config"]
    result.artifacts.append(Artifact.from_path(
        "training_config", config_path, "Validated case-specific SIDISH training configuration"))
    result.record("analysis_scope_selected", actor="user",
                  detail={"patient_column": patient_column,
                          "patient_id": result.metadata.patient_id,
                          "analysis_scope": analysis_scope,
                          "training_iterations": result.metadata.training_iterations})
    result.record("training_prepared", detail={"config": str(config_path),
                                                "bulk_source": route.bulk_source})
    return result, {**route.to_dict(), "training_config": str(config_path),
                    "patient_column": patient_column, "patient_id": result.metadata.patient_id,
                    "analysis_scope": analysis_scope,
                    "training_iterations": result.metadata.training_iterations}


def change_training_iterations(result: SIDISHCaseResult, single_cell_path: str | Path,
                               iterations: int | str) -> SIDISHCaseResult:
    """Change an unstarted case's training iterations and update its job config."""
    if result.status != "training_required":
        raise ValueError("training iterations can only be changed before training is queued")
    value = validate_training_iterations(iterations)
    config_path = Path(single_cell_path).resolve().parent.parent / "jobs" / "training.yaml"
    if not config_path.is_file():
        raise ValueError("the case training configuration has not been created yet")
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config.setdefault("training", {})["iterations"] = value
    tmp = config_path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    tmp.replace(config_path)
    previous = result.metadata.training_iterations
    result.metadata.training_iterations = value
    result.record("training_iterations_changed", actor="user",
                  detail={"previous": previous, "iterations": value})
    return result


def change_selected_patient(result: SIDISHCaseResult, single_cell_path: str | Path,
                            patient_id: str | None,
                            analysis_scope: str = "patient") -> SIDISHCaseResult:
    """Switch between one patient and the entire dataset without retraining."""
    column = result.metadata.patient_column
    if not column:
        raise ValueError("a patient column has not been configured")
    choices = inspect_patient_columns(single_cell_path)["columns"].get(column, [])
    if analysis_scope not in {"patient", "cohort"}:
        raise ValueError("analysis_scope must be 'patient' or 'cohort'")
    if analysis_scope == "patient" and str(patient_id) not in choices:
        raise ValueError("selected patient is not present in the configured patient column")
    previous = result.metadata.patient_id
    previous_scope = result.metadata.analysis_scope
    result.metadata.analysis_scope = analysis_scope
    result.metadata.patient_id = str(patient_id) if analysis_scope == "patient" else None
    scope_dependent = (
        "sample.", "cohort.high_risk", "cohort.celltype", "model.target_perturbation",
        "model.pathway_perturbation", "hypothesis.drug_mapping", "hypothesis.mechanism")
    result.evidence = {key: item for key, item in result.evidence.items()
                       if not key.startswith(scope_dependent)}
    result.artifacts = [a for a in result.artifacts
                        if a.kind not in {"report_html", "report_pdf"} and a.kind != "figure"]
    if result.status == "analysis_complete":
        result.status = "analysis_partial" if result.evidence else "training_complete"
    result.record("analysis_scope_changed", actor="user",
                  detail={"previous_scope": previous_scope, "previous_patient": previous,
                          "analysis_scope": analysis_scope,
                          "patient_id": result.metadata.patient_id,
                          "patient_column": column})
    return result


def prepare_uploaded_case(result: SIDISHCaseResult, single_cell_path: str,
                          user_bulk_path: str | None = None,
                          device: str = "cuda:1") -> tuple[SIDISHCaseResult, dict]:
    """Validate inputs and prepare a reproducible training config.

    Known breast/lung/pancreatic cases always use their locked manuscript bulk
    reference. Other diseases require a validated user bulk-survival table.
    """
    import scanpy as sc

    route = route_case_data(result.metadata.cancer_type, user_bulk_path)
    result.record("data_routed", detail=route.to_dict())
    if not route.ready:
        result.status = "input_required" if route.needs_user_bulk and not user_bulk_path else "input_invalid"
        result.errors = list(route.validation.errors if route.validation else [route.message])
        return result, route.to_dict()

    p = Path(single_cell_path).resolve()
    try:
        ad = sc.read_h5ad(p, backed="r")
        n_cells, n_genes = int(ad.n_obs), int(ad.n_vars)
        single_cell_genes = list(map(str, ad.var_names))
        ad.file.close()
    except Exception as exc:
        result.status = "input_invalid"
        result.errors = [f"single-cell h5ad could not be read: {exc}"]
        return result, route.to_dict()

    with Path(route.bulk_path).open("r", encoding="utf-8-sig", newline="") as handle:
        bulk_header = next(csv.reader(handle))
    bulk_genes = [str(g).strip() for g in bulk_header[2:]]
    single_gene_set = set(single_cell_genes)
    common_genes = [g for g in bulk_genes if g in single_gene_set]
    bulk_overlap = len(common_genes) / max(len(bulk_genes), 1)
    alignment_ok = (len(single_cell_genes) == len(single_gene_set)
                    and len(common_genes) >= 500 and bulk_overlap >= 0.80)

    result.qc.extend([
        QualityCheck("single-cell file readable", QualityStatus.PASS, str(p)),
        QualityCheck("cell count", QualityStatus.PASS if n_cells >= 100 else QualityStatus.WARNING,
                     f"{n_cells:,} cells", n_cells, ">=100"),
        QualityCheck("gene count", QualityStatus.PASS if n_genes >= 500 else QualityStatus.WARNING,
                     f"{n_genes:,} genes", n_genes, ">=500"),
        QualityCheck("bulk-survival schema", QualityStatus.PASS,
                     route.validation.path if route.validation else route.message,
                     route.validation.to_dict() if route.validation else None),
        QualityCheck(
            "single-cell/bulk gene alignment",
            QualityStatus.PASS if alignment_ok else QualityStatus.FAIL,
            (f"{len(common_genes):,} of {len(bulk_genes):,} bulk genes present in single-cell "
             f"data ({bulk_overlap:.1%}); training will use bulk-column order"),
            {"common_genes": len(common_genes), "bulk_genes": len(bulk_genes),
             "single_cell_genes": len(single_cell_genes), "bulk_overlap": bulk_overlap},
            ">=500 common genes, >=80% bulk overlap, unique single-cell gene names",
        ),
    ])
    if not alignment_ok:
        result.status = "input_invalid"
        result.errors = [
            "Single-cell and bulk gene spaces are not sufficiently aligned for SIDISH training. "
            "Provide unique single-cell gene names with at least 500 genes and 80% of the routed "
            "bulk reference genes represented."
        ]
        result.record("input_validation_failed", detail={"reason": "gene_alignment"})
        return result, route.to_dict()
    patient_schema = inspect_patient_columns(p)
    if not patient_schema["columns"]:
        result.qc.append(QualityCheck(
            "patient/sample column", QualityStatus.FAIL,
            "No column with a usable set of patient/sample identifiers was found."))
        result.status = "input_invalid"
        result.errors = ["No usable patient/sample column was found in AnnData.obs."]
        return result, route.to_dict()
    result.qc.append(QualityCheck(
        "patient/sample column", QualityStatus.PASS,
        f"{len(patient_schema['columns'])} selectable column(s) found",
        sorted(patient_schema["columns"]), "at least one selectable column"))

    schema_path = p.parent / "single_cell_schema.json"
    schema_path.write_text(json.dumps(patient_schema, indent=2), encoding="utf-8")
    result.artifacts.append(Artifact.from_path(
        "single_cell_schema", schema_path, "Selectable patient columns and identifiers"))
    result.status = "patient_selection_required"
    result.errors = []
    result.warnings.extend(route.validation.warnings if route.validation else [])
    result.record("patient_selection_required",
                  detail={"columns": sorted(patient_schema["columns"])})
    return result, {**route.to_dict(), "patient_schema": str(schema_path)}


def run_reference_case(result: SIDISHCaseResult, patient_id: str,
                       adata_path: str | None = None, bulk_path: str | None = None,
                       run_dir: str | None = None, device: str = "cpu",
                       patient_specific: bool = True,
                       include_figures: bool = True) -> SIDISHCaseResult:
    """Run the full decision-support analysis and populate a structured bundle."""
    import sidish_tools as T

    T.CONFIG["demo_mode"] = False
    T.CONFIG["force_mock"] = False
    boot = T.init_sidish(adata_path=adata_path, bulk_path=bulk_path, path=run_dir,
                         device=device)
    T.CONTEXT["case_id"] = result.metadata.case_id
    source = boot["source_dataset"]
    model_version = boot["model_version"]
    hashes = {}
    for label, path in (("single_cell", adata_path or T.CONFIG["adata_path"]),
                        ("bulk_survival", bulk_path or T.CONFIG["bulk_csv"])):
        try:
            hashes[label] = _sha256(path)
        except OSError:
            pass

    overview = T.highrisk_overview()
    sample = T.patient_features(patient_id)
    if sample.get("n_cells", 0) == 0:
        raise ValueError(f"patient/sample {patient_id!r} is not present in the loaded dataset")
    markers = T.marker_genes()
    pathways = T.pathway_enrichment()
    survival = T.survival_km()
    target_pert = T.perturbation(patient=patient_id if patient_specific else None)
    pathway_pert = T.pathway_perturbation(patient=patient_id if patient_specific else None)
    drugs = T.drug_perturbation()
    mechanism = T.mechanism()

    common = {"source": source, "model_version": model_version, "hashes": hashes}
    result.evidence = {}
    result.add_evidence(EvidenceItem(
        "sample.high_risk_burden", "Model-defined high-risk-cell burden",
        {"fraction": sample["high_risk_fraction"], "n_high_risk": sample["n_high_risk"],
         "n_cells": sample["n_cells"]}, EvidenceScope.SAMPLE,
        _prov("SIDISH cell-risk thresholding", **common,
              parameters={"percentile": T.CONFIG["percentile"], "sample_id": patient_id}),
        quality=QualityStatus.PASS,
        limitations=["Cell fraction is not an individual's probability of recurrence, death, or response."],
        supports=["Review whether the model-defined cellular state warrants orthogonal confirmation."],
    ))
    result.add_evidence(EvidenceItem(
        "sample.celltype_composition", "High-risk-cell composition",
        {"composition": sample["celltype_composition"], "enriched": sample["enriched"],
         "reduced": sample["reduced"]}, EvidenceScope.SAMPLE,
        _prov("Descriptive composition versus cohort high-risk cells", **common,
              parameters={"sample_id": patient_id}), quality=QualityStatus.WARNING,
        limitations=["Enriched/reduced labels are descriptive and do not currently include inferential statistics."],
        supports=["Select pathology markers to localize the nominated cellular compartment."],
    ))
    result.add_evidence(EvidenceItem(
        "cohort.marker_program", "High-risk marker program", markers,
        EvidenceScope.COHORT, _prov("Wilcoxon differential expression with adjusted p-value filtering",
                                    **common, parameters={"logfc": 1.5, "fdr": 0.05}),
        quality=QualityStatus.PASS,
        limitations=["Markers are cohort-level unless a validated per-sample differential analysis is run."],
    ))
    result.add_evidence(EvidenceItem(
        "cohort.pathways", "Pathway context", pathways, EvidenceScope.COHORT,
        _prov(pathways.get("method", "pathway-table overlap"), **common),
        quality=QualityStatus.WARNING, limitations=[pathways.get("limitation", "")],
        supports=["Prioritize pathway-level orthogonal assays."],
    ))
    result.add_evidence(EvidenceItem(
        "cohort.survival_association", "Survival association", survival,
        EvidenceScope.COHORT, _prov("Penalized Cox score, median split, log-rank test", **common),
        quality=QualityStatus.WARNING, limitations=[survival.get("limitation", "")],
    ))
    pert_scope = EvidenceScope.PATIENT_MODEL if patient_specific else EvidenceScope.COHORT
    result.add_evidence(EvidenceItem(
        "model.target_perturbation", "Target-network perturbation hypotheses", target_pert,
        pert_scope, _prov(target_pert.get("perturbation_model", "SIDISH perturbation"), **common,
                          parameters={"patient_specific": patient_specific,
                                      "gene_scope": target_pert.get("gene_scope")}),
        quality=QualityStatus.WARNING, limitations=[target_pert.get("limitation", "")],
        supports=["Select targets for orthogonal molecular or functional validation."],
    ))
    result.add_evidence(EvidenceItem(
        "model.pathway_perturbation", "Pathway perturbation hypotheses", pathway_pert,
        pert_scope, _prov(pathway_pert.get("method", "SIDISH pathway perturbation"), **common,
                          parameters={"patient_specific": patient_specific}),
        quality=QualityStatus.WARNING, limitations=[pathway_pert.get("limitation", "")],
        supports=["Select pathway programs for functional study."],
    ))
    result.add_evidence(EvidenceItem(
        "hypothesis.drug_mapping", "Drug perturbation and mapping hypotheses", drugs,
        EvidenceScope.HYPOTHESIS, _prov(drugs.get("method", "target-drug mapping"), **common),
        quality=QualityStatus.WARNING, limitations=[drugs.get("limitation", "")],
        supports=["Literature review or trial-screening discussion after target confirmation."],
    ))
    result.add_evidence(EvidenceItem(
        "hypothesis.mechanism", "Mechanistic network hypothesis", mechanism,
        EvidenceScope.HYPOTHESIS, _prov(mechanism.get("method", "network overlap"), **common),
        quality=QualityStatus.WARNING, limitations=[mechanism.get("limitation", "")],
    ))

    result.qc = [
        QualityCheck("model reload", QualityStatus.PASS,
                     f"{model_version} loaded on {boot['device']}"),
        QualityCheck("sample cell count", QualityStatus.PASS if sample["n_cells"] >= 100 else QualityStatus.WARNING,
                     f"{sample['n_cells']:,} cells", sample["n_cells"], ">=100"),
        QualityCheck("sample high-risk cell count",
                     QualityStatus.PASS if sample["n_high_risk"] >= 20 else QualityStatus.WARNING,
                     f"{sample['n_high_risk']:,} high-risk cells", sample["n_high_risk"], ">=20"),
        QualityCheck("cell-type annotation",
                     QualityStatus.PASS if boot["has_celltype"] else QualityStatus.WARNING,
                     "celltype_major present" if boot["has_celltype"] else "celltype_major unavailable"),
    ]
    if include_figures:
        figs = T.show_figures(patient_id, patient_specific=patient_specific)
        for kind, path in figs.get("figures", {}).items():
            if path and not str(path).startswith("error") and Path(path).is_file():
                result.artifacts.append(Artifact.from_path("figure", path, kind))
    result.metadata.patient_id = patient_id
    result.metadata.data_source = source
    result.status = "analysis_complete"
    result.errors = []
    result.record("analysis_completed", detail={"patient_id": patient_id,
                                                "patient_specific": patient_specific,
                                                "evidence_keys": sorted(result.evidence)})
    return result
