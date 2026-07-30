"""Run only the SIDISH capability requested in the active case chat."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import difflib
import hashlib
import os
import re
import threading

from sidish_case_analysis import _prov, _sha256
from sidish_case_store import CaseStore
from sidish_contracts import Artifact, EvidenceItem, EvidenceScope, QualityCheck, QualityStatus
from sidish_data_router import route_case_data


Progress = Callable[[str], None]
_MODEL_LOCK = threading.RLock()


def _paths(store: CaseStore, result) -> tuple[Path, Path, Path]:
    import yaml

    case_dir = store.case_dir(result.metadata.case_id)
    training = case_dir / "jobs" / "training.yaml"
    if training.exists():
        cfg = yaml.safe_load(training.read_text(encoding="utf-8"))
        run_dir = case_dir / "run"
        adata = run_dir / "adata_SIDISH.h5ad"
        return adata, Path(cfg["data"]["bulk_csv"]), run_dir
    route = route_case_data(result.metadata.cancer_type)
    if not route.ready or not route.run_dir:
        raise RuntimeError(route.message)
    run_dir = Path(route.run_dir)
    return run_dir / "adata_SIDISH.h5ad", Path(route.bulk_path), run_dir


def _ensure_model(store: CaseStore, result, progress: Progress):
    import sidish_tools as T

    adata, bulk, run_dir = _paths(store, result)
    if not adata.is_file() or not (run_dir / "vae_transfer").exists() or not (run_dir / "deepCox").exists():
        raise RuntimeError("The case model is not trained yet. Ask SIDISH to start or check training first.")
    if T.CONTEXT.get("mode") == "live" and T.CONTEXT.get("case_id") == result.metadata.case_id:
        return T, T.CONTEXT.get("boot", {})
    progress("Loading the trained SIDISH model and case cells…")
    boot = T.init_sidish(adata_path=str(adata), bulk_path=str(bulk),
                         path=str(run_dir) + "/", device=os.environ.get("SIDISH_DEVICE", "cpu"))
    T.CONTEXT["case_id"] = result.metadata.case_id
    T.CONTEXT["boot"] = boot
    return T, boot


def _common(result, boot, adata: Path, bulk: Path) -> dict:
    hashes = {}
    artifact_hashes = {a.kind.removeprefix("input_"): a.sha256 for a in result.artifacts
                       if a.kind.startswith("input_") and a.sha256}
    for key, path in (("single_cell", adata), ("bulk_survival", bulk)):
        if artifact_hashes.get(key):
            hashes[key] = artifact_hashes[key]
            continue
        try:
            hashes[key] = _sha256(path)
        except OSError:
            pass
    return {"source": boot.get("source_dataset", str(adata)),
            "model_version": boot.get("model_version", "SIDISH"), "hashes": hashes}


def _upsert_qc(result, check: QualityCheck) -> None:
    result.qc = [q for q in result.qc if q.name != check.name]
    result.qc.append(check)


def _entity_key(base: str, entity: str | None) -> str:
    if not entity:
        return base
    slug = re.sub(r"[^a-z0-9]+", "-", entity.lower()).strip("-")[:44] or "entity"
    digest = hashlib.sha256(entity.encode("utf-8")).hexdigest()[:8]
    return f"{base}.{slug}-{digest}"


def _without_chat_artifacts(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "_chat_artifacts"}


def _merge_target_results(baseline: dict[str, Any], focused: dict[str, Any]) -> dict[str, Any]:
    """Keep the requested target and the baseline sweep without duplicate rows."""
    merged = _without_chat_artifacts(baseline)
    for field in ("single", "dual"):
        rows = []
        seen = set()
        for row in list(focused.get(field, [])) + list(baseline.get(field, [])):
            key = str(row.get("target", "")).casefold()
            if key and key not in seen:
                rows.append(row)
                seen.add(key)
        merged[field] = rows
    return merged


def _merge_drug_results(baseline: dict[str, Any], focused: dict[str, Any]) -> dict[str, Any]:
    """Merge target-to-compound mappings, prioritizing the requested target."""
    merged = _without_chat_artifacts(baseline)
    rows = []
    seen = set()
    for row in list(focused.get("by_target", [])) + list(baseline.get("by_target", [])):
        key = str(row.get("target", "")).casefold()
        if key and key not in seen:
            rows.append(row)
            seen.add(key)
    merged["by_target"] = rows
    return merged


def _resolve_genes(T, requested: list[str]) -> list[str]:
    available = list(map(str, T.CONTEXT["sdh"].adata.var_names))
    by_upper = {gene.upper(): gene for gene in available}
    resolved, missing = [], []
    for raw in requested:
        gene = by_upper.get(str(raw).strip().upper())
        (resolved if gene else missing).append(gene or str(raw).strip())
    if missing:
        suggestions = {gene: difflib.get_close_matches(gene.upper(), list(by_upper), n=4,
                                                       cutoff=0.55) for gene in missing}
        detail = "; ".join(
            f"{gene}: {', '.join(suggestions[gene]) or 'no close match'}" for gene in missing)
        raise ValueError(f"Gene(s) not found in this model: {', '.join(missing)}. Close matches — {detail}.")
    return list(dict.fromkeys(resolved))


def _resolve_pathway(T, requested: str) -> str:
    enrichment = T.pathway_enrichment(top_n=50)
    names = [str(row.get("name")) for row in enrichment.get("activated", []) if row.get("name")]
    exact = next((name for name in names if name.casefold() == requested.casefold()), None)
    partial = [name for name in names if requested.casefold() in name.casefold()]
    if exact:
        return exact
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError("That pathway name is ambiguous. Choose one of: " + "; ".join(partial[:8]))
    suggestions = difflib.get_close_matches(requested, names, n=5, cutoff=0.35)
    raise ValueError(
        f"Pathway {requested!r} was not found in this case's enrichment results. "
        + ("Closest matches: " + "; ".join(suggestions) if suggestions else
           "Run pathway context first and choose one of the returned pathway names."))


def _resolve_drug_targets(T, requested: str) -> list[str]:
    hits, names = [], []
    for target, compounds in T.CMAP_DRUGS.items():
        for compound in compounds:
            name = str(compound.get("drug", ""))
            names.append(name)
            if requested.casefold() in name.casefold() or name.casefold() in requested.casefold():
                hits.append(target)
    if not hits:
        suggestions = difflib.get_close_matches(requested, names, n=5, cutoff=0.35)
        raise ValueError(
            f"Drug/compound {requested!r} is not in the configured SIDISH mapping. "
            + ("Closest matches: " + "; ".join(suggestions) if suggestions else
               "Try a target gene or the reverse-signature method."))
    return _resolve_genes(T, list(dict.fromkeys(hits)))


def _transition_artifacts(store: CaseStore, result, T, capability: str,
                          entity: str, row: dict[str, Any]) -> dict[str, Any]:
    transition = T.CONTEXT["cache"].get(f"transition::{capability}::{entity}")
    if not transition:
        return {"figures": [], "table": [row]}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import generate_figures as figures

        adata = figures.ensure_embedding()
        names = list(map(str, adata.obs_names))
        positions = {name: i for i, name in enumerate(names)}
        keep = [(i, positions[name]) for i, name in enumerate(transition["obs_names"])
                if name in positions]
        if not keep:
            raise RuntimeError("the transition cells are absent from the active embedding")
        src, dst = zip(*keep)
        xy = np.asarray(adata.obsm["X_umap"])[list(dst)]
        before = np.asarray(transition["before"], dtype=str)[list(src)]
        after = np.asarray(transition["after"], dtype=str)[list(src)]
        categories = np.full(len(src), "Unchanged background", dtype=object)
        categories[(before == "h") & (after == "h")] = "Remaining high-risk"
        categories[(before == "h") & (after == "b")] = "High-risk → background"
        categories[(before == "b") & (after == "h")] = "Background → high-risk"
        palette = {"Unchanged background": "#c9ced6", "Remaining high-risk": "#c0392b",
                   "High-risk → background": "#178f83", "Background → high-risk": "#e67e22"}
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for label in palette:
            mask = categories == label
            if mask.any():
                ax.scatter(xy[mask, 0], xy[mask, 1], s=10, alpha=.78,
                           c=palette[label], label=f"{label} (n={int(mask.sum())})")
        subject = ("Entire dataset" if result.metadata.analysis_scope == "cohort"
                   else result.metadata.patient_id)
        ax.set(title=f"{subject}: {entity} perturbation",
               xlabel="UMAP 1", ylabel="UMAP 2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        out_dir = store.case_dir(result.metadata.case_id) / "artifacts" / "chat"
        out_dir.mkdir(parents=True, exist_ok=True)
        token = hashlib.sha256(f"{capability}:{entity}".encode()).hexdigest()[:10]
        path = out_dir / f"{capability}-{token}-transition-umap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        result.artifacts = [a for a in result.artifacts if a.path != str(path)]
        result.artifacts.append(Artifact.from_path(
            "chat_transition_umap", path,
            f"{subject} {capability} transition UMAP for {entity}"))
        return {"figures": [str(path)], "table": [row]}
    except Exception as exc:
        result.warnings.append(f"Inline transition UMAP was not generated for {entity}: {exc}")
        return {"figures": [], "table": [row]}


def run_capability(case_id: str, capability: str, store: CaseStore | None = None,
                   progress: Progress | None = None,
                   params: dict[str, Any] | None = None) -> dict:
    """Compute and persist one requested capability; expensive results are cached."""
    store = store or CaseStore()
    progress = progress or (lambda _message: None)
    params = dict(params or {})
    result = store.get(case_id)
    analysis_scope = getattr(result.metadata, "analysis_scope", "patient")
    if analysis_scope not in {"patient", "cohort"}:
        raise RuntimeError(f"Unsupported analysis scope: {analysis_scope!r}")
    patient = result.metadata.patient_id if analysis_scope == "patient" else None
    subject = patient or "the entire dataset"
    if analysis_scope == "patient" and not patient:
        raise RuntimeError("Select a patient/sample before asking SIDISH to analyse it.")
    with _MODEL_LOCK:
        adata, bulk, run_dir = _paths(store, result)
        T, boot = _ensure_model(store, result, progress)
        common = _common(result, boot, adata, bulk)
        scope = EvidenceScope.PATIENT_MODEL

        if capability == "highrisk":
            progress(f"Counting model-labelled high-risk cells for {subject}…")
            sample = T.patient_features(patient) if patient else T.cohort_features()
            if not sample.get("n_cells"):
                if patient:
                    raise ValueError(f"Patient/sample {patient!r} is absent from the selected column.")
                raise ValueError("The uploaded single-cell dataset contains no cells to analyse.")
            evidence_scope = EvidenceScope.SAMPLE if patient else EvidenceScope.DATASET
            burden_key = "sample.high_risk_burden" if patient else "cohort.high_risk_burden"
            composition_key = ("sample.celltype_composition" if patient
                               else "cohort.celltype_composition")
            result.add_evidence(EvidenceItem(
                burden_key,
                ("Model-defined high-risk-cell burden" if patient
                 else "Dataset-wide model-defined high-risk-cell burden"),
                {"fraction": sample["high_risk_fraction"], "n_high_risk": sample["n_high_risk"],
                 "n_cells": sample["n_cells"]}, evidence_scope,
                _prov("SIDISH cell-risk thresholding", **common,
                      parameters={"analysis_scope": analysis_scope, "sample_id": patient,
                                  "patient_column": result.metadata.patient_column}),
                quality=QualityStatus.PASS,
                limitations=["Cell fraction is not an individual probability of outcome or response."]))
            result.add_evidence(EvidenceItem(
                composition_key,
                ("High-risk-cell composition" if patient
                 else "Dataset-wide high-risk-cell composition"),
                {"composition": sample["celltype_composition"], "enriched": sample["enriched"],
                 "reduced": sample["reduced"],
                 "celltype_available": bool(sample.get("celltype_available", True)),
                 "celltype_field": sample.get("celltype_field")}, evidence_scope,
                _prov("Descriptive composition versus cohort high-risk cells", **common,
                      parameters={"analysis_scope": analysis_scope, "sample_id": patient}),
                quality=QualityStatus.WARNING,
                limitations=(["Composition is descriptive and requires orthogonal confirmation."]
                             if sample.get("celltype_available", True) else
                             ["No cell-type annotation was supplied; cell-type composition is not reported."]))
            )
            _upsert_qc(result, QualityCheck(
                "selected patient cell count" if patient else "entire dataset cell count",
                QualityStatus.PASS if sample["n_cells"] >= 100 else QualityStatus.WARNING,
                f"{sample['n_cells']:,} cells for {subject}", sample["n_cells"], ">=100"))
            _upsert_qc(result, QualityCheck(
                "cell-type annotation",
                QualityStatus.PASS if sample.get("celltype_available", True) else QualityStatus.WARNING,
                ("celltype_major is available" if sample.get("celltype_available", True)
                 else "celltype_major is unavailable; burden and perturbation remain reportable without composition"),
                sample.get("celltype_field"), "optional"))
            payload = sample

        elif capability == "markers":
            progress("Calculating adjusted marker genes and pathway context…")
            markers = T.marker_genes()
            pathways = T.pathway_enrichment()
            survival = T.survival_km()
            result.add_evidence(EvidenceItem(
                "cohort.marker_program", "High-risk marker program", markers, EvidenceScope.COHORT,
                _prov("Wilcoxon differential expression with adjusted p-value filtering", **common),
                quality=QualityStatus.PASS,
                limitations=["Markers describe the analysed dataset rather than an individual patient."]))
            result.add_evidence(EvidenceItem(
                "cohort.pathways", "Pathway context", pathways, EvidenceScope.COHORT,
                _prov(pathways.get("method", "pathway enrichment"), **common),
                quality=QualityStatus.WARNING, limitations=[pathways.get("limitation", "")]))
            result.add_evidence(EvidenceItem(
                "cohort.survival_association", "Survival association", survival, EvidenceScope.COHORT,
                _prov("Penalized Cox score, median split, log-rank test", **common),
                quality=QualityStatus.WARNING, limitations=[survival.get("limitation", "")]))
            payload = {"markers": markers, "pathways": pathways, "survival": survival}

        elif capability == "target":
            genes = _resolve_genes(T, list(params.get("genes") or []))
            entity = ", ".join(genes) if genes else None
            progress(f"Running target-network perturbation for {entity or 'the marker sweep'} on {subject}…")
            value = T.perturbation(patient=patient, genes=genes or None,
                                   scope=str(params.get("scope") or "markers"))
            if value.get("error"):
                raise ValueError(value["error"])
            scope = EvidenceScope.PATIENT_MODEL if patient else EvidenceScope.DATASET_MODEL
            provenance = _prov(value.get("perturbation_model", "SIDISH perturbation"), **common,
                               parameters={"analysis_scope": analysis_scope,
                                           "patient_specific": bool(patient), "genes": genes})
            item = EvidenceItem(
                "model.target_perturbation", "Target-network perturbation hypotheses", value, scope,
                provenance, quality=QualityStatus.WARNING,
                limitations=[value.get("limitation", "Computational hypothesis.")])
            result.add_evidence(item)
            if entity:
                result.add_evidence(EvidenceItem(
                    _entity_key("model.target_perturbation", entity),
                    f"Target-network perturbation: {entity}", value, scope, provenance,
                    quality=QualityStatus.WARNING,
                    limitations=[value.get("limitation", "Computational hypothesis.")]))
            row = value.get("single", [{}])[0] if value.get("single") else {}
            artifacts = _transition_artifacts(store, result, T, "target",
                                               row.get("target", entity or "target sweep"), row)
            payload = {**value, "_chat_artifacts": artifacts}

        elif capability == "pathway":
            genes = _resolve_genes(T, list(params.get("genes") or []))
            pathway = str(params.get("pathway") or "").strip() or None
            if pathway and not genes:
                pathway = _resolve_pathway(T, pathway)
            entity = pathway or (", ".join(genes) if genes else None)
            progress(f"Running pathway perturbation for {entity or 'the enriched pathway sweep'} on {subject}…")
            value = T.pathway_perturbation(patient=patient, pathway=pathway, genes=genes or None)
            if value.get("error"):
                raise ValueError(value["error"] + (" Available: " + "; ".join(value.get("available", []))
                                                   if value.get("available") else ""))
            scope = EvidenceScope.PATIENT_MODEL if patient else EvidenceScope.DATASET_MODEL
            provenance = _prov(value.get("method", "SIDISH pathway perturbation"), **common,
                               parameters={"analysis_scope": analysis_scope,
                                           "patient_specific": bool(patient), "pathway": pathway,
                                           "genes": genes})
            result.add_evidence(EvidenceItem(
                "model.pathway_perturbation", "Pathway perturbation hypotheses", value, scope,
                provenance, quality=QualityStatus.WARNING,
                limitations=[value.get("limitation", "Computational hypothesis.")]))
            if entity:
                result.add_evidence(EvidenceItem(
                    _entity_key("model.pathway_perturbation", entity),
                    f"Pathway perturbation: {entity}", value, scope, provenance,
                    quality=QualityStatus.WARNING,
                    limitations=[value.get("limitation", "Computational hypothesis.")]))
            row = value.get("pathways", [{}])[0] if value.get("pathways") else {}
            artifacts = _transition_artifacts(store, result, T, "pathway",
                                               row.get("pathway", entity or "pathway sweep"), row)
            payload = {**value, "_chat_artifacts": artifacts}

        elif capability == "drug":
            method = str(params.get("method") or "targets")
            if method not in {"targets", "reverse_signature"}:
                raise ValueError("Drug mapping method must be 'targets' or 'reverse_signature'.")
            targets = _resolve_genes(T, list(params.get("targets") or []))
            requested_drug = str(params.get("drug") or "").strip() or None
            if requested_drug:
                targets = _resolve_drug_targets(T, requested_drug)
            progress("Mapping the requested target/signature hypothesis to perturbagens…")
            value = T.drug_perturbation(targets=targets or None, method=method)
            if value.get("error"):
                raise ValueError(value["error"])
            if requested_drug:
                value = {**value, "requested_drug": requested_drug,
                         "matched_targets": targets,
                         "interpretation": "Target/compound knowledge mapping; no pharmacologic response was simulated."}
            entity = requested_drug or (", ".join(targets) if targets else method)
            provenance = _prov(value.get("method", "target-drug mapping"), **common,
                               parameters={"targets": targets, "method": method,
                                           "requested_drug": requested_drug})
            result.add_evidence(EvidenceItem(
                "hypothesis.drug_mapping", "Drug perturbation and mapping hypotheses", value,
                EvidenceScope.HYPOTHESIS,
                provenance,
                quality=QualityStatus.WARNING,
                limitations=[value.get("limitation", "Not a treatment recommendation.")]))
            result.add_evidence(EvidenceItem(
                _entity_key("hypothesis.drug_mapping", entity),
                f"Drug mapping: {entity}", value, EvidenceScope.HYPOTHESIS, provenance,
                quality=QualityStatus.WARNING,
                limitations=[value.get("limitation", "Not a treatment recommendation.")]))
            table = value.get("ranked_drugs") or [
                {"target": row.get("target"), **compound}
                for row in value.get("by_target", []) for compound in row.get("compounds", [])]
            payload = {**value, "_chat_artifacts": {"figures": [], "table": table}}

        elif capability == "report":
            progress("Checking which report evidence has already been calculated in this chat…")
            focus_targets = _resolve_genes(T, list(params.get("focus_targets") or []))
            include_other = bool(params.get("include_other_perturbations", False))
            highrisk_keys = ({"sample.high_risk_burden", "sample.celltype_composition"}
                             if patient else
                             {"cohort.high_risk_burden", "cohort.celltype_composition"})
            requirements = {
                "highrisk": highrisk_keys,
                "markers": {"cohort.marker_program", "cohort.pathways",
                            "cohort.survival_association"},
            }
            for needed_capability, keys in requirements.items():
                current = store.get(case_id)
                if not keys.issubset(current.evidence):
                    run_capability(case_id, needed_capability, store, progress)
            baseline_target = baseline_drug = None
            if not focus_targets:
                # Re-activate the complete cached sweeps in case a prior focused
                # request replaced the canonical evidence slots.
                run_capability(case_id, "target", store, progress)
                run_capability(case_id, "pathway", store, progress)
                run_capability(case_id, "drug", store, progress)
            elif include_other:
                baseline_target = run_capability(case_id, "target", store, progress)
                run_capability(case_id, "pathway", store, progress)
                baseline_drug = run_capability(case_id, "drug", store, progress)
            result = store.get(case_id)
            if focus_targets:
                progress("Running the requested focused target perturbation for the report…")
                focused_target = run_capability(
                    case_id, "target", store, progress, params={"genes": focus_targets})
                progress("Updating target-specific compound mapping for the focused report…")
                focused_drug = run_capability(
                    case_id, "drug", store, progress,
                    params={"targets": focus_targets, "method": "targets"})
                result = store.get(case_id)
                if include_other and baseline_target is not None and baseline_drug is not None:
                    result.evidence["model.target_perturbation"].value = _merge_target_results(
                        baseline_target, focused_target)
                    result.evidence["hypothesis.drug_mapping"].value = _merge_drug_results(
                        baseline_drug, focused_drug)
                    store.save(result)
            if focus_targets or "hypothesis.mechanism" not in result.evidence:
                mechanism_target = focus_targets[0] if focus_targets else None
                progress("Connecting the report target to pathway and regulatory context…")
                mechanism = T.mechanism(target=mechanism_target)
                result.add_evidence(EvidenceItem(
                    "hypothesis.mechanism", "Mechanistic network hypothesis", mechanism,
                    EvidenceScope.HYPOTHESIS,
                    _prov(mechanism.get("method", "network overlap"), **common),
                    quality=QualityStatus.WARNING,
                    limitations=[mechanism.get("limitation", "Descriptive, not causal.")]))
            result.status = "analysis_complete"
            result.record("chat_report_evidence_completed",
                          detail={"analysis_scope": analysis_scope, "patient_id": patient,
                                  "focus_targets": focus_targets,
                                  "include_other_perturbations": include_other})
            store.save(result)
            progress("Rendering the clinician decision-support report…")
            from decision_report_writer import render_case_report
            return render_case_report(
                store.case_dir(case_id) / "result.json", make_pdf=True,
                focus_targets=focus_targets,
                include_other_perturbations=include_other)
        else:
            raise ValueError(f"unknown SIDISH capability: {capability}")

        result.metadata.data_source = common["source"]
        result.status = "analysis_partial"
        result.errors = []
        result.record("chat_analysis_completed", detail={"capability": capability,
                                                          "patient_id": patient,
                                                          "analysis_scope": analysis_scope,
                                                          "parameters": params})
        store.save(result)
        return payload
