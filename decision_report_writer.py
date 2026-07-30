"""Render a structured SIDISHCaseResult as a clinician decision-support report."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import datetime
import json
import os
import re
import sys

from jinja2 import Environment, FileSystemLoader, select_autoescape

from sidish_contracts import Artifact, SIDISHCaseResult
from sidish_policy import DISCLAIMER, enforce_text

try:
    from report_figures import build_summary_figures
except Exception:  # pragma: no cover - figures are optional
    def build_summary_figures(ctx):  # type: ignore
        return {"burden": None, "perturbation": None, "enrichment": None}


HERE = Path(__file__).resolve().parent


def _value(result: SIDISHCaseResult, key: str, default=None):
    item = result.evidence.get(key)
    return item.value if item else default


def _scope(result: SIDISHCaseResult, key: str) -> str:
    item = result.evidence.get(key)
    return item.scope.value if item else "not available"


def _top_drugs(drug_value: dict) -> list[dict]:
    rows = []
    for entry in drug_value.get("by_target", []) if isinstance(drug_value, dict) else []:
        for compound in entry.get("compounds", []):
            rows.append({"target": entry.get("target"), **compound})
    return rows[:12]


def _normalize_focus_targets(focus_targets: list[str] | None) -> list[str]:
    return list(dict.fromkeys(
        str(target).strip() for target in (focus_targets or []) if str(target).strip()))


def _row_matches_focus(row: dict, focus: set[str], require_all: bool = False) -> bool:
    targets = {item for item in re.split(
        r"\s*\+\s*|\s*,\s*", str(row.get("target", "")).casefold()) if item}
    return bool(targets) and (targets.issubset(focus) if require_all else bool(targets & focus))


def _summary_context(result: SIDISHCaseResult, focus_targets: list[str] | None = None,
                     include_other_perturbations: bool = False) -> dict[str, Any]:
    focus_targets = _normalize_focus_targets(focus_targets)
    focus = {target.casefold() for target in focus_targets}
    analysis_scope = getattr(result.metadata, "analysis_scope", "patient")
    burden_key = ("cohort.high_risk_burden" if analysis_scope == "cohort"
                  else "sample.high_risk_burden")
    composition_key = ("cohort.celltype_composition" if analysis_scope == "cohort"
                       else "sample.celltype_composition")
    burden = _value(result, burden_key, {})
    comp = _value(result, composition_key, {})
    marker = _value(result, "cohort.marker_program", {})
    pathways = _value(result, "cohort.pathways", {})
    target = _value(result, "model.target_perturbation", {})
    path_pert = _value(result, "model.pathway_perturbation", {})
    survival = _value(result, "cohort.survival_association", {})
    drugs = _value(result, "hypothesis.drug_mapping", {})
    mechanism = _value(result, "hypothesis.mechanism", {})
    singles = target.get("single", []) if isinstance(target, dict) else []
    pathway_rows = path_pert.get("pathways", []) if isinstance(path_pert, dict) else []
    composition = comp.get("composition", {}) if isinstance(comp, dict) else {}
    top_cell = max(composition, key=composition.get) if composition else "not available"
    activated = pathways.get("activated", []) if isinstance(pathways, dict) else []
    drug_rows = _top_drugs(drugs)
    dual_rows = target.get("dual", []) if isinstance(target, dict) else []
    if focus and not include_other_perturbations:
        singles = [row for row in singles if _row_matches_focus(row, focus)]
        dual_rows = [row for row in dual_rows if _row_matches_focus(row, focus, require_all=True)]
        drug_rows = [row for row in drug_rows
                     if str(row.get("target", "")).casefold() in focus]
        pathway_rows = []
        if str(mechanism.get("target", "")).casefold() not in focus:
            mechanism = {}
    celltype_available = bool(comp.get("celltype_available", bool(composition))) \
        if isinstance(comp, dict) else False
    top_single = singles[0] if singles else {}
    return {
        "analysis_scope": analysis_scope,
        "training_iterations": result.metadata.training_iterations,
        "focus_targets": focus_targets,
        "focused_report": bool(focus_targets),
        "include_other_perturbations": bool(include_other_perturbations),
        "report_focus_label": (", ".join(focus_targets) + " target-network perturbation"
                               if focus_targets else "Complete available SIDISH evidence"),
        "scope_label": ("Entire single-cell dataset" if analysis_scope == "cohort"
                        else "Selected patient/sample"),
        "subject_label": ("Entire single-cell dataset" if analysis_scope == "cohort"
                          else result.metadata.patient_id or "Selected patient/sample"),
        "observation_scope": ("Dataset-wide" if analysis_scope == "cohort"
                              else "Sample-specific"),
        "burden": burden, "composition": composition, "top_cell": top_cell,
        "celltype_available": celltype_available,
        "markers": marker.get("genes", []) if isinstance(marker, dict) else [],
        "activated_pathways": activated, "target_rows": singles,
        "dual_rows": dual_rows,
        "pathway_perturbations": pathway_rows, "top_target": top_single,
        "survival": survival, "drug_rows": drug_rows,
        "drug_method": drugs.get("method") if isinstance(drugs, dict) else None,
        "mechanism": mechanism,
        "scopes": {k: _scope(result, k) for k in result.evidence},
    }


def _deterministic_summary(result: SIDISHCaseResult, ctx: dict) -> str:
    b = ctx["burden"]
    frac = float(b.get("fraction", 0)) * 100
    target = ctx["top_target"].get("target", "no target available")
    location = ("across the entire analysed single-cell dataset"
                if ctx["analysis_scope"] == "cohort" else "in the selected sample")
    compartment = ("dataset-wide high-risk compartment"
                   if ctx["analysis_scope"] == "cohort" else "high-risk compartment")
    celltype_sentence = (
        f"The {compartment} is dominated by {ctx['top_cell']} cells. "
        if ctx["celltype_available"] and ctx["top_cell"] != "not available" else
        "No cell-type annotation was supplied, so a dominant cellular compartment is not reported. ")
    focus_sentence = (f"This report is focused on the {', '.join(ctx['focus_targets'])} "
                      "target-network perturbation. " if ctx["focused_report"] else "")
    return (
        f"SIDISH labelled {b.get('n_high_risk', 0):,} of {b.get('n_cells', 0):,} cells "
        f"({frac:.1f}%) as model-defined high-risk {location}. {celltype_sentence}{focus_sentence}"
        "A separate model perturbation nominates "
        f"{target} and associated pathways for orthogonal validation. These results should "
        "guide questions for pathology, molecular confirmation, functional study, or trial "
        "review; they do not select treatment."
    )


def _llm_summary(result: SIDISHCaseResult, ctx: dict) -> str:
    """Optional narrative only. Numbers and tables remain deterministic."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=os.environ.get("OPENAI_BASE_URL"))
    payload = {
        "case": result.metadata.case_id,
        "cancer": result.metadata.disease_label,
        "burden": ctx["burden"], "composition": ctx["composition"],
        "celltype_available": ctx["celltype_available"],
        "report_focus": ctx["focus_targets"],
        "markers": ctx["markers"], "pathways": ctx["activated_pathways"],
        "target_perturbations": ctx["target_rows"],
        "pathway_perturbations": ctx["pathway_perturbations"],
        "scopes": ctx["scopes"],
    }
    prompt = (
        "Write one concise clinician-facing paragraph from this locked JSON. Respect the declared analysis "
        "scope and separate sample-specific or dataset-wide observations "
        "from cohort associations and model hypotheses. Do not recommend, start, stop, "
        "or select treatment; do not predict response or absolute prognosis. Use no facts outside JSON.\n"
        + json.dumps(payload, indent=2)
    )
    response = client.chat.completions.create(
        model=os.environ.get("SIDISH_LLM_MODEL", "gpt-4o"),
        messages=[{"role": "system", "content": "You explain locked SIDISH evidence without changing claims."},
                  {"role": "user", "content": prompt}], temperature=0,
    )
    return enforce_text(response.choices[0].message.content or "")


def _html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            page.pdf(path=str(pdf_path), format="A4", print_background=True,
                     margin={"top": "8mm", "bottom": "8mm", "left": "8mm", "right": "8mm"})
            browser.close()
        return True
    except Exception as first:
        try:
            from weasyprint import HTML
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            return True
        except Exception as second:
            print(f"[warn] PDF rendering failed: {first}; fallback: {second}", file=sys.stderr)
            return False


def render_case_report(result_path: str | Path, out_dir: str | Path | None = None,
                       use_llm: bool = False, make_pdf: bool = True,
                       focus_targets: list[str] | None = None,
                       include_other_perturbations: bool = False) -> dict[str, str]:
    result = SIDISHCaseResult.load(result_path)
    errors = result.validate(for_report=True)
    if errors:
        raise ValueError("report release gate failed: " + "; ".join(errors))
    ctx = _summary_context(result, focus_targets, include_other_perturbations)
    if ctx["focused_report"] and not ctx["target_rows"]:
        raise ValueError(
            "requested report focus is not present in the available target-perturbation evidence: "
            + ", ".join(ctx["focus_targets"]))
    summary = _deterministic_summary(result, ctx)
    if use_llm:
        try:
            summary = _llm_summary(result, ctx)
        except Exception as exc:
            result.warnings.append(f"LLM summary unavailable; deterministic summary used ({exc})")
    enforce_text(summary)

    env = Environment(loader=FileSystemLoader(HERE / "templates"),
                      autoescape=select_autoescape(["html", "xml"]),
                      trim_blocks=True, lstrip_blocks=True)
    tmpl = env.get_template("decision_support_report.html.j2")
    report_keys = {
        "sample.high_risk_burden", "cohort.high_risk_burden",
        "sample.celltype_composition", "cohort.celltype_composition",
        "cohort.marker_program", "cohort.pathways", "cohort.survival_association",
        "model.target_perturbation", "hypothesis.drug_mapping", "hypothesis.mechanism",
    }
    if not ctx["focused_report"] or ctx["include_other_perturbations"]:
        report_keys.add("model.pathway_perturbation")
    report_evidence = [item for key, item in result.evidence.items() if key in report_keys]
    html = tmpl.render(
        result=result, metadata=result.metadata, generated_date=datetime.date.today().isoformat(),
        summary=summary, disclaimer=DISCLAIMER, ctx=ctx, figures=build_summary_figures(ctx),
        evidence=report_evidence, qc=result.qc,
        release_status=("REVIEWED - research-use discussion only"
                        if result.signoff.status == "reviewed" else
                        "DRAFT - scientific and clinical review required"),
    )
    target_dir = Path(out_dir) if out_dir else Path(result_path).resolve().parent / "artifacts"
    target_dir.mkdir(parents=True, exist_ok=True)
    focus_slug = ""
    if ctx["focus_targets"]:
        focus_slug = "_" + "-".join(
            re.sub(r"[^A-Za-z0-9]+", "-", target).strip("-")
            for target in ctx["focus_targets"])
    html_path = target_dir / f"{result.metadata.case_id}_SIDISH{focus_slug}_decision_support.html"
    tmp = html_path.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(html_path)
    outputs = {"html": str(html_path), "report_focus": ctx["report_focus_label"]}
    description = f"Draft clinician decision-support report - {ctx['report_focus_label']}"
    result.artifacts = [artifact for artifact in result.artifacts
                        if not (artifact.kind in {"report_html", "report_pdf"}
                                and Path(artifact.path).stem == html_path.stem)]
    result.artifacts.append(Artifact.from_path("report_html", html_path, description))
    if make_pdf:
        pdf_path = html_path.with_suffix(".pdf")
        if _html_to_pdf(html_path, pdf_path):
            outputs["pdf"] = str(pdf_path)
            result.artifacts.append(Artifact.from_path("report_pdf", pdf_path, description))
    result.record("report_generated", detail={
        "outputs": outputs, "use_llm": use_llm, "release_status": "draft",
        "focus_targets": ctx["focus_targets"],
        "include_other_perturbations": bool(include_other_perturbations)})
    result.save(result_path)
    return outputs
