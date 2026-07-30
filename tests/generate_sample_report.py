"""Generate a non-patient synthetic report for visual regression review."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_report_writer import render_case_report
from sidish_contracts import (
    CaseMetadata, EvidenceItem, EvidenceScope, Provenance,
    QualityCheck, QualityStatus, SIDISHCaseResult,
)


OUT = ROOT / "output" / "pdf"
TMP = ROOT / "tmp" / "pdfs"
OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)

r = SIDISHCaseResult("2.0", CaseMetadata(
    "SYNTHETIC-QA", "breast", "Synthetic breast cancer QA case", patient_id="SAMPLE-QA",
    cancer_subtype="Not applicable - synthetic", data_source="Synthetic visual-regression fixture",
    research_use_only=True, demo=True), status="analysis_complete")
r.qc = [QualityCheck("model reload", QualityStatus.PASS, "synthetic fixture"),
        QualityCheck("sample cell count", QualityStatus.PASS, "1,200 cells", 1200, ">=100"),
        QualityCheck("high-risk cell count", QualityStatus.PASS, "186 cells", 186, ">=20")]

def add(key, title, value, scope, method, limitations):
    r.add_evidence(EvidenceItem(key, title, value, scope,
        Provenance(method, "Synthetic visual-regression fixture", "SIDISH QA model"),
        quality=QualityStatus.WARNING, limitations=limitations))

add("sample.high_risk_burden", "Model-defined high-risk-cell burden",
    {"fraction": .155, "n_high_risk": 186, "n_cells": 1200}, EvidenceScope.SAMPLE,
    "SIDISH thresholding", ["Not an individual probability of outcome or response."])
add("sample.celltype_composition", "High-risk-cell composition",
    {"composition": {"CAFs": .817, "PVL": .183}, "enriched": ["CAFs", "PVL"], "reduced": []},
    EvidenceScope.SAMPLE, "Descriptive composition", ["Requires pathology confirmation."])
add("cohort.marker_program", "High-risk marker program",
    {"genes": ["MDK", "CALD1", "MYL9", "TPM2", "COL6A1", "COL6A2", "CRIP2", "IGFBP7"]},
    EvidenceScope.COHORT, "Adjusted differential expression", ["Cohort-level markers."])
add("cohort.pathways", "Pathway context", {"activated": [
    {"name": "extracellular matrix organization", "q_value": "2.0e-08", "genes": ["FN1", "COL1A1", "COL6A1"]},
    {"name": "collagen fibril organization", "q_value": "8.0e-06", "genes": ["COL1A1", "COL1A2", "COL6A1"]}]},
    EvidenceScope.COHORT, "Synthetic pathway enrichment", ["Synthetic q-values for layout testing only."])
add("cohort.survival_association", "Survival association", {"km_pvalue": "6.05e-10"},
    EvidenceScope.COHORT, "Same-cohort exploratory association", ["Not independent prognostic validation."])
add("model.target_perturbation", "Target-network perturbation hypotheses",
    {"single": [{"target": "FN1", "reduction": 42.4}, {"target": "COL6A1", "reduction": 42.1},
                {"target": "COL1A2", "reduction": 42.0}],
     "dual": [{"target": "COL6A1 + FN1", "reduction": 39.4}]},
    EvidenceScope.PATIENT_MODEL, "Target plus PPI-neighbour network ablation",
    ["Not a literal single-gene knockout or response prediction."])
add("model.pathway_perturbation", "Pathway perturbation hypotheses",
    {"pathways": [{"pathway": "extracellular matrix organization", "genes": ["FN1", "COL1A1", "COL6A1"], "reduction": 48.2},
                  {"pathway": "collagen organization", "genes": ["COL1A1", "COL1A2"], "reduction": 36.7}]},
    EvidenceScope.PATIENT_MODEL, "Pathway-network ablation", ["Computational hypothesis."])
add("hypothesis.drug_mapping", "Drug mapping hypotheses", {"method": "curated indirect mapping", "by_target": [
    {"target": "FN1", "compounds": [{"drug": "Cilengitide", "moa": "integrin-axis inhibitor", "status": "investigational"}]},
    {"target": "COL6A1", "compounds": [{"drug": "Simtuzumab", "moa": "LOXL2/collagen-axis strategy", "status": "investigational"}]}]},
    EvidenceScope.HYPOTHESIS, "Curated target-to-compound mapping", ["Indirect mapping; verify current evidence."])
add("hypothesis.mechanism", "Mechanistic hypothesis", {}, EvidenceScope.HYPOTHESIS,
    "Network overlap", ["Descriptive, not causal."])

path = r.save(TMP / "synthetic-result.json")
print(render_case_report(path, out_dir=OUT, make_pdf=True))
