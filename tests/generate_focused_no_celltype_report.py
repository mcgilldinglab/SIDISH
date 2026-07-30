"""Generate a synthetic ERBB2-focused report without cell-type annotations."""
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
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

result = SIDISHCaseResult(
    "2.0",
    CaseMetadata(
        "SYNTHETIC-LUNG-ERBB2",
        "lung",
        "Synthetic lung cancer QA case",
        patient_id="SAMPLE-QA-01",
        patient_column="sample_id",
        training_iterations=6,
        cancer_subtype="Not specified",
        data_source="Synthetic visual-regression fixture",
        research_use_only=True,
        demo=True,
    ),
    status="analysis_complete",
)
result.qc = [
    QualityCheck("model reload", QualityStatus.PASS,
                 "Synthetic trained-model fixture loaded"),
    QualityCheck("selected patient cell count", QualityStatus.PASS,
                 "2,400 cells for SAMPLE-QA-01", 2400, ">=100"),
    QualityCheck("cell-type annotation", QualityStatus.WARNING,
                 "celltype_major is unavailable; burden and perturbation remain reportable",
                 None, "optional"),
    QualityCheck("high-risk cell count", QualityStatus.PASS,
                 "444 model-labelled cells", 444, ">=20"),
]


def add(key, title, value, scope, method, limitations):
    result.add_evidence(EvidenceItem(
        key, title, value, scope,
        Provenance(method, "Synthetic visual-regression fixture", "SIDISH QA model"),
        quality=QualityStatus.WARNING,
        limitations=limitations,
    ))


add(
    "sample.high_risk_burden", "Model-defined high-risk-cell burden",
    {"fraction": 0.185, "n_high_risk": 444, "n_cells": 2400},
    EvidenceScope.SAMPLE, "SIDISH cell-risk thresholding",
    ["Cell fraction is not an individual probability of outcome or response."],
)
add(
    "sample.celltype_composition", "High-risk-cell composition",
    {"composition": {}, "enriched": [], "reduced": [],
     "celltype_available": False, "celltype_field": None},
    EvidenceScope.SAMPLE, "Cell-type metadata availability check",
    ["No cell-type annotation was supplied; cell-type composition is not reported."],
)
add(
    "cohort.marker_program", "High-risk marker program",
    {"genes": ["ERBB2", "EPCAM", "KRT8", "KRT18", "MKI67", "TOP2A"]},
    EvidenceScope.COHORT, "Adjusted differential expression",
    ["Markers characterize the analysed cohort and require orthogonal confirmation."],
)
add(
    "cohort.pathways", "Pathway context",
    {"activated": [
        {"name": "ERBB receptor signalling", "q_value": "1.8e-05",
         "genes": ["ERBB2", "EGFR", "GRB2"]},
        {"name": "cell-cycle progression", "q_value": "7.2e-04",
         "genes": ["MKI67", "TOP2A", "CDK1"]},
    ]},
    EvidenceScope.COHORT, "Bundled reference pathway-table overlap",
    ["Displayed q-values are synthetic layout values, not patient findings."],
)
add(
    "cohort.survival_association", "Survival association",
    {"km_pvalue": "0.031"}, EvidenceScope.COHORT,
    "Penalized Cox score and exploratory log-rank comparison",
    ["This cohort association is not an absolute or individual prognosis."],
)
add(
    "model.target_perturbation", "Target-network perturbation hypotheses",
    {"single": [
        {"target": "ERBB2", "reduction": 18.5, "n_before_high": 444,
         "n_after_high": 362, "n_high_to_background": 82},
        {"target": "SMC6", "reduction": 7.0},
    ], "dual": [{"target": "ERBB2 + SMC6", "reduction": 20.1}]},
    EvidenceScope.PATIENT_MODEL, "Target plus PPI-neighbour network ablation",
    ["Computational network ablation is not literal gene editing or a response prediction."],
)
add(
    "model.pathway_perturbation", "Pathway perturbation hypotheses",
    {"pathways": [{"pathway": "cell-cycle progression",
                    "genes": ["MKI67", "TOP2A", "CDK1"], "reduction": 11.2}]},
    EvidenceScope.PATIENT_MODEL, "Pathway-network ablation",
    ["Pathway sweeps are exploratory computational hypotheses."],
)
add(
    "hypothesis.drug_mapping", "Drug perturbation and mapping hypotheses",
    {"method": "curated target-to-compound knowledge mapping", "by_target": [
        {"target": "ERBB2", "compounds": [
            {"drug": "Trastuzumab", "moa": "ERBB2-directed antibody",
             "status": "reference mapping - independently verify indication"},
            {"drug": "Lapatinib", "moa": "EGFR/ERBB2 kinase inhibitor",
             "status": "reference mapping - independently verify indication"},
        ]},
        {"target": "SMC6", "compounds": [
            {"drug": "Synthetic-other-target", "moa": "layout-only example",
             "status": "not clinical evidence"},
        ]},
    ]},
    EvidenceScope.HYPOTHESIS, "Curated target-to-compound mapping",
    ["Mapping does not simulate pharmacologic response or recommend treatment."],
)
add(
    "hypothesis.mechanism", "Mechanistic network hypothesis",
    {"target": "ERBB2", "narrative": (
        "The ERBB2-centred PPI-network perturbation changed the model label in a subset "
        "of cells. This nominates ERBB2-network activity for molecular and functional "
        "validation; it does not establish causality or treatment sensitivity.")},
    EvidenceScope.HYPOTHESIS, "Target-to-pathway network overlap",
    ["The mechanism is a computational hypothesis and is not a causal conclusion."],
)

result_path = result.save(TMP / "synthetic-lung-erbb2-result.json")
print(render_case_report(
    result_path,
    out_dir=OUT,
    make_pdf=True,
    focus_targets=["ERBB2"],
    include_other_perturbations=False,
))
