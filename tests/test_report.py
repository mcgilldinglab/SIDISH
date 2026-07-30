import tempfile
import unittest
from pathlib import Path

from decision_report_writer import render_case_report
from sidish_contracts import EvidenceItem, EvidenceScope, Provenance, QualityStatus
from test_contracts import complete_result


def add(r, key, title, value, scope=EvidenceScope.COHORT):
    r.add_evidence(EvidenceItem(key, title, value, scope,
        Provenance("test method", "test dataset", "test model"), quality=QualityStatus.WARNING,
        limitations=["test limitation"]))


class ReportTests(unittest.TestCase):
    def test_structured_report_has_scope_and_no_fake_demographics(self):
        with tempfile.TemporaryDirectory() as d:
            r = complete_result(); r.metadata.patient_id = "P1"
            add(r, "sample.celltype_composition", "Composition", {"composition": {"CAF": .8}, "enriched": ["CAF"], "reduced": []}, EvidenceScope.SAMPLE)
            add(r, "cohort.marker_program", "Markers", {"genes": ["FN1", "COL6A1"]})
            add(r, "cohort.pathways", "Pathways", {"activated": []})
            add(r, "cohort.survival_association", "Survival", {"km_pvalue": "0.1"})
            add(r, "model.target_perturbation", "Targets", {"single": [{"target": "FN1", "reduction": 10}], "dual": []}, EvidenceScope.PATIENT_MODEL)
            add(r, "model.pathway_perturbation", "Pathway perturbation", {"pathways": []}, EvidenceScope.PATIENT_MODEL)
            add(r, "hypothesis.drug_mapping", "Drugs", {"by_target": []}, EvidenceScope.HYPOTHESIS)
            add(r, "hypothesis.mechanism", "Mechanism", {}, EvidenceScope.HYPOTHESIS)
            path = r.save(Path(d) / "result.json")
            outputs = render_case_report(path, make_pdf=False)
            html = Path(outputs["html"]).read_text(encoding="utf-8")
            self.assertIn("patient-specific model perturbation", html)
            self.assertIn("<strong>Training iterations:</strong> 10", html)
            self.assertNotIn("illustrative: 54", html)
            self.assertIn("Not a drug recommendation", html)

    def test_entire_dataset_report_uses_cohort_burden_and_scope_wording(self):
        with tempfile.TemporaryDirectory() as d:
            r = complete_result()
            r.metadata.analysis_scope = "cohort"
            r.metadata.patient_id = None
            add(r, "cohort.high_risk_burden", "Dataset burden",
                {"fraction": .25, "n_high_risk": 50, "n_cells": 200},
                EvidenceScope.DATASET)
            add(r, "cohort.celltype_composition", "Dataset composition",
                {"composition": {"CAF": .8}, "enriched": ["CAF"], "reduced": []},
                EvidenceScope.DATASET)
            add(r, "cohort.marker_program", "Markers", {"genes": ["FN1"]})
            add(r, "cohort.pathways", "Pathways", {"activated": []})
            add(r, "cohort.survival_association", "Survival", {"km_pvalue": "0.1"})
            add(r, "model.target_perturbation", "Targets",
                {"single": [{"target": "FN1", "reduction": 10}], "dual": []},
                EvidenceScope.DATASET_MODEL)
            add(r, "model.pathway_perturbation", "Pathway perturbation", {"pathways": []},
                EvidenceScope.DATASET_MODEL)
            add(r, "hypothesis.drug_mapping", "Drugs", {"by_target": []},
                EvidenceScope.HYPOTHESIS)
            add(r, "hypothesis.mechanism", "Mechanism", {}, EvidenceScope.HYPOTHESIS)
            path = r.save(Path(d) / "result.json")
            outputs = render_case_report(path, make_pdf=False)
            html = Path(outputs["html"]).read_text(encoding="utf-8")
            self.assertIn("Entire single-cell dataset", html)
            self.assertIn("50 of 200 cells (25.0%)", html)
            self.assertIn("Total dataset cells", html)
            self.assertNotIn("<strong>Sample/patient:</strong>", html)

    def test_focused_report_works_without_celltype_annotation(self):
        with tempfile.TemporaryDirectory() as d:
            r = complete_result(); r.metadata.patient_id = "P1"
            add(r, "sample.celltype_composition", "Composition", {
                "composition": {}, "enriched": [], "reduced": [],
                "celltype_available": False, "celltype_field": None,
            }, EvidenceScope.SAMPLE)
            add(r, "cohort.marker_program", "Markers", {"genes": ["ERBB2", "EGFR"]})
            add(r, "cohort.pathways", "Pathways", {"activated": []})
            add(r, "cohort.survival_association", "Survival", {"km_pvalue": "0.2"})
            add(r, "model.target_perturbation", "Targets", {
                "single": [{"target": "ERBB2", "reduction": 18.5},
                           {"target": "SMC6", "reduction": 7.0}],
                "dual": [{"target": "ERBB2 + SMC6", "reduction": 20.0}],
            }, EvidenceScope.PATIENT_MODEL)
            add(r, "model.pathway_perturbation", "Pathway perturbation", {
                "pathways": [{"pathway": "DNA repair", "genes": ["SMC6"],
                              "reduction": 4.0}]}, EvidenceScope.PATIENT_MODEL)
            add(r, "hypothesis.drug_mapping", "Drugs", {"by_target": [
                {"target": "ERBB2", "compounds": [{"drug": "Trastuzumab",
                 "moa": "ERBB2-directed antibody", "status": "reference mapping"}]},
                {"target": "SMC6", "compounds": [{"drug": "Example-SMC6",
                 "moa": "example", "status": "example"}]},
            ]}, EvidenceScope.HYPOTHESIS)
            add(r, "hypothesis.mechanism", "Mechanism", {
                "target": "ERBB2", "narrative": "ERBB2 network hypothesis for validation."},
                EvidenceScope.HYPOTHESIS)
            path = r.save(Path(d) / "result.json")
            outputs = render_case_report(
                path, make_pdf=False, focus_targets=["ERBB2"])
            html = Path(outputs["html"]).read_text(encoding="utf-8")
            self.assertIn("Cell-type composition unavailable", html)
            self.assertIn("ERBB2 target-network perturbation", html)
            self.assertIn("18.5%", html)
            self.assertIn("Trastuzumab", html)
            self.assertNotIn("Example-SMC6", html)
            self.assertNotIn("ERBB2 + SMC6", html)
            self.assertNotIn("Pathway perturbation hypotheses", html)
            self.assertIn("Pathway perturbation sweeps are excluded", html)
            self.assertIn("ERBB2", Path(outputs["html"]).name)


if __name__ == "__main__": unittest.main()
