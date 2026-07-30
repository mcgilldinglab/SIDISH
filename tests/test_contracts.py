import tempfile
import unittest
from pathlib import Path

from sidish_contracts import (
    CaseMetadata, EvidenceItem, EvidenceScope, Provenance,
    QualityCheck, QualityStatus, SIDISHCaseResult,
)


def complete_result():
    r = SIDISHCaseResult("2.0", CaseMetadata("CASE-1", "breast", "Breast cancer"),
                         status="analysis_complete")
    r.qc.append(QualityCheck("input", QualityStatus.PASS, "ok"))
    r.add_evidence(EvidenceItem(
        "sample.high_risk_burden", "Burden",
        {"fraction": .1, "n_high_risk": 10, "n_cells": 100}, EvidenceScope.SAMPLE,
        Provenance("method", "dataset", "model"), quality=QualityStatus.PASS))
    return r


class ContractTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = complete_result().save(Path(d) / "result.json")
            loaded = SIDISHCaseResult.load(path)
            self.assertEqual(loaded.metadata.case_id, "CASE-1")
            self.assertIs(loaded.evidence["sample.high_risk_burden"].scope, EvidenceScope.SAMPLE)
            self.assertEqual(loaded.validate(for_report=True), [])

    def test_report_gate_rejects_failed_qc(self):
        r = complete_result(); r.qc.append(QualityCheck("critical", QualityStatus.FAIL, "bad"))
        self.assertTrue(any("failed QC" in e for e in r.validate(for_report=True)))

    def test_invalid_case_identifier_is_rejected(self):
        r = complete_result(); r.metadata.case_id = "../../patient"
        self.assertTrue(r.validate())


if __name__ == "__main__": unittest.main()
