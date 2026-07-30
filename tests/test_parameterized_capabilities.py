from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from sidish_case_store import CaseStore
from sidish_contracts import CaseMetadata
from sidish_on_demand import run_capability


class ParameterizedCapabilityTests(unittest.TestCase):
    def _store(self, root: Path):
        store = CaseStore(root / "cases")
        result = store.create(CaseMetadata("PARAM-1", "breast", "Breast cancer",
                                           patient_id="P1", patient_column="patient"))
        result.status = "analysis_ready"
        store.save(result)
        return store

    @staticmethod
    def _tools():
        tools = SimpleNamespace()
        tools.CONTEXT = {"sdh": SimpleNamespace(adata=SimpleNamespace(var_names=["TP53", "EGFR"])),
                         "cache": {}}
        tools.perturbation = Mock(return_value={
            "perturbation_model": "test perturbation", "limitation": "test only",
            "single": [{"target": "TP53", "reduction": 12.0, "p_flip": .04,
                        "affected_cells": 6}], "dual": []})
        tools.cohort_features = Mock(return_value={
            "analysis_scope": "cohort", "n_cells": 200, "n_high_risk": 40,
            "high_risk_fraction": .2, "celltype_composition": {"CAF": .75},
            "enriched": ["CAF"], "reduced": []})
        tools.pathway_enrichment = Mock(return_value={
            "activated": [{"name": "cell adhesion", "genes": ["TP53", "EGFR"]}]})
        tools.pathway_perturbation = Mock(return_value={
            "method": "test pathway", "limitation": "test only",
            "pathways": [{"pathway": "cell adhesion", "reduction": 8.0,
                          "p_flip": .1, "affected_cells": 4}]})
        tools.CMAP_DRUGS = {"EGFR": [{"drug": "Erlotinib", "moa": "EGFR inhibitor",
                                      "status": "approved"}]}
        tools.drug_perturbation = Mock(return_value={
            "method": "test mapping", "limitation": "not response",
            "by_target": [{"target": "EGFR", "compounds": tools.CMAP_DRUGS["EGFR"]}]})
        return tools

    def _patches(self, tools):
        return patch.multiple(
            "sidish_on_demand", _paths=Mock(return_value=(Path("adata"), Path("bulk"), Path("run"))),
            _ensure_model=Mock(return_value=(tools, {"source_dataset": "test", "model_version": "test"})),
            _transition_artifacts=Mock(return_value={"figures": ["figure.png"],
                                                     "table": [{"target": "TP53"}]}))

    def test_named_gene_is_passed_and_keyed(self):
        with TemporaryDirectory() as td:
            store, tools = self._store(Path(td)), self._tools()
            with self._patches(tools):
                value = run_capability("PARAM-1", "target", store, params={"genes": ["tp53"]})
            tools.perturbation.assert_called_once_with(patient="P1", genes=["TP53"], scope="markers")
            self.assertEqual(value["_chat_artifacts"]["figures"], ["figure.png"])
            keys = store.get("PARAM-1").evidence
            self.assertTrue(any(key.startswith("model.target_perturbation.tp53-") for key in keys))

    def test_named_pathway_and_drug_are_passed(self):
        with TemporaryDirectory() as td:
            store, tools = self._store(Path(td)), self._tools()
            with self._patches(tools):
                run_capability("PARAM-1", "pathway", store, params={"pathway": "cell adhesion"})
                value = run_capability("PARAM-1", "drug", store, params={"drug": "Erlotinib"})
            tools.pathway_perturbation.assert_called_once_with(
                patient="P1", pathway="cell adhesion", genes=None)
            tools.drug_perturbation.assert_called_once_with(targets=["EGFR"], method="targets")
            self.assertEqual(value["requested_drug"], "Erlotinib")
            self.assertIn("no pharmacologic response", value["interpretation"])

    def test_unknown_gene_fails_with_suggestions(self):
        with TemporaryDirectory() as td:
            store, tools = self._store(Path(td)), self._tools()
            with self._patches(tools), self.assertRaisesRegex(ValueError, "not found"):
                run_capability("PARAM-1", "target", store, params={"genes": ["TP5"]})

    def test_entire_dataset_scope_uses_all_cells(self):
        with TemporaryDirectory() as td:
            store, tools = self._store(Path(td)), self._tools()
            result = store.get("PARAM-1")
            result.metadata.analysis_scope = "cohort"
            result.metadata.patient_id = None
            store.save(result)
            with self._patches(tools):
                burden = run_capability("PARAM-1", "highrisk", store)
                run_capability("PARAM-1", "target", store, params={"genes": ["TP53"]})
            tools.cohort_features.assert_called_once_with()
            tools.perturbation.assert_called_once_with(
                patient=None, genes=["TP53"], scope="markers")
            self.assertEqual(burden["n_cells"], 200)
            evidence = store.get("PARAM-1").evidence
            self.assertIn("cohort.high_risk_burden", evidence)
            self.assertEqual(evidence["model.target_perturbation"].scope.value,
                             "dataset-wide model perturbation")


if __name__ == "__main__":
    unittest.main()
