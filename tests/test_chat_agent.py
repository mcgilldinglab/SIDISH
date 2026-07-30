from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from unittest.mock import patch

from sidish_agent_session import (
    CaseTools, SIDISHAgentSession, _extract_params, _strip_hidden_reasoning,
)
from sidish_case_store import CaseStore
from sidish_contracts import CaseMetadata


class ChatAgentTests(unittest.TestCase):
    def _case(self, root: Path):
        store = CaseStore(root / "cases")
        result = store.create(CaseMetadata("CHAT-1", "breast", "Breast cancer",
                                           patient_id="P1", patient_column="donor"))
        adata = ad.AnnData(
            np.ones((3, 2)),
            obs=pd.DataFrame({"donor": ["P1", "P1", "P2"]},
                             index=["c1", "c2", "c3"]),
            var=pd.DataFrame(index=["TP53", "EGFR"]),
        )
        source = root / "source.h5ad"
        adata.write_h5ad(source)
        store.save_upload("CHAT-1", source.name, source.read_bytes(), "single_cell")
        jobs = store.case_dir("CHAT-1") / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        (jobs / "training.yaml").write_text(
            "training:\n  iterations: 10\n", encoding="utf-8")
        result = store.get("CHAT-1")
        result.status = "training_required"
        store.save(result)
        return store

    def test_training_requires_consent_and_raw_count_does_not_train(self):
        with TemporaryDirectory() as td:
            store = self._case(Path(td))
            agent = SIDISHAgentSession("CHAT-1", store=store)
            count = agent.send("How many cells are in this patient?")
            self.assertIn("2 cells", count["content"])
            self.assertEqual(store.get("CHAT-1").status, "training_required")

            reply = agent.send("Perturb TP53")
            self.assertIn("Reply **yes**", reply["content"])
            result = store.get("CHAT-1")
            self.assertEqual(result.status, "training_required")
            self.assertEqual(result.audit[-1]["action"], "training_consent_requested")
            self.assertEqual(result.audit[-1]["detail"]["parameters"]["genes"], ["TP53"])

    def test_chat_turns_are_backward_compatible_and_artifact_ready(self):
        with TemporaryDirectory() as td:
            store = self._case(Path(td))
            store.append_chat("CHAT-1", "assistant", "legacy-style content")
            event = store.read_chat("CHAT-1")[-1]
            self.assertEqual(event["figures"], [])
            self.assertIsNone(event["table"])

    def test_raw_cell_count_can_cover_entire_dataset(self):
        with TemporaryDirectory() as td:
            store = self._case(Path(td))
            result = store.get("CHAT-1")
            result.metadata.analysis_scope = "cohort"
            result.metadata.patient_id = None
            store.save(result)
            answer = SIDISHAgentSession("CHAT-1", store=store).send(
                "How many cells are in the entire dataset?")
            self.assertIn("3 cells across the entire dataset", answer["content"])
            self.assertIn("dataset-wide raw data count", answer["content"])

    def test_tool_schemas_and_fallback_extraction_carry_entities(self):
        with TemporaryDirectory() as td:
            store = self._case(Path(td))
            specs = {item["function"]["name"]: item for item in CaseTools(store, "CHAT-1", lambda _: None).spec}
            target_props = specs["target_perturbation"]["function"]["parameters"]["properties"]
            report_props = specs["generate_report"]["function"]["parameters"]["properties"]
            self.assertIn("genes", target_props)
            self.assertIn("focus_targets", report_props)
            self.assertIn("filter", report_props)
            self.assertEqual(_extract_params("Please perturb TP53 and EGFR", "target")["genes"],
                             ["TP53", "EGFR"])
            self.assertEqual(_extract_params("Please perturb tp53", "target")["genes"], ["tp53"])
            self.assertEqual(_extract_params("Perturb cell adhesion pathway", "pathway")["pathway"],
                             "cell adhesion")
            self.assertEqual(_extract_params("Map drug cisplatin", "drug")["drug"], "cisplatin")
            self.assertEqual(
                _extract_params("Generate a report focused only on ERBB2", "report")["focus_targets"],
                ["ERBB2"])

    def test_hidden_model_reasoning_is_removed(self):
        visible = _strip_hidden_reasoning(
            "<think>private chain of thought</think>\nThe report is ready.")
        self.assertEqual(visible, "The report is ready.")
        self.assertNotIn("think", visible.lower())

    def test_legacy_report_filter_is_accepted_and_mapped_to_focus(self):
        with TemporaryDirectory() as td:
            store = self._case(Path(td))
            tools = CaseTools(store, "CHAT-1", lambda _: None)
            with patch.object(tools, "_run", return_value={"html": "example.html"}) as run:
                tools.generate_report(filter="ERBB2 only")
            run.assert_called_once_with(
                "report", focus_targets=["ERBB2"],
                include_other_perturbations=False)

    def test_iteration_count_can_be_changed_in_chat_before_training(self):
        with TemporaryDirectory() as td:
            store = self._case(Path(td))
            answer = SIDISHAgentSession("CHAT-1", store=store).send(
                "Use 6 iterations")
            self.assertIn("configured for **6 iterations**", answer["content"])
            result = store.get("CHAT-1")
            self.assertEqual(result.metadata.training_iterations, 6)
            config = (store.case_dir("CHAT-1") / "jobs" / "training.yaml").read_text(
                encoding="utf-8")
            self.assertIn("iterations: 6", config)
            self.assertEqual(result.audit[-1]["action"], "training_iterations_changed")


if __name__ == "__main__":
    unittest.main()
