from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

import anndata as ad
import numpy as np
import pandas as pd

from sidish_case_analysis import (
    change_selected_patient, change_training_iterations, configure_patient_selection,
    inspect_patient_columns, prepare_uploaded_case, validate_training_iterations,
)
from sidish_case_store import CaseStore
from sidish_contracts import CaseMetadata


class PatientSelectionTests(unittest.TestCase):
    def test_patient_column_and_patient_are_persisted_in_training_config(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            genes = [f"G{i}" for i in range(500)]
            adata = ad.AnnData(
                X=np.ones((4, len(genes))),
                obs=pd.DataFrame({"donor": ["P1", "P1", "P2", "P2"],
                                  "batch": ["B1", "B1", "B1", "B1"]},
                                 index=[f"cell{i}" for i in range(4)]),
                var=pd.DataFrame(index=genes),
            )
            h5ad = root / "source.h5ad"
            adata.write_h5ad(h5ad)
            bulk = root / "bulk.csv"
            with bulk.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["duration", "event", *genes])
                for i in range(10):
                    writer.writerow([i + 1, i % 2, *([1.0] * len(genes))])

            case_store = CaseStore(root / "cases")
            result = case_store.create(CaseMetadata(
                "CASE-1", "other-disease", "Other disease", research_use_only=True))
            stored_h5ad = case_store.save_upload("CASE-1", "source.h5ad",
                                                 h5ad.read_bytes(), "single_cell")
            stored_bulk = case_store.save_upload("CASE-1", "bulk.csv",
                                                 bulk.read_bytes(), "bulk_survival")
            result, _ = prepare_uploaded_case(case_store.get("CASE-1"), str(stored_h5ad),
                                              str(stored_bulk))
            self.assertEqual(result.status, "patient_selection_required")
            schema = inspect_patient_columns(stored_h5ad)
            self.assertEqual(schema["columns"]["donor"], ["P1", "P2"])

            result, info = configure_patient_selection(
                result, stored_h5ad, "donor", "P2", str(stored_bulk),
                training_iterations=6)
            self.assertEqual(result.status, "training_required")
            self.assertEqual(result.metadata.patient_column, "donor")
            self.assertEqual(result.metadata.patient_id, "P2")
            self.assertEqual(result.metadata.analysis_scope, "patient")
            self.assertEqual(result.metadata.training_iterations, 6)
            self.assertTrue(Path(info["training_config"]).is_file())
            config = Path(info["training_config"]).read_text(encoding="utf-8")
            self.assertIn("iterations: 6", config)

            result = change_training_iterations(result, stored_h5ad, 8)
            self.assertEqual(result.metadata.training_iterations, 8)
            updated = Path(info["training_config"]).read_text(encoding="utf-8")
            self.assertIn("iterations: 8", updated)

            result = change_selected_patient(
                result, stored_h5ad, None, analysis_scope="cohort")
            self.assertEqual(result.metadata.analysis_scope, "cohort")
            self.assertIsNone(result.metadata.patient_id)
            self.assertEqual(result.audit[-1]["action"], "analysis_scope_changed")

    def test_entire_dataset_scope_is_persisted_in_training_config(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            genes = [f"G{i}" for i in range(500)]
            adata = ad.AnnData(
                X=np.ones((4, len(genes))),
                obs=pd.DataFrame({"donor": ["P1", "P1", "P2", "P2"]},
                                 index=[f"cell{i}" for i in range(4)]),
                var=pd.DataFrame(index=genes),
            )
            h5ad = root / "source.h5ad"
            adata.write_h5ad(h5ad)
            bulk = root / "bulk.csv"
            with bulk.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["duration", "event", *genes])
                for i in range(10):
                    writer.writerow([i + 1, i % 2, *([1.0] * len(genes))])

            store = CaseStore(root / "cases")
            result = store.create(CaseMetadata(
                "CASE-ALL", "other-disease", "Other disease", research_use_only=True))
            stored_h5ad = store.save_upload("CASE-ALL", "source.h5ad",
                                            h5ad.read_bytes(), "single_cell")
            stored_bulk = store.save_upload("CASE-ALL", "bulk.csv",
                                            bulk.read_bytes(), "bulk_survival")
            result, _ = prepare_uploaded_case(result, str(stored_h5ad), str(stored_bulk))
            result, info = configure_patient_selection(
                result, stored_h5ad, "donor", None, str(stored_bulk), analysis_scope="cohort")

            self.assertEqual(result.metadata.analysis_scope, "cohort")
            self.assertIsNone(result.metadata.patient_id)
            config = Path(info["training_config"]).read_text(encoding="utf-8")
            self.assertIn("analysis_scope: cohort", config)
            self.assertIn("selected_patient: null", config)

    def test_training_iterations_are_bounded_whole_numbers(self):
        self.assertEqual(validate_training_iterations("6"), 6)
        for invalid in (0, 101, -1, "six", "6.5"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_training_iterations(invalid)


if __name__ == "__main__":
    unittest.main()
