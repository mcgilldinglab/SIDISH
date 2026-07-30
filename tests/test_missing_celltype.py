from types import SimpleNamespace
import unittest

import anndata as ad
import numpy as np
import pandas as pd

import sidish_tools as tools


class MissingCelltypeTests(unittest.TestCase):
    def setUp(self):
        self.original = dict(tools.CONTEXT)
        adata = ad.AnnData(
            X=np.ones((4, 2)),
            obs=pd.DataFrame({
                "patient": ["P1", "P1", "P2", "P2"],
                "SIDISH": ["h", "b", "h", "b"],
            }, index=["c1", "c2", "c3", "c4"]),
            var=pd.DataFrame(index=["ERBB2", "EGFR"]),
        )
        tools.CONTEXT.update(
            mode="live", adata=adata, sdh=SimpleNamespace(adata=adata), cache={})

    def tearDown(self):
        tools.CONTEXT.clear()
        tools.CONTEXT.update(self.original)

    def test_patient_and_dataset_features_do_not_require_celltype_major(self):
        patient = tools.patient_features("P1")
        cohort = tools.cohort_features()
        self.assertEqual(patient["n_cells"], 2)
        self.assertEqual(patient["n_high_risk"], 1)
        self.assertFalse(patient["celltype_available"])
        self.assertEqual(patient["celltype_composition"], {})
        self.assertEqual(cohort["n_cells"], 4)
        self.assertFalse(cohort["celltype_available"])
        self.assertEqual(cohort["celltype_composition"], {})


if __name__ == "__main__":
    unittest.main()
