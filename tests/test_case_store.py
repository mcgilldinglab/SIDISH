import tempfile
import unittest
from pathlib import Path

from sidish_case_store import CaseStore
from sidish_contracts import CaseMetadata


class CaseStoreTests(unittest.TestCase):
    def test_case_isolation_and_upload(self):
        with tempfile.TemporaryDirectory() as d:
            store = CaseStore(d); result = store.create(CaseMetadata("CASE 1", "breast", "Breast cancer"))
            self.assertEqual(result.metadata.case_id, "CASE-1")
            path = store.save_upload("CASE-1", "sample.h5ad", b"test", "single_cell")
            self.assertEqual(path.parent, store.case_dir("CASE-1") / "inputs")
            self.assertEqual(store.get("CASE-1").audit[-1]["action"], "file_uploaded")

    def test_case_path_escape_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            store = CaseStore(d); store.create(CaseMetadata("CASE-2", "breast", "Breast cancer"))
            with self.assertRaises(PermissionError):
                store.assert_case_path("CASE-2", Path(d).parent / "outside")


if __name__ == "__main__": unittest.main()
