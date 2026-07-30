import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sidish_data_router import route_case_data, validate_bulk_survival


def bulk_csv(path, rows=12, genes=60):
    header = ["duration", "event"] + [f"G{i}" for i in range(genes)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(header)
        for i in range(rows): writer.writerow([i + 1, i % 2] + [float(i + g) for g in range(genes)])
    return path


class RouterTests(unittest.TestCase):
    def test_valid_bulk(self):
        with tempfile.TemporaryDirectory() as d:
            result = validate_bulk_survival(bulk_csv(Path(d) / "bulk.csv"))
            self.assertTrue(result.ok); self.assertEqual(result.n_samples, 12); self.assertEqual(result.n_genes, 60)

    def test_bad_survival_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.csv"; path.write_text("time,status,G\n1,dead,2\n")
            result = validate_bulk_survival(path)
            self.assertFalse(result.ok); self.assertTrue(any("duration,event" in e for e in result.errors))

    def test_other_disease_requires_user_bulk(self):
        route = route_case_data("glioblastoma")
        self.assertTrue(route.needs_user_bulk); self.assertFalse(route.ready)

    def test_known_cancer_uses_locked_manuscript_bulk(self):
        with tempfile.TemporaryDirectory() as d:
            path = bulk_csv(Path(d) / "breast.csv")
            with patch.dict(os.environ, {"SIDISH_BREAST_BULK": str(path)}):
                route = route_case_data("TNBC", user_bulk_path="/tmp/should-not-be-used.csv")
            self.assertTrue(route.ready); self.assertTrue(route.known_cancer)
            self.assertEqual(route.bulk_path, str(path.resolve()))


if __name__ == "__main__": unittest.main()
