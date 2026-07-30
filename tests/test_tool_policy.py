import unittest

import sidish_tools as tools


class ToolPolicyTests(unittest.TestCase):
    def setUp(self):
        self.old_context = dict(tools.CONTEXT)
        self.old_demo = tools.CONFIG.get("demo_mode")
        self.old_force = tools.CONFIG.get("force_mock")
        tools.CONTEXT.update(sdh=None, adata=None, mode="uninitialized", cache={})
        tools.CONFIG["demo_mode"] = False; tools.CONFIG["force_mock"] = False

    def tearDown(self):
        tools.CONTEXT.clear(); tools.CONTEXT.update(self.old_context)
        tools.CONFIG["demo_mode"] = self.old_demo; tools.CONFIG["force_mock"] = self.old_force

    def test_clinician_mode_fails_closed(self):
        with self.assertRaises(RuntimeError): tools.highrisk_overview()

    def test_demo_must_be_explicit(self):
        tools.CONFIG["demo_mode"] = True
        result = tools.highrisk_overview()
        self.assertTrue(result["demo"]); self.assertEqual(result["mode"], "mock")

    def test_pathway_and_drug_tools_are_first_class(self):
        names = {x["function"]["name"] for x in tools.TOOLS_SPEC}
        self.assertIn("pathway_perturbation", names)
        self.assertIn("drug_perturbation", names)
        pathway = next(x for x in tools.TOOLS_SPEC if x["function"]["name"] == "pathway_perturbation")
        self.assertIn("patient", pathway["function"]["parameters"]["properties"])


if __name__ == "__main__": unittest.main()
