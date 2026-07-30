import unittest

from sidish_policy import audit_text, enforce_text


class PolicyTests(unittest.TestCase):
    def test_prohibited_claims_are_blocked(self):
        for text in ["Start cilengitide immediately.", "I recommend pembrolizumab for the patient.",
                     "The patient is predicted to respond.", "Avoid chemotherapy.",
                     "The diagnosis is confirmed.", "I recommend cilengitide.",
                     "The best treatment is immunotherapy.",
                     "This patient has a poor prognosis.",
                     "The patient is likely to respond."]:
            with self.subTest(text=text):
                self.assertTrue(audit_text(text))
                with self.assertRaises(ValueError): enforce_text(text)

    def test_decision_support_wording_passes(self):
        text = "SIDISH nominates the FN1-associated network for orthogonal validation."
        self.assertEqual(audit_text(text), []); self.assertEqual(enforce_text(text), text)


if __name__ == "__main__": unittest.main()
