"""Dependency-free checks for the versioned text-prompt study definition."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE = ROOT / "scripts" / "open_vocab" / "aquarium_prompts.json"


class PromptStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))

    def test_all_sets_cover_same_categories_in_order(self):
        study = self.study
        categories = study["category_id_to_name"]
        self.assertEqual(list(categories), [str(i) for i in range(1, 8)])
        self.assertEqual(study["bare"], list(categories.values()))
        for key in ("bare", "synonym_variant", "descriptions"):
            with self.subTest(set=key):
                self.assertEqual(len(study[key]), 7)
                self.assertTrue(all(isinstance(s, str) and s.strip() for s in study[key]))

    def test_synonym_variant_only_changes_documented_classes(self):
        study = self.study
        changes = {str(i): [before, after] for i, (before, after) in enumerate(
            zip(study["bare"], study["synonym_variant"]), start=1) if before != after}
        self.assertEqual(changes, study["changed_only"])
        self.assertEqual(set(changes), {"2", "6"})

    def test_description_variant_changes_all_classes(self):
        study = self.study
        self.assertTrue(all(a != b for a, b in zip(study["bare"], study["descriptions"])))


if __name__ == "__main__":
    unittest.main()
