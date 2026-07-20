import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareDatasetTests(unittest.TestCase):
    def test_normalize_json_rubrics(self):
        value = json.dumps([
            {"question": "Contains the diagnosis?", "pass_criteria": "YES"},
            "Does not invent medication",
        ])
        result = MODULE.normalize_rubrics(value)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["pass_criteria"], "YES")

    def test_convert_row_routes_to_clinical_agent(self):
        row = {
            "session_id": "abc",
            "text": "Doctor: Fever? Patient: No fever.",
            "sample_prompt": "Write the clinical note.",
            "rubrics": '["The note preserves the fever negation"]',
        }
        result = MODULE.convert_row(row, 0)
        self.assertEqual(result["uuid"], "abc")
        self.assertEqual(
            result["agent_ref"]["name"], "clinical_note_simple_agent"
        )
        self.assertEqual(len(result["rubric"]), 1)


if __name__ == "__main__":
    unittest.main()

