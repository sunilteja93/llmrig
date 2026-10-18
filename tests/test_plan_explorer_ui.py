"""Regression check for explicit interaction in the offline Plan Explorer."""
from pathlib import Path
import unittest


class PlanExplorerUiTests(unittest.TestCase):
    def test_file_selection_does_not_trigger_explore(self):
        content = (Path(__file__).resolve().parents[1] / "docs" / "plan-explorer.html").read_text(encoding="utf-8")
        self.assertIn('id="inspect">Explore Plan</button>', content)
        self.assertIn('Plan loaded. Click Explore Plan to view its evidence.', content)
        self.assertNotIn('document.getElementById("inspect").click()', content)


if __name__ == "__main__":
    unittest.main()
