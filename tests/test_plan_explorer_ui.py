"""Regression check for explicit interaction in the offline Plan Explorer."""
from pathlib import Path
import unittest


class PlanExplorerUiTests(unittest.TestCase):
    def test_file_selection_does_not_trigger_explore(self):
        content = (Path(__file__).resolve().parents[1] / "docs" / "plan-explorer.html").read_text(encoding="utf-8")
        self.assertIn('id="inspect">Explore Plan</button>', content)
        self.assertIn('Plan loaded. Click Explore Plan to view its evidence.', content)
        self.assertNotIn('document.getElementById("inspect").click()', content)

    def test_report_file_selection_does_not_auto_render(self):
        content = (Path(__file__).resolve().parents[1] / "docs" / "rig-report-viewer.html").read_text(encoding="utf-8")
        self.assertIn('Report loaded. Click View report to display it.', content)
        self.assertNotIn('reader.onload=function(){editor.value=reader.result;fromInput();}', content)


if __name__ == "__main__":
    unittest.main()
