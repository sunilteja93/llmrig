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

    def test_report_viewer_collapses_input_after_rendering_and_reveals_full_revision(self):
        content = (Path(__file__).resolve().parents[1] / "docs" / "rig-report-viewer.html").read_text(encoding="utf-8")
        self.assertIn('id="loader" aria-label="Load a report" open', content)
        self.assertIn('document.getElementById("loader").open=false', content)
        self.assertIn('id="revision-details"', content)
        self.assertIn('id="revision-value"', content)
        self.assertIn('revision.slice(0,12)', content)
        self.assertIn('not recorded in this receipt', content)
        self.assertIn('LLMRig does not estimate tokens/s from latency', content)

    def test_ollama_tag_and_catalog_format_caveat_in_viewer(self):
        content = (Path(__file__).resolve().parents[1] / "docs" / "rig-report-viewer.html").read_text(encoding="utf-8")
        self.assertIn('id="artifact-tag"', content)
        self.assertIn('payload.ollama_model_tag', content)
        self.assertIn('Catalog format', content)
        self.assertIn('not runtime-verified quantization', content)
        self.assertIn('tag.textContent=', content)


if __name__ == "__main__":
    unittest.main()
