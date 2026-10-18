"""Guard the newcomer install path and evidence-backed README demo."""
from pathlib import Path
import unittest
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]


class OnboardingDocsTests(unittest.TestCase):
    def test_readme_links_to_real_demo_and_current_source_install(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/VERIFIED_DEMOS.md", readme)
        self.assertIn("assets/verified-dual-runtime.svg", readme)
        self.assertIn("docs/INSTALL.md", readme)
        self.assertIn("pipx install 'git+https://github.com/sunilteja93/llmrig.git@main'", readme)
        self.assertIn("different models and workloads", readme.lower())
        self.assertTrue((ROOT / "docs/INSTALL.md").exists())
        self.assertTrue((ROOT / "docs/VERIFIED_DEMOS.md").exists())

    def test_example_discloses_uncertainty_and_does_not_invent_metrics(self):
        demo = (ROOT / "docs/VERIFIED_DEMOS.md").read_text(encoding="utf-8")
        self.assertIn("receipt-38bdf66af4a4a9ce", demo)
        self.assertIn("mlx-community/Qwen3-0.6B-4bit", demo)
        self.assertIn("qwen3.8:27b-mlx", demo)
        self.assertIn("49.93", demo)
        self.assertIn("0.29 s", demo)
        self.assertIn("HTTP 401", demo)
        self.assertIn("not a controlled runtime", demo)
        self.assertIn("not recorded", demo.lower())
        self.assertIn("catalog", demo.lower())
        self.assertNotIn("5,048,673.82", demo)

    def test_vector_card_parses_and_labels_reconstruction(self):
        path = ROOT / "assets/verified-dual-runtime.svg"
        svg = ElementTree.fromstring(path.read_text(encoding="utf-8"))
        self.assertEqual(svg.tag, "{http://www.w3.org/2000/svg}svg")
        text = path.read_text(encoding="utf-8")
        self.assertIn("Reconstructed illustration", text)
        self.assertIn("Not recorded", text)
        self.assertIn("49.93 tok/s", text)
        self.assertIn("not a controlled comparison", text.lower())

    def test_installation_guide_does_not_confuse_unreleased_main_and_pypi(self):
        guide = (ROOT / "docs/INSTALL.md").read_text(encoding="utf-8")
        self.assertIn("PyPI", guide)
        self.assertIn("0.9.3", guide)
        self.assertIn("unreleased moving branch", guide)
        self.assertIn("(cd /tmp", guide)
        self.assertIn("pipx ensurepath", guide)
        self.assertIn("Windows PowerShell", guide)


if __name__ == "__main__":
    unittest.main()
