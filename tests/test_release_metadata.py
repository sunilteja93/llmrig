import re
import unittest
from pathlib import Path

import llmrig


class ReleaseMetadataTests(unittest.TestCase):
    def test_release_metadata_matches_source_version(self):
        root = Path(__file__).resolve().parents[1]
        citation = (root / "CITATION.cff").read_text(encoding="utf-8")
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")

        citation_match = re.search(r"^version:\s*(\S+)\s*$", citation, re.MULTILINE)
        self.assertIsNotNone(citation_match)
        self.assertEqual(citation_match.group(1), llmrig.VERSION)
        self.assertIn(f"## {llmrig.VERSION} - ", changelog)
        if llmrig.VERSION == "0.10.0":
            self.assertIn("date-released: 2026-10-18", citation)
            readme = (root / "README.md").read_text(encoding="utf-8")
            self.assertIn("pipx install llmrig", readme)
            self.assertIn("docs/VERIFIED_DEMOS.md", readme)
            self.assertTrue((root / "docs/RELEASE_NOTES_v0.10.0.md").exists())
            self.assertIn("docs/VERIFIED_DEMOS.md", readme)
            self.assertIn("GitHub Release", (root / "RELEASING.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
