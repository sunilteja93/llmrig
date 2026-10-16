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


if __name__ == "__main__":
    unittest.main()
