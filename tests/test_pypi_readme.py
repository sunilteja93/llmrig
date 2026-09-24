import re
import unittest
from pathlib import Path


class PyPIReadmeTests(unittest.TestCase):
    def test_readme_links_are_portable_outside_github(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
            encoding="utf-8"
        )

        markdown_targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
        html_targets = re.findall(r"(?:href|src)=\"([^\"]+)\"", readme)

        relative = []
        for target in markdown_targets + html_targets:
            target = target.strip()
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            relative.append(target)

        self.assertEqual(
            relative,
            [],
            "README contains repository-relative links that will break on PyPI",
        )


if __name__ == "__main__":
    unittest.main()
