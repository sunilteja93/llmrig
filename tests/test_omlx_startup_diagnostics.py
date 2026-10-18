"""Privacy-preserving diagnostics for approved oMLX runtime actions."""
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from _llmrig.runtime_actions import OmlxRuntimeActions


def stopped_probe():
    return SimpleNamespace(installed=True, cli_path="/opt/homebrew/bin/omlx", service_available=False)


class OmlxStartupDiagnosticsTests(unittest.TestCase):
    def test_start_success_but_health_check_unavailable_gives_actionable_guidance(self):
        with mock.patch("_llmrig.runtime_actions.OmlxRuntimeAdapter") as adapter, \
             mock.patch("_llmrig.runtime_actions.subprocess.run", return_value=SimpleNamespace(
                 returncode=0, stdout="started", stderr="/Users/private-user/secret"
             )), \
             mock.patch("_llmrig.runtime_actions.time.sleep"):
            adapter.return_value.probe.return_value = stopped_probe()
            result = OmlxRuntimeActions().start(None)
        self.assertEqual(result.status, "failed")
        self.assertIn("model-inventory API", result.detail)
        self.assertIn("authorization", result.detail)
        self.assertIn("port binding", result.detail)
        self.assertNotIn("private-user", result.detail)

    def test_start_failure_mentions_return_code_not_private_stderr(self):
        with mock.patch("_llmrig.runtime_actions.OmlxRuntimeAdapter") as adapter, \
             mock.patch("_llmrig.runtime_actions.subprocess.run", return_value=SimpleNamespace(
                 returncode=3, stdout="", stderr="API_KEY=sensitive /Users/private-user"
             )):
            adapter.return_value.probe.return_value = stopped_probe()
            result = OmlxRuntimeActions().start(None)
        self.assertEqual(result.status, "failed")
        self.assertIn("status 3", result.detail)
        self.assertIn("service logs", result.detail)
        self.assertNotIn("API_KEY", result.detail)
        self.assertNotIn("/Users/", result.detail)

    def test_start_timeout_is_distinguished_from_failed_health_check(self):
        with mock.patch("_llmrig.runtime_actions.OmlxRuntimeAdapter") as adapter, \
             mock.patch("_llmrig.runtime_actions.subprocess.run", side_effect=subprocess.TimeoutExpired(
                 cmd=["omlx", "start"], timeout=60
             )):
            adapter.return_value.probe.return_value = stopped_probe()
            result = OmlxRuntimeActions().start(None)
        self.assertEqual(result.status, "failed")
        self.assertIn("timed out", result.detail)

    def test_refresh_error_keeps_subprocess_output_private(self):
        with mock.patch("_llmrig.runtime_actions.OmlxRuntimeAdapter") as adapter, \
             mock.patch("_llmrig.runtime_actions.subprocess.run", return_value=SimpleNamespace(
                 returncode=7, stdout="", stderr="Authorization: Bearer private-key"
             )):
            adapter.return_value.probe.return_value = stopped_probe()
            result = OmlxRuntimeActions().refresh_after_acquisition(None)
        self.assertEqual(result.status, "failed")
        self.assertIn("status 7", result.detail)
        self.assertNotIn("private-key", result.detail)


if __name__ == "__main__":
    unittest.main()
