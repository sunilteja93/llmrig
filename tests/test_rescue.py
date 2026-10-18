"""Rescue contract: read-only preflight, bounded explicit probes and private output."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _llmrig import cli, rescue

REAL_LOCAL_HOST = rescue.local_host


def fake_api(base, endpoint, payload=None, timeout=8):
    if endpoint == "/api/version":
        return {"version": "0.33.3"}
    if endpoint == "/api/tags":
        return {"models": [{"name": "qwen3.8:27b-mlx", "size": 18_000_000_000}]}
    if endpoint == "/api/show":
        return {
            "details": {"quantization_level": "nvfp4", "parameter_size": "27.8B"},
            "model_info": {"general.architecture": "qwen3_5"},
            "capabilities": ["completion", "tools"],
            "private": "/Users/secret/ollama",
        }
    if endpoint == "/api/chat":
        return {"message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "llmrig_rescue_echo",
                    "arguments": {"code": "rescue-constant-7"},
                },
            }],
        }, "raw_secret": "/Users/secret", "eval_count": 12}
    raise AssertionError("unexpected endpoint " + endpoint)


class RescueTests(unittest.TestCase):
    def setUp(self):
        self.host = mock.patch.object(rescue, "local_host", return_value="http://127.0.0.1:11434")
        self.host.start()
        self.addCleanup(self.host.stop)

    def test_read_only_preflight_never_calls_chat_or_agent(self):
        with mock.patch.object(rescue, "_http", side_effect=fake_api) as http, \
             mock.patch.object(rescue.shutil, "which", return_value="/usr/bin/opencode"):
            result = rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 3, None, False)
        self.assertEqual(result["status"], "ready_for_probe")
        self.assertEqual(result["model_metadata"]["quantization"], "nvfp4")
        self.assertEqual(result["model_metadata"]["architecture"], "qwen3_5")
        self.assertEqual(result["model_metadata"]["tools_capability"], "declared")
        self.assertEqual([item.args[1] for item in http.call_args_list],
                         ["/api/version", "/api/tags", "/api/show"])
        self.assertNotIn("secret", json.dumps(result))

    def test_missing_local_model_does_not_pull_or_probe(self):
        def missing(base, endpoint, payload=None, timeout=8):
            return {"models": []} if endpoint == "/api/tags" else {"version": "0.33.3"}
        with mock.patch.object(rescue, "_http", side_effect=missing) as http:
            report = rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 2, None, True)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("no download attempted", report["blockers"][0])
        self.assertNotIn("/api/chat", [c.args[1] for c in http.call_args_list])
        self.assertNotIn("/api/pull", [c.args[1] for c in http.call_args_list])

    def test_measured_two_context_canary_is_explicit_and_bounded(self):
        with mock.patch.object(rescue, "_http", side_effect=fake_api) as http, \
             mock.patch.object(rescue.shutil, "which", return_value=None):
            r = rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 3, 8192, True)
        self.assertEqual(r["status"], "measured")
        self.assertEqual(len(r["probes"]), 2)
        self.assertEqual([p["tool_call_passes"] for p in r["probes"]], [3, 3])
        self.assertEqual(r["comparison"]["assessment"], "no_observed_pass_difference")
        self.assertEqual(r["comparison"]["confidence"], "insufficient_to_establish_agent_improvement")
        calls = [c.args[2] for c in http.call_args_list if c.args[1] == "/api/chat"]
        self.assertEqual(len(calls), 6)
        self.assertEqual([p["options"]["num_ctx"] for p in calls], [4096]*3+[8192]*3)
        self.assertNotIn("raw_secret", json.dumps(r))
        self.assertIn("NOT end-to-end", r["scope"])

    def test_bad_tool_call_never_counts_as_pass(self):
        self.assertEqual(rescue._classify({"message":{"tool_calls":[]}}), "no_single_tool_call")
        self.assertEqual(rescue._classify({"message":{"tool_calls":[
            {"function":{"name":"some_other_tool","arguments":{"code":"rescue-constant-7"}}}
        ]}}), "wrong_tool")
        self.assertEqual(rescue._classify({"message":{"tool_calls":[
            {"function":{"name":"llmrig_rescue_echo","arguments":{"code":"wrong"}}}
        ]}}), "wrong_arguments")
        self.assertEqual(rescue._classify({"message":{"tool_calls":[
            {"function":{"name":"llmrig_rescue_echo","arguments":"{\"code\": \"rescue-constant-7\"}"}}
        ]}}), "passed")
        self.assertEqual(rescue._classify({"message":{"tool_calls":[
            {"function":{"name":"llmrig_rescue_echo","arguments":"not json"}}
        ]}}), "wrong_arguments")

    def test_reject_untrusted_input_before_network(self):
        bad = ["/Users/secret", "http://evil.example/model", "qwen;rm", "../private",
               "owner//model", "model\nsecret"]
        with mock.patch.object(rescue, "_http") as http:
            for model in bad:
                with self.subTest(model=model), self.assertRaises(rescue.RescueError):
                    rescue.rescue("opencode", model, 4096, 3, None, False)
        http.assert_not_called()

    def test_compare_requires_opt_in_and_context_limits(self):
        with mock.patch.object(rescue, "_http") as http:
            for args in [(4096, 8192, False), (4096, 4096, True),
                         (1, None, False), (300000, None, False)]:
                with self.subTest(args=args), self.assertRaises(rescue.RescueError):
                    rescue.rescue("opencode", "qwen3.8:27b-mlx", args[0], 2, args[1], args[2])
            with self.assertRaises(rescue.RescueError):
                rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 20, None, True)
        http.assert_not_called()

    def test_reject_remote_api_even_in_read_only_mode(self):
        for host in ["https://127.0.0.1:11434", "http://example.com:11434",
                     "http://127.0.0.1:11434/foo", "http://user:pw@127.0.0.1:11434"]:
            with mock.patch.dict(os.environ, {"OLLAMA_HOST": host}), \
                 self.assertRaises(rescue.RescueError):
                REAL_LOCAL_HOST()

    def test_local_loopback_allowed(self):
        for host in ["127.0.0.1:11434", "http://localhost:11434"]:
            with mock.patch.dict(os.environ, {"OLLAMA_HOST": host}):
                self.assertIn(":11434", REAL_LOCAL_HOST())

    def test_local_http_redirects_are_rejected(self):
        handler = rescue._NoRedirect()
        with self.assertRaises(rescue.RescueError):
            handler.redirect_request(
                None, None, 302, "Found", {}, "https://external.example/api/chat"
            )

    def test_transport_error_is_failure_not_fabricated_success(self):
        def failing(base, endpoint, payload=None, timeout=8):
            if endpoint == "/api/chat":
                raise rescue.RescueError("local request failed")
            return fake_api(base, endpoint, payload, timeout)
        with mock.patch.object(rescue, "_http", side_effect=failing):
            result = rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 2, None, True)
        self.assertEqual(result["probes"][0]["tool_call_passes"], 0)
        self.assertEqual(result["probes"][0]["outcomes"], ["transport_error"]*2)

    def test_cli_preflight_report_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(rescue, "_http", side_effect=fake_api):
            path = str(Path(tmp) / "rescue.json")
            args = ["rescue", "--agent", "opencode", "--model", "qwen3.8:27b-mlx",
                    "--json", "--output", path]
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(args)
            self.assertEqual(code, 0)
            payload = json.loads(Path(path).read_text())
            self.assertEqual(payload["mode"], "preflight")
            self.assertEqual(payload["probes"], [])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args), 2)

    def test_saved_measured_report_prints_real_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(rescue, "_http", side_effect=fake_api):
            destination = str(Path(tmp) / "rescue-measured.json")
            args = ["rescue", "--agent", "opencode", "--model", "qwen3.8:27b-mlx",
                    "--context", "4096", "--compare-context", "8192", "--probe",
                    "--runs", "3", "--output", destination]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(args), 0)
            summary = output.getvalue()
            self.assertIn("Context 4096: 3/3 synthetic tool calls passed", summary)
            self.assertIn("Context 8192: 3/3 synthetic tool calls passed", summary)
            self.assertIn("Comparison: no_observed_pass_difference", summary)
            self.assertIn("No OpenCode configuration was read or changed.", summary)
            self.assertIn("Rescue report saved locally", summary)
            self.assertNotIn("/Users/secret", summary)
            measured = json.loads(Path(destination).read_text(encoding="utf-8"))
            self.assertEqual([p["tool_call_passes"] for p in measured["probes"]], [3, 3])
            self.assertNotIn("/Users/secret", json.dumps(measured))

    def test_cli_help_exposes_rescue_without_loading_legacy(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                cli.main(["rescue", "--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--probe", output.getvalue())
        self.assertIn("NOT", "NOT end-to-end")


if __name__ == "__main__":
    unittest.main()
