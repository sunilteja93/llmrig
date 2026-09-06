import ast
import copy
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import llmrig
from _llmrig import (
    CandidateAssessment,
    CandidateAssessments,
    CompatibilityState,
    DiscoveryState,
    ExecutionState,
    LocalAvailabilityState,
    MeasurementCapabilityState,
    MeasurementState,
    PlanningCandidate,
    RecommendationState,
    RuntimeAvailabilityState,
)


class AutopilotFoundationTests(unittest.TestCase):
    def evidence(self, detail="evidenced fact"):
        return llmrig.RecommendationEvidence("verified", "test", detail)

    def known(self, state, detail=None):
        return CandidateAssessment(
            state,
            llmrig.Confidence.HIGH,
            (self.evidence(detail or state.value),),
        )

    def unknown(self, state, detail=None):
        return CandidateAssessment(
            state,
            llmrig.Confidence.UNKNOWN,
            unknowns=(detail or f"{state.__class__.__name__} is unknown",),
        )

    def assessments(
        self,
        compatibility=CompatibilityState.COMPATIBLE,
        local=LocalAvailabilityState.AVAILABLE,
        execution=ExecutionState.EXECUTABLE,
        measurement=MeasurementState.NOT_MEASURED,
    ):
        return CandidateAssessments(
            discovery=self.known(DiscoveryState.DISCOVERED),
            compatibility=self.known(compatibility),
            runtime_availability=self.known(RuntimeAvailabilityState.AVAILABLE),
            local_availability=self.known(local),
            execution=self.known(execution),
            measurement_capability=self.known(
                MeasurementCapabilityState.MEASURABLE
            ),
            measurement=self.known(measurement),
            recommendation=self.known(RecommendationState.INCONCLUSIVE),
        )

    def test_private_package_and_existing_import_identity_are_unchanged(self):
        import _llmrig

        self.assertEqual(Path(llmrig.__file__).name, "llmrig.py")
        self.assertTrue(_llmrig.__spec__.submodule_search_locations)
        self.assertEqual(llmrig.VERSION, "0.6.0")
        for symbol in (
            "Confidence",
            "RecommendationEvidence",
            "Model",
            "ModelArtifact",
            "RuntimeCapability",
            "RuntimeCandidate",
            "RaceWorkload",
            "RaceConfiguration",
            "RaceCompetitor",
            "RaceResult",
            "ExecutionTarget",
            "AnalysisConfiguration",
            "ParetoResult",
            "DecisionResult",
        ):
            self.assertEqual(getattr(llmrig, symbol).__module__, "llmrig")

    def test_private_package_is_declared_for_installed_distribution(self):
        project = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        setuptools = project.split("[tool.setuptools]", 1)[1].split(
            "[tool.setuptools.dynamic]", 1
        )[0]
        self.assertIn('py-modules = ["llmrig"]', setuptools)
        self.assertIn('packages = ["_llmrig"]', setuptools)
        self.assertIn('llmrig = "llmrig:main"', project)

    def test_imports_do_not_probe_hardware_network_or_runtimes(self):
        root = Path(__file__).resolve().parents[1]
        script = """
from unittest import mock
with mock.patch('subprocess.run') as run, \\
     mock.patch('subprocess.Popen') as popen, \\
     mock.patch('urllib.request.urlopen') as network, \\
     mock.patch('webbrowser.open') as browser:
    import llmrig
    import _llmrig
    assert not run.called
    assert not popen.called
    assert not network.called
    assert not browser.called
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_candidate_states_are_orthogonal(self):
        states = self.assessments(
            compatibility=CompatibilityState.COMPATIBLE,
            local=LocalAvailabilityState.NOT_AVAILABLE,
            execution=ExecutionState.NOT_EXECUTABLE,
            measurement=MeasurementState.NOT_MEASURED,
        )
        self.assertEqual(states.compatibility.state, CompatibilityState.COMPATIBLE)
        unavailable_runtime = CandidateAssessments(
            discovery=states.discovery,
            compatibility=states.compatibility,
            runtime_availability=self.known(RuntimeAvailabilityState.UNAVAILABLE),
            local_availability=states.local_availability,
            execution=states.execution,
            measurement_capability=states.measurement_capability,
            measurement=states.measurement,
            recommendation=states.recommendation,
        )
        self.assertEqual(
            unavailable_runtime.runtime_availability.state,
            RuntimeAvailabilityState.UNAVAILABLE,
        )
        self.assertEqual(
            unavailable_runtime.compatibility.state, CompatibilityState.COMPATIBLE
        )
        self.assertEqual(
            states.local_availability.state, LocalAvailabilityState.NOT_AVAILABLE
        )
        self.assertNotEqual(
            states.local_availability.state.value, states.compatibility.state.value
        )
        self.assertEqual(states.execution.state, ExecutionState.NOT_EXECUTABLE)
        self.assertEqual(states.measurement.state, MeasurementState.NOT_MEASURED)

    def test_local_does_not_imply_executable_or_measured(self):
        states = self.assessments(
            local=LocalAvailabilityState.AVAILABLE,
            execution=ExecutionState.NOT_EXECUTABLE,
            measurement=MeasurementState.NOT_MEASURED,
        )
        self.assertEqual(states.local_availability.state, LocalAvailabilityState.AVAILABLE)
        self.assertEqual(states.execution.state, ExecutionState.NOT_EXECUTABLE)
        self.assertEqual(states.measurement.state, MeasurementState.NOT_MEASURED)

        executable = self.assessments(
            execution=ExecutionState.EXECUTABLE,
            measurement=MeasurementState.NOT_MEASURED,
        )
        self.assertEqual(executable.execution.state, ExecutionState.EXECUTABLE)
        self.assertEqual(executable.measurement.state, MeasurementState.NOT_MEASURED)

    def test_unknown_stays_unknown_and_requires_an_explicit_unknown(self):
        assessment = self.unknown(CompatibilityState.UNKNOWN, "static fit is unknown")
        self.assertEqual(assessment.to_dict()["state"], "unknown")
        self.assertEqual(assessment.to_dict()["confidence"], "unknown")
        self.assertEqual(assessment.to_dict()["unknowns"], ["static fit is unknown"])
        with self.assertRaises(ValueError):
            CandidateAssessment(CompatibilityState.UNKNOWN, llmrig.Confidence.HIGH)
        with self.assertRaises(ValueError):
            CandidateAssessment(CompatibilityState.UNKNOWN, llmrig.Confidence.UNKNOWN)

    def test_candidate_serialization_is_deterministic(self):
        candidate = PlanningCandidate(
            logical_model_id="logical/model",
            artifact_id="artifact:one",
            runtime="ollama",
            artifact_format="Ollama",
            quantization=None,
            assessments=self.assessments(),
        )
        first = json.dumps(candidate.to_dict(), sort_keys=True)
        second = json.dumps(candidate.to_dict(), sort_keys=True)
        self.assertEqual(first, second)
        self.assertNotIn("locator", first.lower())
        with self.assertRaises(ValueError):
            PlanningCandidate(
                logical_model_id="logical/model",
                artifact_id=str(Path.home() / "private.gguf"),
                runtime="llama.cpp",
                artifact_format="GGUF",
                quantization=None,
                assessments=self.assessments(),
            )
        with self.assertRaises(ValueError):
            CandidateAssessment(
                DiscoveryState.DISCOVERED,
                llmrig.Confidence.HIGH,
                (
                    llmrig.RecommendationEvidence(
                        "verified",
                        "test",
                        "observed at path=/Users/private/model.gguf",
                    ),
                ),
            )

    def test_ollama_inventory_is_read_only_and_preserves_evidenced_identity(self):
        installed = [
            {"name": "qwen3.8:27b-mtp-q4_K_M", "id": "abcdef123456", "raw": "private raw"},
            {"name": "qwen3.8:latest", "id": "abcdef123456", "raw": "private alias raw"},
            {"name": "other/model:latest", "id": "123456abcdef", "raw": "private unknown raw"},
        ]
        with mock.patch.object(
            llmrig, "installed_ollama_models", return_value=installed
        ) as listing, mock.patch.object(llmrig, "pull_model") as pull, mock.patch.object(
            llmrig.OLLAMA_RUNTIME, "ensure_available"
        ) as ensure, mock.patch.object(llmrig, "ollama_generate") as inference:
            targets = llmrig._autopilot_ollama_inventory()

        self.assertEqual(listing.call_count, 1)
        pull.assert_not_called()
        ensure.assert_not_called()
        inference.assert_not_called()
        self.assertEqual(len(targets), 2)
        records = [target.record for target in targets]
        curated = next(record for record in records if record.logical_model_id == "qwen3.8")
        unknown = next(record for record in records if record.logical_model_id is None)
        self.assertFalse(curated.identity_attested)
        self.assertEqual(curated.association_kind, "curated_name_match")
        self.assertIn(
            "installed artifact content identity is not independently attested",
            curated.unknowns,
        )
        self.assertFalse(unknown.identity_attested)
        self.assertEqual(unknown.identity_confidence, llmrig.Confidence.UNKNOWN)
        payload = json.dumps([record.to_dict() for record in records], sort_keys=True)
        self.assertNotIn("private raw", payload)
        self.assertNotIn("locator", payload.lower())
        with mock.patch.object(
            llmrig, "installed_ollama_models", return_value=list(reversed(installed))
        ):
            reversed_targets = llmrig._autopilot_ollama_inventory()
        reversed_payload = json.dumps(
            [target.record.to_dict() for target in reversed_targets], sort_keys=True
        )
        self.assertEqual(payload, reversed_payload)

    def test_explicit_native_inventory_keeps_paths_private_and_identity_unattested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf = root / "private-model.gguf"
            gguf.write_bytes(b"gguf")
            mlx = root / "private-mlx"
            mlx.mkdir()
            (mlx / "config.json").write_text("{}", encoding="utf-8")
            (mlx / "model.safetensors").write_bytes(b"weights")
            values = (f"llama.cpp={gguf}", f"mlx-lm={mlx}")

            with mock.patch("os.walk") as walk, mock.patch.object(
                Path, "rglob"
            ) as recursive_glob, mock.patch.object(
                llmrig, "pull_model"
            ) as pull, mock.patch.object(
                llmrig, "ollama_generate"
            ) as inference:
                targets = llmrig._autopilot_explicit_native_inventory(
                    "org/logical-model", values
                )

            walk.assert_not_called()
            recursive_glob.assert_not_called()
            pull.assert_not_called()
            inference.assert_not_called()
            self.assertEqual(len(targets), 2)
            for target in targets:
                record = target.record
                self.assertFalse(record.identity_attested)
                self.assertEqual(record.association_kind, "user_supplied")
                self.assertEqual(record.identity_confidence, llmrig.Confidence.UNKNOWN)
                self.assertIsNone(record.quantization)
                payload = json.dumps(record.to_dict(), sort_keys=True)
                self.assertNotIn(str(root), payload)
                self.assertNotIn(str(Path.home()), payload)
                self.assertNotIn("private-model.gguf", payload)
                self.assertNotIn("private-mlx", payload)
                self.assertNotIn(str(root), repr(target))
                with self.assertRaises(TypeError):
                    pickle.dumps(target)
                with self.assertRaises(TypeError):
                    copy.copy(target)

    def test_inventory_does_not_scan_when_no_native_paths_are_supplied(self):
        with mock.patch("os.walk") as walk, mock.patch.object(
            Path, "iterdir"
        ) as iterdir, mock.patch.object(Path, "rglob") as recursive_glob:
            targets = llmrig._autopilot_explicit_native_inventory("org/model", ())
        self.assertEqual(targets, ())
        walk.assert_not_called()
        iterdir.assert_not_called()
        recursive_glob.assert_not_called()

    def test_new_internal_sources_parse_on_supported_python_syntax(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "llmrig.py",
            "_llmrig/__init__.py",
            "_llmrig/inventory.py",
            "_llmrig/planning.py",
            "_llmrig/privacy.py",
        ):
            ast.parse(
                (root / relative).read_text(encoding="utf-8"),
                filename=relative,
                feature_version=(3, 9),
            )


if __name__ == "__main__":
    unittest.main()
