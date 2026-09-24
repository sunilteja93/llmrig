from __future__ import annotations

import unittest

from _llmrig.autopilot_actions import ActionKind, AutopilotPlan, PlannedAction
from _llmrig.hf_artifacts import classify_hub_siblings, summarize_hub_artifacts
from _llmrig.riggraph import RigEdge, RigGraph, RigNode


class HuggingFaceArtifactTests(unittest.TestCase):
    def test_gguf_quantization_and_runtime_hint_are_evidence_bounded(self) -> None:
        artifacts = classify_hub_siblings(
            "owner/model",
            [
                {"rfilename": "model-Q4_K_M.gguf", "size": 10},
                {"rfilename": "README.md", "size": 20},
            ],
        )
        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertEqual(artifact.format, "GGUF")
        self.assertEqual(artifact.quantization, "Q4_K_M")
        self.assertEqual(artifact.runtime_hints, ("llama.cpp",))

    def test_generic_safetensors_does_not_claim_runtime(self) -> None:
        artifacts = classify_hub_siblings(
            "owner/model",
            [{"rfilename": "model-00001-of-00002.safetensors", "size": 100}],
        )
        self.assertEqual(artifacts[0].runtime_hints, ())
        self.assertIn("does not establish runtime compatibility", artifacts[0].unknowns[0])

    def test_mlx_tag_is_only_a_runtime_hint(self) -> None:
        artifacts = classify_hub_siblings(
            "owner/model",
            [{"rfilename": "model.safetensors", "size": 100}],
            tags=("mlx",),
        )
        summary = summarize_hub_artifacts(artifacts)
        self.assertEqual(summary["runtime_hints"], ["mlx-lm", "omlx"])
        self.assertIn("not proof", summary["warning"])


class RigGraphTests(unittest.TestCase):
    def test_graph_is_deterministic(self) -> None:
        hardware = RigNode.from_mapping("hw:1", "hardware", {"ram_gib": 48})
        runtime = RigNode.from_mapping("runtime:omlx", "runtime", {"name": "omlx"})
        edge = RigEdge.from_mapping(
            "hw:1",
            "supports",
            "runtime:omlx",
            "verified",
            {"platform": "Darwin"},
        )
        first = RigGraph.from_records((hardware, runtime), (edge,))
        second = RigGraph.from_records((runtime, hardware), (edge,))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_edge_requires_existing_nodes(self) -> None:
        graph = RigGraph()
        graph.add_node(RigNode("a", "hardware"))
        with self.assertRaises(ValueError):
            graph.add_edge(RigEdge("a", "supports", "missing", "inferred"))


class AutopilotActionTests(unittest.TestCase):
    def test_mutating_plan_requires_explicit_approval(self) -> None:
        plan = AutopilotPlan(
            "plan-1",
            (
                PlannedAction(
                    "download",
                    ActionKind.ACQUIRE_ARTIFACT,
                    "omlx",
                    "Acquire an explicitly selected artifact",
                    True,
                ),
            ),
            "user-selected candidate",
        )
        with self.assertRaises(PermissionError):
            plan.assert_apply_allowed(explicit_user_approval=False)
        plan.assert_apply_allowed(explicit_user_approval=True)

    def test_blocked_plan_cannot_apply_even_with_approval(self) -> None:
        plan = AutopilotPlan(
            "plan-2",
            (
                PlannedAction(
                    "install",
                    ActionKind.INSTALL_RUNTIME,
                    "omlx",
                    "Install runtime",
                    True,
                    blockers=("installation path is not trusted",),
                ),
            ),
            "candidate needs a runtime",
        )
        with self.assertRaises(RuntimeError):
            plan.assert_apply_allowed(explicit_user_approval=True)


if __name__ == "__main__":
    unittest.main()
