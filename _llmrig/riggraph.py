"""Graph-shaped local evidence, persistence, and calibration for LLMRig."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .acquisition import acquisition_registry_file
from .privacy import validate_public_text


RIGGRAPH_SCHEMA_VERSION = "0.9"


@dataclass(frozen=True)
class RigNode:
    node_id: str
    kind: str
    attributes: Tuple[Tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, node_id: str, kind: str, attributes: Dict[str, Any]) -> "RigNode":
        return cls(node_id=node_id, kind=kind, attributes=tuple(sorted(attributes.items())))

    def to_dict(self) -> dict:
        return {"id": self.node_id, "kind": self.kind, "attributes": dict(self.attributes)}


@dataclass(frozen=True)
class RigEdge:
    source: str
    relation: str
    target: str
    evidence_level: str
    attributes: Tuple[Tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        source: str,
        relation: str,
        target: str,
        evidence_level: str,
        attributes: Dict[str, Any],
    ) -> "RigEdge":
        return cls(
            source=source,
            relation=relation,
            target=target,
            evidence_level=evidence_level,
            attributes=tuple(sorted(attributes.items())),
        )

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "evidence_level": self.evidence_level,
            "attributes": dict(self.attributes),
        }


@dataclass
class RigGraph:
    """Deterministic graph representation used by persisted evidence records."""

    schema_version: str = RIGGRAPH_SCHEMA_VERSION
    _nodes: Dict[str, RigNode] = field(default_factory=dict)
    _edges: Dict[Tuple[str, str, str, str], RigEdge] = field(default_factory=dict)

    def add_node(self, node: RigNode) -> None:
        existing = self._nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"node id collision: {node.node_id}")
        self._nodes[node.node_id] = node

    def add_edge(self, edge: RigEdge) -> None:
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise ValueError("RigGraph edges require both endpoint nodes")
        key = (edge.source, edge.relation, edge.target, edge.evidence_level)
        existing = self._edges.get(key)
        if existing is not None and existing != edge:
            raise ValueError("edge identity collision")
        self._edges[key] = edge

    @property
    def nodes(self) -> Tuple[RigNode, ...]:
        return tuple(self._nodes[key] for key in sorted(self._nodes))

    @property
    def edges(self) -> Tuple[RigEdge, ...]:
        return tuple(self._edges[key] for key in sorted(self._edges))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_records(
        cls,
        nodes: Iterable[RigNode],
        edges: Iterable[RigEdge],
        *,
        schema_version: str = RIGGRAPH_SCHEMA_VERSION,
    ) -> "RigGraph":
        graph = cls(schema_version=schema_version)
        for node in nodes:
            graph.add_node(node)
        for edge in edges:
            graph.add_edge(edge)
        return graph


def _metric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def calibration_delta(predicted: Any, measured: Any) -> Optional[float]:
    """Return measured - predicted only when both values are comparable numbers."""

    predicted_value = _metric(predicted)
    measured_value = _metric(measured)
    if predicted_value is None or measured_value is None:
        return None
    return measured_value - predicted_value


@dataclass(frozen=True)
class CalibrationRecord:
    generation_tps_delta: Optional[float]
    prompt_eval_tps_delta: Optional[float]
    total_latency_s_delta: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "generation_tps_delta": self.generation_tps_delta,
            "prompt_eval_tps_delta": self.prompt_eval_tps_delta,
            "total_latency_s_delta": self.total_latency_s_delta,
        }


@dataclass(frozen=True)
class RigGraphEvidenceRecord:
    record_id: str
    observed_at: str
    machine: Tuple[Tuple[str, Any], ...]
    model: str
    logical_model_id: Optional[str]
    runtime: str
    artifact_id: str
    artifact_format: str
    quantization: Optional[str]
    context_tokens: Optional[int]
    prediction: Tuple[Tuple[str, Optional[float]], ...]
    measurement: Tuple[Tuple[str, Optional[float]], ...]
    calibration: CalibrationRecord
    receipt_id: Optional[str]
    plan_id: Optional[str]
    schema_version: str = RIGGRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for label, value, identity in (
            ("RigGraph record id", self.record_id, True),
            ("RigGraph timestamp", self.observed_at, False),
            ("RigGraph model", self.model, True),
            ("RigGraph logical model", self.logical_model_id, True),
            ("RigGraph runtime", self.runtime, False),
            ("RigGraph artifact id", self.artifact_id, True),
            ("RigGraph artifact format", self.artifact_format, False),
            ("RigGraph quantization", self.quantization, False),
            ("RigGraph receipt id", self.receipt_id, True),
            ("RigGraph plan id", self.plan_id, True),
        ):
            validate_public_text(value, label, identity=identity)
        for _, value in self.machine:
            if isinstance(value, str):
                validate_public_text(value, "RigGraph machine field")
        if self.context_tokens is not None and self.context_tokens <= 0:
            raise ValueError("RigGraph context must be positive")

    @classmethod
    def build(
        cls,
        *,
        observed_at: str,
        machine: Mapping[str, Any],
        model: str,
        logical_model_id: Optional[str],
        runtime: str,
        artifact_id: str,
        artifact_format: str,
        quantization: Optional[str],
        context_tokens: Optional[int],
        prediction: Mapping[str, Any],
        measurement: Mapping[str, Any],
        receipt_id: Optional[str],
        plan_id: Optional[str],
    ) -> "RigGraphEvidenceRecord":
        prediction_values = tuple(
            (key, _metric(prediction.get(key)))
            for key in ("generation_tps", "prompt_eval_tps", "total_latency_s")
        )
        measurement_values = tuple(
            (key, _metric(measurement.get(key)))
            for key in ("generation_tps", "prompt_eval_tps", "total_latency_s")
        )
        predicted = dict(prediction_values)
        measured = dict(measurement_values)
        calibration = CalibrationRecord(
            calibration_delta(predicted["generation_tps"], measured["generation_tps"]),
            calibration_delta(predicted["prompt_eval_tps"], measured["prompt_eval_tps"]),
            calibration_delta(predicted["total_latency_s"], measured["total_latency_s"]),
        )
        machine_values = tuple(
            (key, machine.get(key)) for key in ("os", "arch", "cpu", "ram_gib")
        )
        body = {
            "observed_at": observed_at,
            "machine": dict(machine_values),
            "model": model,
            "logical_model_id": logical_model_id,
            "runtime": runtime,
            "artifact_id": artifact_id,
            "artifact_format": artifact_format,
            "quantization": quantization,
            "context_tokens": context_tokens,
            "prediction": dict(prediction_values),
            "measurement": dict(measurement_values),
            "calibration": calibration.to_dict(),
            "receipt_id": receipt_id,
            "plan_id": plan_id,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        record_id = "rig-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return cls(
            record_id=record_id,
            observed_at=observed_at,
            machine=machine_values,
            model=model,
            logical_model_id=logical_model_id,
            runtime=runtime,
            artifact_id=artifact_id,
            artifact_format=artifact_format,
            quantization=quantization,
            context_tokens=context_tokens,
            prediction=prediction_values,
            measurement=measurement_values,
            calibration=calibration,
            receipt_id=receipt_id,
            plan_id=plan_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "observed_at": self.observed_at,
            "machine": dict(self.machine),
            "model": self.model,
            "logical_model_id": self.logical_model_id,
            "runtime": self.runtime,
            "artifact_id": self.artifact_id,
            "artifact_format": self.artifact_format,
            "quantization": self.quantization,
            "context_tokens": self.context_tokens,
            "prediction": dict(self.prediction),
            "measurement": dict(self.measurement),
            "calibration": self.calibration.to_dict(),
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
        }


def evidence_dir() -> Path:
    return acquisition_registry_file().parent / "riggraph"


def save_evidence_record(record: RigGraphEvidenceRecord) -> None:
    directory = evidence_dir()
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
    path = directory / f"{record.record_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    latest = directory / "latest.json"
    latest_tmp = directory / "latest.tmp"
    latest_tmp.write_text(text, encoding="utf-8")
    latest_tmp.replace(latest)


def load_evidence_records() -> Tuple[Dict[str, Any], ...]:
    directory = evidence_dir()
    if not directory.exists():
        return ()
    records = []
    for path in sorted(directory.glob("rig-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") == RIGGRAPH_SCHEMA_VERSION:
            records.append(payload)
    return tuple(records)
