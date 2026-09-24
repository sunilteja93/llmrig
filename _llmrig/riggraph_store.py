"""Privacy-safe local persistence for v0.9 RigGraph evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .acquisition import acquisition_registry_file
from .autopilot_plan import AutopilotExecutionPlan
from .privacy import validate_public_text
from .riggraph import RigEdge, RigGraph, RigNode


RIGGRAPH_STORE_SCHEMA_VERSION = "0.9"


def riggraph_file() -> Path:
    return acquisition_registry_file().parent / "riggraph.json"


def _canonical_id(kind: str, attributes: Mapping[str, Any]) -> str:
    safe: Dict[str, Any] = {}
    for key, value in sorted(attributes.items()):
        validate_public_text(str(key), "RigGraph attribute key")
        if isinstance(value, str):
            validate_public_text(value, "RigGraph attribute value")
        elif value is not None and not isinstance(value, (bool, int, float)):
            value = str(value)
            validate_public_text(value, "RigGraph attribute value")
        safe[str(key)] = value
    payload = json.dumps(
        {"kind": kind, "attributes": safe},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{kind}:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _node(kind: str, **attributes: Any) -> RigNode:
    return RigNode.from_mapping(_canonical_id(kind, attributes), kind, attributes)


def _edge(
    source: RigNode,
    relation: str,
    target: RigNode,
    evidence_level: str,
    **attributes: Any,
) -> RigEdge:
    validate_public_text(relation, "RigGraph relation")
    validate_public_text(evidence_level, "RigGraph evidence level")
    for key, value in attributes.items():
        validate_public_text(str(key), "RigGraph edge key")
        if isinstance(value, str):
            validate_public_text(value, "RigGraph edge value")
    return RigEdge.from_mapping(
        source.node_id,
        relation,
        target.node_id,
        evidence_level,
        attributes,
    )


def _graph_from_payload(payload: Mapping[str, Any]) -> RigGraph:
    if payload.get("schema_version") != RIGGRAPH_STORE_SCHEMA_VERSION:
        return RigGraph(schema_version=RIGGRAPH_STORE_SCHEMA_VERSION)
    nodes = []
    for row in payload.get("nodes", ()):
        if not isinstance(row, Mapping):
            continue
        attributes = row.get("attributes")
        if not isinstance(attributes, Mapping):
            attributes = {}
        try:
            node = RigNode.from_mapping(
                str(row.get("id") or ""),
                str(row.get("kind") or ""),
                dict(attributes),
            )
            validate_public_text(node.node_id, "RigGraph node id")
            validate_public_text(node.kind, "RigGraph node kind")
            nodes.append(node)
        except (TypeError, ValueError):
            continue

    known = {node.node_id for node in nodes}
    edges = []
    for row in payload.get("edges", ()):
        if not isinstance(row, Mapping):
            continue
        source = str(row.get("source") or "")
        target = str(row.get("target") or "")
        if source not in known or target not in known:
            continue
        attributes = row.get("attributes")
        if not isinstance(attributes, Mapping):
            attributes = {}
        try:
            edge = RigEdge.from_mapping(
                source,
                str(row.get("relation") or ""),
                target,
                str(row.get("evidence_level") or ""),
                dict(attributes),
            )
            edges.append(edge)
        except (TypeError, ValueError):
            continue
    try:
        return RigGraph.from_records(
            nodes,
            edges,
            schema_version=RIGGRAPH_STORE_SCHEMA_VERSION,
        )
    except ValueError:
        return RigGraph(schema_version=RIGGRAPH_STORE_SCHEMA_VERSION)


def load_riggraph() -> RigGraph:
    try:
        payload = json.loads(riggraph_file().read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return RigGraph(schema_version=RIGGRAPH_STORE_SCHEMA_VERSION)
    return _graph_from_payload(payload if isinstance(payload, Mapping) else {})


def save_riggraph(graph: RigGraph) -> None:
    path = riggraph_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(graph.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _merge(graph: RigGraph, nodes: Iterable[RigNode], edges: Iterable[RigEdge]) -> None:
    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge)


def record_plan(plan: AutopilotExecutionPlan) -> str:
    """Persist one path-free planning snapshot and return its graph fingerprint."""
    graph = load_riggraph()
    machine_attrs = dict(plan.machine)
    machine = _node("machine", **machine_attrs)
    model = _node("model", model_id=plan.logical_model_id or plan.model)
    plan_node = _node(
        "plan",
        plan_id=plan.plan_id,
        status=plan.recommendation_status,
        blocked=plan.blocked,
    )
    nodes = [machine, model, plan_node]
    edges = [
        _edge(plan_node, "plans_model", model, "observed"),
        _edge(plan_node, "on_machine", machine, "observed"),
    ]

    if plan.selected_candidate_id:
        candidate = next(
            item
            for item in plan.candidates
            if item.candidate_id == plan.selected_candidate_id
        )
        artifact = _node(
            "artifact",
            artifact_id=candidate.artifact_id,
            artifact_format=candidate.artifact_format,
            quantization=candidate.quantization,
        )
        runtime = _node("runtime", runtime=candidate.runtime)
        context = _node("context", tokens=candidate.context_tokens)
        nodes.extend((artifact, runtime, context))
        edges.extend(
            (
                _edge(artifact, "represents_model", model, "discovered"),
                _edge(plan_node, "selects_artifact", artifact, "planned"),
                _edge(plan_node, "selects_runtime", runtime, "planned"),
                _edge(plan_node, "uses_context", context, "planned"),
            )
        )

    _merge(graph, nodes, edges)
    save_riggraph(graph)
    return graph.fingerprint()


def record_receipt(receipt: Mapping[str, Any]) -> str:
    """Persist one public apply/verify receipt into the local evidence graph."""
    graph = load_riggraph()
    candidate = receipt.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("RigGraph receipt requires public candidate metadata")

    machine_attrs = receipt.get("machine")
    if not isinstance(machine_attrs, Mapping):
        machine_attrs = {}
    machine = _node("machine", **dict(machine_attrs))
    model = _node("model", model_id=receipt.get("logical_model_id") or receipt.get("model"))
    artifact = _node(
        "artifact",
        artifact_id=candidate.get("artifact_id"),
        artifact_format=candidate.get("artifact_format"),
        quantization=candidate.get("quantization"),
    )
    runtime = _node(
        "runtime",
        runtime=candidate.get("runtime"),
        runtime_version=candidate.get("runtime_version"),
    )
    context = _node("context", tokens=candidate.get("context_tokens"))
    receipt_node = _node(
        "receipt",
        receipt_id=receipt.get("receipt_id"),
        plan_id=receipt.get("plan_id"),
        status=receipt.get("status"),
    )
    nodes = [machine, model, artifact, runtime, context, receipt_node]
    edges = [
        _edge(artifact, "represents_model", model, "discovered"),
        _edge(receipt_node, "applies_plan", model, "observed", plan_id=receipt.get("plan_id")),
        _edge(receipt_node, "used_artifact", artifact, "observed"),
        _edge(receipt_node, "used_runtime", runtime, "observed"),
        _edge(receipt_node, "on_machine", machine, "observed"),
        _edge(receipt_node, "used_context", context, "observed"),
    ]

    verification = receipt.get("verification")
    if isinstance(verification, Mapping):
        measurement = _node(
            "measurement",
            method_version=verification.get("method_version"),
            generation_tps=verification.get("generation_tps"),
            prompt_eval_tps=verification.get("prompt_eval_tps"),
            total_latency_s=verification.get("total_latency_s"),
            measured_runs=verification.get("measured_runs"),
        )
        calibration = _node(
            "calibration",
            status="prediction_unavailable",
            detail="no comparable numeric performance prediction was recorded before measurement",
        )
        nodes.extend((measurement, calibration))
        edges.extend(
            (
                _edge(receipt_node, "produced_measurement", measurement, "measured"),
                _edge(measurement, "measured_runtime", runtime, "measured"),
                _edge(measurement, "measured_artifact", artifact, "measured"),
                _edge(measurement, "calibration", calibration, "measured"),
            )
        )

    _merge(graph, nodes, edges)
    save_riggraph(graph)
    return graph.fingerprint()


def graph_summary() -> Dict[str, Any]:
    graph = load_riggraph()
    counts: Dict[str, int] = {}
    for node in graph.nodes:
        counts[node.kind] = counts.get(node.kind, 0) + 1
    return {
        "schema_version": graph.schema_version,
        "fingerprint": graph.fingerprint(),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "node_kinds": dict(sorted(counts.items())),
    }
