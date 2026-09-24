"""Small graph-shaped evidence model for future LLMRig calibration and sharing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple


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
    """Deterministic in-memory graph; persistence is intentionally separate."""

    schema_version: str = "0.1"
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
        cls, nodes: Iterable[RigNode], edges: Iterable[RigEdge], *, schema_version: str = "0.1"
    ) -> "RigGraph":
        graph = cls(schema_version=schema_version)
        for node in nodes:
            graph.add_node(node)
        for edge in edges:
            graph.add_edge(edge)
        return graph
