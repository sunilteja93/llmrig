"""Read-only local artifact inventory for solve planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

from .planning import ConfidenceLike, EvidenceLike, _confidence_value, _evidence_payloads
from .privacy import (
    contains_private_path,
    privacy_safe_public_id,
    safe_optional_public_id,
    validate_public_text,
)


_SAFE_FINGERPRINT = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{6,128}$")


def _safe_fingerprint(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text if _SAFE_FINGERPRINT.fullmatch(text) else None


def _safe_messages(values: Tuple[str, ...]) -> Tuple[str, ...]:
    output = []
    for value in values:
        text = str(value).strip()
        validate_public_text(text, "inventory public metadata")
        output.append(text)
    return tuple(sorted(dict.fromkeys(output)))


def _safe_evidence(evidence: Tuple[EvidenceLike, ...]) -> Tuple[Dict[str, str], ...]:
    payloads = _evidence_payloads(evidence)
    if any(
        contains_private_path(payload[key])
        for payload in payloads
        for key in ("kind", "source", "detail")
    ):
        raise ValueError("inventory evidence contains private data")
    return payloads


@dataclass(frozen=True)
class InventoryRecord:
    """Serializable local observation with no execution locator."""

    runtime: str
    public_artifact_id: str
    artifact_fingerprint: Optional[str]
    logical_model_id: Optional[str]
    artifact_format: Optional[str]
    quantization: Optional[str]
    association_kind: str
    identity_attested: bool
    identity_confidence: ConfidenceLike
    identity_evidence: Tuple[EvidenceLike, ...]
    evidence: Tuple[EvidenceLike, ...]
    unknowns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.runtime.strip() or not self.association_kind.strip():
            raise ValueError("inventory runtime and association kind must not be empty")
        validate_public_text(self.runtime, "inventory runtime")
        validate_public_text(self.association_kind, "inventory association kind")
        validate_public_text(self.artifact_format, "inventory artifact format")
        validate_public_text(self.quantization, "inventory quantization")
        object.__setattr__(
            self,
            "public_artifact_id",
            privacy_safe_public_id(self.runtime, self.public_artifact_id),
        )
        object.__setattr__(
            self, "logical_model_id", safe_optional_public_id(self.logical_model_id)
        )
        object.__setattr__(
            self, "artifact_fingerprint", _safe_fingerprint(self.artifact_fingerprint)
        )
        object.__setattr__(self, "unknowns", _safe_messages(self.unknowns))
        _confidence_value(self.identity_confidence)
        _safe_evidence(self.identity_evidence)
        _safe_evidence(self.evidence)
        if self.identity_attested and self.logical_model_id is None:
            raise ValueError("attested inventory identity requires a logical model id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime": self.runtime,
            "public_artifact_id": self.public_artifact_id,
            "artifact_fingerprint": self.artifact_fingerprint,
            "logical_model_id": self.logical_model_id,
            "artifact_format": self.artifact_format,
            "quantization": self.quantization,
            "association_kind": self.association_kind,
            "identity_attested": self.identity_attested,
            "identity_confidence": _confidence_value(self.identity_confidence),
            "identity_evidence": list(_safe_evidence(self.identity_evidence)),
            "evidence": list(_safe_evidence(self.evidence)),
            "unknowns": list(self.unknowns),
        }


class InventoryTarget:
    """Private locator paired with a public inventory record."""

    __slots__ = ("record", "_locator")

    def __init__(self, record: InventoryRecord, locator: str) -> None:
        self.record = record
        self._locator = locator

    def __repr__(self) -> str:
        return f"InventoryTarget(record={self.record!r})"

    @staticmethod
    def _serialization_error() -> TypeError:
        return TypeError("private inventory targets cannot be serialized or copied")

    def __reduce__(self) -> Any:
        raise self._serialization_error()

    def __reduce_ex__(self, protocol: int) -> Any:
        raise self._serialization_error()

    def __getstate__(self) -> Any:
        raise self._serialization_error()


def _execution_locator(target: InventoryTarget) -> str:
    """Return a locator only for the internal inventory-to-execution bridge."""
    if not isinstance(target, InventoryTarget):
        raise TypeError("execution locator requires an inventory target")
    return target._locator


class InventoryProvider(Protocol):
    """Observation-only source of artifacts already present on this machine."""

    name: str

    def observe(self) -> Tuple[InventoryTarget, ...]: ...


EvidenceFactory = Callable[[str, str, str], EvidenceLike]


class OllamaInventoryProvider:
    """Read the existing Ollama inventory without starting or mutating Ollama."""

    name = "ollama"

    def __init__(
        self,
        installed_models: Callable[[], Sequence[Dict[str, str]]],
        curated_specs: Callable[[], Sequence[Any]],
        model_name_matches: Callable[[str, str], bool],
        evidence_factory: EvidenceFactory,
        known_confidence: ConfidenceLike,
        unknown_confidence: ConfidenceLike,
    ) -> None:
        self._installed_models = installed_models
        self._curated_specs = curated_specs
        self._model_name_matches = model_name_matches
        self._evidence_factory = evidence_factory
        self._known_confidence = known_confidence
        self._unknown_confidence = unknown_confidence

    def _curated_match(self, name: str) -> Optional[Any]:
        for spec in self._curated_specs():
            if any(self._model_name_matches(identifier, name) for identifier in spec.ids):
                return spec
        return None

    def observe(self) -> Tuple[InventoryTarget, ...]:
        try:
            rows = tuple(self._installed_models())
        except Exception:
            return ()
        selected: Dict[Tuple[str, str, str], InventoryTarget] = {}
        for row in sorted(rows, key=lambda item: str(item.get("name") or "")):
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            fingerprint = _safe_fingerprint(row.get("id"))
            spec = self._curated_match(name)
            observed = self._evidence_factory(
                "verified-local-inventory",
                "Ollama installed-model listing",
                "the artifact name was returned by the local Ollama inventory",
            )
            if spec is not None:
                artifact = spec.artifact
                identity = self._evidence_factory(
                    "curated-identity-match",
                    "LLMRig curated model IDs and aliases",
                    "the installed Ollama name exactly matches a trusted curated ID or alias",
                )
                record = InventoryRecord(
                    runtime=self.name,
                    public_artifact_id=name,
                    artifact_fingerprint=fingerprint,
                    logical_model_id=spec.model.model_id,
                    artifact_format=artifact.format,
                    quantization=artifact.quantization,
                    association_kind="curated_name_match",
                    identity_attested=False,
                    identity_confidence=self._known_confidence,
                    identity_evidence=(identity,),
                    evidence=(observed,),
                    unknowns=(
                        "installed artifact content identity is not independently attested",
                    )
                    + (
                        ("artifact content digest is unknown",) if fingerprint is None else ()
                    ),
                )
            else:
                record = InventoryRecord(
                    runtime=self.name,
                    public_artifact_id=name,
                    artifact_fingerprint=fingerprint,
                    logical_model_id=None,
                    artifact_format=None,
                    quantization=None,
                    association_kind="unresolved",
                    identity_attested=False,
                    identity_confidence=self._unknown_confidence,
                    identity_evidence=(observed,),
                    evidence=(observed,),
                    unknowns=(
                        "logical model identity is unknown",
                        "artifact format is unknown",
                        "quantization is unknown",
                    )
                    + (("artifact content digest is unknown",) if fingerprint is None else ()),
                )
            key = (
                record.logical_model_id or record.public_artifact_id,
                record.artifact_fingerprint or record.public_artifact_id,
                record.quantization or "unknown",
            )
            selected.setdefault(key, InventoryTarget(record, name))
        return tuple(
            sorted(
                selected.values(),
                key=lambda item: (
                    item.record.runtime,
                    item.record.public_artifact_id,
                ),
            )
        )


class ExplicitNativeInventoryProvider:
    """Observe only explicit GGUF/MLX locators supplied by the user."""

    name = "explicit-native"

    def __init__(
        self,
        requested_logical_model_id: str,
        values: Sequence[str],
        parse_values: Callable[[Sequence[str]], Tuple[Tuple[str, str], ...]],
        validate_locator: Callable[[str, str], str],
        public_id_for_runtime: Callable[[str], str],
        evidence_factory: EvidenceFactory,
        unknown_confidence: ConfidenceLike,
    ) -> None:
        self._requested_logical_model_id = requested_logical_model_id
        self._values = tuple(values)
        self._parse_values = parse_values
        self._validate_locator = validate_locator
        self._public_id_for_runtime = public_id_for_runtime
        self._evidence_factory = evidence_factory
        self._unknown_confidence = unknown_confidence

    def observe(self) -> Tuple[InventoryTarget, ...]:
        targets = []
        for runtime, supplied in self._parse_values(self._values):
            locator = self._validate_locator(runtime, supplied)
            association = self._evidence_factory(
                "user-supplied-local-association",
                "explicit local artifact input",
                "the user associated this local artifact with the requested logical model; its content identity was not independently attested",
            )
            structure = self._evidence_factory(
                "verified-local-artifact-structure",
                "LLMRig explicit local artifact validation",
                "the supplied artifact passed the runtime-specific local structural check",
            )
            record = InventoryRecord(
                runtime=runtime,
                public_artifact_id=self._public_id_for_runtime(runtime),
                artifact_fingerprint=None,
                logical_model_id=self._requested_logical_model_id,
                artifact_format="GGUF" if runtime == "llama.cpp" else "MLX",
                quantization=None,
                association_kind="user_supplied",
                identity_attested=False,
                identity_confidence=self._unknown_confidence,
                identity_evidence=(association,),
                evidence=(structure,),
                unknowns=(
                    "artifact content identity is not independently verified",
                    "artifact content digest is unknown",
                    "quantization is unknown",
                ),
            )
            targets.append(InventoryTarget(record, locator))
        return tuple(
            sorted(
                targets,
                key=lambda item: (
                    item.record.runtime,
                    item.record.public_artifact_id,
                ),
            )
        )
