"""Explicit, provenance-preserving artifact acquisition for Autopilot apply.

The module has no import-time dependency on ``huggingface_hub``.  It is loaded
only after the user approves an apply plan. Public records contain exact Hub
identity and revision but never the private local download path or credentials.
"""

from __future__ import annotations

import copy
import datetime as dt
import importlib
import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .privacy import validate_public_text


ACQUISITION_SCHEMA_VERSION = "0.9"
_REPO_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class AcquisitionError(RuntimeError):
    """Raised when an approved acquisition cannot be completed safely."""


@dataclass(frozen=True)
class HuggingFaceArtifactSource:
    repository_id: str
    artifact_path: str

    def __post_init__(self) -> None:
        validate_public_text(self.repository_id, "Hugging Face repository", identity=True)
        validate_public_text(self.artifact_path, "Hugging Face artifact", identity=True)
        parts = self.repository_id.split("/")
        if (
            len(parts) != 2
            or any(part in {"", ".", ".."} for part in parts)
            or any(_REPO_PART_RE.fullmatch(part) is None for part in parts)
        ):
            raise ValueError("Hugging Face repository id must be an exact owner/repository id")
        path_parts = Path(self.artifact_path).parts
        if (
            not self.artifact_path
            or self.artifact_path.startswith(("/", "\\"))
            or any(part in {"..", "."} for part in path_parts)
        ):
            raise ValueError("Hugging Face artifact path must be repository-relative")

    @property
    def is_snapshot(self) -> bool:
        return self.artifact_path in {"mlx", "safetensors"}

    @property
    def is_gguf_file(self) -> bool:
        return self.artifact_path.lower().endswith(".gguf")


@dataclass(frozen=True)
class AcquisitionRecord:
    artifact_id: str
    repository_id: str
    revision: str
    runtime: str
    acquisition_kind: str
    status: str
    completed_at: str

    def __post_init__(self) -> None:
        for label, value in (
            ("artifact id", self.artifact_id),
            ("repository id", self.repository_id),
            ("revision", self.revision),
            ("runtime", self.runtime),
            ("acquisition kind", self.acquisition_kind),
            ("status", self.status),
            ("completed at", self.completed_at),
        ):
            validate_public_text(value, f"acquisition {label}", identity=label in {"artifact id", "repository id"})
        if self.status not in {"completed", "failed"}:
            raise ValueError("acquisition status must be completed or failed")

    def to_dict(self) -> Dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "runtime": self.runtime,
            "acquisition_kind": self.acquisition_kind,
            "status": self.status,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AcquisitionRecord":
        return cls(
            artifact_id=str(payload.get("artifact_id") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            revision=str(payload.get("revision") or ""),
            runtime=str(payload.get("runtime") or ""),
            acquisition_kind=str(payload.get("acquisition_kind") or ""),
            status=str(payload.get("status") or ""),
            completed_at=str(payload.get("completed_at") or ""),
        )


class AcquiredArtifact:
    """Private execution locator paired with a safe public acquisition record."""

    __slots__ = ("record", "_locator")

    def __init__(self, record: AcquisitionRecord, locator: str) -> None:
        self.record = record
        self._locator = str(locator)

    def __repr__(self) -> str:
        return f"AcquiredArtifact(record={self.record!r})"

    @property
    def locator(self) -> str:
        return self._locator

    @staticmethod
    def _serialization_error() -> TypeError:
        return TypeError("private acquired artifact locators cannot be serialized or copied")

    def __reduce__(self) -> Any:
        raise self._serialization_error()

    def __reduce_ex__(self, protocol: int) -> Any:
        raise self._serialization_error()

    def __getstate__(self) -> Any:
        raise self._serialization_error()

    def __copy__(self) -> Any:
        raise self._serialization_error()

    def __deepcopy__(self, memo: Any) -> Any:
        raise self._serialization_error()


def parse_hf_artifact_id(artifact_id: str) -> HuggingFaceArtifactSource:
    prefix = "hf://"
    if not isinstance(artifact_id, str) or not artifact_id.startswith(prefix):
        raise ValueError("artifact is not an exact Hugging Face source")
    remainder = artifact_id[len(prefix) :]
    parts = remainder.split("/", 2)
    if len(parts) != 3:
        raise ValueError("Hugging Face artifact id must include owner/repository/artifact")
    return HuggingFaceArtifactSource(
        repository_id=f"{parts[0]}/{parts[1]}",
        artifact_path=parts[2],
    )


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _state_root() -> Path:
    if platform.system() == "Windows" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "llmrig"


def acquisition_registry_file() -> Path:
    return _state_root() / "acquisitions.json"


def _load_registry() -> Tuple[AcquisitionRecord, ...]:
    path = acquisition_registry_file()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ()
    if not isinstance(payload, dict) or payload.get("schema_version") != ACQUISITION_SCHEMA_VERSION:
        return ()
    rows = payload.get("acquisitions")
    if not isinstance(rows, list):
        return ()
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            output.append(AcquisitionRecord.from_dict(row))
        except (TypeError, ValueError):
            continue
    return tuple(output)


def completed_acquisitions(
    *, runtime: Optional[str] = None, repository_id: Optional[str] = None
) -> Tuple[AcquisitionRecord, ...]:
    rows = tuple(item for item in _load_registry() if item.status == "completed")
    if runtime is not None:
        rows = tuple(item for item in rows if item.runtime == runtime)
    if repository_id is not None:
        rows = tuple(item for item in rows if item.repository_id == repository_id)
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.runtime,
                item.repository_id,
                item.revision,
                item.completed_at,
            ),
        )
    )


def record_acquisition(record: AcquisitionRecord) -> None:
    """Persist only path-free acquisition provenance after explicit apply."""
    path = acquisition_registry_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = list(_load_registry())
    identity = (record.runtime, record.artifact_id, record.revision)
    existing = [
        item
        for item in existing
        if (item.runtime, item.artifact_id, item.revision) != identity
    ]
    existing.append(record)
    existing.sort(
        key=lambda item: (item.runtime, item.repository_id, item.revision, item.completed_at)
    )
    payload = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "acquisitions": [item.to_dict() for item in existing],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _hub_module() -> Any:
    try:
        return importlib.import_module("huggingface_hub")
    except (ImportError, ModuleNotFoundError) as exc:
        raise AcquisitionError(
            "Hugging Face acquisition requires the optional 'huggingface_hub' package"
        ) from exc


def _exact_revision(hub: Any, repository_id: str) -> str:
    try:
        info = hub.HfApi().model_info(repository_id, files_metadata=True)
    except Exception as exc:
        raise AcquisitionError("could not resolve the exact Hugging Face repository revision") from exc
    revision = str(getattr(info, "sha", "") or "").strip()
    if len(revision) < 20:
        raise AcquisitionError("Hugging Face did not return a stable repository commit revision")
    validate_public_text(revision, "Hugging Face revision")
    return revision


def _omlx_destination(repository_id: str) -> Path:
    configured = os.environ.get("OMLX_MODEL_DIR")
    root = Path(configured).expanduser() if configured else Path.home() / ".omlx" / "models"
    owner, name = repository_id.split("/", 1)
    return root / owner / name


def acquire_huggingface_artifact(
    artifact_id: str,
    runtime: str,
) -> AcquiredArtifact:
    """Acquire one exact Hub artifact after the caller has enforced approval."""
    source = parse_hf_artifact_id(artifact_id)
    hub = _hub_module()
    revision = _exact_revision(hub, source.repository_id)

    try:
        if source.is_gguf_file:
            locator = hub.hf_hub_download(
                repo_id=source.repository_id,
                filename=source.artifact_path,
                revision=revision,
            )
            acquisition_kind = "huggingface-file"
        elif source.is_snapshot:
            kwargs: Dict[str, Any] = {
                "repo_id": source.repository_id,
                "revision": revision,
            }
            if runtime == "omlx":
                kwargs["local_dir"] = str(_omlx_destination(source.repository_id))
            locator = hub.snapshot_download(**kwargs)
            acquisition_kind = "huggingface-snapshot"
        else:
            raise AcquisitionError(
                "the selected Hugging Face artifact shape is not supported for automatic acquisition"
            )
    except AcquisitionError:
        raise
    except Exception as exc:
        raise AcquisitionError("Hugging Face artifact acquisition failed") from exc

    if not locator:
        raise AcquisitionError("Hugging Face acquisition did not return a local artifact")

    record = AcquisitionRecord(
        artifact_id=artifact_id,
        repository_id=source.repository_id,
        revision=revision,
        runtime=runtime,
        acquisition_kind=acquisition_kind,
        status="completed",
        completed_at=_now_iso(),
    )
    record_acquisition(record)
    return AcquiredArtifact(record, str(locator))
