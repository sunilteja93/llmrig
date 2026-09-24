"""Minimal stdlib-only oMLX OpenAI-compatible client for LLMRig v0.8.

The client is intentionally narrow. It can observe the server model inventory and
run an explicitly requested bounded text-completion measurement. It never installs
or starts oMLX, downloads weights, scans model directories, or exposes filesystem
paths.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


DEFAULT_OMLX_BASE_URL = os.environ.get(
    "OMLX_BASE_URL", "http://127.0.0.1:8000"
).rstrip("/")
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class OmlxApiError(RuntimeError):
    """Raised when an oMLX API response cannot be used safely."""


@dataclass(frozen=True)
class OmlxModelInfo:
    """Privacy-safe model metadata exposed by ``GET /v1/models``."""

    model_id: str
    max_model_len: Optional[int]
    owned_by: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_model_len": self.max_model_len,
            "owned_by": self.owned_by,
        }


@dataclass(frozen=True)
class OmlxMeasurement:
    """One non-streaming server-reported oMLX completion measurement."""

    model_id: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    generation_tps: Optional[float]
    prompt_tps: Optional[float]
    total_time_s: Optional[float]
    prompt_eval_duration_s: Optional[float]
    generation_duration_s: Optional[float]
    time_to_first_token_s: Optional[float]

    @property
    def race_latency_s(self) -> Optional[float]:
        """Return latency aligned with native race-v2 execution semantics.

        Native LLMRig adapters compose latency from prompt-evaluation plus
        generation duration. Prefer the same composition here and fall back to
        oMLX's server-reported total time only when those component durations are
        unavailable.
        """
        if (
            self.prompt_eval_duration_s is not None
            and self.generation_duration_s is not None
        ):
            return self.prompt_eval_duration_s + self.generation_duration_s
        return self.total_time_s

    @property
    def race_ready(self) -> bool:
        return all(
            value is not None and value > 0
            for value in (
                self.generation_tps,
                self.prompt_tps,
                self.race_latency_s,
            )
        )

    def race_sample(self) -> Dict[str, Any]:
        if not self.race_ready:
            raise OmlxApiError(
                "oMLX did not return the complete server-side metrics required for race-v2"
            )
        return {
            "generation_tps": self.generation_tps,
            "prompt_tps": self.prompt_tps,
            "wall_seconds": self.race_latency_s,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.completion_tokens,
            "measurement_source": "omlx-server-usage",
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "generation_tps": self.generation_tps,
            "prompt_tps": self.prompt_tps,
            "total_time_s": self.total_time_s,
            "prompt_eval_duration_s": self.prompt_eval_duration_s,
            "generation_duration_s": self.generation_duration_s,
            "time_to_first_token_s": self.time_to_first_token_s,
            "race_latency_s": self.race_latency_s,
            "race_ready": self.race_ready,
        }


def _positive_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _nonnegative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _positive_int(value: Any) -> Optional[int]:
    parsed = _nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "llmrig-omlx-adapter",
    }
    api_key = os.environ.get("OMLX_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=_headers(),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                raise OmlxApiError(
                    "oMLX returned HTTP status %s" % response.status
                )
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise OmlxApiError("oMLX returned HTTP status %s" % error.code) from None
    except (urllib.error.URLError, OSError, TimeoutError):
        raise OmlxApiError("the local oMLX API is unavailable") from None

    if len(body) > _MAX_RESPONSE_BYTES:
        raise OmlxApiError("oMLX returned an unexpectedly large JSON response")
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise OmlxApiError("oMLX returned invalid JSON") from None
    if not isinstance(decoded, dict):
        raise OmlxApiError("oMLX returned an unexpected JSON shape")
    return decoded


def list_models(
    base_url: str = DEFAULT_OMLX_BASE_URL,
    timeout: float = 3.0,
) -> Tuple[OmlxModelInfo, ...]:
    """Observe API-visible oMLX model IDs without scanning local files."""
    payload = _request_json(base_url, "/v1/models", timeout=timeout)
    data = payload.get("data")
    if not isinstance(data, list):
        raise OmlxApiError("oMLX /v1/models response is missing a model list")

    models = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        if model_id in seen:
            continue
        seen.add(model_id)
        owned_by = item.get("owned_by")
        models.append(
            OmlxModelInfo(
                model_id=model_id,
                max_model_len=_positive_int(item.get("max_model_len")),
                owned_by=owned_by if isinstance(owned_by, str) else None,
            )
        )
    return tuple(sorted(models, key=lambda item: item.model_id))


def measure_completion(
    model_id: str,
    prompt: str,
    max_tokens: int,
    *,
    seed: int = 42,
    base_url: str = DEFAULT_OMLX_BASE_URL,
    timeout: float = 120.0,
) -> OmlxMeasurement:
    """Run one explicit bounded non-streaming completion and parse server metrics."""
    if not model_id.strip():
        raise ValueError("oMLX model id must not be empty")
    if not prompt:
        raise ValueError("oMLX benchmark prompt must not be empty")
    if max_tokens < 1 or max_tokens > 512:
        raise ValueError("oMLX benchmark max_tokens must be between 1 and 512")

    payload = _request_json(
        base_url,
        "/v1/completions",
        method="POST",
        payload={
            "model": model_id,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        },
        timeout=timeout,
    )
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise OmlxApiError("oMLX completion response is missing usage metrics")

    response_model = payload.get("model")
    if isinstance(response_model, str) and response_model.strip():
        observed_model_id = response_model.strip()
    else:
        observed_model_id = model_id.strip()

    return OmlxMeasurement(
        model_id=observed_model_id,
        prompt_tokens=_nonnegative_int(usage.get("prompt_tokens")),
        completion_tokens=_nonnegative_int(usage.get("completion_tokens")),
        generation_tps=_positive_float(usage.get("generation_tokens_per_second")),
        prompt_tps=_positive_float(usage.get("prompt_tokens_per_second")),
        total_time_s=_positive_float(usage.get("total_time")),
        prompt_eval_duration_s=_positive_float(usage.get("prompt_eval_duration")),
        generation_duration_s=_positive_float(usage.get("generation_duration")),
        time_to_first_token_s=_positive_float(usage.get("time_to_first_token")),
    )
