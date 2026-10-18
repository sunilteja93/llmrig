"""Privacy-safe structural diagnostics for bounded OpenCode JSONL events.

Never retain raw event values, generated text, file paths, timestamps,
tool arguments or arbitrary event/tool names in the returned report.
A matched expression is provisional answer evidence, not proof of a
finished agent turn or successfully executed code.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Pattern, Union

from .rescue_agent import MAX_EVENTS

KINDS = ("step_start", "step_finish", "tool_use", "text", "error", "other")
TOOLS = ("read", "glob", "grep", "list", "other")


def summarize(raw: bytes, fix_pattern: Union[str, Pattern[str]]) -> Dict[str, Any]:
    """Summarize event ordering only, without echoing user-controlled values."""
    expression = re.compile(fix_pattern) if isinstance(fix_pattern, str) else fix_pattern
    counts = {key: 0 for key in KINDS}
    tool_counts = {key: 0 for key in TOOLS}
    post_fix_counts = {key: 0 for key in KINDS}
    text_buffer = ""
    events_seen = 0
    first_fix_at: Optional[int] = None
    last_kind = "none"
    malformed = False

    for line in raw.splitlines():
        if not line.strip():
            continue
        events_seen += 1
        if events_seen > MAX_EVENTS:
            malformed = True
            break
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            malformed = True
            break
        if not isinstance(event, dict):
            malformed = True
            break

        kind_value = event.get("type")
        kind = kind_value if isinstance(kind_value, str) and kind_value in KINDS else "other"
        counts[kind] += 1
        last_kind = kind
        part = event.get("part")
        if kind == "tool_use":
            tool = part.get("tool") if isinstance(part, dict) else None
            safe_tool = tool.lower() if isinstance(tool, str) else "other"
            tool_counts[safe_tool if safe_tool in TOOLS else "other"] += 1
        if kind == "text" and isinstance(part, dict) and isinstance(part.get("text"), str):
            # Match only in generated answer text (not tool output). Discard
            # this buffer when the function returns; never put it in the report.
            text_buffer = (text_buffer + "\n" + part["text"][:16000])[-131072:]
            if first_fix_at is None and expression.search(text_buffer):
                first_fix_at = events_seen
        elif first_fix_at is not None:
            post_fix_counts[kind] += 1
        # Count text events after fix too, without double-counting the match.
        if kind == "text" and first_fix_at is not None and events_seen > first_fix_at:
            post_fix_counts[kind] += 1

    return {
        "events_seen": min(events_seen, MAX_EVENTS),
        "event_type_counts": counts,
        "tool_category_counts": tool_counts,
        "first_fix_evidence_event_index": first_fix_at,
        "events_after_first_fix_evidence": (
            max(0, min(events_seen, MAX_EVENTS) - first_fix_at)
            if first_fix_at is not None else None
        ),
        "post_fix_event_type_counts": post_fix_counts,
        "last_event_type_category": last_kind,
        "stream_truncated_or_malformed": malformed,
        "interpretation": (
            "provisional_fix_evidence_not_final_agent_completion"
            if first_fix_at is not None else "no_fix_evidence_in_text_events"
        ),
    }
