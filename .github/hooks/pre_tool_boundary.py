#!/usr/bin/env python3
"""PreToolUse hook that enforces project-root boundary for autonomous agents."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from typing import Any

ROOT = os.path.realpath("/home/david/Documents/trade")
WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
META_KEYS = {
    "transcript_path",
    "transcriptpath",
    "session_path",
    "conversation_path",
    "workspace_root",
}


def emit(decision: str, reason: str | None = None) -> int:
    payload: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
        }
    }
    if reason:
        payload["hookSpecificOutput"]["permissionDecisionReason"] = reason
        payload["systemMessage"] = reason

    sys.stdout.write(json.dumps(payload))
    return 0


def is_inside_root(path: str) -> bool:
    real = os.path.realpath(path)
    return real == ROOT or real.startswith(ROOT + os.sep)


def has_parent_traversal(value: str) -> bool:
    return (
        value == ".."
        or value.startswith("../")
        or value.endswith("/..")
        or "/../" in value
        or value.startswith("..\\")
        or value.endswith("\\..")
        or "\\..\\" in value
    )


def check_path_value(value: str) -> str | None:
    s = value.strip()
    if not s:
        return None
    if URL_SCHEME.match(s):
        return None
    if has_parent_traversal(s):
        return f"Parent traversal is not allowed: {s}"
    if WINDOWS_PATH.match(s):
        return f"External absolute path is not allowed: {s}"
    if s.startswith("/") and not is_inside_root(s):
        return f"Absolute path outside project root: {s}"
    return None


def check_command(cmd: str) -> str | None:
    s = cmd.strip()
    if not s:
        return None
    if has_parent_traversal(s):
        return "Parent traversal is not allowed in command input"

    try:
        tokens = shlex.split(s)
    except ValueError:
        tokens = s.split()

    for token in tokens:
        t = token.strip()
        if not t or URL_SCHEME.match(t):
            continue
        if has_parent_traversal(t):
            return f"Parent traversal is not allowed in command token: {t}"
        if WINDOWS_PATH.match(t):
            return f"External absolute path is not allowed in command token: {t}"
        if t.startswith("/") and not is_inside_root(t):
            return f"Absolute path outside project root in command token: {t}"
    return None


def inspect(node: Any, field_hint: str = "") -> str | None:
    if isinstance(node, dict):
        for key, value in node.items():
            key_lower = key.lower()

            if key_lower in META_KEYS:
                continue

            if isinstance(value, str):
                if key_lower in {"command", "shellcommand", "script"}:
                    problem = check_command(value)
                elif any(
                    tag in key_lower
                    for tag in ("path", "file", "dir", "cwd", "workspace", "workingdirectory", "resource")
                ):
                    problem = check_path_value(value)
                else:
                    problem = None

                if problem:
                    return f"{problem} (field: {key})"

            elif isinstance(value, list) and key_lower == "args":
                for arg in value:
                    if isinstance(arg, str):
                        problem = check_path_value(arg)
                        if problem:
                            return f"{problem} (field: {key})"

            nested = inspect(value, key)
            if nested:
                return nested

    elif isinstance(node, list):
        for item in node:
            nested = inspect(item, field_hint)
            if nested:
                return nested

    return None


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return emit("allow")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return emit("ask", "Unable to parse hook payload; user confirmation required")

    tool_payload: Any
    if isinstance(data, dict):
        tool_payload = (
            data.get("tool_input")
            or data.get("toolInput")
            or data.get("input")
            or (data.get("toolUse") or {}).get("input")
            or {}
        )
    else:
        tool_payload = data

    if isinstance(tool_payload, str):
        violation = check_command(tool_payload)
    else:
        violation = inspect(tool_payload)

    if violation:
        return emit("deny", violation)

    return emit("allow")


if __name__ == "__main__":
    raise SystemExit(main())
