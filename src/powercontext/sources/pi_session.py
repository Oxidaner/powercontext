# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Captured evidence from one selected branch of a Pi v3 session."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Never, cast

from pydantic import BaseModel, JsonValue

from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
    Source,
    SourceMaterialization,
    TextEvidence,
)

PI_EVENT_SOURCE_NAME = "pi-event"


class PiSessionInput(BaseModel):
    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identity(session_id: str, event_id: str) -> str:
    digest = hashlib.sha256(_json_text([session_id, event_id]).encode("utf-8")).hexdigest()
    return f"pi:{digest}"


def _reject_non_json_number(value: str) -> Never:
    raise ValueError(f"invalid JSON number {value}")  # noqa: TRY003


def _visible_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if block.get("type") == "text" and isinstance(text, str):
            parts.append(text)
        elif block.get("type") == "toolCall":
            name, call_id = block.get("name"), block.get("id")
            if isinstance(name, str) and isinstance(call_id, str):
                parts.append(f"[tool_call {name} {call_id}] {_json_text(block.get('arguments', {}))}")
    return "\n".join(parts)


def _event_body(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    event_type = record.get("type")
    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        return (event_type, summary) if isinstance(event_type, str) and isinstance(summary, str) else None
    if event_type == "custom_message":
        return "custom", _visible_text(record.get("content"))
    if event_type != "message":
        return None
    message = record.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("role"), str):
        return None
    role = message["role"]
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        body = f"$ {command}" if isinstance(command, str) else ""
        if isinstance(output, str) and output:
            body = f"{body}\n{output}" if body else output
    elif role == "toolResult":
        body = _visible_text(message.get("content"))
        tool_name, call_id = message.get("toolName"), message.get("toolCallId")
        if body and isinstance(tool_name, str) and isinstance(call_id, str):
            body = f"[tool_result {tool_name} {call_id}] {body}"
    else:
        body = _visible_text(message.get("content"))
    return role, body


def _load_nodes(content: bytes) -> tuple[str, dict[str, tuple[str | None, str, dict[str, JsonValue]]]]:
    try:
        lines = content.decode("utf-8-sig").splitlines()
        records = [json.loads(line, parse_constant=_reject_non_json_number) for line in lines]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Pi session JSONL") from error  # noqa: TRY003
    if not records or not isinstance(records[0], dict):
        raise ValueError("missing Pi session header")  # noqa: TRY003
    header = records[0]
    session_id = header.get("id")
    if (
        header.get("type") != "session"
        or header.get("version") != 3
        or not isinstance(session_id, str)
        or not session_id
    ):
        raise ValueError("invalid Pi v3 session header")  # noqa: TRY003

    nodes: dict[str, tuple[str | None, str, dict[str, JsonValue]]] = {}
    for record in records[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi session entries must be objects") from None  # noqa: TRY003, TRY004
        event_id, parent_id, event_type = record.get("id"), record.get("parentId"), record.get("type")
        if (
            not isinstance(event_id, str)
            or not event_id
            or not isinstance(event_type, str)
            or not event_type
            or "parentId" not in record
            or (parent_id is not None and (not isinstance(parent_id, str) or parent_id not in nodes))
            or event_id in nodes
        ):
            raise ValueError("invalid Pi session entry identity or parent")  # noqa: TRY003
        nodes[event_id] = (parent_id, event_type, cast(dict[str, JsonValue], record))
    return session_id, nodes


def _parse_pi_session(
    value: PiSessionInput,
) -> tuple[str, tuple[PiEventInput, ...], dict[str, tuple[str | None, str, dict[str, JsonValue]]]]:
    session_id, nodes = _load_nodes(value.content)
    leaf_id = value.leaf_id if value.leaf_id is not None else next(reversed(nodes), None)
    if leaf_id is None:
        return session_id, (), nodes
    if leaf_id not in nodes:
        raise ValueError("selected Pi session leaf does not exist")  # noqa: TRY003
    path: list[tuple[str, str | None, str, dict[str, JsonValue]]] = []
    while leaf_id is not None:
        parent_id, event_type, record = nodes[leaf_id]
        path.append((leaf_id, parent_id, event_type, record))
        leaf_id = parent_id
    events: list[PiEventInput] = []
    for event_id, parent_id, event_type, record in reversed(path):
        visible = _event_body(record)
        if visible is None:
            continue
        role, body = visible
        if not body.strip():
            continue
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parent_id,
                event_type=event_type,
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return session_id, tuple(events), nodes


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Validate the complete JSONL tree before returning visible ancestors."""

    return _parse_pi_session(value)[1]


class PiEventSourceAdapter:
    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=_identity(value.session_id, value.event_id),
            materialization=SourceMaterialization.CAPTURED,
            **value.model_dump(),
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    projections=(PiEventTextProjection(),),
)


class PiSessionConnector:
    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        session_id, events, nodes = _parse_pi_session(
            PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id)
        )
        checkpoint = session.checkpoint
        if checkpoint is None:
            previous: dict[str, str] = {}
        elif (
            not isinstance(checkpoint, dict)
            or checkpoint.get("session_id") != session_id
            or not isinstance(checkpoint.get("events"), dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in checkpoint["events"].items())
        ):
            raise ValueError("invalid Pi session checkpoint")  # noqa: TRY003
        else:
            previous = checkpoint["events"]

        for event_id, digest in previous.items():
            record = nodes.get(event_id)
            if record is None or digest != hashlib.sha256(_json_text(record[2]).encode("utf-8")).hexdigest():
                raise ValueError(f"previously captured Pi event {event_id!r} was changed")  # noqa: TRY003
        current = {
            event.event_id: hashlib.sha256(_json_text(event.record).encode("utf-8")).hexdigest() for event in events
        }
        for event in events:
            if event.event_id in previous:
                continue
            outcome = await session.submit(_identity(event.session_id, event.event_id), PI_EVENT_SOURCE_NAME, event)
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                return ConnectorRunCompletion(status=ConnectorRunStatus.INCOMPLETE)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"session_id": session_id, "events": previous | current},
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
