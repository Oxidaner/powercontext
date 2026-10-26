from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from powercontext.sources import ConnectorBinding, SourceRef
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.pi_session import PiSessionConnector


def test_off_branch_history_mutation_preserves_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "session.jsonl"
        root = {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "start"}}
        left = {
            "type": "message",
            "id": "left",
            "parentId": "root",
            "message": {"role": "assistant", "content": "left"},
        }
        right = {
            "type": "message",
            "id": "right",
            "parentId": "root",
            "message": {"role": "assistant", "content": "right"},
        }

        def write() -> None:
            records = [{"type": "session", "version": 3, "id": "session"}, root, left, right]
            path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

        class Sink:
            async def submit(
                self, binding: ConnectorBinding, item_id: str, definition_name: str, value: object
            ) -> SourceRef:
                return SourceRef(source_type=definition_name, source_id=item_id)

        class Checkpoints:
            checkpoint: JsonValue | None = None

            async def load(self, binding: ConnectorBinding) -> JsonValue | None:
                return self.checkpoint

            async def save(
                self, binding: ConnectorBinding, checkpoint: JsonValue | None, *, expected: JsonValue | None
            ) -> None:
                assert self.checkpoint == expected
                self.checkpoint = checkpoint

        write()
        checkpoints = Checkpoints()
        lifecycle = ConnectorLifecycle(sink=Sink(), checkpoints=checkpoints)
        binding = ConnectorBinding(
            scope_id="scope", binding_id="pi", connector_name="pi-session-history", connector_version="1"
        )
        first = await lifecycle.run(PiSessionConnector(path, "left"), binding)
        assert len(first.items) == 2
        left["message"]["content"] = "rewritten"
        write()
        with pytest.raises(ValueError, match="previously captured"):
            await lifecycle.run(PiSessionConnector(path, "right"), binding)
        assert checkpoints.checkpoint == first.committed_checkpoint
        left["message"]["content"] = "left"
        write()
        recovered = await lifecycle.run(PiSessionConnector(path, "right"), binding)
        assert len(recovered.items) == 1

    asyncio.run(scenario())
