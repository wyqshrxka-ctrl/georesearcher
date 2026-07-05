"""capabilities/interpret package（见 docs/plan-m3--20260705--v1.md §3.6）。"""
from georesearcher.capabilities.interpret.interpret import (
    StructuredNote,
    _build_prompt,
    _parse_note_json,
    interpret_paper,
)

__all__ = ["interpret_paper", "_build_prompt", "_parse_note_json", "StructuredNote"]
