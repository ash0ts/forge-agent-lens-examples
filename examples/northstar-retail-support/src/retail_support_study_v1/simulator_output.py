from __future__ import annotations

from typing import Any

STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")


def project_simulator_output(message: dict[str, Any]) -> tuple[str, bool]:
    """Project a simulator response into visible customer text and control state."""

    text = str(message.get("content") or "").strip()
    stopped = False
    for token in STOP_TOKENS:
        if token in text:
            text = text.replace(token, "").strip()
            stopped = True
    return text, stopped
