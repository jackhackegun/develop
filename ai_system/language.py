"""Natural language understanding layer using a lightweight parser."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Command:
    intent: str
    target: Optional[str] = None


INTENT_PATTERNS = {
    "navigate": re.compile(r"\b(go|navigate|move)\b|이동|가(?:라|줘)?|도착", re.I),
    "recharge": re.compile(r"\b(recharge|battery)\b|충전|배터리", re.I),
    "report": re.compile(r"\b(report|status)\b|상태|보고", re.I),
}

TARGET_PATTERNS = {
    "goal": re.compile(r"\bgoal|target\b|목표|목적지|골", re.I),
    "charger": re.compile(r"\bcharger|station\b|충전(소|기)", re.I),
}


def parse_command(text: str) -> Command:
    """Extract a structured command from free-form text."""

    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(text):
            break
    else:
        intent = "navigate"

    target = None
    for name, pattern in TARGET_PATTERNS.items():
        if pattern.search(text):
            target = name
            break
    return Command(intent=intent, target=target)
