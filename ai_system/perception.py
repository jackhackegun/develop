"""Perception module that converts raw grid information into symbolic features."""
from __future__ import annotations

from typing import Dict, Tuple

from .data import GridWorld


def perceive(env: GridWorld) -> Dict[str, Tuple[int, int] | bool]:
    """Return simple symbolic features describing the surroundings."""

    x, y = env.agent
    # Immediate neighborhood features capture spatial perception.
    neighborhood = {
        "north_blocked": (x, y + 1) in env.config.obstacles or y + 1 >= env.config.height,
        "south_blocked": (x, y - 1) in env.config.obstacles or y - 1 < 0,
        "east_blocked": (x + 1, y) in env.config.obstacles or x + 1 >= env.config.width,
        "west_blocked": (x - 1, y) in env.config.obstacles or x - 1 < 0,
    }
    return {
        "agent_position": env.agent,
        "goal_position": env.config.goal,
        "charger_position": env.config.charger,
        **neighborhood,
    }
