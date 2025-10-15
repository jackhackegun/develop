"""Utilities for building datasets and environments for the demo AI system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class GridWorldConfig:
    """Configuration describing the grid-world environment."""

    width: int
    height: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    obstacles: Tuple[Tuple[int, int], ...]
    charger: Tuple[int, int]


class GridWorld:
    """Simple grid world used by multiple subsystems.

    The environment provides deterministic dynamics.  It is intentionally kept
    small so that Q-learning converges rapidly and A* search completes almost
    instantly.
    """

    ACTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))

    def __init__(self, config: GridWorldConfig):
        self.config = config
        self.reset()

    @property
    def states(self) -> Iterable[Tuple[int, int]]:
        for x in range(self.config.width):
            for y in range(self.config.height):
                if (x, y) not in self.config.obstacles:
                    yield x, y

    def reset(self) -> Tuple[int, int]:
        self.agent = self.config.start
        return self.agent

    def step(self, action: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool, Dict[str, bool]]:
        nx = self.agent[0] + action[0]
        ny = self.agent[1] + action[1]
        if not (0 <= nx < self.config.width and 0 <= ny < self.config.height):
            nx, ny = self.agent  # hits a wall
        if (nx, ny) in self.config.obstacles:
            nx, ny = self.agent  # blocked by obstacle
        self.agent = (nx, ny)
        reward = -0.04  # mild step penalty
        done = False
        info = {"at_goal": False, "at_charger": False}
        if self.agent == self.config.goal:
            reward = 1.0
            done = True
            info["at_goal"] = True
        elif self.agent == self.config.charger:
            reward = 0.2
            info["at_charger"] = True
        return self.agent, reward, done, info

    def possible_actions(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        actions = []
        x, y = state
        for ax, ay in self.ACTIONS:
            nx, ny = x + ax, y + ay
            if 0 <= nx < self.config.width and 0 <= ny < self.config.height and (nx, ny) not in self.config.obstacles:
                actions.append((ax, ay))
        return actions


def demo_world() -> GridWorld:
    """Create the demo environment used across the modules."""

    config = GridWorldConfig(
        width=5,
        height=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=((1, 2), (2, 2), (3, 1)),
        charger=(0, 4),
    )
    return GridWorld(config)


def encode_state(state: Tuple[int, int], config: GridWorldConfig) -> int:
    """Convert a 2-D state to a single integer index for tabular methods."""

    return state[1] * config.width + state[0]
