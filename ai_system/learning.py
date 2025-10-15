"""Learning subsystem implementing a compact Q-learning agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import random

from .data import GridWorld, encode_state


@dataclass
class QLearningConfig:
    learning_rate: float = 0.3
    discount: float = 0.95
    epsilon: float = 0.2
    episodes: int = 200
    max_steps: int = 75


class QLearningAgent:
    """Tabular Q-learning agent that learns a navigation policy."""

    def __init__(self, env: GridWorld, config: QLearningConfig | None = None):
        self.env = env
        self.config = config or QLearningConfig()
        self.q_table: Dict[Tuple[int, Tuple[int, int]], float] = {}
        self.state_index = lambda s: encode_state(s, env.config)

    def _q(self, state: Tuple[int, int], action: Tuple[int, int]) -> float:
        return self.q_table.get((self.state_index(state), action), 0.0)

    def policy(self, state: Tuple[int, int]) -> Tuple[int, int]:
        actions = self.env.possible_actions(state)
        if not actions:
            raise ValueError("No actions available")
        if random.random() < self.config.epsilon:
            return random.choice(actions)
        q_values = [self._q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self) -> None:
        for _ in range(self.config.episodes):
            state = self.env.reset()
            done = False
            steps = 0
            while not done and steps < self.config.max_steps:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                self._update(state, action, reward, next_state)
                state = next_state
                steps += 1

    def _update(self, state: Tuple[int, int], action: Tuple[int, int], reward: float, next_state: Tuple[int, int]) -> None:
        state_idx = self.state_index(state)
        next_actions = self.env.possible_actions(next_state)
        max_next = 0.0
        if next_actions:
            max_next = max(self._q(next_state, a) for a in next_actions)
        old_value = self._q(state, action)
        target = reward + self.config.discount * max_next
        new_value = old_value + self.config.learning_rate * (target - old_value)
        self.q_table[(state_idx, action)] = new_value

    def greedy_action(self, state: Tuple[int, int]) -> Tuple[int, int]:
        """Return the best-known action without exploration."""

        actions = self.env.possible_actions(state)
        if not actions:
            raise ValueError("No actions available")
        q_values = [self._q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)
