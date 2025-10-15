"""Autonomous decision-making that integrates all subsystems."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .data import GridWorld
from .language import Command
from .learning import QLearningAgent
from .perception import perceive
from .problem_solving import a_star
from .reasoning import infer


@dataclass
class AgentState:
    battery: float = 1.0
    history: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.history.append(message)


class AutonomousController:
    """Coordinate perception, reasoning, learning, and planning."""

    def __init__(self, env: GridWorld, learner: QLearningAgent, state: AgentState | None = None):
        self.env = env
        self.learner = learner
        self.state = state or AgentState()

    def decide(self, command: Command) -> Dict[str, object]:
        features = perceive(self.env)
        inferences = infer(features)
        self.state.log(f"perceived {features}")
        self.state.log(f"inferred {inferences}")

        target = self._resolve_target(command, features, inferences)
        plan = a_star(self.env, self.env.agent, target)
        if plan is None:
            raise RuntimeError("No plan found")

        self.state.battery -= 0.05 * max(len(plan) - 1, 0)
        if self.env.agent == self.env.config.charger:
            self.state.battery = min(1.0, self.state.battery + 0.3)

        action = None
        if len(plan) > 1:
            next_state = plan[1]
            dx = next_state[0] - self.env.agent[0]
            dy = next_state[1] - self.env.agent[1]
            action = (dx, dy)
        else:
            action = (0, 0)

        self.state.log(f"plan {plan}")
        return {
            "features": features,
            "inferences": inferences,
            "plan": plan,
            "action": action,
            "battery": round(self.state.battery, 2),
        }

    def _resolve_target(self, command: Command, features: Dict[str, object], inferences: List[str]) -> Tuple[int, int]:
        if command.intent == "recharge" or command.target == "charger" or self.state.battery < 0.3:
            return features["charger_position"]  # type: ignore[return-value]
        if "goal_reached" in inferences:
            return features["charger_position"]  # celebrate by recharging
        if command.target == "goal":
            return features["goal_position"]  # type: ignore[return-value]
        return features["goal_position"]  # type: ignore[return-value]
