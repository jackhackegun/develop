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
        plan, planner = self._policy_plan(target)
        if plan is None:
            plan = a_star(self.env, self.env.agent, target)
            planner = "a_star"
        if plan is None:
            raise RuntimeError("No plan found")

        action = self._action_from_plan(plan)
        policy_action = self._policy_action()
        if policy_action is not None:
            policy_next = self.env.transition(self.env.agent, policy_action)
            plan_next = self.env.transition(self.env.agent, action)
            policy_dist = self._manhattan_distance(policy_next, target)
            plan_dist = self._manhattan_distance(plan_next, target)
            if planner == "policy" or policy_next == plan_next or policy_dist <= plan_dist:
                action = policy_action
            else:
                self.state.log(
                    f"policy suggested {policy_action} but planner action {action} was closer to target"
                )

        next_state = self.env.transition(self.env.agent, action)
        if next_state != self.env.agent:
            self.state.battery = max(0.0, self.state.battery - 0.05)
        if next_state == self.env.config.charger:
            self.state.battery = min(1.0, self.state.battery + 0.3)

        self.state.log(f"plan {plan}")
        return {
            "features": features,
            "inferences": inferences,
            "plan": plan,
            "action": action,
            "planner": planner,
            "policy_action": policy_action,
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

    def _policy_plan(self, target: Tuple[int, int], max_steps: int = 25) -> tuple[List[Tuple[int, int]] | None, str]:
        """Simulate the learned policy to build a plan if possible."""

        path = [self.env.agent]
        state = self.env.agent
        visited = {state}
        try:
            for _ in range(max_steps):
                if state == target:
                    return path, "policy"
                action = self.learner.greedy_action(state)
                next_state = self.env.transition(state, action)
                if next_state == state or next_state in visited:
                    break
                path.append(next_state)
                visited.add(next_state)
                state = next_state
        except ValueError:
            return None, "policy"
        if state == target:
            return path, "policy"
        return None, "policy"

    def _action_from_plan(self, plan: List[Tuple[int, int]]) -> Tuple[int, int]:
        if len(plan) > 1:
            next_state = plan[1]
            dx = next_state[0] - self.env.agent[0]
            dy = next_state[1] - self.env.agent[1]
            return (dx, dy)
        return (0, 0)

    def _policy_action(self) -> Tuple[int, int] | None:
        try:
            return self.learner.greedy_action(self.env.agent)
        except ValueError:
            return None

    @staticmethod
    def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
