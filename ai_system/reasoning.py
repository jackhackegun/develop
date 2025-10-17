"""Rule-based reasoning over perceived features."""
from __future__ import annotations

from typing import Dict, List

FeatureMap = Dict[str, object]


class Rule:
    """Simple rule represented as predicate and conclusion."""

    def __init__(self, predicate, conclusion):
        self.predicate = predicate
        self.conclusion = conclusion

    def applies(self, features: FeatureMap) -> bool:
        return self.predicate(features)

    def conclude(self, features: FeatureMap) -> str:
        return self.conclusion(features)


DEFAULT_RULES: List[Rule] = [
    Rule(
        lambda f: f["agent_position"] == f["goal_position"],
        lambda _: "goal_reached",
    ),
    Rule(
        lambda f: f["agent_position"] == f["charger_position"],
        lambda _: "recharging",
    ),
    Rule(
        lambda f: f.get("north_blocked") and f.get("east_blocked"),
        lambda _: "avoid_northeast",
    ),
]


def infer(features: FeatureMap, rules: List[Rule] | None = None) -> List[str]:
    """Return symbolic conclusions derived from features."""

    fired = []
    for rule in rules or DEFAULT_RULES:
        if rule.applies(features):
            fired.append(rule.conclude(features))
    if not fired:
        fired.append("navigate")
    return fired
