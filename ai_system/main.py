"""Entry point showing the integrated AI system in action."""
from __future__ import annotations

import argparse
import json
from typing import Iterable, List, Sequence

from .autonomy import AgentState, AutonomousController
from .data import demo_world
from .language import parse_command
from .learning import QLearningAgent


COMMANDS: Sequence[str] = (
    "목표 지점으로 이동해줘",
    "지금 상태 보고해",
    "충전소로 이동해",
)


def run_demo(commands: Iterable[str] | None = None) -> List[dict]:
    env = demo_world()
    learner = QLearningAgent(env)
    learner.learn()
    env.reset()

    controller = AutonomousController(env, learner, AgentState())
    results = []

    for text in list(commands) if commands is not None else COMMANDS:
        command = parse_command(text)
        decision = controller.decide(command)
        env.step(decision["action"])
        results.append(
            {
                "command": text,
                "parsed": command.__dict__,
                **decision,
            }
        )
    return results


def describe_action(action: tuple[int, int]) -> str:
    mapping = {
        (0, 1): "북쪽으로 한 칸 이동",
        (0, -1): "남쪽으로 한 칸 이동",
        (1, 0): "동쪽으로 한 칸 이동",
        (-1, 0): "서쪽으로 한 칸 이동",
        (0, 0): "제자리 유지",
    }
    return mapping.get(action, f"이동 벡터 {action} 실행")


def format_results_korean(results: Sequence[dict]) -> str:
    lines: List[str] = []
    for idx, entry in enumerate(results, start=1):
        lines.append(f"# 명령 {idx}")
        lines.append(f"사용자 입력: {entry['command']}")
        intent = entry["parsed"].get("intent")  # type: ignore[index]
        target = entry["parsed"].get("target")  # type: ignore[index]
        target_text = target if target else "(명시되지 않음)"
        lines.append(f"- 해석된 의도: {intent} / 목표: {target_text}")
        features = entry.get("features", {})
        agent_pos = tuple(features.get("agent_position", ("?", "?")))
        goal_pos = tuple(features.get("goal_position", ("?", "?")))
        charger_pos = tuple(features.get("charger_position", ("?", "?")))
        lines.append(f"- 현재 위치: {agent_pos} / 목표 위치: {goal_pos} / 충전소: {charger_pos}")
        inferences = ", ".join(entry.get("inferences", [])) or "없음"
        lines.append(f"- 추론 결과: {inferences}")
        plan = entry.get("plan", [])
        if plan:
            plan_str = " -> ".join(str(tuple(step)) for step in plan)
        else:
            plan_str = "경로 없음"
        lines.append(f"- 계획 경로: {plan_str}")
        planner = entry.get("planner")
        if planner:
            lines.append(f"- 계획 생성기: {planner}")
        policy_action = entry.get("policy_action")
        if policy_action:
            action_text = describe_action(tuple(policy_action))
            lines.append(f"- 학습된 정책 제안: {action_text}")
        action_desc = describe_action(tuple(entry.get("action", (0, 0))))
        lines.append(f"- 실행 행동: {action_desc}")
        lines.append(f"- 배터리 잔량: {entry.get('battery', 0.0)}")
        lines.append("")
    return "\n".join(lines).strip()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="그리드 월드 AI 데모 실행기")
    parser.add_argument(
        "--format",
        choices=["korean", "json"],
        default="korean",
        help="출력 형식을 선택합니다 (기본: korean)",
    )
    parser.add_argument(
        "commands",
        nargs="*",
        help="사용자 정의 명령 목록을 지정하면 해당 순서로 실행합니다.",
    )
    args = parser.parse_args(argv)

    commands = args.commands if args.commands else None
    results = run_demo(commands)

    if args.format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(format_results_korean(results))


if __name__ == "__main__":
    main()
