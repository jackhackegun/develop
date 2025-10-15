"""한국어 중심의 대화형 콘솔 인터페이스."""
from __future__ import annotations

from typing import Sequence

from .autonomy import AgentState, AutonomousController
from .data import demo_world
from .language import parse_command
from .learning import QLearningAgent
from .main import describe_action


def run_cli(intro: bool = True) -> None:
    env = demo_world()
    learner = QLearningAgent(env)
    learner.learn()

    controller = AutonomousController(env, learner, AgentState())

    if intro:
        print("그리드 월드 AI 데모에 오신 것을 환영합니다!")
        print("한국어나 영어로 이동, 충전, 상태 보고 명령을 내려보세요.")
        print("예시: '목표 지점으로 이동해줘', '상태 보고해', '충전소로 가'.")
        print("종료하려면 '종료', 'quit', 'exit' 중 하나를 입력하세요.")

    while True:
        try:
            text = input("명령> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n세션을 종료합니다.")
            break

        if not text:
            continue

        normalized = text.lower()
        if normalized in {"quit", "exit"} or text in {"종료", "끝"}:
            print("AI 세션을 종료합니다. 이용해주셔서 감사합니다!")
            break

        command = parse_command(text)
        decision = controller.decide(command)
        env.step(decision["action"])

        features = decision["features"]
        plan = " -> ".join(str(tuple(step)) for step in decision["plan"]) or "경로 없음"

        print(f"의도/목표: {command.intent} / {command.target or '지정되지 않음'}")
        print(f"현재 위치: {tuple(features['agent_position'])} / 목표: {tuple(features['goal_position'])}")
        print(f"추론 결과: {', '.join(decision['inferences'])}")
        print(f"계획 경로: {plan}")
        print(f"실행 행동: {describe_action(tuple(decision['action']))}")
        print(f"배터리 잔량: {decision['battery']}")

        if env.agent == env.config.goal:
            print("✔ 목표 지점에 도달했습니다! 환경을 초기화합니다.")
            env.reset()
        elif env.agent == env.config.charger:
            print("🔋 충전소에 도착했습니다. 배터리를 회복합니다.")


def main(argv: Sequence[str] | None = None) -> None:  # noqa: D401 - CLI 진입점
    run_cli()


if __name__ == "__main__":
    main()
