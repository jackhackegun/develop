"""GitHub 업로드를 도와주는 유틸리티 모듈."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """저장소 루트에서 Git 명령을 실행해 결과를 반환한다."""

    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _format_lines(lines: Iterable[str]) -> str:
    return "\n".join(line.rstrip() for line in lines if line is not None)


def publish_to_github(
    *,
    remote: str = "origin",
    branch: str | None = None,
    dry_run: bool = False,
) -> str:
    """원격 저장소에 현재 브랜치를 업로드하거나 필요한 단계를 안내한다."""

    messages: list[str] = []

    current_branch = branch
    if current_branch is None:
        branch_proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if branch_proc.returncode != 0:
            messages.append("현재 브랜치를 확인하지 못했습니다. Git 저장소인지 확인하세요.")
            messages.append(branch_proc.stderr.strip())
            return _format_lines(messages)
        current_branch = branch_proc.stdout.strip()

    remotes_proc = _run_git(["remote"])
    if remotes_proc.returncode != 0:
        messages.append("원격 저장소 목록을 확인하지 못했습니다. Git이 초기화되어 있는지 확인하세요.")
        messages.append(remotes_proc.stderr.strip())
        return _format_lines(messages)

    remotes = {line.strip() for line in remotes_proc.stdout.splitlines() if line.strip()}
    if remote not in remotes:
        messages.append(
            f"원격 '{remote}' 이(가) 설정되어 있지 않습니다. 아래 명령으로 추가하세요:"
        )
        messages.append(f"  git remote add {remote} <깃허브_저장소_URL>")
        messages.append(
            f"그 다음 'python -m ai_system.github --remote {remote}' 명령으로 다시 업로드를 시도하세요."
        )
        return _format_lines(messages)

    messages.append(f"현재 브랜치: {current_branch}")
    messages.append(f"원격: {remote}")

    if dry_run:
        status_proc = _run_git(["status", "-sb"])
        if status_proc.returncode == 0:
            messages.append("현재 커밋 상태:")
            messages.append(status_proc.stdout.strip())
        else:
            messages.append("git status 실행에 실패했습니다:")
            messages.append(status_proc.stderr.strip())
        messages.append("실제 푸시를 원하면 --dry-run 옵션을 제거하고 실행하세요.")
        return _format_lines(messages)

    push_proc = _run_git(["push", "-u", remote, current_branch])
    if push_proc.returncode == 0:
        messages.append("✅ 깃허브로 푸시가 완료되었습니다.")
        if push_proc.stdout:
            messages.append(push_proc.stdout.strip())
    else:
        messages.append("⚠️ 푸시가 실패했습니다. 아래 Git 메시지를 확인하세요:")
        if push_proc.stderr:
            messages.append(push_proc.stderr.strip())
        if push_proc.stdout:
            messages.append(push_proc.stdout.strip())
        messages.append(
            "원격 주소와 인증 토큰이 정확한지 확인한 뒤 다시 시도하세요."
        )

    return _format_lines(messages)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="깃허브 업로드 도우미")
    parser.add_argument("--remote", default="origin", help="사용할 원격 이름 (기본값: origin)")
    parser.add_argument("--branch", default=None, help="푸시할 브랜치 (기본값: 현재 브랜치)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 푸시 없이 상태만 확인",
    )
    args = parser.parse_args(argv)

    print(
        publish_to_github(
            remote=args.remote,
            branch=args.branch,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
