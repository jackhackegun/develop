"""Problem solving via heuristic search."""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from .data import GridWorld

State = Tuple[int, int]
Action = Tuple[int, int]


def heuristic(a: State, b: State) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: Dict[State, State], current: State) -> List[State]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(env: GridWorld, start: State, goal: State) -> Optional[List[State]]:
    """Perform A* search to find a path from start to goal."""

    open_set: List[Tuple[float, State]] = []
    heapq.heappush(open_set, (0, start))
    came_from: Dict[State, State] = {}
    g_score: Dict[State, float] = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for action in env.possible_actions(current):
            neighbor = (current[0] + action[0], current[1] + action[1])
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
    return None
