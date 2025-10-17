"""Integrated AI system demonstrating multiple intelligence criteria."""

from importlib import import_module
from typing import Any

__all__ = ["run_demo", "describe_action", "format_results_korean"]


def _load_main_attr(name: str) -> Any:
    module = import_module("ai_system.main")
    return getattr(module, name)


def run_demo(*args, **kwargs):  # type: ignore[override]
    return _load_main_attr("run_demo")(*args, **kwargs)


def describe_action(*args, **kwargs):  # type: ignore[override]
    return _load_main_attr("describe_action")(*args, **kwargs)


def format_results_korean(*args, **kwargs):  # type: ignore[override]
    return _load_main_attr("format_results_korean")(*args, **kwargs)
