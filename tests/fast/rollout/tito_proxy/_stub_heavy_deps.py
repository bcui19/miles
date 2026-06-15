"""Stub heavy optional deps (ray, sglang) so the tito_proxy fast tests can run
on dev machines without the GPU stack. Real modules always win: each stub is
only installed when the import fails, so full-dependency CI is unaffected.

Import this module before any ``miles.rollout`` import in a test file.
"""

from __future__ import annotations

import sys
import types


def _ensure_module(name: str, attrs: dict | None = None) -> None:
    try:
        __import__(name)
        return
    except ImportError:
        pass
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        partial = ".".join(parts[:depth])
        if partial in sys.modules:
            continue
        module = types.ModuleType(partial)
        module.__path__ = []  # mark as package so submodule imports work
        sys.modules[partial] = module
        if depth > 1:
            setattr(sys.modules[".".join(parts[: depth - 1])], parts[depth - 1], module)
    if attrs:
        for key, value in attrs.items():
            setattr(sys.modules[name], key, value)


def _ray_remote(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda cls: cls


_ensure_module("ray", {"remote": _ray_remote})
_ensure_module(
    "ray.util.scheduling_strategies",
    {"NodeAffinitySchedulingStrategy": type("NodeAffinitySchedulingStrategy", (), {})},
)
_ensure_module("sglang.srt.entrypoints.openai.protocol", {"Tool": type("Tool", (), {})})


def _permissive_getattr(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return types.SimpleNamespace()


if not getattr(sys.modules["sglang"], "__file__", None):
    # permissive: any `from sglang.srt.entrypoints.openai import X` yields a namespace stub
    sys.modules["sglang.srt.entrypoints.openai"].__getattr__ = _permissive_getattr
