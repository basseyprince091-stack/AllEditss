"""Renderer interface (Principle 4: rendering engines are replaceable).

A Remotion/WebGL renderer for motion graphics and typography can implement this
same interface later and be selected per-project without touching the planner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RenderResult:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    preview: bool
    segments: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class Renderer(ABC):
    name = "base"

    @abstractmethod
    def render(self, timeline, out_path, preview: bool = True,
               progress=None) -> RenderResult: ...
