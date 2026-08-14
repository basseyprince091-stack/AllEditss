"""Shoot assistant: planning, directing and inspecting footage that does not exist yet."""
from .shotspec import ShotSpec, SEQUENCES, build_sequence
from .coverage import (assess_coverage, inspect_recording, CoverageReport,
                       ShotCoverage, MISSING, LIKELY, UNVERIFIABLE)

__all__ = ["ShotSpec", "SEQUENCES", "build_sequence", "assess_coverage",
           "inspect_recording", "CoverageReport", "ShotCoverage",
           "MISSING", "LIKELY", "UNVERIFIABLE"]
