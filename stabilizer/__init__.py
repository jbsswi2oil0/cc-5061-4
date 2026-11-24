"""Handheld video stabilization kernel.

Supported motion models are restricted to 2D plane / near-planar camera
motion: similarity (4 DOF) and affine (6 DOF). Nothing here claims a single
transform explains arbitrary depth / parallax; the tracking layer reports
model residuals so parallax-heavy footage is flagged.
"""

from .geometry import (
    aff_compose,
    apply_aff,
    blend_toward_identity,
    coverage_zoom,
    cumulative_trajectory,
    decompose_similarity,
    gaussian_smooth,
    solve_max_alpha,
    build_plan,
    fov_cap_to_zoom,
)
from .tracking import analyze_video, MotionAnalysis, PairMotion
from .pipeline import (
    compute_plan,
    render_plan,
    render_to_file,
    plan_frames_json,
    plan_metrics,
    DEFAULT_PARAMS,
)

__all__ = [
    "aff_compose",
    "apply_aff",
    "blend_toward_identity",
    "coverage_zoom",
    "cumulative_trajectory",
    "decompose_similarity",
    "gaussian_smooth",
    "solve_max_alpha",
    "build_plan",
    "fov_cap_to_zoom",
    "analyze_video",
    "MotionAnalysis",
    "PairMotion",
    "compute_plan",
    "render_plan",
    "render_to_file",
    "plan_frames_json",
    "plan_metrics",
    "DEFAULT_PARAMS",
]
