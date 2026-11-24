"""2D affine geometry, camera path accumulation/smoothing and FOV adjudication.

Conventions
-----------
A motion matrix ``A_i`` maps homogeneous coordinates in frame ``i-1`` to
frame ``i`` (same convention as ``cv2.estimateAffine*2D``).

The cumulative trajectory ``T_i = A_i @ ... @ A_1`` maps frame-0 coordinates
to frame ``i`` coordinates.  A smoothed target path ``S_i`` is a low-pass
version of ``T_i``.  The correction ::

    C_i = S_i @ inv(T_i)

maps original frame-i pixels into the stabilized canvas (source -> dest).

The final source->output matrix is ``K_i = D(center, z_i) @ C_i(alpha_i)``
where ``z_i`` is an explicit zoom/crop factor.  Coverage is checked
geometrically: every output corner, mapped back through ``K_i``, must lie
inside the source frame rectangle.  When full correction needs more zoom than
the declared minimum field of view allows, ``alpha_i < 1`` partially reverts
the correction (local stability sacrifice) so coverage still holds.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

Rect = tuple[int, int]


def I3() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def as33(m: np.ndarray | Sequence) -> np.ndarray:
    a = np.asarray(m, dtype=np.float64)
    if a.shape == (3, 3):
        return a.copy()
    out = I3()
    out[:2, :] = a.reshape(2, 3)
    return out


def aff_compose(x: float, y: float, theta: float, scale: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [scale * c, -scale * s, x],
            [scale * s, scale * c, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def decompose_similarity(M: np.ndarray) -> tuple[float, float, float, float]:
    """Return tx, ty, theta, uniform scale for a (partial) affine matrix."""
    a = as33(M)
    l = a[:2, :2]
    sx = math.hypot(l[0, 0], l[1, 0])
    sy = math.hypot(l[0, 1], l[1, 1])
    scale = 0.5 * (sx + sy)
    theta = math.atan2(l[1, 0], l[0, 0])
    return float(a[0, 2]), float(a[1, 2]), float(theta), float(scale)


def apply_aff(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 2D affine to an (N,2) array, returning (N,2)."""
    a = as33(M)
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return p @ a[:2, :2].T + a[:2, 2]


def cumulative_trajectory(pair_motions: Sequence[np.ndarray]) -> List[np.ndarray]:
    """T_0 = I, T_i = A_i @ T_{i-1}."""
    traj = [I3()]
    cur = I3()
    for A in pair_motions:
        cur = as33(A) @ cur
        traj.append(cur.copy())
    return traj


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 1e-6:
        return np.ones(1)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Edge-replicated Gaussian smoothing over axis 0. Accepts (N,) or (N,C)."""
    v = np.asarray(values, dtype=np.float64)
    scalar = v.ndim == 1
    if scalar:
        v = v[:, None]
    n = v.shape[0]
    if n < 3 or sigma <= 1e-6:
        return values.copy() if not scalar else np.asarray(values, dtype=np.float64).copy()
    k = gaussian_kernel_1d(sigma)
    r = len(k) // 2
    padded = np.pad(v, ((r, r), (0, 0)), mode="edge")
    out = np.zeros_like(v)
    for j, w in enumerate(k):
        out += w * padded[j : j + n]
    if scalar:
        return out[:, 0]
    return out


def smooth_trajectory(
    traj: Sequence[np.ndarray], strength: float, model: str = "similarity"
) -> List[np.ndarray]:
    """Low-pass the cumulative camera path.

    ``strength`` in [0, 1] maps to sigma in frames.  strength=0 returns the
    raw path (correction = identity).  The cap keeps maximum smoothing finite
    so deliberate slow camera moves survive (no frozen-frame "stabilization").
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    sigma = strength * 18.0
    n = len(traj)
    mats = [as33(m) for m in traj]
    if sigma <= 1e-6 or n < 3:
        return [m.copy() for m in mats]

    if model == "affine":
        params = np.array(
            [[m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 2], m[1, 2]] for m in mats]
        )
        sm = gaussian_smooth(params, sigma)
        out: List[np.ndarray] = []
        for r in sm:
            out.append(
                np.array(
                    [
                        [r[0], r[1], r[4]],
                        [r[2], r[3], r[5]],
                        [0.0, 0.0, 1.0],
                    ]
                )
            )
        return out

    pose = np.array([decompose_similarity(m) for m in mats])
    # unwrap angle channel so rotations do not tear at +/- pi
    pose[:, 2] = np.unwrap(pose[:, 2])
    pose[:, 3] = np.log(np.maximum(pose[:, 3], 1e-6))
    sm = gaussian_smooth(pose, sigma)
    return [aff_compose(x, y, th, math.exp(ls)) for x, y, th, ls in sm]


def correction_matrices(
    orig: Sequence[np.ndarray], target: Sequence[np.ndarray]
) -> List[np.ndarray]:
    return [as33(S) @ np.linalg.inv(as33(T)) for T, S in zip(orig, target)]


def zoom_about_center(w: int, h: int, zoom: float) -> np.ndarray:
    cx, cy = w * 0.5, h * 0.5
    return np.array(
        [
            [zoom, 0.0, cx * (1.0 - zoom)],
            [0.0, zoom, cy * (1.0 - zoom)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def output_corners(w: int, h: int) -> np.ndarray:
    return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)


def mapped_back_corners(
    C: np.ndarray, w: int, h: int, zoom: float
) -> np.ndarray:
    K = zoom_about_center(w, h, zoom) @ as33(C)
    return apply_aff(np.linalg.inv(K), output_corners(w, h))


def is_covered(
    C: np.ndarray, w: int, h: int, zoom: float, safety: float = 1.5
) -> bool:
    q = mapped_back_corners(C, w, h, zoom)
    return bool(
        np.all(q[:, 0] >= safety)
        and np.all(q[:, 0] <= w - safety)
        and np.all(q[:, 1] >= safety)
        and np.all(q[:, 1] <= h - safety)
    )


def coverage_zoom(
    C: np.ndarray, w: int, h: int, safety: float = 1.5
) -> float:
    """Smallest zoom >= 1 for which no output pixel samples outside source."""
    C = as33(C)
    if is_covered(C, w, h, 1.0, safety):
        return 1.0
    lo, hi = 1.0, 1.05
    for _ in range(40):
        if is_covered(C, w, h, hi, safety):
            break
        hi *= 1.5
        if hi > 64.0:
            break
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if is_covered(C, w, h, mid, safety):
            hi = mid
        else:
            lo = mid
    return hi


def blend_toward_identity(C: np.ndarray, alpha: float, model: str = "similarity") -> np.ndarray:
    """Scale the correction: alpha=1 full correction, alpha=0 identity."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha >= 1.0 - 1e-12:
        return as33(C)
    if alpha <= 1e-12:
        return I3()
    C = as33(C)
    if model == "affine":
        return (1.0 - alpha) * I3() + alpha * C
    dx, dy, dth, ds = decompose_similarity(C)
    return aff_compose(
        alpha * dx,
        alpha * dy,
        alpha * dth,
        math.exp(alpha * math.log(max(ds, 1e-6))),
    )


def solve_max_alpha(
    C: np.ndarray, z_cap: float, w: int, h: int, model: str = "similarity",
    safety: float = 1.5,
) -> float:
    """Largest correction strength whose coverage zoom stays <= z_cap."""
    if coverage_zoom(C, w, h, safety) <= z_cap + 1e-9:
        return 1.0
    lo, hi = 0.0, 1.0  # lo feasible (identity needs ~1), hi infeasible
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        Cm = blend_toward_identity(C, mid, model)
        if coverage_zoom(Cm, w, h, safety) <= z_cap + 1e-9:
            lo = mid
        else:
            hi = mid
    return lo


def fov_cap_to_zoom(min_fov: float) -> float:
    """Linear FOV retention fraction -> maximum permitted zoom factor."""
    min_fov = float(np.clip(min_fov, 0.5, 1.0))
    return 1.0 / min_fov


def _rolling_max(x: np.ndarray, radius: int) -> np.ndarray:
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = np.max(x[lo:hi])
    return out


def build_plan(
    traj: Sequence[np.ndarray],
    w: int,
    h: int,
    strength: float,
    min_fov: float,
    model: str = "similarity",
    zoom_radius: int = 22,
    zoom_sigma: float = 9.0,
    safety: float = 1.5,
) -> dict:
    """Compute target path, per-frame correction, zoom, alpha and crop quad.

    Zoom/crop envelope
    ------------------
    The geometric minimum zoom ``z_raw_i`` oscillates with the shake. Applying
    it directly causes visible "zoom pumping" and injects high-frequency scale
    jumps, so the raw minima are spread with a centered rolling maximum and a
    Gaussian transition. The final zoom is then guaranteed to be at least the
    raw minimum. Once that envelope is fixed, each frame takes the largest
    correction ``alpha`` its zoom allows; a frame whose best correction still
    cannot fit the declared FOV cap is locally sacrificed (``alpha < 1``).

    Returns a dict with numpy arrays / matrices plus JSON-friendly pose data.
    """
    n = len(traj)
    target = smooth_trajectory(traj, strength, model)
    corr = correction_matrices(traj, target)
    z_cap = fov_cap_to_zoom(min_fov)

    # Stage 1: unrestricted geometric need and FOV-cap adjudication.
    z_raw = np.ones(n)
    alpha_cap = np.ones(n)  # max alpha allowed by the declared z_cap
    for i, C in enumerate(corr):
        z_full = coverage_zoom(C, w, h, safety)
        if z_full <= z_cap + 1e-9:
            z_raw[i] = z_full
        else:
            a = solve_max_alpha(C, z_cap, w, h, model, safety)
            alpha_cap[i] = a
            Ca = blend_toward_identity(C, a, model)
            z_raw[i] = min(z_cap, coverage_zoom(Ca, w, h, safety))

    # Stage 2: slow zoom envelope that never undershoots a geometric need.
    if n >= 3:
        z_maxenv = _rolling_max(z_raw, int(zoom_radius))
        z_env = gaussian_smooth(z_maxenv, zoom_sigma)
        z_actual = np.maximum(z_env, z_raw)
    else:
        z_actual = z_raw.copy()
    z_actual = np.clip(z_actual, 1.0, z_cap)

    # Stage 3: at the chosen zoom, take the largest feasible alpha, then smooth
    # alpha slightly but only downward relative to that per-frame maximum.
    alpha_zoom = np.ones(n)
    for i, C in enumerate(corr):
        if coverage_zoom(C, w, h, safety) <= z_actual[i] + 1e-9:
            alpha_zoom[i] = 1.0
        else:
            alpha_zoom[i] = solve_max_alpha(
                C, z_actual[i], w, h, model, safety
            )
    alpha_zoom = np.minimum(alpha_zoom, alpha_cap)
    a_smooth = gaussian_smooth(alpha_zoom, 2.0) if n >= 3 else alpha_zoom.copy()
    alpha_actual = np.clip(np.minimum(a_smooth, alpha_zoom), 0.0, 1.0)

    sacrificed = alpha_cap < 0.999

    # Stage 4: final matrices + source-frame crop quads, with a hard coverage
    # repair as a last-resort invariant.
    quad_src = np.zeros((n, 4, 2))
    K_final: List[np.ndarray] = []
    for i in range(n):
        Ci = blend_toward_identity(corr[i], alpha_actual[i], model)
        Ki = zoom_about_center(w, h, z_actual[i]) @ Ci
        if not is_covered(Ci, w, h, z_actual[i], safety):
            a = solve_max_alpha(Ci, z_cap, w, h, model, safety)
            alpha_actual[i] = min(alpha_actual[i], a)
            Ci = blend_toward_identity(corr[i], alpha_actual[i], model)
            z_fix = min(z_cap, coverage_zoom(Ci, w, h, safety))
            z_actual[i] = max(z_actual[i], z_fix)
            Ki = zoom_about_center(w, h, z_actual[i]) @ Ci
        K_final.append(Ki)
        quad_src[i] = apply_aff(np.linalg.inv(Ki), output_corners(w, h))

    orig_pose = np.array([decompose_similarity(m) for m in traj])
    target_pose = np.array([decompose_similarity(m) for m in target])
    orig_pose[:, 2] = np.unwrap(orig_pose[:, 2])
    target_pose[:, 2] = np.unwrap(target_pose[:, 2])

    return {
        "strength": float(strength),
        "min_fov": float(min_fov),
        "z_cap": float(z_cap),
        "model": model,
        "orig_pose": orig_pose,
        "target_pose": target_pose,
        "z_raw": z_raw,
        "zoom": z_actual,
        "fov": 1.0 / z_actual,
        "alpha": alpha_actual,
        "alpha_max": alpha_cap,
        "sacrificed": sacrificed,
        "quad_src": quad_src,
        "K": K_final,
        "target_mats": target,
        "corr_mats": corr,
    }
