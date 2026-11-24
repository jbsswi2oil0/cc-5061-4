"""Quantitative comparison helpers: path high-frequency energy, GT error, FOV."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .geometry import decompose_similarity


def pose_series_from_mats(mats) -> np.ndarray:
    pose = np.array([decompose_similarity(m) for m in mats], dtype=np.float64)
    pose[:, 2] = np.unwrap(pose[:, 2])
    pose[:, 3] = np.log(np.maximum(pose[:, 3], 1e-6))
    return pose  # x[px], y[px], theta[rad], log-scale


def highfreq_report(pose: np.ndarray, width: int, height: int) -> dict:
    """Second-difference RMS (jerk) and spectral HF energy share.

    Rotation / scale channels are converted to pixel-equivalent displacement
    using half the frame diagonal so the combined index has a px meaning.
    """
    diag = math.hypot(width, height)
    rad = diag * 0.5
    if len(pose) >= 3:
        d2 = np.diff(pose, n=2, axis=0)
    else:
        d2 = np.zeros((1, 4))
    chan = {
        "x_px": float(np.sqrt(np.mean(d2[:, 0] ** 2))),
        "y_px": float(np.sqrt(np.mean(d2[:, 1] ** 2))),
        "angle_deg": float(np.degrees(np.sqrt(np.mean(d2[:, 2] ** 2)))),
        "scale_pct_log": float(100.0 * np.sqrt(np.mean(d2[:, 3] ** 2))),
    }
    px_eq = np.column_stack(
        [d2[:, 0], d2[:, 1], d2[:, 2] * rad, d2[:, 3] * rad]
    )
    hf_index_px = float(np.sqrt(np.mean(px_eq**2)))

    # Spectral share above ~1.8 Hz at 30 fps (0.06 cycles/frame).
    cutoff = 0.06
    shares = []
    for c in range(4):
        x = pose[:, c] - np.mean(pose[:, c])
        if len(x) >= 16:
            spec = np.abs(np.fft.rfft(x)) ** 2
            freqs = np.fft.rfftfreq(len(x), d=1.0)
            total = spec.sum() + 1e-12
            shares.append(float(spec[freqs >= cutoff].sum() / total))
    hf_spectral_share = float(np.mean(shares)) if shares else 0.0

    return {
        "second_diff_rms": chan,
        "hf_index_px": hf_index_px,
        "hf_spectral_share": hf_spectral_share,
    }


def trajectory_rmse_vs_gt(est_mats, gt_mats, gauge_normalize: bool = True) -> dict:
    """Compare cumulative paths.

    Both trajectories are gauge-fixed to frame 0 unless requested otherwise:
    relative matrices ``G_i @ inv(G_0)`` describe the frame0 -> frame-i map,
    which is what the estimator produces (its T_0 = I).
    """
    n = min(len(est_mats), len(gt_mats))
    est = [np.asarray(m, dtype=np.float64) for m in est_mats[:n]]
    gt = [np.asarray(m, dtype=np.float64) for m in gt_mats[:n]]
    if gauge_normalize:
        gt0_inv = np.linalg.inv(gt[0])
        gt = [g @ gt0_inv for g in gt]
        est0_inv = np.linalg.inv(est[0])
        est = [e @ est0_inv for e in est]
    est_pose = pose_series_from_mats(est)
    gt_pose = pose_series_from_mats(gt)
    dx = est_pose[:, 0] - gt_pose[:, 0]
    dy = est_pose[:, 1] - gt_pose[:, 1]
    da = np.degrees(
        np.arctan2(
            np.sin(est_pose[:, 2] - gt_pose[:, 2]),
            np.cos(est_pose[:, 2] - gt_pose[:, 2]),
        )
    )
    ds = (np.exp(est_pose[:, 3]) - np.exp(gt_pose[:, 3])) * 100.0
    return {
        "frames": n,
        "tx_rmse_px": float(np.sqrt(np.mean(dx**2))),
        "ty_rmse_px": float(np.sqrt(np.mean(dy**2))),
        "translation_rmse_px": float(np.sqrt(np.mean(dx**2 + dy**2))),
        "angle_rmse_deg": float(np.sqrt(np.mean(da**2))),
        "scale_rmse_pct": float(np.sqrt(np.mean(ds**2))),
        "tx_max_abs_px": float(np.max(np.abs(dx))),
    }


def fov_report(zoom: np.ndarray, alpha: np.ndarray, sacrificed: np.ndarray,
               z_cap: float) -> dict:
    fov = 1.0 / np.asarray(zoom)
    return {
        "mean_retained_fov": float(np.mean(fov)),
        "min_retained_fov": float(np.min(fov)),
        "mean_retained_area": float(np.mean(fov**2)),
        "worst_retained_area": float(np.min(fov**2)),
        "mean_zoom": float(np.mean(zoom)),
        "max_zoom": float(np.max(zoom)),
        "declared_max_zoom": float(z_cap),
        "sacrificed_frames": int(np.sum(sacrificed)),
        "sacrificed_fraction": float(np.mean(sacrificed)),
        "mean_correction_strength": float(np.mean(alpha)),
    }


def count_sentinel_pixels(frame_bgr: np.ndarray, sentinel=(255, 0, 255),
                          tol: int = 8) -> int:
    s = np.array(sentinel, dtype=np.int16)
    d = np.abs(frame_bgr.astype(np.int16) - s).max(axis=2)
    return int(np.count_nonzero(d <= tol))


def motion_direction_score(centers_x: list[float]) -> dict:
    """Monotonic-trend evidence for a moving object (temporal order preserved)."""
    x = np.asarray(centers_x, dtype=np.float64)
    if len(x) < 4:
        return {"frames": len(x)}
    t = np.arange(len(x))
    slope = float(np.polyfit(t, x, 1)[0])
    corr = float(np.corrcoef(t, x)[0, 1])
    return {
        "frames": len(x),
        "slope_px_per_frame": slope,
        "trend_corr": corr,
        "net_displacement_px": float(x[-1] - x[0]),
        "monotonic_run_fraction": float(
            max(np.mean(np.diff(x) >= -1.0), np.mean(np.diff(x) <= 1.0))
        ),
    }
