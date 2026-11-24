"""End-to-end stabilization pipeline and JSON-safe result assembly."""

from __future__ import annotations

import json
import math
import os
from typing import Callable, Optional

import cv2
import numpy as np

from .geometry import build_plan, cumulative_trajectory
from .metrics import fov_report, highfreq_report, pose_series_from_mats
from .tracking import MotionAnalysis, analyze_video
from .videoio import FFmpegWriter, has_audio_stream

ProgressCb = Optional[Callable[[float, str], None]]

DEFAULT_PARAMS = {
    "smoothness": 55,   # UI slider 0..100
    "min_fov": 0.85,    # linear retained field of view lower bound
    "model": "similarity",
}


def strength_from_slider(value) -> float:
    return float(np.clip(float(value) / 100.0, 0.0, 1.0))


def trajectory_of(analysis: MotionAnalysis):
    return cumulative_trajectory(analysis.motions)


def compute_plan(analysis: MotionAnalysis, smoothness, min_fov):
    strength = strength_from_slider(smoothness)
    traj = trajectory_of(analysis)
    return build_plan(
        traj,
        analysis.width,
        analysis.height,
        strength=strength,
        min_fov=float(min_fov),
        model=analysis.model,
    )


def _r(x, nd=2):
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if not math.isfinite(x) else round(x, nd)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def plan_frames_json(analysis: MotionAnalysis, plan: dict) -> list[dict]:
    n = analysis.n_frames
    op, tp = plan["orig_pose"], plan["target_pose"]
    pair_by_frame = {p.index: p for p in analysis.pairs}
    frames = []
    for i in range(n):
        pair = pair_by_frame.get(i)
        quad = np.round(plan["quad_src"][i], 1).tolist()
        row = {
            "i": i,
            "ox": _r(op[i, 0], 2),
            "oy": _r(op[i, 1], 2),
            "oa": _r(math.degrees(op[i, 2]), 3),
            "os": _r(math.exp(op[i, 3]), 5),
            "tx": _r(tp[i, 0], 2),
            "ty": _r(tp[i, 1], 2),
            "ta": _r(math.degrees(tp[i, 2]), 3),
            "ts": _r(math.exp(tp[i, 3]), 5),
            "zoom": _r(plan["zoom"][i], 4),
            "fov": _r(plan["fov"][i], 4),
            "alpha": _r(plan["alpha"][i], 3),
            "sacrificed": bool(
                plan["sacrificed"][i] or plan["alpha"][i] < 0.985
            ),
            "quad": quad,
        }
        if pair is not None:
            row.update(
                {
                    "ok": pair.ok,
                    "imputed": pair.imputed,
                    "method": pair.method,
                    "n_verified": pair.n_verified,
                    "n_inliers": pair.n_inliers,
                    "inlier_ratio": _r(pair.inlier_ratio, 3),
                    "rmse_in": _r(pair.rmse_inliers_px, 3),
                    "residual_all": _r(pair.residual_rms_all_px, 3),
                    "residual_p90": _r(pair.residual_p90_px, 3),
                    "bad_fraction": _r(pair.bad_fraction, 3),
                    "spatial_std": _r(pair.spatial_residual_std_px, 3),
                    "warning": pair.error,
                }
            )
        else:
            row.update({"ok": True, "imputed": False, "method": "first-frame"})
        frames.append(row)
    return frames


def plan_metrics(analysis: MotionAnalysis, plan: dict) -> dict:
    traj = trajectory_of(analysis)
    orig_pose = pose_series_from_mats(traj)
    target_pose = pose_series_from_mats(plan["target_mats"])
    diag = math.hypot(analysis.width, analysis.height)

    orig_hf = highfreq_report(orig_pose, analysis.width, analysis.height)
    target_hf = highfreq_report(target_pose, analysis.width, analysis.height)
    fov = fov_report(
        plan["zoom"], plan["alpha"], plan["sacrificed"], plan["z_cap"]
    )

    # geometric coverage violations (must be zero)
    violations = 0
    margin = 0.4
    for i in range(analysis.n_frames):
        q = plan["quad_src"][i]
        if (
            np.any(q[:, 0] < -margin)
            or np.any(q[:, 0] > analysis.width + margin)
            or np.any(q[:, 1] < -margin)
            or np.any(q[:, 1] > analysis.height + margin)
        ):
            violations += 1

    summary = analysis.summary()
    parallax_flag = _parallax_flag(summary, analysis.pairs, diag, analysis.model)
    return {
        "orig_hf": orig_hf,
        "target_hf": target_hf,
        "fov": fov,
        "coverage_violations": violations,
        "analysis": summary,
        "parallax": parallax_flag,
    }


def _parallax_flag(summary: dict, pairs, diag: float, model: str = "similarity") -> dict:
    """Heuristic, calibrated against the bundled parallax sample."""
    res = [p.residual_rms_all_px for p in pairs if p.n_verified > 0]
    spa = [p.spatial_residual_std_px for p in pairs if p.n_verified > 0]
    inl = [p.inlier_ratio for p in pairs if p.n_verified > 0]
    bad = [p.bad_fraction for p in pairs if p.n_verified > 0]
    p90 = [p.residual_p90_px for p in pairs if p.n_verified > 0]
    mean_res = float(np.mean(res)) if res else 0.0
    p95_res = float(np.percentile(res, 95)) if res else 0.0
    mean_spa = float(np.mean(spa)) if spa else 0.0
    mean_inl = float(np.mean(inl)) if inl else 1.0
    mean_bad = float(np.mean(bad)) if bad else 0.0
    mean_p90 = float(np.mean(p90)) if p90 else 0.0
    rel = mean_res / diag
    score = min(
        1.0,
        0.45 * min(1.0, mean_bad / 0.15)
        + 0.35 * min(1.0, mean_spa / 4.0)
        + 0.20 * min(1.0, rel / 0.006),
    )
    level = "low"
    if mean_bad >= 0.12 or mean_spa >= 2.5:
        level = "high"
    elif mean_bad >= 0.03 or mean_spa >= 1.0 or mean_p90 >= 2.5:
        level = "medium"
    return {
        "level": level,
        "score": round(score, 3),
        "mean_model_residual_px": round(mean_res, 2),
        "p95_model_residual_px": round(p95_res, 2),
        "mean_p90_residual_px": round(mean_p90, 2),
        "mean_bad_fraction": round(mean_bad, 3),
        "mean_spatial_std_px": round(mean_spa, 2),
        "mean_inlier_ratio": round(mean_inl, 3),
        "supported_model": (
            "仿射变换（平移/旋转/缩放/剪切，平面运动）"
            if model == "affine"
            else "相似变换（平移/旋转/统一缩放，近似平面运动）"
        ),
        "message": (
            "残差偏高且呈空间分布：存在明显视差或独立运动，单一 2D 变换无法解释所有深度，"
            "稳定结果以主平面为准，近景/前景可能残留运动。"
            if level == "high"
            else (
                "检测到中等模型残差，可能有轻微视差或运动物体；请查看逐帧残差曲线。"
                if level == "medium"
                else "模型残差较低，画面接近平面/近似相机运动假设。"
            )
        ),
    }


def iter_warped_frames(
    source_path: str,
    width: int,
    height: int,
    K_mats: list[np.ndarray],
    border_value=(0, 0, 0),
    progress: ProgressCb = None,
):
    """Yield stabilized BGR frames. Pure generator (tests can consume directly)."""
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"重采样阶段无法打开视频: {source_path}")
    i = 0
    n = len(K_mats)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            Ki = K_mats[min(i, n - 1)]
            warped = cv2.warpAffine(
                frame,
                Ki[:2, :].astype(np.float64),
                (width, height),
                flags=cv2.INTER_LINEAR,  # default: provided M is src->dst
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
            yield i, warped
            i += 1
            if progress is not None and i % 5 == 0:
                progress(min(0.99, i / max(n, 1)), f"重采样输出 {i} 帧")
    finally:
        cap.release()


def render_to_file(
    analysis: MotionAnalysis,
    plan: dict,
    out_path: str,
    progress: ProgressCb = None,
    keep_audio: bool = True,
) -> int:
    audio_src = analysis.path if keep_audio and has_audio_stream(analysis.path) else None
    n_written = 0
    tmp_out = out_path + ".part.mp4"
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    with FFmpegWriter(
        tmp_out,
        analysis.width,
        analysis.height,
        analysis.fps,
        audio_source=audio_src,
    ) as writer:
        for i, frame in iter_warped_frames(
            analysis.path,
            analysis.width,
            analysis.height,
            plan["K"],
        ):
            writer.write(frame)
            n_written += 1
            if progress is not None and n_written % 5 == 0:
                progress(
                    min(0.97, n_written / max(analysis.n_frames, 1)),
                    f"编码输出 {n_written} 帧",
                )
    os.replace(tmp_out, out_path)
    return n_written


def render_plan(
    analysis: MotionAnalysis,
    job_dir: str,
    smoothness,
    min_fov,
    progress: ProgressCb = None,
    verify: bool = True,
) -> dict:
    """Compute plan, render stabilized.mp4, optionally re-analyze the output."""
    os.makedirs(job_dir, exist_ok=True)
    plan = compute_plan(analysis, smoothness, min_fov)
    out_video = os.path.join(job_dir, "stabilized.mp4")

    def rp(frac, msg):
        if progress:
            progress(0.15 + 0.7 * frac, msg)

    n_written = render_to_file(analysis, plan, out_video, rp)

    result = {
        "params": {
            "smoothness": int(smoothness),
            "min_fov": float(min_fov),
            "model": analysis.model,
        },
        "video": "stabilized.mp4",
        "frames": plan_frames_json(analysis, plan),
        "metrics": plan_metrics(analysis, plan),
        "n_written": n_written,
        "verification": None,
    }

    if verify and os.path.exists(out_video):
        if progress:
            progress(0.9, "复核输出视频轨迹与残差")
        try:
            out_an = analyze_video(out_video, model="similarity")
            out_traj = trajectory_of(out_an)
            out_pose = pose_series_from_mats(out_traj)
            result["verification"] = {
                "output_hf": highfreq_report(
                    out_pose, analysis.width, analysis.height
                ),
                "analysis": out_an.summary(),
            }
            before = result["metrics"]["orig_hf"]["hf_index_px"]
            after = result["verification"]["output_hf"]["hf_index_px"]
            result["verification"]["hf_reduction_pct"] = (
                round(100.0 * (1.0 - after / max(before, 1e-9)), 1)
            )
        except Exception as e:  # verification is advisory
            result["verification"] = {"error": str(e)}

    tmp_json = os.path.join(job_dir, "result.json.tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    os.replace(tmp_json, os.path.join(job_dir, "result.json"))
    if progress:
        progress(1.0, "完成")
    return result
