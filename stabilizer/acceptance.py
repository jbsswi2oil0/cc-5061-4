"""对抗性轨迹验收套件：把验收器、测试场与真实稳定内核串成可判定的验收流程。

Cases
-----
* ``boundary_pixel_center`` / ``boundary_half_pixel``
  参考边分别落在像素中心与半像素位置，校准插值采样越界模型。
* ``shortest_pan_search``
  搜索使画面露边的最短摇移（解析 + 二分 + 实渲三重确认）并反推裁切窗口。
* ``jitter_field`` / ``pan_field``
  真值视场测试场（边缘标记环）分别走小幅高频抖动与长距离摇移，真实内核
  求解轨迹后作用到实际帧，逐帧读回覆盖边界；空像素风险与有效视野损失
  分开判定。
* ``shuffled_reset``
  打乱观测顺序 + 注入一次状态复位，验收器逐帧判定必须保持不变。
* ``reversed_path``
  倒序播放同一路径，边界结论必须一致。
* ``forced_interpolation_oob`` / ``forced_edge_exposed`` / ``forced_insufficient_fov``
  人为制造插值采样越界、露边与视野不足，确认验收器能分别指出各类失败，
  并还原对应源帧、相机变换与裁切矩形。
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Optional, Sequence

import cv2
import numpy as np

from .geometry import (
    apply_aff,
    as33,
    build_plan,
    output_corners,
    zoom_about_center,
)
from .pipeline import compute_plan, strength_from_slider
from .testfield import (
    H as TF_H,
    TESTFIELD_INFO,
    W as TF_W,
    detect_markers,
    ensure_testfield_samples,
    inject_state_reset,
    reverse_path,
    shuffle_observation_order,
)
from .tracking import analyze_video
from .validator import (
    PIX_CLEAN,
    PIX_CONTAMINATED,
    PIX_EMPTY,
    VERDICT_EDGE_EXPOSED,
    VERDICT_INTERP_OOB,
    VERDICT_INSUFFICIENT_FOV,
    VERDICT_OK,
    bisect_exposing_pan,
    check_border_correspondence,
    count_sentinel_pixels,
    crop_window_for_pan,
    pan_exposure_distances,
    reference_edge_cases,
    render_sentinel_frame,
    reverse_lookup_mask,
    save_failure_artifact,
    validate_frame,
    zoomed_translation,
)

CASE_TITLES = {
    "boundary_pixel_center": "参考边落在像素中心",
    "boundary_half_pixel": "参考边落在半像素位置",
    "shortest_pan_search": "最短露边摇移搜索与裁切窗口反推",
    "jitter_field": "测试场 · 小幅高频抖动",
    "pan_field": "测试场 · 长距离摇移",
    "shuffled_reset": "打乱观测顺序 + 状态复位",
    "reversed_path": "倒序播放同一路径",
    "forced_interpolation_oob": "人为插值采样越界（应被指出）",
    "forced_edge_exposed": "人为露边（应被指出）",
    "forced_insufficient_fov": "人为视野不足（应被指出）",
}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def _read_all_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    frames = []
    try:
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            frames.append(fr)
    finally:
        cap.release()
    return frames


def _load_gt(kind: str) -> dict:
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "samples", TESTFIELD_INFO[kind]["gt"])
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _marker_readback(rendered: np.ndarray, K: np.ndarray, gt_mat: np.ndarray,
                     spec: dict, w: int, h: int) -> dict:
    """读回覆盖边界：实测标记可见集合 vs 裁切四边形解析预测。

    ``spec["positions"]`` 是标记在参考帧（T=I）的帧内坐标；真值矩阵
    ``T_i`` 把参考帧坐标映到第 i 帧，故第 i 帧源图坐标为 ``T_i @ p_ref``。
    """
    ref_pos = np.array(spec["positions"], dtype=np.float64)
    src_pos = apply_aff(as33(gt_mat), ref_pos)  # marker pos in source frame i
    band = spec["radius"] + 3.0
    predicted, uncertain = set(), set()
    expected_xy = {}
    for mid, p in enumerate(src_pos):
        ox, oy = apply_aff(as33(K), np.array([p], dtype=np.float64))[0]
        expected_xy[mid] = (float(ox), float(oy))
        if band <= ox <= w - band and band <= oy <= h - band:
            predicted.add(mid)
        elif ox < -band or ox > w + band or oy < -band or oy > h + band:
            pass  # confidently gone
        else:
            uncertain.add(mid)
    measured = detect_markers(rendered, spec)
    missing = sorted(predicted - set(measured))
    unexpected = sorted(set(measured) - predicted - uncertain)
    pos_err = [
        math.hypot(measured[m][0] - expected_xy[m][0],
                   measured[m][1] - expected_xy[m][1])
        for m in predicted
        if m in measured
    ]
    return {
        "predicted_visible": len(predicted),
        "measured_visible": len(set(measured) & (predicted | uncertain)),
        "missing": missing,
        "unexpected": unexpected,
        "uncertain_boundary": len(uncertain),
        "max_pos_err_px": round(max(pos_err), 2) if pos_err else None,
        "ok": not missing and not unexpected,
    }


def _validate_plan_frames(frames: list[np.ndarray], Ks, zooms, min_fov,
                          gt_mats=None, marker_spec=None,
                          artifact_dir: Optional[str] = None, case: str = "",
                          order: Optional[Sequence[int]] = None) -> dict:
    """把求得的轨迹作用到实际帧，逐帧读回覆盖边界并分开判定两类风险。"""
    w, h = frames[0].shape[1], frames[0].shape[0]
    n = len(frames)
    idx_order = list(order) if order is not None else list(range(n))
    per_frame: list[Optional[dict]] = [None] * n
    worst_margin = float("inf")
    worst_corr = 0.0
    worst_fov = float("inf")
    marker_stats = {"checked": 0, "mismatched": 0, "max_pos_err_px": 0.0}
    failures = []

    for i in idx_order:
        K = as33(Ks[i])
        rendered = render_sentinel_frame(frames[i], K)
        v = validate_frame(K, w, h, zoom=float(zooms[i]), min_fov=min_fov,
                           rendered=rendered, check_render=True)
        v["sentinel_pixels"] = count_sentinel_pixels(rendered)
        if gt_mats is not None and marker_spec is not None:
            rb = _marker_readback(rendered, K, gt_mats[i], marker_spec, w, h)
            v["coverage_readback"] = rb
            marker_stats["checked"] += 1
            if not rb["ok"]:
                marker_stats["mismatched"] += 1
            if rb["max_pos_err_px"] is not None:
                marker_stats["max_pos_err_px"] = max(
                    marker_stats["max_pos_err_px"], rb["max_pos_err_px"])
        per_frame[i] = v

        risk = v["empty_pixel_risk"]
        worst_margin = min(worst_margin, risk["margin_px"])
        worst_corr = max(worst_corr, risk.get("border_correspondence_px", 0.0))
        if "retained_fov" in v["fov_loss"]:
            worst_fov = min(worst_fov, v["fov_loss"]["retained_fov"])

        bad = (risk["status"] != VERDICT_OK
               or v["fov_loss"]["status"] != VERDICT_OK
               or ("coverage_readback" in v and not v["coverage_readback"]["ok"]))
        if bad and artifact_dir is not None and len(failures) < 4:
            art = save_failure_artifact(
                artifact_dir, case, i, frames[i], K, w, h,
                reason=f"{risk['status']}|{v['fov_loss']['status']}",
                rendered=rendered,
                extra={"zoom": float(zooms[i]),
                       "sentinel_pixels": v["sentinel_pixels"]},
            )
            failures.append(art)

    per_frame = [v for v in per_frame]
    agg = {
        "frames": n,
        "empty_pixel_risk": {
            "status": VERDICT_OK,
            "worst_margin_px": round(worst_margin, 3),
            "max_border_correspondence_px": round(worst_corr, 3),
            "frames_with_empty": sum(
                1 for v in per_frame
                if v["empty_pixel_risk"]["analytic_empty_pixels"] > 0),
            "frames_with_contamination": sum(
                1 for v in per_frame
                if v["empty_pixel_risk"]["analytic_contaminated_pixels"] > 0),
            "sentinel_pixels_total": int(
                sum(v["sentinel_pixels"] for v in per_frame)),
        },
        "fov_loss": {
            "status": VERDICT_OK,
            "min_retained_fov": (round(worst_fov, 5)
                                 if math.isfinite(worst_fov) else None),
            "declared_min_fov": min_fov,
            "frames_below_declared": sum(
                1 for v in per_frame
                if v["fov_loss"]["status"] == VERDICT_INSUFFICIENT_FOV),
        },
        "marker_readback": marker_stats,
        "per_frame": per_frame,
        "failures": failures,
    }
    for v in per_frame:
        if v["empty_pixel_risk"]["status"] != VERDICT_OK:
            agg["empty_pixel_risk"]["status"] = v["empty_pixel_risk"]["status"]
            break
    if agg["fov_loss"]["frames_below_declared"] > 0:
        agg["fov_loss"]["status"] = VERDICT_INSUFFICIENT_FOV
    return agg


def _min_zoom(C: np.ndarray, w: int, h: int, target: int) -> float:
    """Minimal zoom whose sampling has no pixel at or below ``target``."""
    def ok(z):
        K = zoom_about_center(w, h, z) @ C
        return not bool(np.any(reverse_lookup_mask(K, w, h) <= target))

    lo, hi = 1.0, 1.02
    while not ok(hi):
        hi *= 1.4
        if hi > 64:
            return float("inf")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _case(cid: str, status: str, summary: str, **kw) -> dict:
    return {
        "id": cid,
        "title": CASE_TITLES[cid],
        "status": status,
        "summary": summary,
        "failures": kw.pop("failures", []),
        **kw,
    }


# ---------------------------------------------------------------------- cases

def case_reference_edges(w: int, h: int) -> list[dict]:
    """像素中心 / 半像素两条参考边，校准插值采样越界模型。"""
    out = []
    for spec in reference_edge_cases(w, h):
        cid = "boundary_" + spec["id"]
        K = spec["K"]
        corr = check_border_correspondence(K, w, h, tol=0.0)
        mask = reverse_lookup_mask(K, w, h)
        empty_cols = 0
        for x in range(w):
            if np.all(mask[:, x] == PIX_EMPTY):
                empty_cols += 1
            else:
                break
        contam_cols = 0
        for x in range(empty_cols, w):
            if np.all(mask[:, x] == PIX_CONTAMINATED):
                contam_cols += 1
            else:
                break
        ok = (
            corr["ok"]
            and empty_cols == spec["expect_empty_cols_left"]
            and contam_cols == spec["expect_contaminated_cols"]
        )
        out.append(_case(
            cid, "pass" if ok else "fail",
            f"空列 {empty_cols}（期望 {spec['expect_empty_cols_left']}），"
            f"插值污染列 {contam_cols}（期望 {spec['expect_contaminated_cols']}），"
            f"解析/实渲边界偏差 {corr['correspondence_px']:.2f}px",
            details={
                "expect_empty_cols_left": spec["expect_empty_cols_left"],
                "expect_contaminated_cols": spec["expect_contaminated_cols"],
                "empty_cols_left": empty_cols,
                "contaminated_cols": contam_cols,
                "correspondence": corr,
            },
        ))
    return out


def case_shortest_pan(w: int, h: int, zoom: float = 1.10) -> dict:
    """最短露边摇移：解析解、二分搜索与实渲三者必须对应；再反推裁切窗口。"""
    dist = pan_exposure_distances(w, h, zoom)
    shortest = dist["shortest_empty"]
    d_analytic = shortest["distance"]
    dx, dy = shortest["direction"]
    d_num = bisect_exposing_pan(w, h, zoom, (dx, dy), target=PIX_EMPTY)
    d_num_contam = bisect_exposing_pan(
        w, h, zoom, dist["shortest_contamination"]["direction"],
        target=PIX_CONTAMINATED)

    # 实渲确认：略低于阈值无空像素，略高于阈值出现空像素
    K_safe = zoomed_translation(w, h, zoom, dx * (d_analytic - 0.3),
                                dy * (d_analytic - 0.3))
    K_over = zoomed_translation(w, h, zoom, dx * (d_analytic + 0.3),
                                dy * (d_analytic + 0.3))
    m_safe = reverse_lookup_mask(K_safe, w, h)
    m_over = reverse_lookup_mask(K_over, w, h)
    render_ok = (not np.any(m_safe == PIX_EMPTY)) and np.any(m_over == PIX_EMPTY)

    # 反推裁切窗口：让该摇移恰好不露边的最小放大
    tx, ty = shortest["vector"]
    win = crop_window_for_pan(w, h, tx, ty)
    win_ok = (
        win["feasible"]
        and abs(win["zoom"] - win["analytic_zoom"]) < 5e-3
    )
    K_win = win.get("K")
    win_clean = bool(K_win is not None
                     and np.all(reverse_lookup_mask(K_win, w, h) == PIX_CLEAN))

    analytic_vs_numeric = abs(d_analytic - d_num)
    interp_gap = d_analytic - d_num_contam
    ok = (
        analytic_vs_numeric <= 0.05
        and render_ok
        and win_ok
        and win_clean
        and 0.9 <= interp_gap <= 1.1  # 插值越界总是恰好早 1px
    )
    return _case(
        "shortest_pan_search", "pass" if ok else "fail",
        f"最短露边摇移 {d_analytic:.2f}px（二分 {d_num:.2f}px，差 "
        f"{analytic_vs_numeric:.3f}px）；插值越界提前 {interp_gap:.2f}px；"
        f"反推裁切窗口 zoom×{win.get('zoom', float('nan')):.4f}"
        f"（FOV {100.0 / max(win.get('zoom') or 1.0, 1e-9):.1f}%）",
        details={
            "zoom": zoom,
            "analytic": dist,
            "numeric_empty_distance": d_num,
            "numeric_contamination_distance": d_num_contam,
            "render_check": render_ok,
            "crop_window": {
                "zoom": win.get("zoom"),
                "analytic_zoom": win.get("analytic_zoom"),
                "retained_fov": win.get("retained_fov"),
                "feasible": win["feasible"],
                "clean_after_render": win_clean,
            },
        },
    )


def _field_case(cid: str, kind: str, smoothness: int, min_fov: float,
                artifact_dir: str) -> tuple:
    """测试场片段走真实内核：求解轨迹 -> 作用到实际帧 -> 逐帧读回判定。"""
    video = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "samples", TESTFIELD_INFO[kind]["file"])
    gt = _load_gt(kind)
    an = analyze_video(video)
    plan = compute_plan(an, smoothness, min_fov)
    frames = _read_all_frames(video)
    spec = gt["marker_ring"]
    gt_mats = [np.array(m, dtype=np.float64) for m in gt["mats"]]

    agg = _validate_plan_frames(
        frames, plan["K"], plan["zoom"], min_fov,
        gt_mats=gt_mats, marker_spec=spec,
        artifact_dir=artifact_dir, case=cid)
    risk = agg["empty_pixel_risk"]
    fov = agg["fov_loss"]
    ok = (
        risk["status"] == VERDICT_OK
        and fov["status"] == VERDICT_OK
        and agg["marker_readback"]["mismatched"] == 0
    )
    case = _case(
        cid, "pass" if ok else "fail",
        f"{agg['frames']} 帧：空像素帧 {risk['frames_with_empty']}，"
        f"插值越界帧 {risk['frames_with_contamination']}，"
        f"最差安全边距 {risk['worst_margin_px']:.2f}px，"
        f"最低保留视野 {100 * (fov['min_retained_fov'] or 0):.1f}%"
        f"（下限 {100 * min_fov:.0f}%），"
        f"标记读回不一致帧 {agg['marker_readback']['mismatched']}",
        empty_pixel_risk={k: v for k, v in risk.items()},
        fov_loss={k: v for k, v in fov.items()},
        details={
            "video": os.path.basename(video),
            "smoothness": smoothness,
            "min_fov": min_fov,
            "sacrificed_frames": int(np.sum(plan["sacrificed"])),
            "marker_readback": agg["marker_readback"],
        },
        failures=agg["failures"],
    )
    ctx = {
        "plan": plan, "frames": frames, "gt_mats": gt_mats, "spec": spec,
        "analysis": an, "agg": agg,
    }
    return case, ctx


def case_shuffled_reset(ctx: dict, min_fov: float, smoothness: int,
                        artifact_dir: str) -> dict:
    """打乱观测顺序 + 注入一次状态复位：逐帧判定必须保持不变。"""
    plan, frames, gt_mats = ctx["plan"], ctx["frames"], ctx["gt_mats"]
    spec, an = ctx["spec"], ctx["analysis"]
    w, h, n = an.width, an.height, len(frames)

    # 1) 打乱观测顺序：同一组 (帧, K) 按置换顺序验收，结论必须逐帧一致
    order = shuffle_observation_order(n)
    fwd = _validate_plan_frames(frames, plan["K"], plan["zoom"], min_fov)
    shuffled = _validate_plan_frames(frames, plan["K"], plan["zoom"], min_fov,
                                     order=order)
    same = all(
        a["empty_pixel_risk"]["status"] == b["empty_pixel_risk"]["status"]
        and a["fov_loss"]["status"] == b["fov_loss"]["status"]
        and a["empty_pixel_risk"]["analytic_empty_pixels"]
        == b["empty_pixel_risk"]["analytic_empty_pixels"]
        for a, b in zip(fwd["per_frame"], shuffled["per_frame"])
    )

    # 2) 状态复位：在真值轨迹第 k 帧注入复位，重新规划后作用到实际帧
    k = n // 2
    traj_reset = inject_state_reset(gt_mats, k)
    plan_reset = build_plan(traj_reset, w, h,
                            strength=strength_from_slider(smoothness),
                            min_fov=min_fov, model=an.model)
    # 复位检测：原始位姿的平移跳变必须在注入帧处最大
    pose = plan_reset["orig_pose"]
    jumps = np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)
    detected_at = int(np.argmax(jumps)) + 1
    reset_ok = abs(detected_at - k) <= 1
    agg_reset = _validate_plan_frames(
        frames, plan_reset["K"], plan_reset["zoom"], min_fov,
        gt_mats=gt_mats, marker_spec=spec,
        artifact_dir=artifact_dir, case="shuffled_reset")
    reset_clean = (agg_reset["empty_pixel_risk"]["status"] == VERDICT_OK
                   and agg_reset["fov_loss"]["status"] == VERDICT_OK)

    ok = same and reset_ok and reset_clean
    return _case(
        "shuffled_reset", "pass" if ok else "fail",
        f"乱序验收逐帧结论一致：{same}；复位注入于帧 {k}，"
        f"检测定位于帧 {detected_at}；复位后空像素帧 "
        f"{agg_reset['empty_pixel_risk']['frames_with_empty']}，"
        f"视野违规帧 {agg_reset['fov_loss']['frames_below_declared']}",
        details={
            "shuffle_order_consistent": same,
            "reset_injected_at": k,
            "reset_detected_at": detected_at,
            "reset_max_jump_px": round(float(jumps.max()), 2),
            "reset_empty_pixel_risk": agg_reset["empty_pixel_risk"]["status"],
            "reset_fov_loss": agg_reset["fov_loss"]["status"],
        },
        failures=agg_reset["failures"],
    )


def case_reversed_path(ctx: dict, min_fov: float, smoothness: int,
                       artifact_dir: str) -> dict:
    """倒序播放同一路径：边界结论必须与正序一致。"""
    frames, gt_mats = ctx["frames"], ctx["gt_mats"]
    spec, an = ctx["spec"], ctx["analysis"]
    w, h = an.width, an.height
    traj_rev = reverse_path(gt_mats)
    plan_rev = build_plan(traj_rev, w, h,
                          strength=strength_from_slider(smoothness),
                          min_fov=min_fov, model=an.model)
    agg = _validate_plan_frames(
        list(reversed(frames)), plan_rev["K"], plan_rev["zoom"], min_fov,
        gt_mats=list(reversed(gt_mats)), marker_spec=spec,
        artifact_dir=artifact_dir, case="reversed_path")
    ok = (agg["empty_pixel_risk"]["status"] == VERDICT_OK
          and agg["fov_loss"]["status"] == VERDICT_OK
          and agg["marker_readback"]["mismatched"] == 0)
    return _case(
        "reversed_path", "pass" if ok else "fail",
        f"倒序 {agg['frames']} 帧：空像素帧 "
        f"{agg['empty_pixel_risk']['frames_with_empty']}，"
        f"最差安全边距 {agg['empty_pixel_risk']['worst_margin_px']:.2f}px，"
        f"最低保留视野 {100 * (agg['fov_loss']['min_retained_fov'] or 0):.1f}%，"
        f"标记读回不一致帧 {agg['marker_readback']['mismatched']}",
        empty_pixel_risk=agg["empty_pixel_risk"],
        fov_loss=agg["fov_loss"],
        failures=agg["failures"],
    )


def case_forced_failures(ctx: dict, min_fov: float,
                         artifact_dir: str) -> list[dict]:
    """人为制造三类失败：验收器必须分别指出，且空像素风险与视野损失互不染指。"""
    plan, frames, an = ctx["plan"], ctx["frames"], ctx["analysis"]
    w, h = an.width, an.height
    i = int(np.argmax(plan["z_raw"]))  # 校正需求最大的一帧
    C = as33(plan["corr_mats"][i])
    z_clean = _min_zoom(C, w, h, PIX_CONTAMINATED)  # 低于此即有插值越界
    z_noempty = _min_zoom(C, w, h, PIX_EMPTY)       # 低于此即出现全空像素

    out = []
    # A) 插值采样越界：zoom 落在 (z_noempty, z_clean) 之间 -> 越界但无全空；
    #    声明的视野下限放宽到该 zoom 可容纳，确保只有插值越界一类失败。
    z_bad = 0.5 * (z_noempty + z_clean)
    fov_ok_here = 1.0 / z_bad - 0.01
    K_bad = zoom_about_center(w, h, z_bad) @ C
    rendered = render_sentinel_frame(frames[i], K_bad)
    v = validate_frame(K_bad, w, h, zoom=z_bad, min_fov=fov_ok_here,
                       rendered=rendered)
    status = v["empty_pixel_risk"]["status"]
    sentinel = count_sentinel_pixels(rendered)
    detected = (status == VERDICT_INTERP_OOB and sentinel == 0
                and v["fov_loss"]["status"] == VERDICT_OK)
    art = save_failure_artifact(
        artifact_dir, "forced_interpolation_oob", i, frames[i], K_bad, w, h,
        reason=status, rendered=rendered,
        extra={"zoom": z_bad, "z_clean": z_clean, "z_noempty": z_noempty,
               "sentinel_pixels": sentinel})
    out.append(_case(
        "forced_interpolation_oob", "pass" if detected else "fail",
        f"帧 {i}：zoom×{z_bad:.4f} ∈ (×{z_noempty:.4f}, ×{z_clean:.4f}) → "
        f"判定 {status}（全空像素 {sentinel}，视野判定独立为 "
        f"{v['fov_loss']['status']}）；源帧/相机变换/裁切矩形已还原",
        empty_pixel_risk={"status": status},
        fov_loss={"status": v["fov_loss"]["status"]},
        details={"frame": i, "zoom": z_bad, "z_clean": z_clean,
                 "z_noempty": z_noempty, "verdict": status,
                 "sentinel_pixels": sentinel,
                 "fov_verdict": v["fov_loss"]["status"]},
        failures=[art],
    ))

    # B) 露边：zoom 明显低于 z_noempty -> 出现全空像素
    z_low = 0.94 * z_noempty
    K_low = zoom_about_center(w, h, z_low) @ C
    rendered = render_sentinel_frame(frames[i], K_low)
    v = validate_frame(K_low, w, h, zoom=z_low, min_fov=min_fov,
                       rendered=rendered)
    status = v["empty_pixel_risk"]["status"]
    sentinel = count_sentinel_pixels(rendered)
    detected = status == VERDICT_EDGE_EXPOSED and sentinel > 0
    art = save_failure_artifact(
        artifact_dir, "forced_edge_exposed", i, frames[i], K_low, w, h,
        reason=status, rendered=rendered,
        extra={"zoom": z_low, "z_noempty": z_noempty,
               "sentinel_pixels": sentinel})
    out.append(_case(
        "forced_edge_exposed", "pass" if detected else "fail",
        f"帧 {i}：zoom×{z_low:.4f} < ×{z_noempty:.4f} → 判定 {status}，"
        f"哨兵空像素 {sentinel}；源帧/相机变换/裁切矩形已还原",
        empty_pixel_risk={"status": status},
        fov_loss={"status": v["fov_loss"]["status"]},
        details={"frame": i, "zoom": z_low, "z_noempty": z_noempty,
                 "verdict": status, "sentinel_pixels": sentinel},
        failures=[art],
    ))

    # C) 视野不足：放大足够覆盖（零空像素）但保留视野低于声明下限
    K_full = zoom_about_center(w, h, z_clean) @ C
    declared = min(0.99, 1.0 / z_clean + 0.05)
    rendered = render_sentinel_frame(frames[i], K_full)
    v = validate_frame(K_full, w, h, zoom=z_clean, min_fov=declared,
                       rendered=rendered)
    sentinel = count_sentinel_pixels(rendered)
    detected = (v["fov_loss"]["status"] == VERDICT_INSUFFICIENT_FOV
                and v["empty_pixel_risk"]["status"] == VERDICT_OK
                and sentinel == 0)
    art = save_failure_artifact(
        artifact_dir, "forced_insufficient_fov", i, frames[i], K_full, w, h,
        reason=VERDICT_INSUFFICIENT_FOV, rendered=rendered,
        extra={"zoom": z_clean, "declared_min_fov": declared,
               "retained_fov": 1.0 / z_clean, "sentinel_pixels": sentinel})
    out.append(_case(
        "forced_insufficient_fov", "pass" if detected else "fail",
        f"帧 {i}：zoom×{z_clean:.4f} 覆盖洁净（空像素 {sentinel}），"
        f"但保留视野 {100.0 / z_clean:.1f}% < 声明下限 {100 * declared:.0f}% → "
        f"判定 {v['fov_loss']['status']}；空像素风险独立判定为 "
        f"{v['empty_pixel_risk']['status']}",
        empty_pixel_risk={"status": v["empty_pixel_risk"]["status"]},
        fov_loss={"status": v["fov_loss"]["status"]},
        details={"frame": i, "zoom": z_clean, "declared_min_fov": declared,
                 "empty_pixel_risk": v["empty_pixel_risk"]["status"],
                 "fov_loss": v["fov_loss"]["status"]},
        failures=[art],
    ))
    return out


# -------------------------------------------------------------------- suite

def run_acceptance(out_dir: str, smoothness: int = 55, min_fov: float = 0.85,
                   progress=None) -> dict:
    """运行完整对抗性轨迹验收，写出 JSON 报告与失败还原 artifacts。"""
    os.makedirs(out_dir, exist_ok=True)
    # 清掉上一轮验收留下的失败还原 artifacts，避免陈旧文件混入本次报告
    for name in os.listdir(out_dir):
        if (name.endswith(("_src.png", "_out.png"))
                or (name.endswith(".json") and "_frame" in name)):
            try:
                os.remove(os.path.join(out_dir, name))
            except OSError:
                pass
    t0 = time.time()
    cases: list[dict] = []
    w, h = TF_W, TF_H

    def tell(frac, msg):
        if progress:
            progress(frac, msg)

    tell(0.02, "校准参考边（像素中心 / 半像素）")
    cases.extend(case_reference_edges(w, h))

    tell(0.08, "搜索最短露边摇移并反推裁切窗口")
    cases.append(case_shortest_pan(w, h, zoom=1.10))

    tell(0.14, "生成/载入真值视场测试场片段")
    ensure_testfield_samples()

    tell(0.22, "测试场：小幅高频抖动（真实内核求解 + 逐帧判定）")
    jitter_case, _jitter_ctx = _field_case(
        "jitter_field", "testfield_jitter", smoothness, min_fov, out_dir)
    cases.append(jitter_case)

    tell(0.48, "测试场：长距离摇移（真实内核求解 + 逐帧判定）")
    pan_case, pan_ctx = _field_case(
        "pan_field", "testfield_pan", smoothness, min_fov, out_dir)
    cases.append(pan_case)

    tell(0.66, "打乱观测顺序 + 注入状态复位")
    cases.append(case_shuffled_reset(pan_ctx, min_fov, smoothness, out_dir))

    tell(0.80, "倒序播放同一路径")
    cases.append(case_reversed_path(pan_ctx, min_fov, smoothness, out_dir))

    tell(0.90, "人为制造插值越界 / 露边 / 视野不足，检查分别判定")
    cases.extend(case_forced_failures(pan_ctx, min_fov, out_dir))

    verdict = "pass" if all(c["status"] == "pass" for c in cases) else "fail"
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "verdict": verdict,
        "params": {"smoothness": smoothness, "min_fov": min_fov},
        "cases": cases,
    }
    tmp = os.path.join(out_dir, "acceptance.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, ensure_ascii=False)
    os.replace(tmp, os.path.join(out_dir, "acceptance.json"))
    tell(1.0, "完成")
    return report
