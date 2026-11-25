"""Acceptance checks for the adversarial trajectory validator.

Run:  PYTHONPATH=/workspace python3 tests/test_validator.py

Covers: the exact sampler-domain model (pixel-center vs half-pixel reference
edges), analytic-vs-rendered border correspondence, the shortest
edge-exposing pan search, crop-window back-computation, and the full
acceptance suite on the ground-truth-FOV test field (jitter / pan /
shuffled / reset / reversed / forced failures).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stabilizer.acceptance import run_acceptance  # noqa: E402
from stabilizer.geometry import aff_compose, zoom_about_center  # noqa: E402
from stabilizer.testfield import (  # noqa: E402
    H as TF_H,
    W as TF_W,
    ensure_testfield_samples,
)
from stabilizer.validator import (  # noqa: E402
    PIX_CLEAN,
    PIX_CONTAMINATED,
    PIX_EMPTY,
    bisect_exposing_pan,
    check_border_correspondence,
    crop_window_for_pan,
    exposure_margin,
    pan_exposure_distances,
    reference_edge_cases,
    reverse_lookup_mask,
    translation,
    zoomed_translation,
)

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


def test_reference_edges():
    """像素中心 vs 半像素参考边：插值采样越界模型必须分别精确命中。"""
    print("[reference edges: pixel-center vs half-pixel]")
    w, h = TF_W, TF_H
    for spec in reference_edge_cases(w, h):
        corr = check_border_correspondence(spec["K"], w, h, tol=0.0)
        mask = reverse_lookup_mask(spec["K"], w, h)
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
        check(f"{spec['id']}: analytic border == rendered first empty pixel",
              corr["ok"] and corr["correspondence_px"] == 0.0,
              f"corr={corr['correspondence_px']:.2f}px")
        check(f"{spec['id']}: empty columns == {spec['expect_empty_cols_left']}",
              empty_cols == spec["expect_empty_cols_left"],
              f"got {empty_cols}")
        check(f"{spec['id']}: contaminated columns == {spec['expect_contaminated_cols']}",
              contam_cols == spec["expect_contaminated_cols"],
              f"got {contam_cols}")


def test_shortest_pan_and_crop_window():
    """最短露边摇移：解析、二分、实渲三者对应；反推裁切窗口恰好洁净。"""
    print("[shortest exposing pan + crop window back-computation]")
    w, h = TF_W, TF_H
    zoom = 1.10
    dist = pan_exposure_distances(w, h, zoom)
    d_empty = dist["shortest_empty"]["distance"]
    d_contam = dist["shortest_contamination"]["distance"]
    check("interpolation reach is exactly 1px",
          abs((d_empty - d_contam) - 1.0) < 1e-9,
          f"gap={d_empty - d_contam:.4f}px")

    dx, dy = dist["shortest_empty"]["direction"]
    d_num = bisect_exposing_pan(w, h, zoom, (dx, dy), target=PIX_EMPTY)
    check("analytic == bisected exposing pan",
          abs(d_empty - d_num) <= 0.05,
          f"analytic={d_empty:.4f} numeric={d_num:.4f}")

    K_safe = zoomed_translation(w, h, zoom, dx * (d_empty - 0.3), dy * (d_empty - 0.3))
    K_over = zoomed_translation(w, h, zoom, dx * (d_empty + 0.3), dy * (d_empty + 0.3))
    check("no empty pixel just below the threshold",
          not np.any(reverse_lookup_mask(K_safe, w, h) == PIX_EMPTY))
    check("empty pixels appear just above the threshold",
          np.any(reverse_lookup_mask(K_over, w, h) == PIX_EMPTY))

    tx, ty = dist["shortest_empty"]["vector"]
    win = crop_window_for_pan(w, h, tx, ty)
    check("back-computed crop window feasible",
          win["feasible"] and np.isfinite(win["zoom"]))
    check("crop window zoom: analytic == bisection",
          abs(win["zoom"] - win["analytic_zoom"]) < 5e-3,
          f"zoom={win['zoom']:.5f} analytic={win['analytic_zoom']:.5f}")
    check("crop window renders fully clean",
          np.all(reverse_lookup_mask(win["K"], w, h) == PIX_CLEAN))
    check("crop window quad stays inside the source",
          bool(np.all(win["quad_src"][:, 0] >= -1.0)
               and np.all(win["quad_src"][:, 0] <= w + 1.0)
               and np.all(win["quad_src"][:, 1] >= -1.0)
               and np.all(win["quad_src"][:, 1] <= h + 1.0)))


def test_exposure_margin_general_affine():
    """一般仿射（含旋转缩放）下，安全边距解析值与二分搜索一致。"""
    print("[exposure margin on a general affine K]")
    w, h = TF_W, TF_H
    C = aff_compose(6.0, -4.0, 0.01, 1.02)
    K = zoom_about_center(w, h, 1.08) @ C
    margin = exposure_margin(K, w, h)
    # source-space translation adversary: K' = K @ T(d) shifts samples by d
    best = float("inf")
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        def exposes(dist):
            KK = K @ translation(dx * dist, dy * dist)
            return bool(np.any(reverse_lookup_mask(KK, w, h) == PIX_EMPTY))

        lo, hi = 0.0, 4.0
        while not exposes(hi) and hi < 4096:
            hi *= 2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if exposes(mid):
                hi = mid
            else:
                lo = mid
        best = min(best, hi)
    check("analytic margin matches bisection",
          abs(margin - best) <= 0.05,
          f"margin={margin:.4f} bisect={best:.4f}")


def test_acceptance_suite():
    """完整对抗验收：测试场 + 乱序/复位 + 倒序 + 人为失败分别判定。"""
    print("[full adversarial acceptance suite]")
    ensure_testfield_samples()
    out_dir = os.path.join(ROOT, "reports", "testfield")
    rep = run_acceptance(out_dir, smoothness=55, min_fov=0.85)
    check("overall verdict is pass", rep["verdict"] == "pass",
          f"verdict={rep['verdict']}")
    by_id = {c["id"]: c for c in rep["cases"]}

    for cid in ("boundary_pixel_center", "boundary_half_pixel",
                "shortest_pan_search", "jitter_field", "pan_field",
                "shuffled_reset", "reversed_path",
                "forced_interpolation_oob", "forced_edge_exposed",
                "forced_insufficient_fov"):
        check(f"case {cid} passes", by_id[cid]["status"] == "pass")

    # 空像素风险与视野损失分开判定：抖动/摇移案例两列都必须是 ok
    for cid in ("jitter_field", "pan_field"):
        c = by_id[cid]
        check(f"{cid}: empty-pixel risk judged separately",
              c["empty_pixel_risk"]["status"] == "ok"
              and c["empty_pixel_risk"]["frames_with_empty"] == 0)
        check(f"{cid}: fov loss judged separately",
              c["fov_loss"]["status"] == "ok"
              and c["fov_loss"]["frames_below_declared"] == 0)

    # 人为失败必须被分别指出
    check("validator flags interpolation out-of-bounds",
          by_id["forced_interpolation_oob"]["details"]["verdict"]
          == "interpolation_out_of_bounds")
    check("validator flags insufficient FOV",
          by_id["forced_insufficient_fov"]["details"]["fov_loss"]
          == "insufficient_fov"
          and by_id["forced_insufficient_fov"]["details"]["empty_pixel_risk"]
          == "ok")

    # 失败还原：源帧 + 相机变换 + 裁切矩形
    for cid in ("forced_interpolation_oob", "forced_edge_exposed",
                "forced_insufficient_fov"):
        arts = by_id[cid]["failures"]
        check(f"{cid}: failure artifacts recorded", len(arts) >= 1)
        art = arts[0]
        src = os.path.join(out_dir, art["source_frame_png"])
        check(f"{cid}: source frame restored",
              os.path.exists(src) and os.path.getsize(src) > 10_000)
        check(f"{cid}: camera transform is 3x3",
              len(art["camera_transform"]) == 3
              and all(len(r) == 3 for r in art["camera_transform"]))
        check(f"{cid}: crop rect has 4 corners",
              len(art["crop_rect_src"]) == 4)

    # 乱序/复位与倒序细节
    det = by_id["shuffled_reset"]["details"]
    check("shuffled observation order keeps per-frame verdicts",
          det["shuffle_order_consistent"] is True)
    check("state reset detected at the injected frame",
          abs(det["reset_detected_at"] - det["reset_injected_at"]) <= 1,
          f"injected={det['reset_injected_at']} detected={det['reset_detected_at']}")
    check("reversed path keeps borders clean",
          by_id["reversed_path"]["empty_pixel_risk"]["frames_with_empty"] == 0)


if __name__ == "__main__":
    test_reference_edges()
    test_shortest_pan_and_crop_window()
    test_exposure_margin_general_affine()
    test_acceptance_suite()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
