"""对抗性轨迹验收器：露边搜索、边界对应与逐帧覆盖判定。

Two independent edge-exposure detectors, tied by one boundary condition:

* Detector A (解析轨迹): trajectory/corner geometry of the source->output
  map ``K``.  The analytic border is the image of the exact sampler-domain
  boundary under ``K`` - no per-pixel work.
* Detector B (逐输出像素反查): every output pixel center is mapped back
  through ``inv(K)`` and tested against the exact bilinear sampling domain,
  yielding the valid sampling domain pixel by pixel.

Boundary condition (边界条件): the analytic border and the first empty pixel
of an actual render (sentinel border, INTER_LINEAR, BORDER_CONSTANT) must
correspond.  Two calibration reference edges - one landing exactly on a
pixel center, one on a half-pixel - pin the interpolation reach, so the
validator can tell 插值采样越界 (interpolation sampling out-of-bounds) from
视野不足 (insufficient field of view) instead of conflating them.

Exact sampler domain (probed against cv2.warpAffine, INTER_LINEAR +
BORDER_CONSTANT; sample point ``s = inv(K) @ q`` for output pixel center
``q``, axis length ``n``):

* clean         : ``0 <= s <= n-1``   (closed; exact integers included)
* contaminated  : ``-1 < s < 0`` or ``n-1 < s < n``  (one tap outside)
* empty         : ``s <= -1`` or ``s >= n``          (all taps outside)

Consequently the interpolation reach is exactly 1 px: the empty boundary
always lies 1 px beyond the contamination boundary.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional, Sequence

import cv2
import numpy as np

from .geometry import apply_aff, as33, output_corners, zoom_about_center

PIX_EMPTY = 0
PIX_CONTAMINATED = 1
PIX_CLEAN = 2

SENTINEL = (255, 0, 255)  # magenta border used to expose empty pixels

VERDICT_OK = "ok"
VERDICT_INTERP_OOB = "interpolation_out_of_bounds"  # 插值采样越界
VERDICT_EDGE_EXPOSED = "edge_exposed"               # 露边（全空像素）
VERDICT_INSUFFICIENT_FOV = "insufficient_fov"       # 视野不足
VERDICT_BORDER_MISMATCH = "border_mismatch"         # 解析/实渲不对应


# ---------------------------------------------------------------- sampler model

def axis_state(s: np.ndarray, n: int) -> np.ndarray:
    """Per-axis sampler state for sample coordinate ``s`` and axis length ``n``."""
    s = np.asarray(s, dtype=np.float64)
    st = np.full(s.shape, PIX_CONTAMINATED, dtype=np.int8)
    st[(s >= 0.0) & (s <= n - 1.0)] = PIX_CLEAN
    st[(s <= -1.0) | (s >= float(n))] = PIX_EMPTY
    return st


def sample_extrema(K: np.ndarray, w: int, h: int) -> dict:
    """Min/max source sample coordinates over the output pixel-center grid.

    For an affine map the extrema over the pixel grid are attained at the
    grid corners, so mapping four corners suffices.
    """
    Ki = np.linalg.inv(as33(K))
    pts = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]]
    )
    s = apply_aff(Ki, pts)
    return {
        "sx_min": float(s[:, 0].min()),
        "sx_max": float(s[:, 0].max()),
        "sy_min": float(s[:, 1].min()),
        "sy_max": float(s[:, 1].max()),
    }


def exposure_margin(K: np.ndarray, w: int, h: int) -> float:
    """Shortest additional source-space translation (px) exposing an empty pixel.

    Exact for a translation adversary applied on the source side
    (``K' = K @ T(d)`` shifts every sample coordinate by ``d``): empty
    begins when a sample coordinate crosses -1 or n.
    """
    e = sample_extrema(K, w, h)
    return min(
        e["sx_min"] + 1.0,
        w - e["sx_max"],
        e["sy_min"] + 1.0,
        h - e["sy_max"],
    )


def contamination_margin(K: np.ndarray, w: int, h: int) -> float:
    """Shortest additional translation (px) that would contaminate a pixel."""
    return exposure_margin(K, w, h) - 1.0


# ------------------------------------------------------------- detector A (解析)

def analytic_domain_polygon(K: np.ndarray, w: int, h: int,
                            kind: str = "nonempty") -> np.ndarray:
    """Detector A: analytic border in output coordinates.

    ``kind="nonempty"`` images the first-empty boundary ``s in {-1, n}``;
    ``kind="clean"`` images the fully-clean domain ``s in [0, n-1]``.
    """
    if kind == "nonempty":
        rect = [(-1.0, -1.0), (float(w), -1.0), (float(w), float(h)), (-1.0, float(h))]
    elif kind == "clean":
        rect = [(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)]
    else:
        raise ValueError(kind)
    return apply_aff(as33(K), np.array(rect, dtype=np.float64))


def corners_covered(K: np.ndarray, w: int, h: int, safety: float = 0.0) -> bool:
    """Detector A corner test: mapped-back output corners inside the source."""
    Ki = np.linalg.inv(as33(K))
    q = apply_aff(Ki, output_corners(w, h))
    return bool(
        np.all(q[:, 0] >= safety)
        and np.all(q[:, 0] <= w - safety)
        and np.all(q[:, 1] >= safety)
        and np.all(q[:, 1] <= h - safety)
    )


# ------------------------------------------------- detector B (逐输出像素反查)

def reverse_lookup_mask(K: np.ndarray, w: int, h: int) -> np.ndarray:
    """Detector B: per-output-pixel valid sampling domain.

    Returns an (h, w) uint8 image of PIX_EMPTY / PIX_CONTAMINATED / PIX_CLEAN.
    """
    Ki = np.linalg.inv(as33(K))
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    src = apply_aff(Ki, pts)
    stx = axis_state(src[:, 0].reshape(h, w), w)
    sty = axis_state(src[:, 1].reshape(h, w), h)
    return np.minimum(stx, sty).astype(np.uint8)


# ------------------------------------------------------------------- 实渲复核

def render_validity_mask(K: np.ndarray, w: int, h: int) -> np.ndarray:
    """Empirical sampling-domain mask via an actual warpAffine of a flat image."""
    flat = np.full((h, w, 3), 255, dtype=np.uint8)
    warped = cv2.warpAffine(
        flat,
        as33(K)[:2, :].astype(np.float64),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    v = warped[:, :, 0]
    m = np.full((h, w), PIX_CONTAMINATED, dtype=np.uint8)
    m[v == 255] = PIX_CLEAN
    m[v == 0] = PIX_EMPTY
    return m


def render_sentinel_frame(frame_bgr: np.ndarray, K: np.ndarray,
                          sentinel=SENTINEL) -> np.ndarray:
    """Warp an actual source frame; out-of-domain pixels come back ``sentinel``."""
    h, w = frame_bgr.shape[:2]
    return cv2.warpAffine(
        frame_bgr,
        as33(K)[:2, :].astype(np.float64),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=sentinel,
    )


def count_sentinel_pixels(frame_bgr: np.ndarray, sentinel=SENTINEL,
                          tol: int = 10) -> int:
    s = np.array(sentinel, dtype=np.int16)
    d = np.abs(frame_bgr.astype(np.int16) - s).max(axis=2)
    return int(np.count_nonzero(d <= tol))


# ------------------------------------------------------------- boundary scans

def edge_runs(mask: np.ndarray, value: int = PIX_EMPTY) -> dict:
    """Per-scanline run length of ``value`` pixels from each frame edge."""
    eq = mask == value
    h, w = mask.shape

    def runs_from_left(rows):
        out = np.zeros(rows.shape[0], dtype=np.int32)
        for i, row in enumerate(rows):
            k = 0
            while k < row.shape[0] and row[k]:
                k += 1
            out[i] = k
        return out

    return {
        "left": runs_from_left(eq),
        "right": runs_from_left(eq[:, ::-1]),
        "top": runs_from_left(eq.T),
        "bottom": runs_from_left(eq.T[::-1]),
    }


def first_empty_pixels(mask: np.ndarray) -> dict:
    """First empty pixel met scanning inward from each edge (None if clean)."""
    runs = edge_runs(mask, PIX_EMPTY)
    h, w = mask.shape
    out = {}
    for edge, r in runs.items():
        if int(r.max()) <= 0:
            out[edge] = None
            continue
        if edge in ("left", "right"):
            y = int(np.argmax(r))
            x = int(runs["left"][y] - 1) if edge == "left" else int(w - runs["right"][y])
        else:
            x = int(np.argmax(r))
            y = int(runs["top"][x] - 1) if edge == "top" else int(h - runs["bottom"][x])
        out[edge] = (x, y)
    return out


def check_border_correspondence(K: np.ndarray, w: int, h: int,
                                tol: float = 1.0) -> dict:
    """边界条件：解析边框（detector B）与实渲首个空像素必须对应。

    Compares per-edge empty-run profiles between the analytic per-pixel
    reverse lookup and an actual render; returns the worst discrepancy.
    """
    analytic = reverse_lookup_mask(K, w, h)
    rendered = render_validity_mask(K, w, h)
    ra = edge_runs(analytic, PIX_EMPTY)
    rr = edge_runs(rendered, PIX_EMPTY)
    max_diff = 0.0
    for edge in ra:
        max_diff = max(
            max_diff,
            float(np.max(np.abs(ra[edge].astype(np.int32) - rr[edge].astype(np.int32)))),
        )
    ca = int(np.count_nonzero(analytic == PIX_CONTAMINATED))
    cr = int(np.count_nonzero(rendered == PIX_CONTAMINATED))
    ea = int(np.count_nonzero(analytic == PIX_EMPTY))
    er = int(np.count_nonzero(rendered == PIX_EMPTY))
    return {
        "correspondence_px": max_diff,
        "ok": max_diff <= tol,
        "analytic_empty": ea,
        "rendered_empty": er,
        "analytic_contaminated": ca,
        "rendered_contaminated": cr,
        "analytic_first_empty": first_empty_pixels(analytic),
        "rendered_first_empty": first_empty_pixels(rendered),
    }


# ------------------------------------------------------- 最短露边摇移（解析）

def pan_exposure_distances(w: int, h: int, zoom: float) -> dict:
    """Analytic pan distances (px) at which sampling leaves the source.

    For ``K = D(zoom) @ translate(t)`` the sample range over pixel centers is
    ``[a - t, b - t]`` with ``a = n(z-1)/(2z)``, ``b = (n(1+z)/2 - 1)/z`` per
    axis.  Contamination starts exactly 1 px before a fully empty pixel.
    """
    z = float(zoom)
    out = {}
    for axis, n in (("x", w), ("y", h)):
        contam_pos = n * (z - 1.0) / (2.0 * z)
        contam_neg = (n - 2.0) * (z - 1.0) / (2.0 * z)
        out[axis] = {
            "contaminate_pos": contam_pos,
            "contaminate_neg": contam_neg,
            "empty_pos": contam_pos + 1.0,
            "empty_neg": contam_neg + 1.0,
        }
    cands = [
        (out["x"]["empty_pos"], (1.0, 0.0)),
        (out["x"]["empty_neg"], (-1.0, 0.0)),
        (out["y"]["empty_pos"], (0.0, 1.0)),
        (out["y"]["empty_neg"], (0.0, -1.0)),
    ]
    dist, direction = min(cands, key=lambda c: c[0])
    out["shortest_empty"] = {
        "distance": dist,
        "direction": direction,
        "vector": (direction[0] * dist, direction[1] * dist),
    }
    cands_c = [
        (out["x"]["contaminate_pos"], (1.0, 0.0)),
        (out["x"]["contaminate_neg"], (-1.0, 0.0)),
        (out["y"]["contaminate_pos"], (0.0, 1.0)),
        (out["y"]["contaminate_neg"], (0.0, -1.0)),
    ]
    distc, dirc = min(cands_c, key=lambda c: c[0])
    out["shortest_contamination"] = {
        "distance": distc,
        "direction": dirc,
        "vector": (dirc[0] * distc, dirc[1] * distc),
    }
    return out


def _translation(tx: float, ty: float) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    m[0, 2] = tx
    m[1, 2] = ty
    return m


def bisect_exposing_pan(w: int, h: int, zoom: float, direction,
                        target: int = PIX_EMPTY,
                        lo: float = 0.0, hi: Optional[float] = None,
                        iters: int = 40) -> float:
    """Numeric shortest exposing pan along ``direction`` via detector B."""
    dx, dy = direction
    D = zoom_about_center(w, h, zoom)

    def exposes(d: float) -> bool:
        K = D @ _translation(dx * d, dy * d)
        m = reverse_lookup_mask(K, w, h)
        return bool(np.any(m <= target))

    if hi is None:
        hi = 4.0
        while not exposes(hi) and hi < 4096.0:
            hi *= 2.0
    if not exposes(hi):
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if exposes(mid):
            hi = mid
        else:
            lo = mid
    return hi


def crop_window_for_pan(w: int, h: int, tx: float, ty: float,
                        iters: int = 48) -> dict:
    """反推裁切窗口：让摇移 (tx, ty) 恰好不露边的最小放大与裁切四边形。

    Analytic: per axis the contamination distance ``(n-2)(z-1)/(2z)`` must
    reach ``|t|`` (a fully clean window also clears the empty boundary,
    which lies exactly 1 px further out), giving
    ``z >= (n-2)/(n-2-2|t|)``; confirmed by detector-B bisection.
    Returns the zoom, retained FOV and the source crop quad.
    """
    need = 1.0
    for t, n in ((abs(tx), w), (abs(ty), h)):
        if t <= 1e-9:
            continue
        if 2.0 * t >= n - 2.0:
            need = float("inf")
            break
        need = max(need, (n - 2.0) / (n - 2.0 - 2.0 * t))
    analytic_zoom = need

    def clean(z: float) -> bool:
        K = zoom_about_center(w, h, z) @ _translation(tx, ty)
        return bool(np.all(reverse_lookup_mask(K, w, h) == PIX_CLEAN))

    if not math.isfinite(analytic_zoom):
        return {"zoom": float("inf"), "feasible": False}
    lo, hi = 1.0, max(analytic_zoom, 1.0 + 1e-6)
    while not clean(hi):
        hi *= 1.5
        if hi > 64.0:
            return {"zoom": float("inf"), "feasible": False}
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if clean(mid):
            hi = mid
        else:
            lo = mid
    z = hi
    K = zoom_about_center(w, h, z) @ _translation(tx, ty)
    quad = apply_aff(np.linalg.inv(K), output_corners(w, h))
    return {
        "zoom": z,
        "analytic_zoom": analytic_zoom,
        "retained_fov": 1.0 / z,
        "quad_src": quad,
        "K": K,
        "feasible": True,
    }


# ------------------------------------------------------------- per-frame 判定

def validate_frame(K: np.ndarray, w: int, h: int,
                   zoom: Optional[float] = None,
                   min_fov: Optional[float] = None,
                   rendered: Optional[np.ndarray] = None,
                   check_render: bool = True) -> dict:
    """Per-frame acceptance: empty-pixel risk and FOV loss judged separately.

    ``rendered`` is the actual sentinel-bordered output frame when available;
    otherwise a validity-mask render of ``K`` stands in for the sampler.
    """
    mask_b = reverse_lookup_mask(K, w, h)
    empty_b = int(np.count_nonzero(mask_b == PIX_EMPTY))
    contam_b = int(np.count_nonzero(mask_b == PIX_CONTAMINATED))
    margin = exposure_margin(K, w, h)

    risk = {
        "analytic_empty_pixels": empty_b,
        "analytic_contaminated_pixels": contam_b,
        "margin_px": round(margin, 3),
        "status": VERDICT_OK,
    }
    if check_render:
        corr = check_border_correspondence(K, w, h, tol=1.0)
        risk["border_correspondence_px"] = corr["correspondence_px"]
        risk["rendered_empty_pixels"] = corr["rendered_empty"]
        risk["rendered_contaminated_pixels"] = corr["rendered_contaminated"]
        if not corr["ok"]:
            risk["status"] = VERDICT_BORDER_MISMATCH
    if rendered is not None:
        risk["sentinel_pixels"] = count_sentinel_pixels(rendered)
    if risk["status"] == VERDICT_OK:
        if empty_b > 0:
            risk["status"] = VERDICT_EDGE_EXPOSED
        elif contam_b > 0:
            risk["status"] = VERDICT_INTERP_OOB

    fov = {"status": VERDICT_OK}
    if zoom is not None:
        retained = 1.0 / float(zoom)
        fov["retained_fov"] = round(retained, 5)
        fov["loss_pct"] = round(100.0 * (1.0 - retained), 2)
        if min_fov is not None:
            fov["declared_min_fov"] = float(min_fov)
            if retained < float(min_fov) - 1e-6:
                fov["status"] = VERDICT_INSUFFICIENT_FOV
    return {"empty_pixel_risk": risk, "fov_loss": fov}


# ------------------------------------------------------------- 失败还原 artifacts

def save_failure_artifact(out_dir: str, case: str, frame_index: int,
                          source_frame: Optional[np.ndarray],
                          K: np.ndarray, w: int, h: int,
                          reason: str,
                          rendered: Optional[np.ndarray] = None,
                          extra: Optional[dict] = None) -> dict:
    """任何失败都还原：源帧 + 相机变换 + 裁切矩形（外加实渲图备查）。"""
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{case}_frame{frame_index:04d}"
    art = {"case": case, "frame": frame_index, "reason": reason}
    if source_frame is not None:
        p_src = os.path.join(out_dir, stem + "_src.png")
        cv2.imwrite(p_src, source_frame)
        art["source_frame_png"] = os.path.basename(p_src)
    if rendered is not None:
        p_out = os.path.join(out_dir, stem + "_out.png")
        cv2.imwrite(p_out, rendered)
        art["rendered_png"] = os.path.basename(p_out)
    K3 = as33(K)
    crop_quad = apply_aff(np.linalg.inv(K3), output_corners(w, h))
    art["camera_transform"] = [[float(v) for v in row] for row in K3]
    art["crop_rect_src"] = [[float(x), float(y)] for x, y in crop_quad]
    if extra:
        art.update(extra)
    p_json = os.path.join(out_dir, stem + ".json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(art, f, ensure_ascii=False, indent=2)
    art["json"] = os.path.basename(p_json)
    return art


# ------------------------------------------------------------ 参考边（校准用）

def reference_edge_cases(w: int, h: int) -> list[dict]:
    """一条参考边落在像素中心，另一条落在半像素位置。

    Pure translations make the analytic empty border ``qx = tx - 1`` land
    exactly on an output pixel center (integer tx) or on a half-pixel
    (tx = k + 0.5).  The pixel-center edge has a sharp empty->clean
    transition with no contaminated column; the half-pixel edge has exactly
    one contaminated column - the interpolation reach made visible.
    """
    return [
        {
            "id": "pixel_center",
            "title": "参考边落在像素中心（tx=3.0）",
            "K": _translation(3.0, 0.0),
            "expect_empty_cols_left": 3,
            "expect_contaminated_cols": 0,
        },
        {
            "id": "half_pixel",
            "title": "参考边落在半像素位置（tx=3.5）",
            "K": _translation(3.5, 0.0),
            "expect_empty_cols_left": 3,
            "expect_contaminated_cols": 1,
        },
    ]


def translation(tx: float, ty: float) -> np.ndarray:
    return _translation(tx, ty)


def zoomed_translation(w: int, h: int, zoom: float, tx: float, ty: float) -> np.ndarray:
    return zoom_about_center(w, h, zoom) @ _translation(tx, ty)


def mats_to_jsonable(Ks: Sequence[np.ndarray]) -> list:
    return [[[float(v) for v in row] for row in as33(K)] for K in Ks]
