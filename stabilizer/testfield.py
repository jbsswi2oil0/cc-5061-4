"""真值视场测试场：边缘标记环 + 已知轨迹的合成片段。

The test field generates clips whose ground-truth field of view is known
exactly:

* the source frames carry a ring of identifiable markers at the footage
  edge (unique hue per marker, positions stored in the GT file);
* the camera path is fully known, so for every output frame the exact set
  of markers that should remain visible is determined analytically by the
  crop quad - the ground-truth field of view;
* two motion profiles are provided: 小幅高频抖动 (small high-frequency
  jitter) and 长距离摇移 (long-distance pan with wobble);
* observation order can be shuffled and a state reset injected to probe
  validator robustness before the solved trajectory is applied to actual
  frames.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional, Sequence

import cv2
import numpy as np

from .geometry import aff_compose, apply_aff, as33
from .videoio import FFmpegWriter

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")

W, H = 640, 360
MARGIN_X, MARGIN_Y = 260, 200
WW, WH = W + 2 * MARGIN_X, H + 2 * MARGIN_Y

RING_INSET = 12       # marker ring inset from the nominal frame edge (px)
N_MARKERS = 24
MARKER_RADIUS = 6

TESTFIELD_INFO = {
    "testfield_jitter": {
        "file": "testfield_jitter.mp4",
        "gt": "testfield_jitter_gt.json",
        "title": "真值视场测试场 · 小幅高频抖动",
        "desc": "原片边缘一圈可识别标记 + 小幅高频抖动；真值视场已知，用于空像素风险判定。",
    },
    "testfield_pan": {
        "file": "testfield_pan.mp4",
        "gt": "testfield_pan_gt.json",
        "title": "真值视场测试场 · 长距离摇移",
        "desc": "边缘标记环 + 长距离摇移与摆动，考验视野裁决；用于露边搜索与裁切窗口反推验收。",
    },
}


# ------------------------------------------------------------------ marker ring

def marker_palette() -> np.ndarray:
    """One unique saturated BGR color per marker (HSV hue encodes the id)."""
    colors = []
    for k in range(N_MARKERS):
        hue = int(round(k * 180.0 / N_MARKERS)) % 180
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))
    return np.array(colors, dtype=np.int32)


def marker_ring_spec(w: int = W, h: int = H, inset: int = RING_INSET,
                     n: int = N_MARKERS) -> dict:
    """Marker positions (frame space at the reference pose) around the edge."""
    cx, cy = w / 2.0, h / 2.0
    rx, ry = w / 2.0 - inset, h / 2.0 - inset
    # perimeter-parameterised rectangle ring
    perim = 4.0 * (rx + ry)
    pos = []
    for k in range(n):
        u = (k + 0.5) / n * perim
        seg = perim / 4.0
        side = int(u // seg) % 4
        v = (u % seg) / seg  # 0..1 along the side
        if side == 0:    # top-left -> top-right (through top edge midpoint)
            x, y = cx - rx + 2 * rx * v, cy - ry
        elif side == 1:  # right
            x, y = cx + rx, cy - ry + 2 * ry * v
        elif side == 2:  # bottom
            x, y = cx + rx - 2 * rx * v, cy + ry
        else:            # left
            x, y = cx - rx, cy + ry - 2 * ry * v
        pos.append((float(x), float(y)))
    return {
        "n": n,
        "inset": inset,
        "radius": MARKER_RADIUS,
        "frame_w": w,
        "frame_h": h,
        "positions": pos,  # frame space at reference pose
        "hues": [int(round(k * 180.0 / n)) % 180 for k in range(n)],
    }


def marker_world_positions(spec: dict) -> np.ndarray:
    """Ring positions in world coordinates (nominal window = frame at identity)."""
    off = np.array([MARGIN_X, MARGIN_Y], dtype=np.float64)
    return np.array(spec["positions"], dtype=np.float64) + off


def draw_marker_ring(world: np.ndarray, spec: dict) -> None:
    colors = marker_palette()
    for (x, y), color in zip(marker_world_positions(spec), colors):
        c = (int(round(x)), int(round(y)))
        cv2.circle(world, c, MARKER_RADIUS + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(world, c, MARKER_RADIUS, tuple(int(v) for v in color), -1,
                   cv2.LINE_AA)


def detect_markers(frame_bgr: np.ndarray, spec: dict,
                   s_min: int = 140, v_min: int = 140,
                   hue_tol: int = 7, min_area: int = 14) -> dict:
    """Blindly detect ring markers in a frame; returns {marker_id: (x, y)}."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min)).astype(np.uint8) * 255
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    hues = spec["hues"]
    found: dict[int, tuple[float, float]] = {}
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        pix = hsv[labels == k]
        hue = int(np.median(pix[:, 0]))
        best, best_d = None, hue_tol + 1
        for mid, mh in enumerate(hues):
            d = abs(hue - mh)
            d = min(d, 180 - d)
            if d < best_d:
                best, best_d = mid, d
        if best is None or best_d > hue_tol:
            continue
        if best in found:  # keep the larger component per id
            continue
        found[best] = (float(cents[k][0]), float(cents[k][1]))
    return found


# ----------------------------------------------------------------------- world

def build_testfield_world(seed: int = 61) -> np.ndarray:
    """Muted, trackable texture; saturation kept low so markers stand out."""
    rng = np.random.default_rng(seed)
    world = np.full((WH, WW, 3), 208, dtype=np.uint8)
    world[:] = (196, 200, 203)
    for _ in range(34):
        x0, y0 = rng.integers(0, WW), rng.integers(0, WH)
        bw, bh = rng.integers(70, 240), rng.integers(50, 200)
        base = rng.integers(120, 220, 3)
        # mute toward gray: keeps S low, markers stay uniquely saturated
        muted = (0.7 * base + 0.3 * 160).astype(np.int32)
        cv2.rectangle(world, (x0, y0), (min(WW, x0 + bw), min(WH, y0 + bh)),
                      tuple(int(c) for c in muted), -1)
    for x in range(0, WW, 36):
        cv2.line(world, (x, 0), (x, WH), (110, 110, 110), 1)
    for y in range(0, WH, 36):
        cv2.line(world, (0, y), (WW, y), (110, 110, 110), 1)
    for _ in range(130):
        cx, cy = rng.integers(16, WW - 16), rng.integers(16, WH - 16)
        r = int(rng.integers(6, 26))
        g = int(rng.integers(40, 200))
        cv2.rectangle(world, (cx - r, cy - r), (cx + r, cy + r),
                      (g, g, g), -1)
        cv2.rectangle(world, (cx - r, cy - r), (cx + r, cy + r),
                      (30, 30, 30), 1)
    for _ in range(9):  # fiducial crosses for tracking texture
        cx, cy = int(rng.integers(60, WW - 60)), int(rng.integers(60, WH - 60))
        cv2.line(world, (cx - 14, cy), (cx + 14, cy), (25, 25, 25), 2)
        cv2.line(world, (cx, cy - 14), (cx, cy + 14), (25, 25, 25), 2)
    noise = rng.normal(0, 4.0, world.shape).astype(np.int16)
    return np.clip(world.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ trajectories

def jitter_path(n: int, fps: float, seed: int = 71) -> list[np.ndarray]:
    """小幅高频抖动：small high-frequency jitter around the reference pose."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    jx = 5.0 * np.sin(2 * np.pi * 7.3 * t + 0.5) + 2.5 * np.sin(2 * np.pi * 11.1 * t)
    jy = 4.0 * np.sin(2 * np.pi * 8.7 * t + 1.9) + 2.0 * np.sin(2 * np.pi * 12.9 * t)
    ja = np.radians(0.5) * np.sin(2 * np.pi * 6.1 * t + 0.3)
    js = 0.006 * np.sin(2 * np.pi * 9.7 * t)
    jx += rng.normal(0, 0.6, n)
    jy += rng.normal(0, 0.6, n)
    return [aff_compose(float(jx[i]), float(jy[i]), float(ja[i]),
                        float(math.exp(js[i]))) for i in range(n)]


def pan_path(n: int, fps: float, seed: int = 83) -> list[np.ndarray]:
    """长距离摇移：long pan with a wobble too slow to vanish, too fast to keep."""
    t = np.arange(n) / fps
    u = np.linspace(0.0, 1.0, n)
    sm = u * u * (3.0 - 2.0 * u)
    px = 170.0 * sm - 45.0 + 26.0 * np.sin(2 * np.pi * 1.15 * t + 0.4)
    py = -58.0 * sm + 18.0 * np.sin(2 * np.pi * 0.9 * t + 2.0)
    pa = np.radians(2.2) * np.sin(2 * np.pi * 0.8 * t + 1.0)
    ps = 1.0 + 0.05 * sm + 0.012 * np.sin(2 * np.pi * 1.7 * t)
    jx = 7.0 * np.sin(2 * np.pi * 6.9 * t + 0.2)
    jy = 6.0 * np.sin(2 * np.pi * 8.3 * t + 1.1)
    return [aff_compose(float(px[i] + jx[i]), float(py[i] + jy[i]),
                        float(pa[i]), float(ps[i])) for i in range(n)]


def shuffle_observation_order(n: int, seed: int = 97) -> list[int]:
    """打乱观测顺序：a fixed permutation of frame indices."""
    rng = np.random.default_rng(seed)
    order = list(range(n))
    rng.shuffle(order)
    return order


def inject_state_reset(mats: Sequence[np.ndarray], at: int) -> list[np.ndarray]:
    """注入一次状态复位：path snaps back to identity at frame ``at``."""
    out = [as33(m) for m in mats]
    inv_at = np.linalg.inv(out[at])
    for i in range(at, len(out)):
        out[i] = out[i] @ inv_at
    return out


def reverse_path(mats: Sequence[np.ndarray]) -> list[np.ndarray]:
    """倒序播放同一路径：reversed trajectory, gauge-fixed to frame 0."""
    mats = [as33(m) for m in mats]
    inv_last = np.linalg.inv(mats[-1])
    return [m @ inv_last for m in reversed(mats)]


# ------------------------------------------------------------------ rendering

def render_testfield(kind: str, out_dir: str = SAMPLE_DIR,
                     n_frames: int = 150, fps: int = 30) -> dict:
    if kind not in TESTFIELD_INFO:
        raise ValueError(f"unknown testfield kind: {kind}")
    os.makedirs(out_dir, exist_ok=True)
    info = TESTFIELD_INFO[kind]
    out_mp4 = os.path.join(out_dir, info["file"])
    out_gt = os.path.join(out_dir, info["gt"])

    world = build_testfield_world()
    spec = marker_ring_spec()
    draw_marker_ring(world, spec)
    mats = jitter_path(n_frames, fps) if kind == "testfield_jitter" else pan_path(n_frames, fps)

    off = np.array([MARGIN_X, MARGIN_Y], dtype=np.float64)
    with FFmpegWriter(out_mp4, W, H, fps, crf=18, preset="veryfast") as wr:
        for i in range(n_frames):
            T = as33(mats[i])
            R = np.eye(3)
            R[:2, :] = (np.array([[1.0, 0, off[0]], [0, 1.0, off[1]]])
                        @ np.linalg.inv(T))[:2, :]
            frame = cv2.warpAffine(
                world, R[:2, :], (W, H),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40),
            )
            grain = np.random.default_rng(5000 + i).normal(0, 1.8, frame.shape)
            frame = np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)
            wr.write(frame)

    gt = {
        "kind": kind,
        "width": W,
        "height": H,
        "fps": fps,
        "n_frames": n_frames,
        "mats": [np.round(as33(m), 8).tolist() for m in mats],
        "marker_ring": spec,
        "marker_world": marker_world_positions(spec).round(4).tolist(),
        "margin": [MARGIN_X, MARGIN_Y],
    }
    with open(out_gt, "w", encoding="utf-8") as f:
        json.dump(gt, f)
    return {"video": out_mp4, "gt": out_gt, **info}


def ensure_testfield_samples(out_dir: str = SAMPLE_DIR,
                             n_frames: int = 150) -> list[dict]:
    out = []
    for kind, info in TESTFIELD_INFO.items():
        path = os.path.join(out_dir, info["file"])
        if not os.path.exists(path):
            out.append(render_testfield(kind, out_dir, n_frames=n_frames))
        else:
            out.append({"video": path,
                        "gt": os.path.join(out_dir, info["gt"]), **info})
    return out


if __name__ == "__main__":
    for r in ensure_testfield_samples():
        print("testfield:", r["video"])
