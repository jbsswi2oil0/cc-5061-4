"""Synthetic but verifiable test footage.

Three built-in clips are generated:

* ``planar``          - richly textured single plane with a *known* camera
                        trajectory (slow intended move + handheld jitter).
* ``handheld_scene``  - authored "self-made short": street scene with a car
                        and walker moving independently, handheld camera.
* ``parallax_layers`` - background + near foreground layer whose shake
                        translation is depth-amplified, so a single global
                        transform leaves systematic residuals.

Ground-truth matrices are written next to the clips for quantitative checks.
In a camera-less sandbox this stands in for a self-shot short; the web UI also
accepts uploads of real footage.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import cv2
import numpy as np

from .geometry import aff_compose
from .videoio import FFmpegWriter

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")

W, H = 640, 360
MARGIN_X, MARGIN_Y = 240, 180
WW, WH = W + 2 * MARGIN_X, H + 2 * MARGIN_Y

SAMPLE_INFO = {
    "planar": {
        "file": "planar.mp4",
        "gt": "planar_gt.json",
        "title": "已知抖动平面序列（真值轨迹）",
        "desc": "单平面高纹理海报，相机路径完全已知，用于轨迹/重投影精度评测。",
    },
    "handheld_scene": {
        "file": "handheld_scene.mp4",
        "gt": "handheld_scene_gt.json",
        "title": "自制手持街景短片（含运动车辆）",
        "desc": "程序化拍摄的街景短片：车辆与行人独立运动，用于验证时间顺序与真实运动保留。",
    },
    "parallax_layers": {
        "file": "parallax_layers.mp4",
        "gt": "parallax_layers_gt.json",
        "title": "视差分层序列（深度冲突）",
        "desc": "近景层抖动位移大于背景，单一变换必然残留，用于残差/视差告警验证。",
    },
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_poster_world(seed: int = 7) -> np.ndarray:
    rng = _rng(seed)
    world = np.full((WH, WW, 3), 235, dtype=np.uint8)
    world[:] = (214, 226, 230)  # BGR warm gray

    # broad color blocks
    for _ in range(26):
        x0, y0 = rng.integers(0, WW), rng.integers(0, WH)
        bw, bh = rng.integers(80, 260), rng.integers(60, 220)
        color = tuple(int(c) for c in rng.integers(120, 230, 3))
        cv2.rectangle(world, (x0, y0), (min(WW, x0 + bw), min(WH, y0 + bh)),
                      color, -1)

    # grid
    for x in range(0, WW, 40):
        cv2.line(world, (x, 0), (x, WH), (120, 120, 120), 1)
    for y in range(0, WH, 40):
        cv2.line(world, (0, y), (WW, y), (120, 120, 120), 1)
    for x in range(0, WW, 200):
        cv2.line(world, (x, 0), (x, WH), (60, 60, 60), 2)
    for y in range(0, WH, 200):
        cv2.line(world, (0, y), (WW, y), (60, 60, 60), 2)

    # random shapes with corners
    for _ in range(150):
        cx, cy = rng.integers(20, WW - 20), rng.integers(20, WH - 20)
        r = int(rng.integers(8, 34))
        color = tuple(int(c) for c in rng.integers(20, 200, 3))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(world, (cx - r, cy - r), (cx + r, cy + r), color, -1)
            cv2.rectangle(world, (cx - r, cy - r), (cx + r, cy + r),
                          (30, 30, 30), 2)
        elif kind == 1:
            cv2.circle(world, (cx, cy), r, color, -1)
            cv2.circle(world, (cx, cy), r, (30, 30, 30), 2)
        else:
            pts = np.array(
                [[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32
            )
            cv2.fillPoly(world, [pts], color)
            cv2.polylines(world, [pts], True, (30, 30, 30), 2)

    # checkerboard patches
    for _ in range(7):
        cx, cy = int(rng.integers(80, WW - 80)), int(rng.integers(80, WH - 80))
        cell = 16
        for i in range(6):
            for j in range(4):
                if (i + j) % 2 == 0:
                    cv2.rectangle(
                        world,
                        (cx + i * cell, cy + j * cell),
                        (cx + (i + 1) * cell, cy + (j + 1) * cell),
                        (20, 20, 20), -1,
                    )
        cv2.rectangle(world, (cx, cy), (cx + 6 * cell, cy + 4 * cell),
                      (0, 0, 0), 2)

    # text signs
    words = ["STAB", "FEATURE", "PLANE", "2026", "OPENCV", "MOTION",
             "RANSAC", "FLOW", "FRAME", "TRACK", "CAM", "PATH"]
    for _ in range(26):
        x, y = int(rng.integers(20, WW - 160)), int(rng.integers(40, WH - 20))
        cv2.rectangle(world, (x - 6, y - 26), (x + 130, y + 10),
                      (245, 245, 245), -1)
        cv2.rectangle(world, (x - 6, y - 26), (x + 130, y + 10),
                      (40, 40, 40), 2)
        cv2.putText(world, str(rng.choice(words)), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2, cv2.LINE_AA)

    noise = rng.normal(0, 5.0, world.shape).astype(np.int16)
    world = np.clip(world.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return world


def build_street_world(seed: int = 11) -> np.ndarray:
    rng = _rng(seed)
    world = np.zeros((WH, WW, 3), dtype=np.uint8)
    horizon = int(WH * 0.62)
    # sky
    world[:horizon] = (206, 196, 178)  # BGR dusty blue
    for y in range(horizon):
        world[y] = tuple(int(c - 20 * (y / horizon)) for c in (214, 205, 188))
    # road
    world[horizon:] = (74, 72, 68)
    cv2.line(world, (0, horizon), (WW, horizon), (90, 90, 90), 3)
    # sidewalk
    cv2.rectangle(world, (0, horizon - 26), (WW, horizon + 6), (150, 148, 140), -1)

    # buildings with window grids
    x = -10
    while x < WW:
        bw = int(rng.integers(120, 210))
        top = int(rng.integers(40, 150))
        color = tuple(int(c) for c in rng.integers(110, 190, 3))
        cv2.rectangle(world, (x, top), (x + bw, horizon - 24), color, -1)
        cv2.rectangle(world, (x, top), (x + bw, horizon - 24), (45, 45, 45), 2)
        for wx in range(x + 16, x + bw - 20, 26):
            for wy in range(top + 18, horizon - 44, 30):
                lit = rng.random() > 0.4
                wc = (200, 225, 235) if lit else (70, 80, 90)
                cv2.rectangle(world, (wx, wy), (wx + 14, wy + 18), wc, -1)
                cv2.rectangle(world, (wx, wy), (wx + 14, wy + 18),
                              (40, 40, 40), 1)
        # rooftop sign with text
        if rng.random() > 0.4:
            cv2.rectangle(world, (x + 10, top - 34), (x + bw - 10, top - 4),
                          (60, 60, 220), -1)
            cv2.putText(world, "SHOP", (x + bw // 2 - 45, top - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
        x += bw + int(rng.integers(6, 20))

    # road lanes
    for lx in range(-40, WW, 90):
        cv2.rectangle(world, (lx, horizon + 70), (lx + 46, horizon + 80),
                      (220, 220, 220), -1)
    # trees / poles for near-background texture
    for tx in range(40, WW, 130):
        cv2.rectangle(world, (tx - 5, horizon - 120), (tx + 5, horizon - 20),
                      (45, 60, 75), -1)
        cv2.circle(world, (tx, horizon - 135), 34, (52, 110, 72), -1)
        cv2.circle(world, (tx - 18, horizon - 120), 24, (58, 122, 80), -1)
        cv2.circle(world, (tx + 20, horizon - 122), 26, (48, 100, 66), -1)

    noise = rng.normal(0, 4.0, world.shape).astype(np.int16)
    return np.clip(world.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def build_foreground_layer(seed: int = 23) -> np.ndarray:
    """Close-depth railing / pergola / panels: many trackable corners across
    the whole frame so depth motion produces spatially spread residuals."""
    rng = _rng(seed)
    layer = np.zeros((WH, WW, 4), dtype=np.uint8)
    dark = (38, 42, 52, 255)
    edge = (150, 158, 168, 255)

    # bottom railing: three horizontal rails + dense vertical bars
    for ry, thick in ((WH - 150, 12), (WH - 92, 8), (WH - 34, 14)):
        cv2.rectangle(layer, (0, ry), (WW, ry + thick), dark, -1)
        cv2.line(layer, (0, ry), (WW, ry), edge, 2)
    for bx in range(14, WW, 44):
        cv2.rectangle(layer, (bx, WH - 168), (bx + 11, WH), dark, -1)
        cv2.line(layer, (bx + 11, WH - 168), (bx + 11, WH), edge, 1)

    # overhead pergola beams + hanging slats (covers top spatial bins)
    cv2.rectangle(layer, (0, 46), (WW, 66), dark, -1)
    cv2.rectangle(layer, (0, 96), (WW, 110), dark, -1)
    for bx in range(30, WW, 64):
        top = 0
        bottom = int(rng.integers(150, 250))
        cv2.rectangle(layer, (bx, top), (bx + 12, bottom), dark, -1)
        cv2.line(layer, (bx + 12, top), (bx + 12, bottom), edge, 1)
    for _ in range(7):  # diagonal near beams
        x0 = int(rng.integers(-100, WW))
        cv2.line(layer, (x0, -20), (x0 + 360, 220), dark, 16)
        cv2.line(layer, (x0, -20), (x0 + 360, 220), edge, 2)

    # high-texture warning plates / boards on the near plane
    for cx, cy in ((250, 300), (840, 250), (560, 150)):
        pw, ph, cell = 150, 96, 12
        x0, y0 = cx - pw // 2, cy - ph // 2
        cv2.rectangle(layer, (x0 - 8, y0 - 8), (x0 + pw + 8, y0 + ph + 8),
                      (20, 20, 20, 255), -1)
        for i in range(pw // cell):
            for j in range(ph // cell):
                if (i + j) % 2 == 0:
                    cv2.rectangle(
                        layer,
                        (x0 + i * cell, y0 + j * cell),
                        (x0 + (i + 1) * cell, y0 + (j + 1) * cell),
                        (60, 210, 240, 255), -1,
                    )
        cv2.rectangle(layer, (x0 - 8, y0 - 8), (x0 + pw + 8, y0 + ph + 8),
                      (235, 235, 235, 255), 3)

    # a couple of close poles at the sides
    for cx in (70, WW - 90):
        cv2.rectangle(layer, (cx, 120), (cx + 34, WH), (30, 34, 44, 255), -1)
        for yy in range(150, WH, 46):
            cv2.rectangle(layer, (cx - 6, yy), (cx + 40, yy + 8), edge, -1)
    return layer


def camera_path(n: int, fps: int, seed: int, energetic: bool = True) -> dict:
    """Return component arrays + cumulative ground-truth matrices T_i."""
    rng = _rng(seed)
    t = np.arange(n) / fps

    # deliberate slow camera move (must survive smoothing)
    pan_x = 95.0 * _smoothstep(np.linspace(0, 1, n)) - 30.0
    pan_x += 14.0 * np.sin(2 * np.pi * t / 5.3 + 0.6)
    pan_y = 16.0 * np.sin(2 * np.pi * t / 6.7) + 8.0 * _smoothstep(np.linspace(0, 1, n))
    pan_a = np.radians(1.6 * np.sin(2 * np.pi * t / 7.1 + 1.3))
    pan_s = 1.0 + 0.07 * _smoothstep(np.linspace(0, 1, n))

    # handheld high-frequency jitter
    amp_x, amp_y, amp_a, amp_s = (
        (20.0, 15.0, 1.15, 0.028) if energetic else (10.0, 8.0, 0.6, 0.014)
    )
    jx = amp_x * (
        0.55 * np.sin(2 * np.pi * (6.3 / fps) * np.arange(n) + 0.4)
        + 0.30 * np.sin(2 * np.pi * (10.7 / fps) * np.arange(n) + 2.1)
    )
    jy = amp_y * (
        0.6 * np.sin(2 * np.pi * (7.9 / fps) * np.arange(n) + 1.2)
        + 0.3 * _filtered_noise(n, rng, 0.18)
    )
    ja = np.radians(amp_a) * np.sin(
        2 * np.pi * (5.4 / fps) * np.arange(n) + 0.2
    ) + np.radians(0.35) * _filtered_noise(n, rng, 0.25)
    js = amp_s * _filtered_noise(n, rng, 0.12)

    xs = pan_x + jx
    ys = pan_y + jy
    ang = pan_a + ja
    scl = pan_s * np.exp(js)

    mats = [aff_compose(float(xs[i]), float(ys[i]), float(ang[i]),
                        float(scl[i])) for i in range(n)]
    comp = {
        "pan_x": pan_x, "pan_y": pan_y, "pan_a": pan_a, "pan_s": pan_s,
        "jx": jx, "jy": jy, "ja": ja, "js": js,
        "x": xs, "y": ys, "angle": ang, "scale": scl,
    }
    return {"mats": mats, "comp": comp}


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return u * u * (3 - 2 * u)


def _filtered_noise(n: int, rng, alpha: float) -> np.ndarray:
    white = rng.standard_normal(n)
    out = np.zeros(n)
    v = 0.0
    for i in range(n):
        v = (1 - alpha) * v + alpha * white[i]
        out[i] = v
    out -= out.mean()
    m = np.max(np.abs(out)) + 1e-9
    return out / m


def _world_to_frame(T: np.ndarray, wx: float, wy: float) -> tuple[int, int]:
    off = np.array([MARGIN_X, MARGIN_Y], dtype=np.float64)
    q = np.array([wx, wy], dtype=np.float64) - off
    p = T[:2, :2] @ q + T[:2, 2]
    return int(round(p[0])), int(round(p[1]))


def _draw_car(frame: np.ndarray, x: int, y: int, scale: float = 1.0,
              color=(60, 70, 220)) -> None:
    w, h = int(74 * scale), int(30 * scale)
    cv2.rectangle(frame, (x - w // 2, y - h), (x + w // 2, y), color, -1)
    cv2.rectangle(frame, (x - w // 3, y - h - int(16 * scale)),
                  (x + w // 4, y - h), (190, 220, 235), -1)
    cv2.rectangle(frame, (x - w // 3, y - h - int(16 * scale)),
                  (x + w // 4, y - h), (30, 30, 30), 1)
    r = int(9 * scale)
    for dx in (-w // 3, w // 3):
        cv2.circle(frame, (x + dx, y), r, (25, 25, 25), -1)
        cv2.circle(frame, (x + dx, y), r // 2, (170, 170, 170), -1)
    cv2.rectangle(frame, (x - w // 2 + 4, y - h + 6),
                  (x - w // 2 + 14, y - h + 12), (200, 230, 255), -1)


def _draw_walker(frame: np.ndarray, x: int, y: int, phase: float) -> None:
    cv2.circle(frame, (x, y - 34), 7, (40, 35, 30), -1)          # head
    cv2.line(frame, (x, y - 27), (x, y - 12), (30, 60, 140), 5)   # body
    swing = int(6 * math.sin(phase))
    cv2.line(frame, (x, y - 12), (x - 6 + swing, y), (30, 30, 30), 3)
    cv2.line(frame, (x, y - 12), (x + 6 - swing, y), (30, 30, 30), 3)
    cv2.line(frame, (x, y - 24), (x - 8 - swing, y - 16), (30, 60, 140), 3)
    cv2.line(frame, (x, y - 24), (x + 8 + swing, y - 18), (30, 60, 140), 3)


def render_sample(kind: str, out_dir: str = SAMPLE_DIR,
                  n_frames: int = 180, fps: int = 30) -> dict:
    if kind not in SAMPLE_INFO:
        raise ValueError(f"unknown sample kind: {kind}")
    os.makedirs(out_dir, exist_ok=True)
    info = SAMPLE_INFO[kind]
    out_mp4 = os.path.join(out_dir, info["file"])
    out_gt = os.path.join(out_dir, info["gt"])

    if kind == "planar":
        world = build_poster_world()
        path = camera_path(n_frames, fps, seed=101, energetic=True)
    elif kind == "handheld_scene":
        world = build_street_world()
        path = camera_path(n_frames, fps, seed=202, energetic=True)
    else:
        world = build_poster_world(seed=303)
        # turn poster into a "distant plane": slight blur + cooler tint
        world = cv2.GaussianBlur(world, (3, 3), 0.6)
        foreground = build_foreground_layer()
        path = camera_path(n_frames, fps, seed=404, energetic=True)

    mats = path["mats"]
    off = np.array([MARGIN_X, MARGIN_Y], dtype=np.float64)
    car_xs, walker_xs = [], []

    with FFmpegWriter(out_mp4, W, H, fps, crf=18, preset="veryfast") as wr:
        for i in range(n_frames):
            T = mats[i]
            R = np.eye(3)
            R[:2, :] = (np.array([[1.0, 0, off[0]], [0, 1, off[1]]]) @
                        np.linalg.inv(T))[:2, :]
            frame = cv2.warpAffine(
                world, R[:2, :], (W, H),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30),
            )

            if kind == "parallax_layers":
                # Near layer: amplified high-frequency translation only.
                c = path["comp"]
                Tf = aff_compose(
                    float(c["pan_x"][i] + 2.4 * c["jx"][i]),
                    float(c["pan_y"][i] + 2.4 * c["jy"][i]),
                    float(c["pan_a"][i] + 1.15 * c["ja"][i]),
                    float(c["pan_s"][i] * math.exp(1.15 * c["js"][i])),
                )
                Rf = np.eye(3)
                Rf[:2, :] = (np.array([[1.0, 0, off[0]], [0, 1, off[1]]]) @
                             np.linalg.inv(Tf))[:2, :]
                fg = cv2.warpAffine(
                    foreground, Rf[:2, :], (W, H),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                alpha = fg[:, :, 3:4].astype(np.float32) / 255.0
                frame = (
                    fg[:, :, :3].astype(np.float32) * alpha
                    + frame.astype(np.float32) * (1 - alpha)
                ).astype(np.uint8)

            if kind == "handheld_scene":
                horizon = int(WH * 0.62)
                car_wx = -80 + (WW + 160) * (i / max(1, n_frames - 1))
                car_wy = horizon + 74
                cx, cy = _world_to_frame(T, car_wx, car_wy)
                _draw_car(frame, cx, cy, 1.0)
                car_xs.append(cx)

                walk_wx = WW + 60 - (WW + 200) * (i / max(1, n_frames - 1))
                walk_wy = horizon + 34
                wx, wy = _world_to_frame(T, walk_wx, walk_wy)
                _draw_walker(frame, wx, wy, i * 0.5)
                walker_xs.append(wx)

            # subtle frame grain + vignette to feel like real footage
            grain = np.random.default_rng(1000 + i).normal(0, 2.2, frame.shape)
            frame = np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)
            wr.write(frame)

    gt = {
        "kind": kind,
        "width": W,
        "height": H,
        "fps": fps,
        "n_frames": n_frames,
        "mats": [np.round(m, 8).tolist() for m in mats],
        "pose": {
            k: [float(v) for v in path["comp"][k]]
            for k in ("x", "y", "angle", "scale")
        },
        "moving_objects": {
            "car_screen_x": car_xs,
            "walker_screen_x": walker_xs,
        },
    }
    with open(out_gt, "w", encoding="utf-8") as f:
        json.dump(gt, f)
    return {"video": out_mp4, "gt": out_gt, **info}


def generate_all(n_frames: int = 180) -> list[dict]:
    out = []
    for kind in SAMPLE_INFO:
        out.append(render_sample(kind, n_frames=n_frames))
    return out


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for r in generate_all():
            print("generated", r["video"])
    else:
        print(render_sample(which))
