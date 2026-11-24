"""Verifiable frame-to-frame feature tracking and robust 2D motion estimation.

Pipeline per frame pair:

1. goodFeaturesToTrack on the previous frame;
2. Lucas-Kanade flow forward *and* backward, reject tracks that fail the
   round-trip check (explicitly verifiable matches);
3. RANSAC similarity / affine fit, keeping an inlier mask;
4. residual statistics on both inliers and all verified matches.  The
   all-match residual is the honest model mismatch: it grows when parallax or
   independently moving objects violate the single-plane assumption;
5. ORB + ratio-test matching as a fallback when LK yields too few tracks.

Failed pairs are imputed with the last good motion (constant velocity) and
flagged, so the UI can surface degraded frames instead of silently hiding
them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .geometry import I3, as33

ANALYSIS_MAX_EDGE = 960
MIN_TRACKS = 24
MIN_INLIERS = 10
FB_THRESHOLD_PX = 2.0
RANSAC_THRESHOLD_PX = 3.0

ProgressCb = Optional[Callable[[float, str], None]]


@dataclass
class PairMotion:
    index: int  # pair frame i-1 -> i, indices start at 1
    M: np.ndarray  # 3x3, full-resolution mapping prev -> curr (imputed if needed)
    ok: bool  # robust estimate succeeded
    imputed: bool
    method: str  # "lk+ransac" | "orb+ransac" | "constant-velocity" | "identity"
    n_candidates: int = 0
    n_verified: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    rmse_inliers_px: float = 0.0
    residual_median_px: float = 0.0
    residual_p90_px: float = 0.0
    residual_p99_px: float = 0.0
    residual_rms_all_px: float = 0.0
    bad_fraction: float = 0.0
    spatial_residual_std_px: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("M", None)
        return d


@dataclass
class MotionAnalysis:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    model: str
    analysis_scale: float
    pairs: list[PairMotion] = field(default_factory=list)

    @property
    def motions(self) -> list[np.ndarray]:
        return [p.M for p in self.pairs]

    def summary(self) -> dict:
        n_ok = sum(p.ok for p in self.pairs)
        n_imp = sum(p.imputed for p in self.pairs)
        rms = [p.rmse_inliers_px for p in self.pairs if p.n_inliers > 0]
        res = [p.residual_rms_all_px for p in self.pairs if p.n_verified > 0]
        inl = [p.inlier_ratio for p in self.pairs if p.n_verified > 0]
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "model": self.model,
            "n_pairs": len(self.pairs),
            "pairs_ok": n_ok,
            "pairs_imputed": n_imp,
            "mean_inlier_rmse_px": float(np.mean(rms)) if rms else None,
            "max_inlier_rmse_px": float(np.max(rms)) if rms else None,
            "mean_model_residual_px": float(np.mean(res)) if res else None,
            "max_model_residual_px": float(np.max(res)) if res else None,
            "mean_inlier_ratio": float(np.mean(inl)) if inl else None,
            "diagonal_px": math.hypot(self.width, self.height),
        }


def probe_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 1.0 or fps > 240.0 or not math.isfinite(fps):
        fps = 30.0
    if w <= 0 or h <= 0:
        raise RuntimeError("无法读取视频尺寸，文件可能已损坏或编码不受支持")
    return {"width": w, "height": h, "fps": fps, "n_frames_hint": n}


def _feature_params(w_small: int) -> dict:
    return dict(
        maxCorners=700,
        qualityLevel=0.004,
        minDistance=max(5.0, w_small / 120.0),
        blockSize=7,
    )


def _lk_verify(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return verified prev/curr point pairs in analysis coordinates."""
    pts0 = cv2.goodFeaturesToTrack(prev_gray, **_feature_params(prev_gray.shape[1]))
    if pts0 is None or len(pts0) < MIN_TRACKS:
        return np.empty((0, 2)), np.empty((0, 2))

    lk = dict(
        winSize=(21, 21),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts0, None, **lk)
    if p1 is None:
        return np.empty((0, 2)), np.empty((0, 2))
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, p1, None, **lk
    )
    if p0r is None:
        return np.empty((0, 2)), np.empty((0, 2))

    fb_err = np.linalg.norm(pts0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
    good = (
        (st1.reshape(-1) == 1)
        & (st2.reshape(-1) == 1)
        & (fb_err < FB_THRESHOLD_PX)
    )
    return pts0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]


_ORB = None
_BF = None


def _orb_match(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global _ORB, _BF
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
        _BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    kp1, des1 = _ORB.detectAndCompute(prev_gray, None)
    kp2, des2 = _ORB.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return np.empty((0, 2)), np.empty((0, 2))
    knn = _BF.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < MIN_TRACKS:
        return np.empty((0, 2)), np.empty((0, 2))
    a = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float64)
    b = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float64)
    return a, b


def _fit_model(
    a: np.ndarray, b: np.ndarray, model: str
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(a) < MIN_INLIERS:
        return None, None
    kwargs = dict(method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD_PX)
    try:
        if model == "affine":
            M23, inliers = cv2.estimateAffine2D(a, b, **kwargs)
        else:
            M23, inliers = cv2.estimateAffinePartial2D(a, b, **kwargs)
    except cv2.error:
        return None, None
    if M23 is None or inliers is None:
        return None, None
    return M23, inliers.reshape(-1).astype(bool)


def _residual_stats(
    M_small: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    inlier_mask: np.ndarray,
    inv_scale: float,
    w_small: int,
    h_small: int,
) -> dict:
    M = as33(M_small)
    pred = a @ M[:2, :2].T + M[:2, 2]
    err_small = np.linalg.norm(pred - b, axis=1)
    err_full = err_small * inv_scale

    inl = inlier_mask & (np.arange(len(err_small)) < len(inlier_mask))
    if inl.sum() >= 1:
        rmse_in = float(
            math.sqrt(float(np.mean(err_full[inl] ** 2)))
        )
    else:
        rmse_in = 0.0

    # Spatial coherence of residuals (3x3 bins), a parallax fingerprint.
    spatial_std = 0.0
    try:
        bx = np.clip((a[:, 0] / max(w_small, 1) * 3).astype(int), 0, 2)
        by = np.clip((a[:, 1] / max(h_small, 1) * 3).astype(int), 0, 2)
        bin_means = []
        for ix in range(3):
            for iy in range(3):
                m = (bx == ix) & (by == iy)
                if m.sum() >= 6:
                    bin_means.append(float(np.mean(err_full[m])))
        if len(bin_means) >= 3:
            spatial_std = float(np.std(bin_means))
    except Exception:
        spatial_std = 0.0

    return {
        "rmse_inliers_px": rmse_in,
        "residual_median_px": float(np.median(err_full)),
        "residual_p90_px": float(np.percentile(err_full, 90)),
        "residual_p99_px": float(np.percentile(err_full, 99)),
        "residual_rms_all_px": float(math.sqrt(float(np.mean(err_full**2)))),
        "bad_fraction": float(
            np.mean(err_full > RANSAC_THRESHOLD_PX * inv_scale)
        ),
        "spatial_residual_std_px": spatial_std,
    }


def analyze_video(
    path: str,
    model: str = "similarity",
    progress: ProgressCb = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> MotionAnalysis:
    meta = probe_video(path)
    w, h, fps = meta["width"], meta["height"], meta["fps"]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")

    long_edge = max(w, h)
    scale = min(1.0, ANALYSIS_MAX_EDGE / float(long_edge))
    inv_scale = 1.0 / scale
    ws, hs = max(2, int(round(w * scale))), max(2, int(round(h * scale)))

    pairs: list[PairMotion] = []
    last_good_M: Optional[np.ndarray] = None
    idx = 0
    prev_gray = None
    n_total = max(1, meta["n_frames_hint"])

    try:
        while True:
            if should_cancel is not None and should_cancel():
                raise RuntimeError("已取消")
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            idx += 1
            small = cv2.resize(frame, (ws, hs), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                pidx = idx - 1
                a, b = _lk_verify(prev_gray, gray)
                method = "lk+ransac"
                M23, inl = _fit_model(a, b, model)
                if M23 is None or int(inl.sum()) < MIN_INLIERS:
                    a2, b2 = _orb_match(prev_gray, gray)
                    if len(a2) >= MIN_TRACKS:
                        M23o, inlo = _fit_model(a2, b2, model)
                        if M23o is not None and int(inlo.sum()) >= MIN_INLIERS:
                            a, b, M23, inl = a2, b2, M23o, inlo
                            method = "orb+ransac"

                n_verified = len(a)
                if M23 is not None and inl is not None and int(inl.sum()) >= MIN_INLIERS:
                    stats = _residual_stats(
                        M23, a, b, inl, inv_scale, ws, hs
                    )
                    M_full = as33(M23)
                    M_full[0, 2] *= inv_scale
                    M_full[1, 2] *= inv_scale
                    pm = PairMotion(
                        index=pidx,
                        M=M_full,
                        ok=True,
                        imputed=False,
                        method=method,
                        n_candidates=700,
                        n_verified=n_verified,
                        n_inliers=int(inl.sum()),
                        inlier_ratio=float(inl.mean()) if len(inl) else 0.0,
                        **stats,
                    )
                    last_good_M = M_full.copy()
                else:
                    if last_good_M is not None:
                        M_imp = last_good_M.copy()
                        how = "constant-velocity"
                    else:
                        M_imp = I3()
                        how = "identity"
                    pm = PairMotion(
                        index=pidx,
                        M=M_imp,
                        ok=False,
                        imputed=True,
                        method=how,
                        n_verified=n_verified,
                        error="可靠匹配不足，已用邻近运动插值",
                    )
                pairs.append(pm)

            prev_gray = gray
            if progress is not None and (idx % 5 == 0):
                progress(min(0.99, idx / max(n_total, idx)), f"特征分析 {idx} 帧")
    finally:
        cap.release()

    n_frames = idx
    if n_frames < 8:
        raise RuntimeError(
            f"有效视频帧过少（{n_frames} 帧），至少需要约 8 帧才能稳定"
        )

    return MotionAnalysis(
        path=path,
        width=w,
        height=h,
        fps=fps,
        n_frames=n_frames,
        model=model,
        analysis_scale=scale,
        pairs=pairs,
    )
