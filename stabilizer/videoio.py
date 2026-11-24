"""Video output through a bundled ffmpeg binary (H.264 / yuv420p for browsers)."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional

import numpy as np

try:
    import imageio_ffmpeg

    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def ffmpeg_exe() -> str:
    return _FFMPEG


def has_audio_stream(path: str) -> bool:
    try:
        p = subprocess.run(
            [_FFMPEG, "-hide_banner", "-i", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return "Audio:" in (p.stderr or "")
    except Exception:
        return False


class FFmpegWriter:
    """Raw BGR24 frames in -> browser-friendly H.264 mp4 out."""

    def __init__(
        self,
        out_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: Optional[str] = None,
        crf: int = 20,
        preset: str = "veryfast",
    ):
        self.out_path = out_path
        self.w, self.h = width, height
        self.audio_source = audio_source
        cmd = [
            _FFMPEG, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", f"{fps:.6f}",
            "-i", "-",
        ]
        if audio_source:
            cmd += ["-i", audio_source, "-map", "0:v:0", "-map", "1:a:0?"]
        else:
            cmd += ["-map", "0:v:0"]
        cmd += [
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if audio_source:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += [out_path]
        self.cmd = cmd
        self.proc: Optional[subprocess.Popen] = None
        self._stderr = b""

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return self

    def write(self, frame_bgr: np.ndarray) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
            raise ValueError(
                f"frame size {frame_bgr.shape[1]}x{frame_bgr.shape[0]} "
                f"!= writer {self.w}x{self.h}"
            )
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except BrokenPipeError:
            err = self._drain_err()
            raise RuntimeError(f"ffmpeg 写入失败: {err[-800:]}")

    def _drain_err(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            return self.proc.stderr.read().decode("utf-8", "ignore")
        except Exception:
            return ""

    def __exit__(self, exc_type, exc, tb):
        if self.proc is None:
            return False
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self._stderr = self.proc.stderr.read() if self.proc.stderr else b""
            rc = self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()
            rc = -9
        if exc_type is None and rc != 0:
            msg = self._stderr.decode("utf-8", "ignore")
            raise RuntimeError(f"ffmpeg 编码失败 (code {rc}): {msg[-800:]}")
        return False


def encode_existing_raw_note() -> None:
    """Marker kept for tooling discovery; no-op."""
    return None
