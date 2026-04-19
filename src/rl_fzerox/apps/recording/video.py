# src/rl_fzerox/apps/recording/video.py
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO

import numpy as np

from fzerox_emulator.arrays import NumpyArray, RgbFrame


@dataclass(frozen=True)
class VideoSettings:
    """MP4 writer settings for one attempt."""

    path: Path
    ffmpeg_path: str
    fps: float


class FfmpegRgbWriter:
    """Encode one attempt by streaming rendered RGB frames into ffmpeg."""

    def __init__(
        self,
        *,
        path: Path,
        ffmpeg_path: str,
        fps: float,
    ) -> None:
        self._path = path
        self._ffmpeg_path = ffmpeg_path
        self._fps = fps
        self._process: subprocess.Popen[bytes] | None = None
        self._stdin: IO[bytes] | None = None
        self._shape: tuple[int, int, int] | None = None

    def __enter__(self) -> FfmpegRgbWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def write(self, frame: RgbFrame) -> None:
        copied_frame = np.ascontiguousarray(frame)
        if self._shape is None:
            if copied_frame.ndim != 3 or copied_frame.shape[2] != 3:
                raise ValueError(f"expected RGB frame with shape HxWx3, got {copied_frame.shape}")
            self._shape = (
                int(copied_frame.shape[0]),
                int(copied_frame.shape[1]),
                int(copied_frame.shape[2]),
            )
            self._open_process(width=self._shape[1], height=self._shape[0])
        if copied_frame.shape != self._shape:
            raise ValueError(
                f"frame shape changed during recording: {copied_frame.shape} != {self._shape}"
            )
        if self._stdin is None:
            raise RuntimeError("ffmpeg writer is not open")
        self._stdin.write(copied_frame.tobytes())

    def _open_process(self, *, width: int, height: int) -> None:
        command = _ffmpeg_command(
            ffmpeg_path=self._ffmpeg_path,
            output_path=self._path,
            width=width,
            height=height,
            fps=self._fps,
        )
        self._process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None:
            raise RuntimeError("failed to open ffmpeg stdin")
        self._stdin = self._process.stdin

    def close(self) -> None:
        if self._stdin is not None:
            self._stdin.close()
            self._stdin = None
        if self._process is None:
            return
        stderr = b""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
        return_code = self._process.wait()
        self._process = None
        if return_code != 0:
            detail = stderr.decode(errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}{suffix}")


def resolve_ffmpeg_path() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg
    bundled_ffmpeg = _imageio_ffmpeg_path()
    if bundled_ffmpeg is not None:
        return bundled_ffmpeg
    raise RuntimeError(
        "ffmpeg is required for MP4 recording. Install system ffmpeg or run "
        "`.venv/bin/python -m pip install imageio-ffmpeg`."
    )


def resolve_video_fps(
    *,
    native_fps: float,
    override: float | None,
) -> float:
    if override is not None:
        return override
    return max(float(native_fps), 1.0)


def attempt_output_path(
    output_path: Path,
    attempt: int,
    *,
    session_id: str | None = None,
) -> Path:
    if session_id is None:
        return output_path.with_name(f".{output_path.stem}.attempt-{attempt:03d}.mp4")
    return output_path.with_name(f".{output_path.stem}.{session_id}.attempt-{attempt:03d}.mp4")


def temp_session_id() -> str:
    return f"session-{os.getpid()}-{time.time_ns()}"


def ensure_attempt_path_available(path: Path) -> None:
    if not path.exists():
        return
    raise FileExistsError(f"temporary recording artifact already exists: {path}")


def discard_attempt_video(path: Path) -> None:
    path.unlink(missing_ok=True)


def as_rgb_frame(frame: NumpyArray) -> RgbFrame:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected RGB frame with shape HxWx3, got {frame.shape}")
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _ffmpeg_command(
    *,
    ffmpeg_path: str,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def _imageio_ffmpeg_path() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return None
    return get_ffmpeg_exe()
