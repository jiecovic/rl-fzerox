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

from fzerox_emulator.arrays import Int16Array, NumpyArray, Pcm16Samples, RgbFrame


@dataclass(frozen=True)
class VideoSettings:
    """Video writer settings for one attempt."""

    path: Path
    ffmpeg_path: str
    fps: float


class FfmpegRgbWriter:
    """Encode rendered RGB frames by streaming them into ffmpeg."""

    def __init__(
        self,
        *,
        path: Path,
        ffmpeg_path: str,
        fps: float,
        audio_sample_rate: int | None = None,
    ) -> None:
        self._path = path
        self._ffmpeg_path = ffmpeg_path
        self._fps = fps
        self._audio_sample_rate = audio_sample_rate
        self._process: subprocess.Popen[bytes] | None = None
        self._stdin: IO[bytes] | None = None
        self._audio_file: IO[bytes] | None = None
        self._audio_path: Path | None = None
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

    def write_audio(self, samples: Pcm16Samples) -> None:
        if self._audio_sample_rate is None:
            return
        copied_samples = as_pcm16_samples(samples)
        if copied_samples.size == 0:
            return
        if self._audio_file is None:
            raise RuntimeError("ffmpeg audio writer is not open")
        self._audio_file.write(copied_samples.tobytes())

    def _open_process(self, *, width: int, height: int) -> None:
        command = _ffmpeg_command(
            ffmpeg_path=self._ffmpeg_path,
            output_path=self._path,
            width=width,
            height=height,
            fps=self._fps,
        )
        try:
            self._process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if self._process.stdin is None:
                raise RuntimeError("failed to open ffmpeg stdin")
            self._stdin = self._process.stdin
            if self._audio_sample_rate is not None:
                self._audio_path = _audio_sidecar_path(self._path)
                self._audio_file = self._audio_path.open("wb")
        except Exception:
            if self._process is not None:
                self._process.kill()
                self._process = None
            self._stdin = None
            if self._audio_file is not None:
                self._audio_file.close()
                self._audio_file = None
            if self._audio_path is not None:
                self._audio_path.unlink(missing_ok=True)
                self._audio_path = None
            raise

    def close(self) -> None:
        if self._stdin is not None:
            self._stdin.close()
            self._stdin = None
        if self._audio_file is not None:
            self._audio_file.close()
            self._audio_file = None
        if self._process is None:
            self._discard_audio_sidecar()
            return
        stderr = b""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
        return_code = self._process.wait()
        self._process = None
        if return_code != 0:
            detail = stderr.decode(errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            self._discard_audio_sidecar()
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}{suffix}")
        try:
            self._mux_audio_sidecar()
        finally:
            self._discard_audio_sidecar()

    def _mux_audio_sidecar(self) -> None:
        if self._audio_sample_rate is None or self._audio_path is None:
            return
        if not self._audio_path.exists() or self._audio_path.stat().st_size == 0:
            return
        mux_path = _audio_mux_path(self._path)
        command = _ffmpeg_audio_mux_command(
            ffmpeg_path=self._ffmpeg_path,
            video_path=self._path,
            audio_path=self._audio_path,
            output_path=mux_path,
            audio_sample_rate=self._audio_sample_rate,
        )
        completed = subprocess.run(  # noqa: S603
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            mux_path.unlink(missing_ok=True)
            detail = completed.stderr.decode(errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"ffmpeg audio mux failed with exit code {completed.returncode}{suffix}"
            )
        mux_path.replace(self._path)

    def _discard_audio_sidecar(self) -> None:
        if self._audio_path is not None:
            self._audio_path.unlink(missing_ok=True)
            self._audio_path = None


def resolve_ffmpeg_path() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg
    bundled_ffmpeg = _imageio_ffmpeg_path()
    if bundled_ffmpeg is not None:
        return bundled_ffmpeg
    raise RuntimeError(
        "ffmpeg is required for video recording. Install system ffmpeg or run "
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


def as_pcm16_samples(samples: Pcm16Samples) -> Int16Array:
    copied_samples = np.ascontiguousarray(samples, dtype=np.int16)
    if copied_samples.ndim != 1:
        raise ValueError(f"expected flat PCM samples, got shape {copied_samples.shape}")
    if copied_samples.size % 2 != 0:
        raise ValueError("expected an even number of interleaved stereo PCM samples")
    return copied_samples


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
        "-map",
        "0:v:0",
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


def _ffmpeg_audio_mux_command(
    *,
    ffmpeg_path: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_sample_rate: int,
) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-f",
        "s16le",
        "-ar",
        str(audio_sample_rate),
        "-ac",
        "2",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]


def _audio_sidecar_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{temp_session_id()}.audio.s16le")


def _audio_mux_path(path: Path) -> Path:
    suffix = path.suffix or ".mkv"
    return path.with_name(f".{path.stem}.{temp_session_id()}.audio-mux{suffix}")


def _imageio_ffmpeg_path() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return None
    return get_ffmpeg_exe()
