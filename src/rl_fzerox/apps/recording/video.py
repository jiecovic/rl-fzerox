# src/rl_fzerox/apps/recording/video.py
from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Sequence
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
        self._audio_stdin: IO[bytes] | None = None
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
        if self._audio_stdin is None:
            raise RuntimeError("ffmpeg audio writer is not open")
        self._audio_stdin.write(copied_samples.tobytes())

    def _open_process(self, *, width: int, height: int) -> None:
        audio_read_fd: int | None = None
        audio_write_fd: int | None = None
        if self._audio_sample_rate is not None:
            audio_read_fd, audio_write_fd = os.pipe()
        command = _ffmpeg_command(
            ffmpeg_path=self._ffmpeg_path,
            output_path=self._path,
            width=width,
            height=height,
            fps=self._fps,
            audio_sample_rate=self._audio_sample_rate,
            audio_pipe_fd=audio_read_fd,
        )
        try:
            self._process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=() if audio_read_fd is None else (audio_read_fd,),
            )
            if audio_read_fd is not None:
                os.close(audio_read_fd)
                audio_read_fd = None
            if self._process.stdin is None:
                raise RuntimeError("failed to open ffmpeg stdin")
            self._stdin = self._process.stdin
            if audio_write_fd is not None:
                self._audio_stdin = os.fdopen(audio_write_fd, "wb")
                audio_write_fd = None
        except Exception:
            if self._process is not None:
                self._process.kill()
                self._process = None
            self._stdin = None
            if self._audio_stdin is not None:
                self._audio_stdin.close()
                self._audio_stdin = None
            if audio_read_fd is not None:
                os.close(audio_read_fd)
            if audio_write_fd is not None:
                os.close(audio_write_fd)
            raise

    def close(self) -> None:
        if self._stdin is not None:
            self._stdin.close()
            self._stdin = None
        if self._audio_stdin is not None:
            self._audio_stdin.close()
            self._audio_stdin = None
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


def remux_recording_to_mp4(
    input_path: Path,
    *,
    ffmpeg_path: str,
    output_path: Path | None = None,
) -> Path:
    source_path = input_path.expanduser()
    if source_path.suffix.lower() == ".mp4" and output_path is None:
        return source_path
    requested_target_path = (output_path or source_path.with_suffix(".mp4")).expanduser()
    if requested_target_path == source_path:
        raise ValueError(f"remux target must differ from source path: {source_path}")
    target_path = _available_output_path(requested_target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-n",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    result = subprocess.run(  # noqa: S603
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"ffmpeg remux failed with exit code {result.returncode}{suffix}")
    return target_path


def concat_mp4_recordings(
    input_paths: Sequence[Path],
    *,
    ffmpeg_path: str,
    output_path: Path,
) -> Path:
    """Concatenate already-finalized MP4 recordings without re-encoding."""

    source_paths = tuple(path.expanduser() for path in input_paths)
    if not source_paths:
        raise ValueError("at least one MP4 input is required")
    requested_target_path = output_path.expanduser()
    target_path = _available_output_path(requested_target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = target_path.with_name(f".{target_path.stem}.concat.txt")
    list_path.write_text(
        "".join(f"file '{_ffmpeg_concat_path(path)}'\n" for path in source_paths),
        encoding="utf-8",
    )
    try:
        command = [
            ffmpeg_path,
            "-n",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(target_path),
        ]
        result = subprocess.run(  # noqa: S603
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"ffmpeg concat failed with exit code {result.returncode}{suffix}")
    return target_path


def _available_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for counter in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}-{counter:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"no available recording output path near {path}")


def _ffmpeg_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


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
    audio_sample_rate: int | None = None,
    audio_pipe_fd: int | None = None,
) -> list[str]:
    command = [
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
    ]
    if audio_sample_rate is not None:
        if audio_pipe_fd is None:
            raise ValueError("audio_pipe_fd is required when audio_sample_rate is set")
        command.extend(
            [
                "-f",
                "s16le",
                "-ar",
                str(audio_sample_rate),
                "-ac",
                "2",
                "-i",
                f"pipe:{audio_pipe_fd}",
            ]
        )
    else:
        command.append("-an")
    command.extend(
        [
            "-fflags",
            "+genpts",
            "-flush_packets",
            "1",
            "-max_interleave_delta",
            "0",
        ]
    )
    command.extend(
        [
            "-map",
            "0:v:0",
        ]
    )
    if audio_sample_rate is not None:
        command.extend(["-map", "1:a:0"])
    command.extend(
        [
            "-vcodec",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-g",
            f"{max(1, round(fps))}",
            "-keyint_min",
            f"{max(1, round(fps))}",
            "-sc_threshold",
            "0",
            "-bf",
            "0",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if audio_sample_rate is not None:
        command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    command.extend(_ffmpeg_output_args(output_path))
    return command


def _ffmpeg_output_args(output_path: Path) -> list[str]:
    if output_path.suffix.lower() == ".mkv":
        return [
            "-f",
            "matroska",
            "-live",
            "1",
            "-cluster_time_limit",
            "1000",
            str(output_path),
        ]
    return [str(output_path)]


def _imageio_ffmpeg_path() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return None
    return get_ffmpeg_exe()
