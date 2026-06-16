# src/rl_fzerox/ui/watch/runtime/career_mode/recording/finalizer.py
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.ui.watch.runtime.career_mode.recording.paths import career_session_video_path
from rl_fzerox.ui.watch.runtime.career_mode.recording.summary import (
    _SegmentSummarySnapshot,
    _SessionSummaryWriter,
    write_segment_summary_files,
)


@dataclass(frozen=True, slots=True)
class _FinalizerJob:
    path: Path
    future: Future[Path]
    summary: _SegmentSummarySnapshot | None = None
    delete_source: bool = False


class _Mp4RecordingFinalizer:
    """Remux closed live Matroska recordings to MP4 without blocking playback."""

    def __init__(
        self,
        *,
        session_source_path: Path | None = None,
        live_source_path: Path | None = None,
        session_mp4_enabled: bool = True,
    ) -> None:
        from rl_fzerox.ui.watch.runtime.career_mode import recording as recording_api

        self._ffmpeg_path = recording_api.resolve_ffmpeg_path()
        self._session_summary = (
            None if session_source_path is None else _SessionSummaryWriter(session_source_path)
        )
        self._session_mp4_enabled = session_mp4_enabled
        if self._session_summary is not None and live_source_path is not None:
            self._session_summary.record_live_video(live_source_path)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="career-recording-remux",
        )
        self._jobs: list[_FinalizerJob] = []
        self._notices: list[str] = []

    def finalize(self, path: Path, *, summary: _SegmentSummarySnapshot | None = None) -> None:
        if path.suffix.lower() != ".mkv":
            if summary is not None:
                write_segment_summary_files(summary, video_path=path)
                if self._session_summary is not None:
                    self._session_summary.record_segment(summary, video_path=path)
            return
        from rl_fzerox.ui.watch.runtime.career_mode import recording as recording_api

        self._jobs.append(
            _FinalizerJob(
                path=path,
                summary=summary,
                delete_source=summary is not None,
                future=self._executor.submit(
                    recording_api.remux_recording_to_mp4,
                    path,
                    ffmpeg_path=self._ffmpeg_path,
                ),
            )
        )

    def drain_notices(self) -> tuple[str, ...]:
        pending: list[_FinalizerJob] = []
        notices = self._notices
        self._notices = []
        for job in self._jobs:
            if not job.future.done():
                pending.append(job)
                continue
            notices.append(_finalizer_job_notice(job, session_summary=self._session_summary))
        self._jobs = pending
        return tuple(notices)

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for job in self._jobs:
            self._notices.append(_finalizer_job_notice(job, session_summary=self._session_summary))
        self._jobs.clear()
        self._finalize_session_video()

    def _finalize_session_video(self) -> None:
        if self._session_summary is None:
            return
        if not self._session_mp4_enabled:
            self._session_summary.write()
            return
        segment_paths = self._session_summary.segment_video_paths()
        if not segment_paths:
            self._session_summary.write()
            return
        try:
            from rl_fzerox.ui.watch.runtime.career_mode import recording as recording_api

            session_video_path = recording_api.concat_mp4_recordings(
                segment_paths,
                ffmpeg_path=self._ffmpeg_path,
                output_path=career_session_video_path(self._session_summary.source_path),
            )
        except Exception as exc:  # pragma: no cover - defensive async error reporting
            self._notices.append(f"session MP4 assembly failed: {exc}")
            self._session_summary.write()
            return
        self._session_summary.record_session_video(session_video_path)
        self._notices.append(f"session MP4 ready: {session_video_path.name}")


def _finalizer_job_notice(
    job: _FinalizerJob,
    *,
    session_summary: _SessionSummaryWriter | None = None,
) -> str:
    try:
        output_path = job.future.result()
        if job.summary is not None:
            write_segment_summary_files(job.summary, video_path=output_path)
            if session_summary is not None:
                session_summary.record_segment(job.summary, video_path=output_path)
        if job.delete_source:
            job.path.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - defensive async error reporting
        return f"MP4 conversion failed: {exc}"
    return f"MP4 ready: {output_path.name}"
