# tests/test_viewer.py
import numpy as np

from rl_fzerox.ui.viewer import _target_display_size


def test_target_display_size_applies_aspect_correction() -> None:
    frame = np.zeros((240, 640, 3), dtype=np.uint8)

    display_size = _target_display_size(frame, 4.0 / 3.0)

    assert display_size == (640, 480)


def test_target_display_size_falls_back_to_raw_frame_size() -> None:
    frame = np.zeros((240, 640, 3), dtype=np.uint8)

    display_size = _target_display_size(frame, 0.0)

    assert display_size == (640, 240)
