# tests/test_seed.py
from rl_fzerox.core.seed import episode_seed


def test_episode_seed_is_stable_offset_from_master_seed() -> None:
    assert episode_seed(11, 0) == 11
    assert episode_seed(11, 3) == 14


def test_episode_seed_propagates_none() -> None:
    assert episode_seed(None, 5) is None
