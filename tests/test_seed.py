# tests/test_seed.py
from rl_fzerox.core.seed import derive_seed, splitmix64


def test_splitmix64_is_stable_for_one_input() -> None:
    assert splitmix64(11) == splitmix64(11)


def test_derive_seed_domain_separates_components() -> None:
    assert derive_seed(11, 1) != derive_seed(11, 2)


def test_derive_seed_propagates_none() -> None:
    assert derive_seed(None, 5) is None
