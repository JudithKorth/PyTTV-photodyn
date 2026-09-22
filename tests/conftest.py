import numpy as np
import pytest

from helpers import TWO_BODY, TWO_PLANET, build_sim  # noqa: F401 (also sets sys.path)


@pytest.fixture
def two_body_sim():
    return build_sim(TWO_BODY)


@pytest.fixture
def two_planet_sim():
    return build_sim(TWO_PLANET)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)
