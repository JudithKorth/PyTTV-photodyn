"""Tests for the helpers shared by the light curve readers in src/io/_common.py.

The CHEOPS reader itself needs real FITS files, so what is covered here is the behaviour
that moved out of `read_cheops` into `_common` when the TESS reader was added.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.io._common import (_check_ephemerides, _exposure_time, _normalise,
                            _transiting_planets)


# ----------------------------------------------------------------------
# _normalise
# ----------------------------------------------------------------------
def test_normalise_standardises_each_column():
    covs = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 60.0]])
    out = _normalise(covs)
    assert_allclose(out.mean(0), 0.0, atol=1e-14)
    assert_allclose(out.std(0), 1.0, rtol=1e-14)


def test_normalise_centres_but_does_not_scale_a_constant_column():
    """Scaling a zero-variance column would divide by zero and poison the baseline fit."""
    covs = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    out = _normalise(covs)
    assert_allclose(out[:, 1], 0.0)
    assert np.isfinite(out).all()


def test_normalise_is_idempotent():
    covs = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 60.0]])
    once = _normalise(covs)
    assert_allclose(_normalise(once), once, rtol=1e-14)


# ----------------------------------------------------------------------
# _transiting_planets
# ----------------------------------------------------------------------
def test_transiting_planets_finds_the_planet_in_the_window():
    time = np.linspace(0.95, 1.05, 50)          # around planet 0's transit at t0 = 1.0
    assert _transiting_planets(time, [1.0, 2.5], [3.4, 7.1], 0.0) == (0,)


def test_transiting_planets_is_empty_between_transits():
    time = np.linspace(1.5, 1.6, 50)
    assert _transiting_planets(time, [1.0, 2.5], [3.4, 7.1], 0.0) == ()


def test_transiting_planets_finds_several():
    """Two planets transiting in the same window are both reported."""
    time = np.linspace(0.9, 1.1, 100)
    assert _transiting_planets(time, [1.0, 1.05], [3.4, 7.1], 0.0) == (0, 1)


def test_transiting_planets_padding_catches_a_partial_transit():
    """A centre just outside the window is caught once the box is widened."""
    time = np.linspace(1.02, 1.10, 50)          # the centre at 1.0 falls before the start
    assert _transiting_planets(time, [1.0], [3.4], 0.0) == ()
    assert _transiting_planets(time, [1.0], [3.4], 0.05) == (0,)


def test_transiting_planets_uses_the_nearest_centre():
    """A window many periods from the zero epoch still resolves to its own transit."""
    time = np.linspace(0.95, 1.05, 50) + 10 * 3.4
    assert _transiting_planets(time, [1.0], [3.4], 0.0) == (0,)


def test_transiting_planets_of_an_empty_light_curve():
    assert _transiting_planets(np.array([]), [1.0], [3.4], 0.0) == ()


# ----------------------------------------------------------------------
# _check_ephemerides
# ----------------------------------------------------------------------
def test_check_ephemerides_accepts_a_valid_set():
    _check_ephemerides([1.0, 2.5], [3.4, 7.1])


def test_check_ephemerides_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match='2 zero epochs but 1 periods'):
        _check_ephemerides([1.0, 2.5], [3.4])


@pytest.mark.parametrize('period', [0.0, -3.4, np.inf, np.nan])
def test_check_ephemerides_rejects_bad_periods(period):
    with pytest.raises(ValueError, match='finite and positive'):
        _check_ephemerides([1.0], [period])


# ----------------------------------------------------------------------
# _exposure_time
# ----------------------------------------------------------------------
def test_exposure_time_converts_from_seconds():
    time = np.linspace(0.0, 1.0, 100)
    assert_allclose(_exposure_time(time, 60.0, 1.0 / 86400.0), 60.0 / 86400.0)


def test_exposure_time_passes_days_through():
    time = np.linspace(0.0, 1.0, 100)
    assert_allclose(_exposure_time(time, 0.00139, 1.0), 0.00139)


@pytest.mark.parametrize('value', [None, 0.0, -1.0, np.nan, np.inf, 'not a number'])
def test_exposure_time_falls_back_to_the_median_cadence(value):
    """An unusable header value gives the median cadence, with a warning."""
    time = np.arange(0.0, 1.0, 0.01)
    with pytest.warns(UserWarning, match='median cadence'):
        assert_allclose(_exposure_time(time, value, 1.0, 'a file'), 0.01, rtol=1e-8)


def test_exposure_time_of_a_single_point_is_zero():
    with pytest.warns(UserWarning, match='median cadence'):
        assert _exposure_time(np.array([1.0]), None) == 0.0
