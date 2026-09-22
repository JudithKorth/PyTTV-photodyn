"""Tests for the Kepler reader in src/io/read_kepler.py.

The mission-agnostic window logic lives in src/io/_common.py and is covered by
tests/test_read_tess.py; these tests cover what is Kepler's own: the metadata mapping
(quarter -> sector), the SPOC-only flux columns, and the cadence-dependent supersampling.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import lightkurve as lk

from src.io.read_kepler import (INSTRUMENT, LONG_CADENCE_THRESHOLD, PASSBAND,
                                _resolve_nsamples, read_kepler, read_lc)

# A two-planet system observed for 30 days.
T0 = np.array([1.0, 2.5])
PER = np.array([3.4, 7.1])
DEPTHS = (0.01, 0.006)
HDUR = 0.05                             # half a transit duration, days
SHORT_CADENCE = 1 / 1440                # Kepler short cadence is 58.8 s
LONG_CADENCE = 29.4 / 60 / 24           # Kepler long cadence is 29.4 min


def photometry(tspan: float = 30.0, cadence: float = SHORT_CADENCE, t0=T0, per=PER):
    """Times, fluxes and errors of a noiseless two-planet light curve."""
    time = np.arange(0.0, tspan, cadence)
    flux = np.ones(time.size)
    for zt, p, d in zip(t0, per, DEPTHS):
        flux[np.abs((time - zt + 0.5 * p) % p - 0.5 * p) < HDUR] -= d
    return time, flux, np.full(time.size, 2e-4)


def make_lc(time=None, flux=None, ferr=None, column='pdcsap_flux', scale=1234.0,
            quarter=16, obsmode='long cadence', cadence=LONG_CADENCE, timedel='cadence'):
    """A lightkurve LightCurve carrying a Kepler-style flux column.

    `scale` puts the flux on an arbitrary instrumental scale, so a test that sees a
    normalised flux knows the reader normalised it. `timedel='cadence'` copies the
    sampling cadence into the TIMEDEL header, as in a real file; `None` omits the header.
    """
    if time is None:
        time, flux, ferr = photometry(cadence=cadence)
    lc = lk.LightCurve(time=time, flux=flux, flux_err=ferr)
    lc[column] = flux * scale
    lc[f'{column}_err'] = ferr * scale
    if quarter is not None:
        lc.meta['QUARTER'] = quarter
    if obsmode is not None:
        lc.meta['OBSMODE'] = obsmode
    if timedel is not None:
        lc.meta['TIMEDEL'] = cadence if timedel == 'cadence' else timedel
    return lc


# ----------------------------------------------------------------------
# Cadence resolution
# ----------------------------------------------------------------------
def test_resolve_nsamples_trusts_obsmode():
    assert _resolve_nsamples({'OBSMODE': 'long cadence'}, LONG_CADENCE, 1, 10) == 10
    assert _resolve_nsamples({'OBSMODE': 'short cadence'}, SHORT_CADENCE, 1, 10) == 1


def test_resolve_nsamples_obsmode_is_case_insensitive():
    assert _resolve_nsamples({'OBSMODE': 'Long Cadence'}, LONG_CADENCE, 1, 10) == 10
    assert _resolve_nsamples({'OBSMODE': 'SHORT CADENCE'}, SHORT_CADENCE, 1, 10) == 1


def test_resolve_nsamples_obsmode_wins_over_the_exposure_time():
    """A header that says short cadence is trusted even with a long exposure time."""
    assert _resolve_nsamples({'OBSMODE': 'short cadence'}, LONG_CADENCE, 1, 10) == 1


def test_resolve_nsamples_falls_back_to_the_exposure_time():
    assert _resolve_nsamples({}, LONG_CADENCE, 1, 10) == 10
    assert _resolve_nsamples({}, SHORT_CADENCE, 1, 10) == 1
    assert _resolve_nsamples({'OBSMODE': 'something else'}, LONG_CADENCE, 1, 10) == 10


def test_resolve_nsamples_never_supersamples_a_zero_exposure_time():
    """LCData warns about nsamples > 1 with exptime == 0, so the resolver returns one."""
    assert _resolve_nsamples({'OBSMODE': 'long cadence'}, 0.0, 1, 10) == 1
    assert _resolve_nsamples({}, 0.0, 1, 10) == 1


# ----------------------------------------------------------------------
# Flux column selection
# ----------------------------------------------------------------------
def test_sap_selects_the_sap_column():
    lcs = read_lc(make_lc(column='sap_flux'), T0, PER, wbaseline=12.0, type='sap')
    assert lcs.size > 0


def test_pdc_does_not_fall_back_to_the_qlp_column():
    """Kepler has no QLP products, so kspsap_flux is an unknown format here."""
    lc = make_lc(column='kspsap_flux')
    with pytest.raises(ValueError, match='Unknown Kepler light curve format'):
        read_lc(lc, T0, PER, wbaseline=12.0)


def test_invalid_type_raises():
    with pytest.raises(ValueError, match="use 'sap' or 'pdc'"):
        read_lc(make_lc(), T0, PER, wbaseline=12.0, type='nonsense')


# ----------------------------------------------------------------------
# read_lc
# ----------------------------------------------------------------------
def test_read_lc_metadata():
    lcs = read_lc(make_lc(), T0, PER, wbaseline=12.0)

    assert lcs.size == 13
    assert lcs.passband_names == [PASSBAND]
    assert all(lc.instrument == INSTRUMENT for lc in lcs)
    assert all(lc.segment == 0 for lc in lcs)
    assert set(lcs.sectors) == {16}                   # the quarter fills the sector slot
    assert_allclose(lcs.exptimes, LONG_CADENCE)       # TIMEDEL, already in days
    assert_array_equal(lcs.ncovs, 0)                  # no covariates
    assert lcs.has_pids


def test_read_lc_normalises_the_flux():
    lcs = read_lc(make_lc(scale=1234.0), T0, PER, wbaseline=12.0)
    for f in lcs.fluxes:
        assert_allclose(np.median(f), 1.0, rtol=1e-12)


def test_long_cadence_windows_are_supersampled():
    lcs = read_lc(make_lc(obsmode='long cadence', cadence=LONG_CADENCE),
                  T0, PER, wbaseline=12.0)
    assert_array_equal(lcs.nsamples, 10)


def test_short_cadence_windows_are_not_supersampled():
    lcs = read_lc(make_lc(obsmode='short cadence', cadence=SHORT_CADENCE),
                  T0, PER, wbaseline=12.0)
    assert_array_equal(lcs.nsamples, 1)


def test_nsamples_defaults_can_be_overridden():
    lcs = read_lc(make_lc(obsmode='long cadence', cadence=LONG_CADENCE),
                  T0, PER, wbaseline=12.0, nsamples_long=5)
    assert_array_equal(lcs.nsamples, 5)
    lcs = read_lc(make_lc(obsmode='short cadence', cadence=SHORT_CADENCE),
                  T0, PER, wbaseline=12.0, nsamples_short=3)
    assert_array_equal(lcs.nsamples, 3)


def test_missing_obsmode_classifies_from_the_exposure_time():
    lcs = read_lc(make_lc(obsmode=None, cadence=LONG_CADENCE), T0, PER, wbaseline=12.0)
    assert_array_equal(lcs.nsamples, 10)
    lcs = read_lc(make_lc(obsmode=None, cadence=SHORT_CADENCE), T0, PER, wbaseline=12.0)
    assert_array_equal(lcs.nsamples, 1)


def test_read_lc_falls_back_to_the_median_cadence():
    """A light curve without TIMEDEL gets its exposure time from the cadence, with a
    warning, and the cadence classification still works on the fallback value."""
    lc = make_lc(obsmode=None, cadence=LONG_CADENCE, timedel=None)
    with pytest.warns(UserWarning, match='median cadence'):
        lcs = read_lc(lc, T0, PER, wbaseline=12.0)
    assert_allclose(lcs.exptimes, LONG_CADENCE, rtol=1e-8)
    assert_array_equal(lcs.nsamples, 10)


def test_read_lc_without_a_quarter():
    lcs = read_lc(make_lc(quarter=None), T0, PER, wbaseline=12.0)
    assert set(lcs.sectors) == {-1}


def test_wnids_group_by_quarter():
    """Windows share a white noise parameter with the rest of their quarter."""
    time, flux, ferr = photometry(cadence=LONG_CADENCE)
    a = read_lc(make_lc(time, flux, ferr, quarter=16), T0, PER, wbaseline=12.0)
    b = read_lc(make_lc(time + 100.0, flux, ferr, quarter=17), T0, PER, wbaseline=12.0)
    lcs = a + b

    assert set(lcs.sectors) == {16, 17}
    assert set(np.unique(lcs.wnids)) == {0, 1}
    assert (lcs.wnids[:a.size] == 0).all()
    assert (lcs.wnids[a.size:] == 1).all()


# ----------------------------------------------------------------------
# read_kepler
# ----------------------------------------------------------------------
def test_read_kepler_warns_and_returns_an_empty_group_when_nothing_matches(tmp_path):
    with pytest.warns(UserWarning, match='No Kepler light curve files'):
        lcs = read_kepler(tmp_path, 4349452, T0, PER, wbaseline=12.0)
    assert lcs.size == 0


def test_read_kepler_validates_the_ephemerides(tmp_path):
    with pytest.raises(ValueError, match='zero epochs'):
        read_kepler(tmp_path, 4349452, [1.0], [3.4, 7.1], wbaseline=12.0)


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------
def test_group_feeds_the_photodynamical_lpf():
    """The supersampling counts survive the trip into the LPF."""
    from src.pdlpf import PhotoDynamicalLPF

    lcs = read_lc(make_lc(), T0, PER, wbaseline=6.0)
    lpf = PhotoDynamicalLPF('kepler', 2, list(T0), list(PER), lcs,
                            is_transiting=[True, True], tref=0.0, lnlikelihood='wn')

    assert lpf.nlc == lcs.size
    assert lpf.pids == list(lcs.pids)
    assert list(lpf.passbands) == ['Kepler']
    assert_array_equal(lpf.nsamples, 10)
    assert_allclose(lpf.exptimes, LONG_CADENCE)
    assert lpf.n_noise_blocks == 1                    # a single quarter
