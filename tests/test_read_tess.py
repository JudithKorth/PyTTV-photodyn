"""Tests for the TESS reader in src/io/read_tess.py."""
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import lightkurve as lk

from src.io.read_tess import (INSTRUMENT, PASSBAND, _extract_windows, _flux_columns,
                              read_lc, read_tess)

# A two-planet system observed for 30 days at a two-minute cadence.
T0 = np.array([1.0, 2.5])
PER = np.array([3.4, 7.1])
DEPTHS = (0.01, 0.006)
HDUR = 0.05                     # half a transit duration, days
CADENCE = 2 / 1440


def photometry(tspan: float = 30.0, cadence: float = CADENCE, t0=T0, per=PER):
    """Times, fluxes and errors of a noiseless two-planet light curve."""
    time = np.arange(0.0, tspan, cadence)
    flux = np.ones(time.size)
    for zt, p, d in zip(t0, per, DEPTHS):
        flux[np.abs((time - zt + 0.5 * p) % p - 0.5 * p) < HDUR] -= d
    return time, flux, np.full(time.size, 2e-4)


def make_lc(time=None, flux=None, ferr=None, column='pdcsap_flux', scale=1234.0,
            sector=17, timedel=CADENCE, mask=None):
    """A lightkurve LightCurve carrying a TESS-style flux column.

    `scale` puts the flux on an arbitrary instrumental scale, so a test that sees a
    normalised flux knows the reader normalised it.
    """
    if time is None:
        time, flux, ferr = photometry()
    lc = lk.LightCurve(time=time, flux=flux, flux_err=ferr)
    values = flux * scale
    if mask is not None:
        values = np.ma.masked_array(values, mask=mask)
    lc[column] = values
    lc[f'{column}_err'] = ferr * scale
    if sector is not None:
        lc.meta['SECTOR'] = sector
    if timedel is not None:
        lc.meta['TIMEDEL'] = timedel
    return lc


# ----------------------------------------------------------------------
# Window extraction
# ----------------------------------------------------------------------
def test_extract_windows_cuts_one_window_per_transit():
    """One window per transit, each containing the transit centre it was cut around."""
    time, flux, ferr = photometry()
    windows = _extract_windows(time, flux, ferr, [T0[0]], [PER[0]], wbaseline=12.0)

    expected = T0[0] + np.arange(9) * PER[0]        # 9 transits in 30 days
    expected = expected[expected < time[-1]]
    assert len(windows) == expected.size

    for (t, _, _), centre in zip(windows, expected):
        assert t.min() <= centre <= t.max()
        assert np.ptp(t) <= 12.0 / 24.0 + CADENCE   # no wider than the requested baseline


def test_extract_windows_drops_short_windows():
    """A window spanning less than min_window is discarded."""
    time, flux, ferr = photometry()
    kw = dict(zero_epochs=[T0[0]], periods=[PER[0]], wbaseline=6.0)
    assert len(_extract_windows(time, flux, ferr, min_window=1.0, **kw)) == 9
    assert len(_extract_windows(time, flux, ferr, min_window=12.0, **kw)) == 0


def test_extract_windows_normalises_flux_and_errors_together():
    """Each window is divided by its own median, errors included."""
    time, flux, ferr = photometry()
    scale = 1234.0
    windows = _extract_windows(time, flux * scale, ferr * scale, T0, PER, wbaseline=12.0)

    for t, f, e in windows:
        assert_allclose(np.median(f), 1.0, rtol=1e-12)
        # The relative uncertainty survives the normalisation.
        assert_allclose(e, 2e-4 / np.median(flux[np.isin(time, t)]), rtol=1e-8)


def test_extract_windows_merges_overlapping_planets():
    """Windows of two planets that overlap become a single contiguous window."""
    time, flux, ferr = photometry()
    # Two planets transiting at almost the same time.
    t0 = [5.0, 5.02]
    separate = _extract_windows(time, flux, ferr, [t0[0]], [50.0], wbaseline=2.0)
    merged = _extract_windows(time, flux, ferr, t0, [50.0, 50.0], wbaseline=2.0)
    assert len(separate) == 1 and len(merged) == 1
    assert merged[0][0].size > separate[0][0].size


def test_extract_windows_needs_at_least_two_points():
    empty = np.array([])
    assert _extract_windows(empty, empty, empty, T0, PER, wbaseline=12.0) == []


# ----------------------------------------------------------------------
# Flux column selection
# ----------------------------------------------------------------------
def test_pdc_falls_back_to_the_qlp_column():
    """A QLP light curve has kspsap_flux but no pdcsap_flux."""
    lc = make_lc(column='kspsap_flux')
    flux, ferr = _flux_columns(lc, 'pdc')
    assert flux.size == ferr.size == len(lc)


def test_sap_on_a_qlp_light_curve_raises_valueerror():
    """Regression: the legacy reader tested for kspsap_flux and then read sap_flux.

    That raised AttributeError on a QLP file rather than reporting an unknown format.
    """
    lc = make_lc(column='kspsap_flux')
    with pytest.raises(ValueError, match='Unknown TESS light curve format'):
        _flux_columns(lc, 'sap')


def test_unknown_flux_format_raises():
    lc = make_lc(column='some_other_flux')
    with pytest.raises(ValueError, match='Unknown TESS light curve format'):
        _flux_columns(lc, 'pdc')


def test_invalid_type_raises():
    with pytest.raises(ValueError, match="use 'sap' or 'pdc'"):
        _flux_columns(make_lc(), 'nonsense')


def test_masked_and_nonfinite_points_are_dropped():
    """Masked cadences and non-finite values never reach LCData."""
    time, flux, ferr = photometry()
    mask = np.zeros(time.size, bool)
    mask[:20] = True
    flux = flux.copy()
    flux[100:110] = np.nan
    lc = make_lc(time, flux, ferr, mask=mask)

    lcs = read_lc(lc, T0, PER, wbaseline=12.0)
    joined = np.concatenate(lcs.fluxes)
    assert np.isfinite(joined).all()
    assert np.isfinite(np.concatenate(lcs.times)).all()


# ----------------------------------------------------------------------
# read_lc
# ----------------------------------------------------------------------
def test_read_lc_metadata():
    lcs = read_lc(make_lc(), T0, PER, wbaseline=12.0)

    assert lcs.size == 13
    assert lcs.passband_names == [PASSBAND]
    assert all(lc.instrument == INSTRUMENT for lc in lcs)
    assert all(lc.segment == 0 for lc in lcs)
    assert set(lcs.sectors) == {17}
    assert_allclose(lcs.exptimes, CADENCE)            # TIMEDEL, already in days
    assert_array_equal(lcs.ncovs, 0)                  # no covariates
    assert lcs.has_pids


def test_read_lc_assigns_planets_to_windows():
    """Every window names the planets whose transit centre falls in it."""
    lcs = read_lc(make_lc(), T0, PER, wbaseline=12.0)
    assert lcs.n_planets == 2
    assert sum(p == (0,) for p in lcs.pids) == 9      # P = 3.4 d over 30 d
    assert sum(p == (1,) for p in lcs.pids) == 4      # P = 7.1 d over 30 d
    assert all(len(p) >= 1 for p in lcs.pids)


def test_read_lc_normalises_the_flux():
    """The arbitrary instrumental flux scale is divided out."""
    lcs = read_lc(make_lc(scale=1234.0), T0, PER, wbaseline=12.0)
    for f in lcs.fluxes:
        assert_allclose(np.median(f), 1.0, rtol=1e-12)


def test_read_lc_falls_back_to_the_median_cadence():
    """A light curve without TIMEDEL gets its exposure time from the cadence, with a warning."""
    lc = make_lc(timedel=None)
    with pytest.warns(UserWarning, match='median cadence'):
        lcs = read_lc(lc, T0, PER, wbaseline=12.0)
    assert_allclose(lcs.exptimes, CADENCE, rtol=1e-8)


def test_read_lc_without_a_sector():
    lcs = read_lc(make_lc(sector=None), T0, PER, wbaseline=12.0)
    assert set(lcs.sectors) == {-1}


def test_read_lc_drop_empty():
    """A window with no transiting planet can be kept or dropped."""
    # A baseline so narrow that only the exact centres are kept, with a third planet whose
    # ephemeris puts no transit in the data.
    lcs = read_lc(make_lc(), T0, PER, wbaseline=12.0, drop_empty=True)
    assert all(p for p in lcs.pids)


def test_read_lc_validates_the_ephemerides():
    lc = make_lc()
    with pytest.raises(ValueError, match='zero epochs'):
        read_lc(lc, [1.0], [3.4, 7.1], wbaseline=12.0)
    with pytest.raises(ValueError, match='finite and positive'):
        read_lc(lc, [1.0], [-3.4], wbaseline=12.0)


def test_wnids_group_by_sector():
    """Windows share a white noise parameter with the rest of their sector."""
    time, flux, ferr = photometry()
    a = read_lc(make_lc(time, flux, ferr, sector=17), T0, PER, wbaseline=12.0)
    b = read_lc(make_lc(time + 100.0, flux, ferr, sector=24), T0, PER, wbaseline=12.0)
    lcs = a + b

    assert set(lcs.sectors) == {17, 24}
    assert set(np.unique(lcs.wnids)) == {0, 1}
    assert (lcs.wnids[:a.size] == 0).all()
    assert (lcs.wnids[a.size:] == 1).all()


# ----------------------------------------------------------------------
# Cadence-dependent supersampling
# ----------------------------------------------------------------------
def ffi_lc(cadence):
    """A light curve sampled at a full-frame-image cadence."""
    time, flux, ferr = photometry(cadence=cadence)
    return make_lc(time, flux, ferr, timedel=cadence)


def test_two_minute_cadence_is_not_supersampled():
    lcs = read_lc(make_lc(), T0, PER, wbaseline=12.0)
    assert_array_equal(lcs.nsamples, 1)


def test_ffi_cadences_are_supersampled():
    for cadence in (10 / 1440, 30 / 1440):
        lcs = read_lc(ffi_lc(cadence), T0, PER, wbaseline=12.0)
        assert_array_equal(lcs.nsamples, 10)


def test_fast_ffi_cadence_is_not_supersampled():
    """The 200 s full-frame images sit below the threshold."""
    lcs = read_lc(ffi_lc(200 / 86400), T0, PER, wbaseline=12.0)
    assert_array_equal(lcs.nsamples, 1)


def test_nsamples_defaults_can_be_overridden():
    lcs = read_lc(ffi_lc(30 / 1440), T0, PER, wbaseline=12.0, nsamples_long=5)
    assert_array_equal(lcs.nsamples, 5)
    lcs = read_lc(make_lc(), T0, PER, wbaseline=12.0, nsamples_short=3)
    assert_array_equal(lcs.nsamples, 3)


def test_missing_timedel_still_classifies_the_cadence():
    """The median-cadence fallback exposure time feeds the cadence classification."""
    time, flux, ferr = photometry(cadence=30 / 1440)
    lc = make_lc(time, flux, ferr, timedel=None)
    with pytest.warns(UserWarning, match='median cadence'):
        lcs = read_lc(lc, T0, PER, wbaseline=12.0)
    assert_array_equal(lcs.nsamples, 10)


# ----------------------------------------------------------------------
# read_tess
# ----------------------------------------------------------------------
def test_read_tess_warns_and_returns_an_empty_group_when_nothing_matches(tmp_path):
    with pytest.warns(UserWarning, match='No TESS light curve files'):
        lcs = read_tess(tmp_path, 12345678, T0, PER, wbaseline=12.0)
    assert lcs.size == 0


def test_read_tess_validates_the_ephemerides(tmp_path):
    with pytest.raises(ValueError, match='zero epochs'):
        read_tess(tmp_path, 12345678, [1.0], [3.4, 7.1], wbaseline=12.0)


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------
def test_group_feeds_the_photodynamical_lpf():
    """The real acceptance test: the LPF rejects light curves with undeclared pids."""
    from src.pdlpf import PhotoDynamicalLPF

    lcs = read_lc(make_lc(), T0, PER, wbaseline=6.0)
    lpf = PhotoDynamicalLPF('tess', 2, list(T0), list(PER), lcs,
                            is_transiting=[True, True], tref=0.0, lnlikelihood='wn')

    assert lpf.nlc == lcs.size
    assert lpf.pids == list(lcs.pids)
    assert list(lpf.passbands) == ['TESS']
    # No covariates, so every window gets an intercept-only baseline.
    assert lpf.lstsq_baseline.ncoef == [1] * lcs.size
    assert lpf.n_noise_blocks == 1                    # a single sector
