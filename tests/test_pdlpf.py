"""Tests for PhotoDynamicalLPF and the njit helpers in pdlpf.py."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")

from helpers import EXPTIME, window
from pytransit.param import NormalPrior as NP
from pytransit.orbits.orbits_py import as_from_rhop, i_from_baew
import astropy.units as u
from astropy.constants import M_sun

from pytransit.utils.io import LCData, LCDataGroup, RVData, RVDataGroup

from src.pdmodel import PhotoDynamicalModel, TransitCenter
from src.pdlpf import PhotoDynamicalLPF, map_ldc, lnlike_normal, nan_lnlike_normal
from src.rvlikelihood import WNRVLikelihood, QPGPRVLikelihood, with_george

# ----------------------------------------------------------------------
# Physical truth system (2 planets, 1 passband)
# ----------------------------------------------------------------------
MSTAR, RSTAR = 1.0, 1.0
Q1, Q2 = 0.36, 0.4                      # -> u, v = 0.48, 0.12
UV = np.squeeze(map_ldc(np.array([[Q1, Q2]])))
MP = np.array([3e-6, 5e-5])
K = np.array([0.1, 0.08])
T0 = np.array([1.0, 2.5])
PER = np.array([3.4, 7.1])
B = np.array([0.25, 0.30])
E = np.array([0.05, 0.03])
W = np.array([0.4, 1.2])
OM = np.array([np.pi, np.pi])
RHO = ((MSTAR * u.M_sun).to(u.g) / (4. / 3. * np.pi * (RSTAR * u.R_sun).to(u.cm) ** 3)).value
A = as_from_rhop(RHO, PER)
INC = i_from_baew(B, A, E, W)

RVTIMES = np.linspace(0.5, 14.0, 8)
TREF = 0.0

PV_VALUES = {
    'mstar': MSTAR, 'rstar': RSTAR, 'q1_tess': Q1, 'q2_tess': Q2,
    'log10mplanet_0': np.log10(MP[0]), 'k_0': K[0], 't0_0': T0[0], 'p_0': PER[0],
    'b_0': B[0], 'secosw_0': np.sqrt(E[0]) * np.cos(W[0]),
    'sesinw_0': np.sqrt(E[0]) * np.sin(W[0]), 'omega_0': OM[0],
    'log10mplanet_1': np.log10(MP[1]), 'k_1': K[1], 't0_1': T0[1], 'p_1': PER[1],
    'b_1': B[1], 'secosw_1': np.sqrt(E[1]) * np.cos(W[1]),
    'sesinw_1': np.sqrt(E[1]) * np.sin(W[1]), 'omega_1': OM[1],
    'srv_0': 0.0, 'log10rvj_0': -1.0,
}


def truth():
    """Evaluate the underlying photodynamical model with the truth parameters."""
    times = [window(T0[0]), window(T0[0] + PER[0])]
    lctimes = np.concatenate(times)
    lcids = np.concatenate([np.full(t.size, i) for i, t in enumerate(times)])
    tm = PhotoDynamicalModel(2, np.array([True, True]), TREF,
                             lctimes=lctimes, pids=[[0, 1], [0, 1]], lcids=lcids,
                             pbids=[0, 0], exptimes=[EXPTIME] * 2, nsamples=[1, 1],
                             rvtimes=RVTIMES, tcs=T0[0] + np.arange(2) * PER[0],
                             tcipl=np.zeros(2, int), with_gr=True)
    fl, rv, tc = tm(MSTAR, RSTAR, UV, MP, K, T0, PER, INC, E, W, OM)
    return times, fl.copy(), rv.copy(), tc.copy()


def make_group(times, fluxes, errors, pids=((0, 1), (0, 1)), passbands=('tess', 'tess')):
    """Pack the canonical two-light-curve photometry into a LCDataGroup."""
    return LCDataGroup([
        LCData(t, f, e, passband=pb, pids=p, exptime=EXPTIME, nsamples=1)
        for t, f, e, p, pb in zip(times, fluxes, errors, pids, passbands)])


def make_rv_group(times, values, errors, instruments=('HARPS',)):
    """Pack radial velocity sets into an RVDataGroup."""
    return RVDataGroup([RVData(t, v, e, instrument=i)
                        for t, v, e, i in zip(times, values, errors, instruments)])


@pytest.fixture(scope="module")
def lpf_and_truth():
    np.random.seed(42)
    times, fl, rv, tc = truth()
    n0 = times[0].size
    fluxes = [fl[:n0] + np.random.normal(0, 2e-4, n0),
              fl[n0:] + np.random.normal(0, 2e-4, times[1].size)]
    errors = [np.full(t.size, 2e-4) for t in times]
    rv_values = rv + np.random.normal(0, 0.5, rv.size)
    rv_errors = np.full(rv.size, 0.5)
    center_times = [tc + np.random.normal(0, 20 / 86400, tc.size)]
    center_errors = [np.full(tc.size, 30 / 86400)]

    lpf = PhotoDynamicalLPF('test', 2, make_group(times, fluxes, errors),
                            make_rv_group([RVTIMES], [rv_values], [rv_errors]),
                            (center_times, center_errors),
                            zero_epochs=list(T0), periods=list(PER),
                            is_transiting=[True, True],
                            tref=TREF, lnlikelihood='wn')

    pv = lpf.ps.sample_from_prior(1)[0]
    for name, value in PV_VALUES.items():
        pv[lpf.ps.find_pid(name)] = value
    return lpf, pv, fl, rv, tc


# ----------------------------------------------------------------------
# njit helpers
# ----------------------------------------------------------------------
def test_map_ldc_triangular_transform():
    uv = np.squeeze(map_ldc(np.array([[0.36, 0.4]])))
    a, b = np.sqrt(0.36), 2 * 0.4
    assert np.allclose(uv, [a * b, a * (1 - b)])  # (0.48, 0.12)


def test_lnlike_normal_matches_scipy(rng):
    o, m, e = rng.normal(0, 1, 50), rng.normal(0, 1, 50), rng.uniform(0.5, 2, 50)
    assert np.isclose(lnlike_normal(o, m, e), norm.logpdf(o, m, e).sum())


def test_nan_lnlike_normal_penalizes_nans(rng):
    o, e = rng.normal(0, 1, 20), np.full(20, 1.0)
    m = o.copy()
    base = nan_lnlike_normal(o, m, e)
    m[3] = np.nan
    assert nan_lnlike_normal(o, m, e) < base - 1e5
    assert np.isfinite(nan_lnlike_normal(o, m, e))


# ----------------------------------------------------------------------
# LPF construction and model evaluation
# ----------------------------------------------------------------------
def test_parameter_set_layout(lpf_and_truth):
    lpf = lpf_and_truth[0]
    names = [p.name for p in lpf.ps]
    assert names[:4] == ['mstar', 'rstar', 'q1_tess', 'q2_tess']
    assert names[4:12] == ['log10mplanet_0', 'k_0', 't0_0', 'p_0', 'b_0',
                           'secosw_0', 'sesinw_0', 'omega_0']
    for n in ('srv_0', 'log10rvj_0'):
        assert n in names
    for n in ('rv_trend', 'rv_sine_amplitude', 'rv_sine_period', 'rv_sine_phase'):
        assert n not in names  # default rv_slope_order is 0; the sinusoid is gone


def test_transit_model_reproduces_truth(lpf_and_truth):
    lpf, pv, fl, rv, tc = lpf_and_truth
    mfl, mrv, mtc = lpf.transit_model(pv)
    assert np.abs(mfl - fl).max() < 1e-8
    assert np.abs(mrv - rv).max() < 1e-6  # srv_0 = 0 and no trend by default
    assert np.abs(mtc - tc).max() * 86400 < 0.01
    assert 0.005 < 1 - mfl.min() < 0.03


def test_transit_model_out_of_bounds_returns_inf(lpf_and_truth):
    lpf, pv = lpf_and_truth[:2]
    bad = pv.copy()
    bad[lpf.ps.find_pid('k_0')] = -0.01
    mfl, mrv, mtc = lpf.transit_model(bad)
    assert np.all(np.isinf(mfl))


def test_transit_model_population_rv_shifts(lpf_and_truth):
    """Each parameter vector in a population must receive only its own systemic shift.

    The RV mean model is applied inside the per-vector loop, so writing to the whole
    model array instead of the current row leaks each row's shift into its neighbours.
    """
    lpf, pv = lpf_and_truth[:2]
    pvp = np.tile(pv, (2, 1))
    i = lpf.ps.find_pid('srv_0')
    pvp[0, i], pvp[1, i] = 10.0, -10.0

    _, mrv, _ = lpf.transit_model(pvp)

    assert mrv.shape == (2, lpf.nrvs)
    assert np.abs((mrv[0] - mrv[1]) - 20.0).max() < 1e-8


def test_lnlikelihood_finite_and_prefers_truth(lpf_and_truth):
    lpf, pv = lpf_and_truth[:2]
    lnl_true = lpf.lnlikelihood(pv)
    assert np.isfinite(lnl_true)
    shifted = pv.copy()
    shifted[lpf.ps.find_pid('t0_0')] += 0.01
    assert lnl_true > lpf.lnlikelihood(shifted)


def test_lnposterior_outside_prior_is_minus_inf(lpf_and_truth):
    lpf, pv = lpf_and_truth[:2]
    bad = pv.copy()
    bad[lpf.ps.find_pid('omega_0')] = 0.0  # outside UP(pi/2, 3pi/2)
    assert lpf.lnposterior(bad) == -np.inf
    assert np.isfinite(lpf.lnposterior(pv))


def test_eccentricity_prior_is_finite(lpf_and_truth):
    lpf, pv = lpf_and_truth[:2]
    assert np.isfinite(lpf.eccentricity_prior(pv))


def test_create_pv_population(lpf_and_truth):
    lpf = lpf_and_truth[0]
    pop = lpf.create_pv_population(20)
    assert pop.shape == (20, len(lpf.ps))
    assert np.all(np.isfinite(pop))


# ----------------------------------------------------------------------
# Derived products: transit times, durations, folding
# ----------------------------------------------------------------------
def test_get_transit_times_within_range(lpf_and_truth):
    lpf, pv, _, _, tc = lpf_and_truth
    times = lpf.get_transit_times_within_range(pv, 0, 0.3, 24.0)
    assert times.size == 7  # transits at ~1.0 + n*3.4 for n = 0..6
    assert np.all(np.abs(np.diff(times) - PER[0]) < 0.02)
    # the first two must agree with the model's fitted centre times
    assert np.abs(times[:2] - tc).max() * 86400 < 0.5


def test_get_transit_durations_within_range(lpf_and_truth):
    lpf, pv = lpf_and_truth[:2]
    tcs_a, dur_a = lpf.get_transit_durations_within_range(pv, 0, 0.3, 12.0, kind='analytical')
    tcs_n, dur_n = lpf.get_transit_durations_within_range(pv, 0, 0.3, 12.0, kind='numerical')
    assert tcs_a.size == tcs_n.size == 4
    assert np.all((0.05 < dur_a) & (dur_a < 0.3))
    assert np.abs(dur_n / dur_a - 1).max() < 0.15
    assert np.abs(tcs_a - tcs_n).max() * 86400 < 0.5


def test_fold_times_centres_on_transit(lpf_and_truth):
    lpf, pv = lpf_and_truth[:2]
    folded, fluxes = lpf.fold_times(pv, 0)
    assert len(folded) == 2
    for ft in folded:
        assert abs(np.median(ft)) < 0.02  # windows are centred on the transits


# ----------------------------------------------------------------------
# RV mean model: polynomial trend and systemic velocities
# ----------------------------------------------------------------------
def build_lpf(rv_slope_order=0, rv_trend_priors=None, rv_lnlikelihood=None,
              with_rvs=True, seed=42, rv_tref=None):
    """Build an LPF over the canonical two-planet system, with configurable RV setup."""
    np.random.seed(seed)
    times, fl, rv, tc = truth()
    n0 = times[0].size
    fluxes = [fl[:n0] + np.random.normal(0, 2e-4, n0),
              fl[n0:] + np.random.normal(0, 2e-4, times[1].size)]
    errors = [np.full(t.size, 2e-4) for t in times]
    center_times = [tc + np.random.normal(0, 20 / 86400, tc.size)]
    center_errors = [np.full(tc.size, 30 / 86400)]

    rvdata = None
    if with_rvs:
        rvdata = make_rv_group([RVTIMES], [rv + np.random.normal(0, 0.5, rv.size)],
                               [np.full(rv.size, 0.5)])

    return PhotoDynamicalLPF('test', 2, make_group(times, fluxes, errors), rvdata,
                             (center_times, center_errors),
                             zero_epochs=list(T0), periods=list(PER),
                             is_transiting=[True, True],
                             tref=TREF, lnlikelihood='wn',
                             rv_slope_order=rv_slope_order,
                             rv_trend_priors=rv_trend_priors,
                             rv_tref=rv_tref,
                             rv_lnlikelihood=rv_lnlikelihood)


def set_pv(lpf, **overrides):
    """Draw a prior sample and force the truth values, plus any overrides, onto it."""
    pv = lpf.ps.sample_from_prior(1)[0]
    for name, value in {**PV_VALUES, **overrides}.items():
        pv[lpf.ps.find_pid(name)] = value
    return pv


@pytest.mark.parametrize('order,expected', [
    (0, ['srv_0']),
    (1, ['rv_trend', 'srv_0']),
    (2, ['rv_trend', 'rv_trend_2', 'srv_0']),
])
def test_rv_block_layout(order, expected):
    """The rv block holds the trend terms then the systemic velocities, in that order."""
    lpf = build_lpf(rv_slope_order=order)
    names = [p.name for p in lpf.ps]
    assert names[lpf._sl_rvs] == expected
    # The planet block starts right after star (2 params) + ldc (2 params for the
    # single passband used here). The strided indexing reads _start_planets rather
    # than assuming this value, so it also holds for several passbands.
    assert lpf._start_planets == 4
    # _start_rvs doubles as the end of the planet block for the strided indexing
    assert lpf._start_rvs == lpf._start_planets + 8 * lpf.nplanets


def test_rv_slope_order_rejects_bad_values():
    with pytest.raises(ValueError):
        build_lpf(rv_slope_order=3)


def test_rv_slope_order_requires_rv_data():
    """A trend order > 0 with no RV data would declare parameters transit_model
    never applies (its nrvsets > 0 guard skips them), so it must be rejected."""
    with pytest.raises(ValueError):
        build_lpf(with_rvs=False, rv_slope_order=1)


def test_rv_trend_priors_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_lpf(rv_slope_order=2, rv_trend_priors=[NP(0, 1)])


def test_rv_trend_priors_custom_success_path():
    """Passing explicit priors of the right length must be used verbatim."""
    from pytransit.param import UniformPrior as UP
    priors = [UP(-5, 5), UP(-2, 2)]
    lpf = build_lpf(rv_slope_order=2, rv_trend_priors=priors)
    names = [p.name for p in lpf.ps]
    assert names[lpf._sl_rvs][:2] == ['rv_trend', 'rv_trend_2']
    assert lpf.rv_trend_priors == priors
    assert lpf.ps[lpf.ps.find_pid('rv_trend')].prior is priors[0]
    assert lpf.ps[lpf.ps.find_pid('rv_trend_2')].prior is priors[1]


@pytest.mark.parametrize('rv_tref, expected', [(None, TREF), (TREF + 3.7, TREF + 3.7)])
def test_rv_trend_applied(rv_tref, expected):
    """Model RVs must shift by sum(c_j * (t - rv_tref) ** (j + 1)), with rv_tref
    defaulting to the model tref rather than to anything derived from the RV epochs."""
    lpf = build_lpf(rv_slope_order=2, rv_tref=rv_tref)
    assert lpf.rv_tref == expected
    assert not np.isclose(expected, lpf._orvtimes.mean())  # otherwise the test is blind
    base = lpf.transit_model(set_pv(lpf, rv_trend=0.0, rv_trend_2=0.0))[1]

    c1, c2 = 0.03, -0.004
    shifted = lpf.transit_model(set_pv(lpf, rv_trend=c1, rv_trend_2=c2))[1]

    rvt = lpf._orvtimes - expected
    assert np.abs((shifted - base) - (c1 * rvt + c2 * rvt ** 2)).max() < 1e-10


def test_rv_block_empty_without_rv_data():
    """No RV data and no trend: the rv block is empty but the planet indexing still works."""
    lpf = build_lpf(with_rvs=False, rv_slope_order=0)
    assert lpf._sl_rvs == slice(lpf._start_rvs, lpf._start_rvs)
    assert lpf._start_rvs == lpf._start_planets + 8 * lpf.nplanets

    pv = lpf.ps.sample_from_prior(1)[0]
    for name, value in PV_VALUES.items():
        if name in ('srv_0', 'log10rvj_0'):
            continue
        pv[lpf.ps.find_pid(name)] = value
    mfl, _, mtc = lpf.transit_model(pv)
    assert np.all(np.isfinite(mfl))
    assert np.all(np.isfinite(mtc))


# ----------------------------------------------------------------------
# Pluggable RV noise model
# ----------------------------------------------------------------------
def test_default_rv_likelihood_is_wn():
    lpf = build_lpf()
    assert isinstance(lpf.rv_lnl, WNRVLikelihood)
    assert [p.name for p in lpf.ps][lpf.rv_lnl.slice] == ['log10rvj_0']
    assert not hasattr(lpf, '_sl_rvj')
    assert not hasattr(lpf, '_start_rvj')


def test_rv_noise_block_follows_the_rv_block():
    """The noise block must come last so the strided planet indexing stays valid."""
    lpf = build_lpf(rv_slope_order=1)
    assert lpf.rv_lnl.start == lpf._sl_rvs.stop


def test_lnlikelihood_separated_sums_to_total():
    lpf = build_lpf()
    pv = set_pv(lpf)
    assert np.isclose(sum(lpf.lnlikelihood_separated(pv)), lpf.lnlikelihood(pv))


def test_wn_path_matches_legacy():
    """The refactored white-noise path must reproduce pdlpf_legacy exactly.

    The legacy class carries the pre-refactor RV model: a linear trend plus a
    sinusoid plus jitter. Setting the trend to zero and the sine amplitude to zero
    reduces it to the systemic-velocity-plus-jitter model that rv_slope_order=0 with
    WNRVLikelihood now expresses.
    """
    from src.pdlpf_legacy import PhotoDynamicalLPF as LegacyLPF

    np.random.seed(42)
    times, fl, rv, tc = truth()
    n0 = times[0].size
    fluxes = [fl[:n0] + np.random.normal(0, 2e-4, n0),
              fl[n0:] + np.random.normal(0, 2e-4, times[1].size)]
    errors = [np.full(t.size, 2e-4) for t in times]
    rv_values = rv + np.random.normal(0, 0.5, rv.size)
    rv_errors = np.full(rv.size, 0.5)
    center_times = [tc + np.random.normal(0, 20 / 86400, tc.size)]
    center_errors = [np.full(tc.size, 30 / 86400)]

    # The legacy class keeps the old array API, the current one takes the same
    # photometry as a group, so this also checks that the group is expanded into
    # exactly the arguments the arrays used to supply.
    shared = dict(is_transiting=[True, True], tref=TREF, lnlikelihood='wn')

    # The legacy class keeps the old array API for both the photometry and the RVs.
    legacy = LegacyLPF('legacy', 2, ['tess'], zero_epochs=list(T0), periods=list(PER),
                       times=times, fluxes=fluxes, errors=errors, pids=[[0, 1], [0, 1]],
                       pbids=[0, 0], nsamples=[1, 1], exptimes=[EXPTIME] * 2,
                       rv_times=[RVTIMES], rv_values=[rv_values], rv_errors=[rv_errors],
                       center_times=center_times, center_time_errors=center_errors,
                       **shared)
    # The legacy class has no baseline model, so the comparison is only like-for-like
    # with the least-squares baseline switched off.
    current = PhotoDynamicalLPF('current', 2, make_group(times, fluxes, errors),
                                make_rv_group([RVTIMES], [rv_values], [rv_errors]),
                                (center_times, center_errors),
                                zero_epochs=list(T0), periods=list(PER),
                                use_lstsq_baseline=False, **shared)

    physical = {k: v for k, v in PV_VALUES.items()}
    pv_legacy = legacy.ps.sample_from_prior(1)[0]
    for name, value in {**physical, 'rv_trend': 0.0, 'rv_sine_amplitude': 0.0,
                        'rv_sine_period': 1.0, 'rv_sine_phase': 0.0}.items():
        pv_legacy[legacy.ps.find_pid(name)] = value

    pv_current = current.ps.sample_from_prior(1)[0]
    for name, value in physical.items():
        pv_current[current.ps.find_pid(name)] = value

    assert np.isclose(current.lnlikelihood(pv_current), legacy.lnlikelihood(pv_legacy),
                      rtol=0, atol=1e-10)
    for a, b in zip(current.lnlikelihood_separated(pv_current),
                    legacy.lnlikelihood_separated(pv_legacy)):
        assert np.isclose(a, b, rtol=0, atol=1e-10)


@pytest.mark.skipif(not with_george, reason='george is not installed')
def test_gp_rv_likelihood_end_to_end():
    """A GP-noise LPF produces a finite likelihood and still prefers the truth."""
    from pytransit.param import UniformPrior as UP
    lpf = build_lpf(rv_lnlikelihood=QPGPRVLikelihood(period_prior=UP(15, 30)))
    gp_values = {'gp_ap_std': 2.0, 'gp_ap_scale': 30.0, 'gp_std': 5.0,
                 'gp_log10_gamma': -0.5, 'gp_period': 20.95, 'gp_coherence': 100.0}
    pv = lpf.ps.sample_from_prior(1)[0]
    for name, value in {**PV_VALUES, **gp_values}.items():
        if name == 'log10rvj_0':
            continue  # the GP model has no jitter parameter by default
        pv[lpf.ps.find_pid(name)] = value

    lnl_true = lpf.lnlikelihood(pv)
    assert np.isfinite(lnl_true)
    assert np.isclose(sum(lpf.lnlikelihood_separated(pv)), lnl_true)

    shifted = pv.copy()
    shifted[lpf.ps.find_pid('t0_0')] += 0.01
    assert lnl_true > lpf.lnlikelihood(shifted)


# ----------------------------------------------------------------------
# predict_rvs: the dynamical RV signal at arbitrary times
# ----------------------------------------------------------------------
STRADDLING_TIMES = np.array([-6.0, -2.5, -0.5, 0.5, 3.0, 9.0])  # tref is 0.0


def test_predict_rvs_matches_transit_model():
    """At the observed epochs, the prediction must reproduce the fitted RV model.

    With srv_0 = 0 and no trend the model RVs are the bare N-body signal, which is
    exactly what predict_rvs returns, so the two paths must agree.
    """
    lpf = build_lpf()
    pv = set_pv(lpf, srv_0=0.0)
    assert np.allclose(lpf.predict_rvs(pv, lpf._orvtimes), lpf.transit_model(pv)[1], atol=1e-10)


def test_predict_rvs_spans_reference_epoch():
    """Times on both sides of tref exercise the backward and forward integrations.

    The observed RV epochs all sit after tref, so nothing else covers the backward
    branch. Evaluating each time in its own call is independent of the split, and must
    give the same answer as evaluating them together.
    """
    lpf = build_lpf()
    pv = set_pv(lpf, srv_0=0.0)

    together = lpf.predict_rvs(pv, STRADDLING_TIMES)
    separately = np.array([lpf.predict_rvs(pv, np.array([t]))[0] for t in STRADDLING_TIMES])

    assert np.all(np.isfinite(together))
    assert np.allclose(together, separately, atol=1e-10)


def test_predict_rvs_handles_unordered_times():
    """The result follows the caller's time order, not the internal sorted order."""
    lpf = build_lpf()
    pv = set_pv(lpf, srv_0=0.0)
    order = np.array([5, 0, 3, 7, 1, 6, 2, 4])

    ordered = lpf.predict_rvs(pv, lpf._orvtimes)
    shuffled = lpf.predict_rvs(pv, lpf._orvtimes[order])

    assert np.allclose(shuffled, ordered[order], atol=1e-12)


def test_predict_rvs_planet_split():
    """`pid` splits the signal into one planet's contribution and everything else.

    The split is exact only for non-interacting planets, so the two parts sum to the
    full signal up to the mutual perturbation.
    """
    lpf = build_lpf()
    pv = set_pv(lpf, srv_0=0.0)

    full = lpf.predict_rvs(pv, lpf._orvtimes)
    rv_planet, rv_others = lpf.predict_rvs(pv, lpf._orvtimes, pid=0)

    assert rv_planet.shape == rv_others.shape == full.shape
    assert np.all(np.isfinite(rv_planet)) and np.all(np.isfinite(rv_others))
    # Both parts must carry real signal, so the sum check below cannot pass trivially.
    assert np.ptp(rv_planet) > 0.1
    assert np.ptp(rv_others) > 0.1
    assert np.abs((rv_planet + rv_others) - full).max() < 5e-3


def test_predict_rvs_without_rv_data():
    """An RV-free fit can still predict RVs: only the N-body system is needed."""
    lpf = build_lpf(with_rvs=False)
    pv = lpf.ps.sample_from_prior(1)[0]
    for name, value in PV_VALUES.items():
        if name in ('srv_0', 'log10rvj_0'):
            continue
        pv[lpf.ps.find_pid(name)] = value

    rvs = lpf.predict_rvs(pv, STRADDLING_TIMES)
    assert rvs.shape == STRADDLING_TIMES.shape
    assert np.all(np.isfinite(rvs))
    assert np.ptp(rvs) > 1.0


# ----------------------------------------------------------------------
# Light curve group ingestion
# ----------------------------------------------------------------------
def photometry():
    """The canonical two-light-curve photometry as plain arrays."""
    np.random.seed(42)
    times, fl, _, _ = truth()
    n0 = times[0].size
    fluxes = [fl[:n0] + np.random.normal(0, 2e-4, n0),
              fl[n0:] + np.random.normal(0, 2e-4, times[1].size)]
    errors = [np.full(t.size, 2e-4) for t in times]
    return times, fluxes, errors


def set_truth_pv(lpf, passbands=('tess',)):
    """Set the truth values on a parameter vector, adapting to the LPF's own blocks.

    The limb-darkening parameters are named after the passbands, and the RV parameters
    only exist when the LPF has RV data, so both are matched against the parameter set
    rather than assumed.
    """
    names = {p.name for p in lpf.ps}
    pv = lpf.ps.sample_from_prior(1)[0]
    for name, value in PV_VALUES.items():
        targets = ([name.replace('_tess', f'_{pb}') for pb in passbands]
                   if name.endswith('_tess') else [name])
        for target in targets:
            if target in names:
                pv[lpf.ps.find_pid(target)] = value
    return pv


def build_from(lcdata, nplanets=2, **kwargs):
    return PhotoDynamicalLPF('lcdata', nplanets, lcdata, None, None,
                             zero_epochs=list(T0), periods=list(PER),
                             is_transiting=[True, True], tref=TREF, lnlikelihood='wn', **kwargs)


def test_group_supplies_the_base_class_arrays():
    """The group is expanded into the per-light-curve arrays the base class builds on."""
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors))

    assert lpf.nlc == 2
    assert lpf.npb == 1
    assert list(lpf.passbands) == ['tess']
    assert_array_equal(lpf.pbids, [0, 0])
    assert lpf.n_noise_blocks == 1
    assert lpf.timea.size == sum(t.size for t in times)
    assert_array_equal(lpf.timea, np.concatenate(times))
    assert_allclose(lpf.errora, np.concatenate(errors))
    assert_array_equal(lpf.exptimes, [EXPTIME] * 2)
    assert lpf.pids == [(0, 1), (0, 1)]
    assert lpf.with_photometry


def test_single_light_curve_is_wrapped_into_a_group():
    """A bare LCData is accepted without the caller building a group."""
    times, fluxes, errors = photometry()
    lc = LCData(times[0], fluxes[0], errors[0], passband='tess', pids=(0, 1),
                        exptime=EXPTIME, nsamples=1)
    lpf = build_from(lc)
    assert lpf.nlc == 1
    assert lpf.pids == [(0, 1)]


def test_light_curve_without_pids_raises():
    """Every light curve must say which planets contribute to it."""
    times, fluxes, errors = photometry()
    lcs = LCDataGroup([
        LCData(times[0], fluxes[0], errors[0], passband='tess', pids=(0, 1),
                       exptime=EXPTIME),
        LCData(times[1], fluxes[1], errors[1], passband='tess', exptime=EXPTIME)])
    with pytest.raises(ValueError, match=r'no pids') as e:
        build_from(lcs)
    assert '[1]' in str(e.value)     # the offending light curve is named


def test_pids_beyond_nplanets_raises():
    """A planet id the system does not have is caught before the N-body model sees it."""
    times, fluxes, errors = photometry()
    lcs = make_group(times, fluxes, errors, pids=((0, 1), (0, 2)))
    with pytest.raises(ValueError, match=r'refers to planet 2'):
        build_from(lcs, nplanets=2)


def test_empty_group_raises():
    """An empty group is an error, not a silent RV/TTV-only fit."""
    with pytest.raises(ValueError, match=r'empty'):
        build_from(LCDataGroup())


def test_non_light_curve_lcdata_raises():
    times, fluxes, errors = photometry()
    with pytest.raises(TypeError, match=r'LCDataGroup'):
        build_from([times, fluxes, errors])


def test_group_without_errors_leaves_errora_nan():
    """Without explicit uncertainties the group's noise estimate is not passed on."""
    times, fluxes, _ = photometry()
    lcs = LCDataGroup([
        LCData(t, f, passband='tess', pids=(0, 1), exptime=EXPTIME)
        for t, f in zip(times, fluxes)])
    assert not lcs.has_errors
    lpf = build_from(lcs)
    assert np.all(np.isnan(lpf.errora))


def test_no_photometry_builds_an_rv_only_lpf():
    """lcdata=None runs the RV/TTV-only path with empty photometric arrays."""
    np.random.seed(42)
    _, _, rv, tc = truth()
    lpf = PhotoDynamicalLPF('rv-only', 2, None,
                            make_rv_group([RVTIMES], [rv + np.random.normal(0, 0.5, rv.size)],
                                          [np.full(rv.size, 0.5)]),
                            ([tc], [np.full(tc.size, 30 / 86400)]),
                            zero_epochs=list(T0), periods=list(PER),
                            is_transiting=[True, True],
                            tref=TREF, lnlikelihood='wn')

    assert not lpf.with_photometry
    assert lpf.lstsq_baseline is None       # nothing to detrend without photometry
    assert lpf.nlc == 0
    assert lpf.npb == 1                     # the ldc block still needs one passband
    assert lpf.timea.size == 0
    assert lpf.pids is None

    # Without photometry the limb-darkening block is named after the dummy passband.
    assert np.isfinite(lpf.lnposterior(set_truth_pv(lpf, passbands=('white',))))


# ----------------------------------------------------------------------
# Multiple passbands
# ----------------------------------------------------------------------
def test_two_passbands_reproduce_the_single_passband_truth():
    """The planet parameters are read by block offset, not from a fixed index.

    With two passbands the limb-darkening block grows from two parameters to four and
    pushes the planet block from index 4 to index 6. Giving both passbands the same
    limb darkening makes the system physically identical to the single-passband truth,
    so the model flux must match it. Reading the planet parameters from the old
    hard-coded offset would pick up the limb-darkening coefficients instead.
    """
    times, fluxes, errors = photometry()
    lcs = make_group(times, fluxes, errors, passbands=('g', 'r'))
    lpf = build_from(lcs)

    assert lpf.npb == 2
    assert lpf._start_planets == 6           # 2 star + 2 * 2 limb darkening

    # Same limb darkening in both passbands -> same physics as the npb=1 truth.
    pv = set_truth_pv(lpf, passbands=('g', 'r'))

    _, fl_truth, _, _ = truth()
    mflux, _, _ = lpf.transit_model(pv)
    assert_allclose(mflux, fl_truth, atol=1e-8)


def test_two_passbands_use_their_own_limb_darkening():
    """Each light curve is modelled with the limb darkening of its own passband."""
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors, passbands=('g', 'r')))

    pv = set_truth_pv(lpf, passbands=('g', 'r'))

    # Perturbing only the second passband's limb darkening must leave the first light
    # curve untouched and change the second.
    sl0, sl1 = lpf.lcslices
    base = lpf.transit_model(pv)[0].copy()
    pv[lpf.ps.find_pid('q1_r')] = 0.8
    perturbed = lpf.transit_model(pv)[0]

    assert_allclose(perturbed[sl0], base[sl0], atol=1e-12)
    assert not np.allclose(perturbed[sl1], base[sl1], atol=1e-8)


# ----------------------------------------------------------------------
# Least-squares baseline
# ----------------------------------------------------------------------
def test_baseline_is_on_by_default_and_covers_every_light_curve():
    """Every light curve gets a baseline; without covariates it is intercept-only."""
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors))

    assert lpf.lstsq_baseline is not None
    assert lpf.lstsq_baseline.nlc == lpf.nlc == 2
    assert lpf.lstsq_baseline.ncoef == [1, 1]        # intercept only
    assert lpf.lstsq_baseline.mask.all()             # no point left uncovered
    assert lpf.covariates is not None                # LSTSQBaseline requires them


def test_baseline_can_be_switched_off():
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors), use_lstsq_baseline=False)
    assert lpf.lstsq_baseline is None
    # The un-baselined model flux is what reaches the likelihood.
    pv = set_truth_pv(lpf)
    mflux, _, _ = lpf.transit_model(pv)
    assert_allclose(lpf.apply_baseline(mflux), mflux, rtol=1e-15)


def test_baseline_adds_no_parameters():
    """LSTSQBaseline profiles its coefficients out instead of sampling them."""
    times, fluxes, errors = photometry()
    on = build_from(make_group(times, fluxes, errors))
    off = build_from(make_group(times, fluxes, errors), use_lstsq_baseline=False)
    assert [p.name for p in on.ps] == [p.name for p in off.ps]


def test_baseline_absorbs_a_constant_rescaling():
    """A light curve at the wrong normalisation is corrected by the fitted intercept."""
    times, fl, _, _ = truth()
    n0 = times[0].size
    scale = 1.05
    fluxes = [scale * fl[:n0], scale * fl[n0:]]
    errors = [np.full(t.size, 2e-4 * scale) for t in times]
    lcs = make_group(times, fluxes, errors)

    lpf = build_from(lcs)
    pv = set_truth_pv(lpf)
    mflux, _, _ = lpf.transit_model(pv)

    # The data is exactly `scale` times the model, so the intercept must be `scale`.
    coefs = lpf.lstsq_baseline.coefficients(mflux)
    assert len(coefs) == 2
    for c in coefs:
        assert c.shape == (1,)
        assert_allclose(c[0], scale, rtol=1e-10)

    # And the baselined model reproduces the rescaled data.
    assert_allclose(lpf.flux_model(pv), scale * fl, atol=1e-10)

    # Which the un-baselined fit cannot do.
    off = build_from(lcs, use_lstsq_baseline=False)
    assert lpf.lnlikelihood(pv) > off.lnlikelihood(set_truth_pv(off))


def test_baseline_recovers_a_covariate_slope():
    """A known linear covariate trend is recovered by the least-squares fit."""
    times, fl, _, _ = truth()
    n0 = times[0].size
    models = [fl[:n0], fl[n0:]]
    slope = 0.01
    covs = [np.linspace(-1, 1, t.size) for t in times]
    fluxes = [m * (1 + slope * c) for m, c in zip(models, covs)]
    errors = [np.full(t.size, 2e-4) for t in times]

    lcs = LCDataGroup([
        LCData(t, f, e, covariates=c, passband='tess', pids=(0, 1), exptime=EXPTIME)
        for t, f, e, c in zip(times, fluxes, errors, covs)])
    lpf = build_from(lcs)
    assert lpf.lstsq_baseline.ncoef == [2, 2]        # intercept + one covariate

    pv = set_truth_pv(lpf)
    mflux, _, _ = lpf.transit_model(pv)

    # The coefficients live in standardised covariate space, so map them back.
    for cov, coef in zip(covs, lpf.lstsq_baseline.coefficients(mflux)):
        raw_slope = coef[1] / cov.std()
        assert_allclose(raw_slope, slope, rtol=1e-8)
        assert_allclose(coef[0] - cov.mean() * raw_slope, 1.0, rtol=1e-8)


def test_flux_model_is_the_transit_model_times_the_baseline():
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors))
    pv = set_truth_pv(lpf)

    mflux, _, _ = lpf.transit_model(pv)
    bl = lpf.lstsq_baseline(mflux).copy()

    assert bl.shape == mflux.shape == (lpf.timea.size,)
    assert_allclose(lpf.flux_model(pv), mflux * bl, rtol=1e-12)

    # Each light curve is fitted separately: constant within a light curve (the design
    # matrix is intercept-only here) but a different constant for each.
    sl0, sl1 = lpf.lcslices
    assert len(np.unique(bl[sl0])) == 1 and len(np.unique(bl[sl1])) == 1
    assert bl[sl0][0] != bl[sl1][0]


def test_fold_times_returns_baseline_corrected_fluxes():
    """fold_times divides the observed flux by the fitted baseline."""
    times, fl, _, _ = truth()
    n0 = times[0].size
    scale = 1.05
    fluxes = [scale * fl[:n0], scale * fl[n0:]]
    errors = [np.full(t.size, 2e-4 * scale) for t in times]
    lpf = build_from(make_group(times, fluxes, errors))

    pv = set_truth_pv(lpf)
    _, corrected = lpf.fold_times(pv, 0)
    # The rescaling is divided out, so the corrected flux is the model flux again.
    assert_allclose(np.concatenate(corrected), fl, atol=1e-10)


def test_baseline_returns_the_baseline_alone():
    """`baseline` hands back the fitted baseline, not the base class's scalar 1."""
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors))
    pv = set_truth_pv(lpf)

    bl = lpf.baseline(pv)
    mflux, _, _ = lpf.transit_model(pv)

    assert isinstance(bl, np.ndarray)
    assert bl.shape == (lpf.timea.size,)
    assert_allclose(bl, lpf.lstsq_baseline(mflux), rtol=1e-14)
    assert_allclose(lpf.flux_model(pv), mflux * bl, rtol=1e-12)


def test_baseline_returns_a_copy_not_the_shared_buffer():
    """LSTSQBaseline hands back a reused buffer, so `baseline` must copy it."""
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors))

    pv0 = set_truth_pv(lpf)
    pv1 = pv0.copy()
    pv1[lpf.ps.find_pid('k_0')] *= 1.5      # a visibly different transit depth

    bl0 = lpf.baseline(pv0)
    bl1 = lpf.baseline(pv1)

    assert bl0 is not bl1
    assert not np.array_equal(bl0, bl1)      # bl0 was not overwritten by the second fit
    assert_allclose(bl0, lpf.baseline(pv0), rtol=1e-14)


def test_baseline_and_flux_model_accept_a_precomputed_model_flux():
    """Passing mflux avoids a second N-body integration and changes nothing."""
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors))
    pv = set_truth_pv(lpf)
    mflux, _, _ = lpf.transit_model(pv)

    assert_allclose(lpf.baseline(pv, mflux), lpf.baseline(pv), rtol=1e-14)
    assert_allclose(lpf.flux_model(pv, mflux), lpf.flux_model(pv), rtol=1e-14)


def test_baseline_is_ones_without_a_baseline_model():
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors), use_lstsq_baseline=False)
    pv = set_truth_pv(lpf)
    mflux, _, _ = lpf.transit_model(pv)

    bl = lpf.baseline(pv)
    assert bl.shape == (lpf.timea.size,)
    assert_allclose(bl, 1.0, rtol=1e-15)
    assert_allclose(lpf.baseline(pv, mflux), 1.0, rtol=1e-15)
    assert_allclose(lpf.flux_model(pv), mflux, rtol=1e-15)


def test_baseline_detrends_the_observed_flux():
    """The visualisation use case: dividing the data by the baseline recovers the model."""
    times, fl, _, _ = truth()
    n0 = times[0].size
    scale = 1.05
    fluxes = [scale * fl[:n0], scale * fl[n0:]]
    errors = [np.full(t.size, 2e-4 * scale) for t in times]
    lpf = build_from(make_group(times, fluxes, errors))

    pv = set_truth_pv(lpf)
    assert_allclose(lpf.ofluxa / lpf.baseline(pv), fl, atol=1e-10)


def test_baseline_shape_for_a_parameter_vector_stack():
    times, fluxes, errors = photometry()
    lpf = build_from(make_group(times, fluxes, errors))
    pv = set_truth_pv(lpf)
    pvp = np.array([pv, pv, pv])

    mflux = np.array([lpf.transit_model(p)[0] for p in pvp])
    assert lpf.baseline(pvp, mflux).shape == (3, lpf.timea.size)
    assert lpf.baseline(pvp).shape == (3, lpf.timea.size)


def test_baseline_without_photometry_is_empty():
    """The RV/TTV-only fit has no light curves, so the baseline has no points."""
    lpf = PhotoDynamicalLPF('rv-only', 2, None, None, None,
                            zero_epochs=list(T0), periods=list(PER),
                            is_transiting=[True, True], tref=TREF, lnlikelihood='wn')
    pv = set_truth_pv(lpf, passbands=('white',))
    assert lpf.baseline(pv).size == 0


# ----------------------------------------------------------------------
# RV data ingestion
# ----------------------------------------------------------------------
def rv_photometry(seed=42):
    """The canonical light curve group plus a matching RV dataset."""
    np.random.seed(seed)
    times, fl, rv, tc = truth()
    n0 = times[0].size
    fluxes = [fl[:n0] + np.random.normal(0, 2e-4, n0),
              fl[n0:] + np.random.normal(0, 2e-4, times[1].size)]
    errors = [np.full(t.size, 2e-4) for t in times]
    return (make_group(times, fluxes, errors),
            rv + np.random.normal(0, 0.5, rv.size), np.full(rv.size, 0.5))


def build_with_rvs(rvdata, nplanets=2, **kwargs):
    lcs, _, _ = rv_photometry()
    return PhotoDynamicalLPF('rv', nplanets, lcs, rvdata, None,
                             zero_epochs=list(T0), periods=list(PER),
                             is_transiting=[True, True], tref=TREF,
                             lnlikelihood='wn', **kwargs)


def test_single_rv_dataset_is_wrapped_into_a_group():
    """A bare RVData is accepted without the caller building a group."""
    _, values, errors = rv_photometry()
    lpf = build_with_rvs(RVData(RVTIMES, values, errors, instrument='HARPS'))
    assert lpf.nrvsets == 1
    assert lpf.nrvs == RVTIMES.size
    assert lpf.rv_instruments == ['HARPS']


def test_empty_rv_group_raises():
    """An empty group is an error, not a silent photometry-only fit."""
    with pytest.raises(ValueError, match='rvdata is empty'):
        build_with_rvs(RVDataGroup())


def test_non_container_rvdata_raises():
    _, values, errors = rv_photometry()
    with pytest.raises(TypeError, match='RVDataGroup'):
        build_with_rvs([RVTIMES, values, errors])


def test_rvdata_none_gives_no_rv_block():
    lpf = build_with_rvs(None)
    assert lpf.nrvsets == 0 and lpf.nrvs == 0
    assert lpf.rv_times is None and lpf.rv_instruments is None
    assert lpf._orvtimes is None and lpf._orvids is None
    assert [p.name for p in lpf.ps][lpf._sl_rvs] == []
    # set_truth_pv skips the parameters this LPF does not have (srv_0, log10rvj_0).
    assert np.isfinite(lpf.lnposterior(set_truth_pv(lpf)))


def test_two_instruments_keep_their_own_sets():
    """A multi-instrument group gives one systemic velocity per set."""
    _, values, errors = rv_photometry()
    # A second instrument, offset in time and in velocity zero point.
    t2 = RVTIMES + 0.25
    v2, e2 = values + 30.0, errors
    rvd = make_rv_group([RVTIMES, t2], [values, v2], [errors, e2],
                        instruments=('HARPS', 'CARMENES'))
    lpf = build_with_rvs(rvd)

    assert lpf.nrvsets == 2
    assert lpf.nrvs == RVTIMES.size + t2.size
    assert lpf.rv_instruments == ['HARPS', 'CARMENES']

    names = [p.name for p in lpf.ps]
    assert names[lpf._sl_rvs] == ['srv_0', 'srv_1']

    # `_orvids` labels each point with its set, in the time-sorted order.
    expected = np.concatenate([np.zeros(RVTIMES.size, int), np.ones(t2.size, int)])
    order = np.argsort(np.concatenate([RVTIMES, t2]))
    assert_array_equal(lpf._orvids, expected[order])
    assert_allclose(lpf._orvtimes, np.concatenate([RVTIMES, t2])[order])


def test_systemic_velocity_priors_come_from_their_own_set():
    """Each srv_i prior uses its own set's mean and scatter, not the pooled values."""
    _, values, errors = rv_photometry()
    v2 = values + 30.0
    rvd = make_rv_group([RVTIMES, RVTIMES + 0.25], [values, v2], [errors, errors],
                        instruments=('HARPS', 'CARMENES'))
    lpf = build_with_rvs(rvd)

    for i, v in enumerate((values, v2)):
        prior = lpf.ps[lpf.ps.find_pid(f'srv_{i}')].prior
        assert_allclose(prior.mean, v.mean(), rtol=1e-12)
        assert_allclose(prior.std, v.std(), rtol=1e-12)
    # The two means genuinely differ, so a pooled prior would fail the check above.
    assert abs(values.mean() - v2.mean()) > 1.0


def test_instrument_labels_reach_the_descriptions():
    """Names stay indexed by set; the instrument shows up in the description."""
    _, values, errors = rv_photometry()
    rvd = make_rv_group([RVTIMES], [values], [errors], instruments=('HARPS',))
    lpf = build_with_rvs(rvd)

    descriptions = {p.name: p.description for p in lpf.ps}
    assert descriptions['srv_0'] == 'systemic_rv_HARPS'
    assert descriptions['log10rvj_0'] == 'log10_rv_jitter_HARPS'
    # The names themselves are unchanged, so scripts setting parameters still work.
    assert 'srv_HARPS' not in descriptions and 'log10rvj_HARPS' not in descriptions


def test_unlabelled_sets_fall_back_to_the_index():
    """RVData defaults its instrument to an empty string."""
    _, values, errors = rv_photometry()
    lpf = build_with_rvs(RVData(RVTIMES, values, errors))
    descriptions = {p.name: p.description for p in lpf.ps}
    assert descriptions['srv_0'] == 'systemic_rv_0'
    assert descriptions['log10rvj_0'] == 'log10_rv_jitter_0'


def test_duplicate_instrument_labels_are_accepted():
    """Unlike RVDataGroup.rvis, this class names parameters by set index."""
    _, values, errors = rv_photometry()
    rvd = make_rv_group([RVTIMES, RVTIMES + 0.25], [values, values], [errors, errors],
                        instruments=('HARPS', 'HARPS'))
    with pytest.raises(ValueError):
        rvd.rvis                      # the label-based naming would collide
    lpf = build_with_rvs(rvd)         # the index-based naming does not
    assert lpf.nrvsets == 2
    assert [p.name for p in lpf.ps][lpf._sl_rvs] == ['srv_0', 'srv_1']


def test_rv_group_order_fixes_the_set_indices():
    """The group is never sorted; only the flattened arrays are."""
    _, values, errors = rv_photometry()
    # The second set is earlier in time than the first.
    rvd = make_rv_group([RVTIMES + 50.0, RVTIMES], [values, values + 30.0],
                        [errors, errors], instruments=('LATE', 'EARLY'))
    lpf = build_with_rvs(rvd)

    assert lpf.rv_instruments == ['LATE', 'EARLY']       # insertion order kept
    assert lpf._orvtimes[0] == RVTIMES.min()             # flattened array is sorted
    assert lpf._orvids[0] == 1                           # ...and points at the second set


# ----------------------------------------------------------------------
# Transit-centre data ingestion
# ----------------------------------------------------------------------
def build_with_centers(ctdata, nplanets=2):
    lcs, _, _ = rv_photometry()
    return PhotoDynamicalLPF('ct', nplanets, lcs, None, ctdata,
                             zero_epochs=list(T0), periods=list(PER),
                             is_transiting=[True, True], tref=TREF, lnlikelihood='wn')


def centers_for(ipl, n=3):
    """Measured centres and uncertainties for one planet."""
    tc = T0[ipl] + np.arange(n) * PER[ipl]
    return tc, np.full(n, 30 / 86400)


def test_ctdata_none_gives_no_centres():
    lpf = build_with_centers(None)
    assert lpf.ncenters == 0
    assert lpf.center_times is None and lpf.ctdata is None
    assert lpf._center_array is None and lpf._center_planet_ids is None
    assert np.isfinite(lpf.lnposterior(set_truth_pv(lpf)))


def test_ctdata_single_planet():
    tc, er = centers_for(0)
    lpf = build_with_centers(([tc], [er]))
    assert lpf.ncenters == tc.size
    assert_array_equal(lpf._center_planet_ids, 0)
    assert_allclose(lpf._center_array, tc)
    assert_allclose(lpf._center_error_array, er)


def test_ctdata_keeps_the_planet_of_each_centre():
    """The outer index identifies the planet, and survives the time sort."""
    tc0, er0 = centers_for(0)
    tc1, er1 = centers_for(1)
    lpf = build_with_centers(([tc0, tc1], [er0, er1]))

    assert lpf.ncenters == tc0.size + tc1.size

    # The arrays are sorted by time, and each centre keeps its own planet id.
    pooled = np.concatenate([tc0, tc1])
    expected_ids = np.concatenate([np.zeros(tc0.size, int), np.ones(tc1.size, int)])
    order = np.argsort(pooled)
    assert_allclose(lpf._center_array, pooled[order])
    assert_array_equal(lpf._center_planet_ids, expected_ids[order])

    # And the ids reach the model: each centre event tracks its own planet.
    events = [e for e in lpf.tm.events if isinstance(e, TransitCenter)]
    assert len(events) == lpf.ncenters
    assert_array_equal([e.ipl for e in sorted(events, key=lambda e: e.center)],
                       lpf._center_planet_ids)


def test_ctdata_allows_a_planet_without_measurements():
    """A planet with no measured centres gets an empty array, not a missing entry."""
    tc1, er1 = centers_for(1)
    lpf = build_with_centers(([np.array([]), tc1], [np.array([]), er1]))
    assert lpf.ncenters == tc1.size
    assert_array_equal(lpf._center_planet_ids, 1)


def test_ctdata_rejects_a_bad_pair():
    tc, er = centers_for(0)
    with pytest.raises(ValueError, match=r'\(centers, errors\) pair'):
        build_with_centers(([tc],))
    with pytest.raises(ValueError, match=r'\(centers, errors\) pair'):
        build_with_centers(([tc], [er], [tc]))


def test_ctdata_rejects_mismatched_sequences():
    tc0, er0 = centers_for(0)
    tc1, _ = centers_for(1)
    with pytest.raises(ValueError, match='centres for 2 planets but uncertainties for 1'):
        build_with_centers(([tc0, tc1], [er0]))


def test_ctdata_rejects_too_many_planets():
    tc, er = centers_for(0)
    with pytest.raises(ValueError, match='centres for 3 planets, but the system has 2'):
        build_with_centers(([tc, tc, tc], [er, er, er]))


def test_ctdata_rejects_a_length_mismatch_within_a_planet():
    tc, er = centers_for(0)
    with pytest.raises(ValueError, match=r'differ in length for planets \[0\]'):
        build_with_centers(([tc], [er[:-1]]))


def test_centre_likelihood_prefers_the_truth(lpf_and_truth):
    """The centre term is live: perturbing the measured centres costs likelihood."""
    lpf, pv, _, _, _ = lpf_and_truth
    _, _, lnlc = lpf.lnlikelihood_separated(pv)
    assert np.isfinite(lnlc) and lnlc != 0.0


def test_ctdata_rejects_flat_arrays():
    """A flat (centers, errors) pair would be read as one planet per measurement."""
    tc, er = centers_for(0)
    with pytest.raises(ValueError, match='one array per planet'):
        build_with_centers((tc, er), nplanets=5)
