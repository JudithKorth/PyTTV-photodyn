"""End-to-end tests for PhotoDynamicalModel: transits, eclipses, RVs, centres."""
import numpy as np
import pytest
from meepmeep.backends.numba.utils import eclipse_time_offset
from pytransit.models.eclipse_model import EclipseModel

from helpers import TWO_BODY, EXPTIME, build_sim, window, model_arrays, nbody_minimum_separation
from src.pdmodel import PhotoDynamicalModel, TransitLC, EclipseLC, RV_CONVERSION, c

P = TWO_BODY
T0, PER, K = P['t0'][0], P['p'][0], P['k'][0]
FR = 0.1
ET = eclipse_time_offset(PER, P['inc'][0], P['e'][0], P['w'][0])
RVTIMES = np.linspace(0.3, 12.0, 7)


def make_model():
    """One transit window, one eclipse window, RVs, and both centre types."""
    t_tr, t_ec = window(T0), window(T0 + ET)
    lctimes = np.concatenate([t_tr, t_ec])
    lcids = np.concatenate([np.zeros(t_tr.size, int), np.ones(t_ec.size, int)])
    model = PhotoDynamicalModel(
        nplanets=1, is_transiting=np.array([True]), tref=0.0,
        lctimes=lctimes, pids=[[0], [0]], lcids=lcids, pbids=[0, 0],
        exptimes=[EXPTIME, EXPTIME], nsamples=[1, 1], lctypes=[0, 1],
        rvtimes=RVTIMES,
        tcs=np.array([T0, T0 + ET]), tcipl=np.array([0, 0]), tctypes=np.array([0, 1]),
        with_gr=False)
    return model, t_tr.size


@pytest.fixture(scope="module")
def evaluated():
    model, ntr = make_model()
    fl, rv, centers = model(*model_arrays(P), fr=np.array([[FR]]))
    return model, fl.copy(), rv.copy(), centers.copy(), ntr


def test_event_types_follow_flags():
    model, _ = make_model()
    lcs = [e for e in model.events if isinstance(e, (TransitLC, EclipseLC))]
    assert sorted(type(e).__name__ for e in lcs) == ['EclipseLC', 'TransitLC']


def test_transit_and_eclipse_depths(evaluated):
    _, fl, _, _, ntr = evaluated
    fl_tr, fl_ec = fl[:ntr], fl[ntr:]
    assert 0.005 < 1 - fl_tr.min() < 0.02          # transit
    assert abs((1 - fl_ec.min()) - FR * K ** 2) < 1e-6  # eclipse depth = fr*k^2
    assert np.all(fl <= 1.0)


def test_transit_center_output(evaluated):
    _, _, _, centers, _ = evaluated
    sim = build_sim(P)
    t_ms = nbody_minimum_separation(sim, T0)
    sim.integrate(t_ms)
    expected = t_ms - sim.particles[1].z / c
    assert abs(centers[0] - expected) * 86400 < 0.01


def test_eclipse_center_output(evaluated):
    _, _, _, centers, _ = evaluated
    sim = build_sim(P)
    t_ms = nbody_minimum_separation(sim, T0 + ET)
    sim.integrate(t_ms)
    expected = t_ms - sim.particles[1].z / c
    assert abs(centers[1] - expected) * 86400 < 0.01


def test_rv_matches_nbody_star_velocity(evaluated):
    _, _, rv, _, _ = evaluated
    sim = build_sim(P)
    expected = np.empty(RVTIMES.size)
    for i, t in enumerate(RVTIMES):
        sim.integrate(t)
        expected[i] = -sim.particles[0].vz * RV_CONVERSION
    assert np.abs(rv - expected).max() < 1e-3  # m/s
    assert np.abs(rv).max() > 0.1  # the signal is actually there


def test_eclipse_flux_matches_pytransit(evaluated):
    """Our eclipse light curve must match PyTransit's classic EclipseModel
    (same kernel and geometry; PyTransit lacks LTT, so shift its t0)."""
    model, fl, _, centers, ntr = evaluated
    t_ec = window(T0 + ET)
    sim = build_sim(P)
    t_ms = nbody_minimum_separation(sim, T0 + ET)
    sim.integrate(t_ms)
    delay = -sim.particles[1].z / c
    a_scaled = sim.particles[1].orbit(sim.particles[0]).a / sim.particles[0].r

    em = EclipseModel()
    em.set_data(t_ec, nsamples=[1], exptimes=[EXPTIME])
    fl_pt = np.squeeze(em.evaluate(K, T0 + delay, PER, a_scaled, P['inc'][0],
                                   P['e'][0], P['w'][0], fr=FR))
    assert np.abs(fl[ntr:] - fl_pt).max() < 1e-6


def test_flat_fr_equals_2d_fr(evaluated):
    model, fl, _, _, _ = evaluated
    fl2 = make_model()[0](*model_arrays(P), fr=np.array([FR]))[0]
    assert np.abs(fl2 - fl).max() < 1e-14


def test_wrong_fr_shape_raises():
    model, _ = make_model()
    with pytest.raises(ValueError, match="nplanets"):
        model(*model_arrays(P), fr=np.ones((3, 1)))


def test_missing_fr_raises_for_eclipse_lcs():
    model, _ = make_model()
    with pytest.raises(ValueError, match="flux ratios"):
        model(*model_arrays(P))


def test_build_only_returns_nones():
    model, _ = make_model()
    out = model(*model_arrays(P), fr=np.array([[FR]]), build_only=True)
    assert out == (None, None, None)
    assert model.sim is not None
