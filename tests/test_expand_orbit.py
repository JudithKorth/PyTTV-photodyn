"""Tests for the single-point Taylor expansion functions in pdmodel.py."""
import numpy as np
from meepmeep.numba3d import pos_c, zpos_c, solve3d, pos, pos_d
from meepmeep.backends.numba.utils import eclipse_time_offset

from helpers import TWO_BODY
from src.pdmodel import expand_orbit, expand_orbit_d


def test_returns_shapes_and_anchor(two_body_sim):
    tc, p, coeffs = expand_orbit(two_body_sim, 1)
    assert coeffs.shape == (3, 5)
    assert abs(p - TWO_BODY['p'][0]) < 1e-9
    # tc must be an integer number of periods away from the true transit centre
    depoch = (tc - TWO_BODY['t0'][0]) / p
    assert abs(depoch - round(depoch)) < 1e-6


def test_expansion_matches_nbody_positions(two_body_sim):
    """The expansion must reproduce rebound's star-relative coordinates: near the
    expansion point to Taylor-truncation accuracy, and without the O(1-10) R*
    errors a sky-frame (lan) convention mismatch would produce."""
    sim = two_body_sim
    tc, p, coeffs = expand_orbit(sim, 1)
    rs = sim.particles[0].r
    for dt in np.linspace(-0.15, 0.15, 31):
        sim.integrate(tc + dt)
        pl, st = sim.particles[1], sim.particles[0]
        xyz_nb = np.array([(pl.x - st.x) / rs, (pl.y - st.y) / rs, (pl.z - st.z) / rs])
        err = np.abs(np.array(pos_c(dt, coeffs)) - xyz_nb).max()
        assert err < (2e-5 if abs(dt) <= 0.05 else 2e-3)


def test_transit_and_eclipse_z_branches(two_body_sim):
    p = TWO_BODY
    _, _, coeffs_tr = expand_orbit(two_body_sim, 1)
    et = eclipse_time_offset(p['p'][0], p['inc'][0], p['e'][0], p['w'][0])
    _, _, coeffs_ec = expand_orbit(two_body_sim, 1, te=et)
    assert zpos_c(0.0, coeffs_tr) > 0.0   # transit side: toward the observer
    assert zpos_c(0.0, coeffs_ec) < 0.0   # eclipse side: behind the star


def test_derivative_expansion_matches_value_expansion(two_body_sim):
    tc, p, coeffs = expand_orbit(two_body_sim, 1)
    tcd, pd, coeffs_d, dcoeffs = expand_orbit_d(two_body_sim, 1)
    assert tcd == tc and pd == p
    assert np.abs(coeffs_d - coeffs).max() < 1e-13
    assert dcoeffs.shape == (7, 3, 5)


def test_element_derivatives_against_finite_differences(two_body_sim):
    sim = two_body_sim
    _, p, coeffs, dcoeffs = expand_orbit_d(sim, 1)
    o = sim.particles[1].orbit(sim.particles[0])
    pars = np.array([p, o.a / sim.particles[0].r, o.inc, o.e, o.omega, o.Omega + np.pi])
    for j in range(6):  # dcoeffs slots 1..6 = (p, a, i, e, w, lan)
        d = 1e-6 * max(1.0, abs(pars[j]))
        pp, pm = pars.copy(), pars.copy()
        pp[j] += d
        pm[j] -= d
        fd = (solve3d(0.0, *pp) - solve3d(0.0, *pm)) / (2 * d)
        an = dcoeffs[j + 1]
        assert np.abs(fd - an).max() / max(np.abs(an).max(), 1e-12) < 1e-4


def test_tc_derivative_slot_against_finite_differences(two_body_sim):
    tc, p, coeffs, dcoeffs = expand_orbit_d(two_body_sim, 1)
    times = tc + np.linspace(-0.05, 0.05, 11)
    _, _, _, dx, dy, dz = pos_d(times, tc, p, coeffs, dcoeffs)
    d = 1e-7
    for grad, axis in zip((dx, dy, dz), range(3)):
        fp = np.asarray(pos(times, tc + d, p, coeffs)[axis])
        fm = np.asarray(pos(times, tc - d, p, coeffs)[axis])
        fd = (fp - fm) / (2 * d)
        an = grad[:, 0]
        assert np.abs(fd - an).max() / max(np.abs(an).max(), 1e-12) < 1e-4
