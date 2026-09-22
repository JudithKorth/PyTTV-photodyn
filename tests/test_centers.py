"""Tests for transit/eclipse centre calculation and light-travel time handling."""
import numpy as np
import pytest
from meepmeep.numba3d import solve3d, find_z_min
from meepmeep.backends.numba.utils import eclipse_time_offset

from helpers import TWO_BODY, TWO_PLANET, build_sim, nbody_minimum_separation
from src.pdmodel import (calculate_center_and_orbit, calculate_eclipse_center_and_orbit,
                         find_separation_minimum, expand_orbit, expand_orbit_d, c)

P = TWO_BODY
SEC = 86400.0


def _planet_z(sim, t):
    sim.integrate(t)
    return sim.particles[1].z


def test_transit_center_two_body(two_body_sim):
    """Transit centre = N-body minimum-separation time - z_planet/c."""
    tc_model, coeffs = calculate_center_and_orbit(two_body_sim.copy(), P['t0'][0], 0)
    t_ms = nbody_minimum_separation(two_body_sim.copy(), P['t0'][0])
    expected = t_ms - _planet_z(two_body_sim, t_ms) / c
    assert abs(tc_model - expected) * SEC < 1e-2
    assert coeffs.shape == (3, 5)


def test_eclipse_center_two_body(two_body_sim):
    """Eclipse centre = N-body minimum-separation time - z_planet/c (planet behind)."""
    et = eclipse_time_offset(P['p'][0], P['inc'][0], P['e'][0], P['w'][0])
    ec_model, _ = calculate_eclipse_center_and_orbit(two_body_sim.copy(), P['t0'][0] + et, 0)
    t_ms = nbody_minimum_separation(two_body_sim.copy(), P['t0'][0] + et)
    zp = _planet_z(two_body_sim, t_ms)
    assert zp < 0.0
    assert abs(ec_model - (t_ms - zp / c)) * SEC < 1e-2


def test_transit_eclipse_interval_carries_2a_over_c(two_body_sim):
    """The Roemer part of the modelled transit-to-eclipse interval must be ~2a/c."""
    et = eclipse_time_offset(P['p'][0], P['inc'][0], P['e'][0], P['w'][0])
    tc, _ = calculate_center_and_orbit(two_body_sim.copy(), P['t0'][0], 0)
    ec, _ = calculate_eclipse_center_and_orbit(two_body_sim.copy(), P['t0'][0] + et, 0)
    t_tr = nbody_minimum_separation(two_body_sim.copy(), P['t0'][0])
    t_ec = nbody_minimum_separation(two_body_sim.copy(), P['t0'][0] + et)
    ltt = (ec - tc) - (t_ec - t_tr)
    two_a_c = 2 * two_body_sim.particles[1].orbit(two_body_sim.particles[0]).a / c
    assert abs(ltt / two_a_c - 1) < 0.1


def test_center_epoch_folding(two_body_sim):
    """Guesses offset by a fraction of a period must converge to the same epoch."""
    tc0, _ = calculate_center_and_orbit(two_body_sim.copy(), P['t0'][0], 0)
    for offset in (-0.3, 0.25):
        tc, _ = calculate_center_and_orbit(two_body_sim.copy(),
                                           P['t0'][0] + offset * P['p'][0], 0)
        assert abs(tc - tc0) * SEC < 1e-2


def test_perturbed_system_center_spacing(two_planet_sim):
    """Consecutive centres of the perturbed inner planet: finite, spaced by ~p."""
    t0, p = TWO_PLANET['t0'][0], TWO_PLANET['p'][0]
    tcs = []
    for n in range(4):
        sim = build_sim(TWO_PLANET)
        tcs.append(calculate_center_and_orbit(sim, t0 + n * p, 0)[0])
    tcs = np.array(tcs)
    assert np.all(np.isfinite(tcs))
    assert np.all(np.abs(np.diff(tcs) - p) < 0.02)  # TTVs are minutes, not hours


def test_find_separation_minimum_recentres_bracket(two_body_sim):
    """A minimum outside find_z_min's +-0.01 d bracket must not be clipped."""
    sim = two_body_sim
    sim.integrate(P['t0'][0])
    o = sim.particles[1].orbit(sim.particles[0])
    # expand 0.02 d before the conjunction: the separation minimum sits at ~+0.02 d,
    # outside the +-0.01 d golden-section bracket of a single find_z_min call
    coeffs = solve3d(-0.02, o.P, o.a / sim.particles[0].r, o.inc, o.e, o.omega,
                     o.Omega + np.pi)
    clipped = find_z_min(0.0, coeffs)[0]
    recentred = find_separation_minimum(coeffs)
    assert abs(clipped) <= 0.0105          # single call clips at the bracket edge
    assert abs(recentred - 0.02) < 2e-3    # re-centred search finds the true minimum


def test_expansion_point_coincides_with_reported_center(two_body_sim):
    """The returned coefficients must be expanded at the (uncorrected) centre:
    the separation minimum of the returned expansion sits at offset ~0."""
    tc_model, coeffs = calculate_center_and_orbit(two_body_sim.copy(), P['t0'][0], 0)
    assert abs(find_z_min(0.0, coeffs)[0]) * SEC < 0.5


# ----------------------------------------------------------------------
# Regression: rebound rejects numpy integers as particle indices
# ----------------------------------------------------------------------
@pytest.mark.parametrize('cast', [int, np.int64, np.int32], ids=['int', 'int64', 'int32'])
def test_center_functions_accept_numpy_integer_ipl(two_body_sim, cast):
    """`ipl` may arrive as a numpy integer (e.g. from concatenated planet-id arrays).

    rebound's `sim.particles[...]` accepts only str, python int, or c_uint32, so the
    planet index must be coerced before it reaches a particle lookup.
    """
    et = eclipse_time_offset(P['p'][0], P['inc'][0], P['e'][0], P['w'][0])
    tc_ref, _ = calculate_center_and_orbit(two_body_sim.copy(), P['t0'][0], 0)
    ec_ref, _ = calculate_eclipse_center_and_orbit(two_body_sim.copy(), P['t0'][0] + et, 0)

    tc, tc_coeffs = calculate_center_and_orbit(two_body_sim.copy(), P['t0'][0], cast(0))
    ec, _ = calculate_eclipse_center_and_orbit(two_body_sim.copy(), P['t0'][0] + et, cast(0))

    assert tc == tc_ref
    assert ec == ec_ref
    assert tc_coeffs.shape == (3, 5)


@pytest.mark.parametrize('cast', [int, np.int64, np.int32], ids=['int', 'int64', 'int32'])
def test_expand_orbit_accepts_numpy_integer_pid(two_body_sim, cast):
    """`expand_orbit` indexes `sim.particles` directly and must coerce `pid` too."""
    tc_ref, p_ref, coeffs_ref = expand_orbit(two_body_sim.copy(), 1)
    tc, p, coeffs = expand_orbit(two_body_sim.copy(), cast(1))
    assert (tc, p) == (tc_ref, p_ref)
    assert np.array_equal(coeffs, coeffs_ref)


@pytest.mark.parametrize('cast', [int, np.int64, np.int32], ids=['int', 'int64', 'int32'])
def test_expand_orbit_d_accepts_numpy_integer_pid(two_body_sim, cast):
    """`expand_orbit_d` indexes `sim.particles` directly and must coerce `pid` too."""
    tc_ref, p_ref, coeffs_ref, dcoeffs_ref = expand_orbit_d(two_body_sim.copy(), 1)
    tc, p, coeffs, dcoeffs = expand_orbit_d(two_body_sim.copy(), cast(1))
    assert (tc, p) == (tc_ref, p_ref)
    assert np.array_equal(coeffs, coeffs_ref)
    assert np.array_equal(dcoeffs, dcoeffs_ref)
