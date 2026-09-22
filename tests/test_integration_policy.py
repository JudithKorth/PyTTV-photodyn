"""Tests for the gap-crossing integration policy in src/pdmodel.py.

The photodynamical model spends most of its time integrating across the empty stretches
between observing seasons, where the system is doing nothing interesting. The policy
swaps IAS15 for a fixed-step symplectic integrator on those long jumps only. These tests
pin down what makes those savings safe: they must not move transit times, and they must
leave everything shorter than the threshold exactly as it was. The IAS15 tolerance used
for the event-local work is held to the same standard at the bottom of the file.
"""
import numpy as np
import pytest
import rebound

from helpers import TWO_PLANET, build_sim, model_arrays
from test_pdlpf import lpf_and_truth  # noqa: F401 (pytest fixture)
from src.pdmodel import (PhotoDynamicalModel, calculate_center_and_orbit,
                         find_first_transit_center, integrate_to, set_integration_policy)

SEC = 86400.0
P_INNER = TWO_PLANET['p'][0]

# A gap comparable to the season gaps in a real TESS + CHEOPS dataset: ~120 inner orbits
# of empty sky, far enough that any per-step phase error has room to accumulate.
GAP = 400.0


def build_system_args(pars):
    """`build_system` positional arguments: `model_arrays` minus the ldc and k it omits."""
    mstar, rstar, _ldc, mp, _k, *elements = model_arrays(pars)
    return (mstar, rstar, mp, *elements)


def transit_time_after_gap(sim):
    """Transit centre of the inner planet on the far side of a long data gap."""
    return find_first_transit_center(sim, GAP, P_INNER, 0, 100)[0]


def test_gap_crossing_preserves_transit_times():
    """A gap crossed under the policy must give the same transit time as pure IAS15."""
    reference = transit_time_after_gap(build_sim(TWO_PLANET))

    sim = build_sim(TWO_PLANET)
    set_integration_policy(sim, gap_threshold=10.0, gap_dt=P_INNER / 100)
    hybrid = transit_time_after_gap(sim)

    assert abs(hybrid - reference) * SEC < 1.0


def test_gap_crossing_leaves_the_simulation_on_ias15():
    """The cheap integrator is for the gap only; the event-local work stays on IAS15."""
    sim = build_sim(TWO_PLANET)
    set_integration_policy(sim, gap_threshold=10.0, gap_dt=P_INNER / 100)

    integrate_to(sim, GAP)

    assert sim.integrator == 'ias15'


def test_backward_gap_crossing_preserves_transit_times():
    """Integrating backward across a gap must work as well as forward."""
    reference = build_sim(TWO_PLANET)
    reference.integrate(-GAP)

    sim = build_sim(TWO_PLANET)
    set_integration_policy(sim, gap_threshold=10.0, gap_dt=P_INNER / 100)
    integrate_to(sim, -GAP)

    tc_ref = calculate_center_and_orbit(reference, -GAP + 1.0, 0)[0]
    tc_hyb = calculate_center_and_orbit(sim, -GAP + 1.0, 0)[0]
    assert abs(tc_hyb - tc_ref) * SEC < 1.0


def test_step_below_the_threshold_is_untouched():
    """Anything shorter than the threshold must integrate exactly as it did before."""
    plain = build_sim(TWO_PLANET)
    plain.integrate(5.0)

    sim = build_sim(TWO_PLANET)
    set_integration_policy(sim, gap_threshold=10.0, gap_dt=P_INNER / 100)
    integrate_to(sim, 5.0)

    assert sim.particles[1].x == plain.particles[1].x
    assert sim.particles[1].vx == plain.particles[1].vx


def test_simulation_without_a_policy_falls_back_to_plain_integrate():
    """`integrate_to` must accept a bare rebound simulation and change nothing."""
    plain = build_sim(TWO_PLANET)
    plain.integrate(GAP)

    sim = build_sim(TWO_PLANET)
    integrate_to(sim, GAP)

    assert sim.particles[1].x == plain.particles[1].x


def test_disabled_policy_falls_back_to_plain_integrate():
    """A `None` threshold turns the policy off without the caller special-casing it."""
    plain = build_sim(TWO_PLANET)
    plain.integrate(GAP)

    sim = build_sim(TWO_PLANET)
    set_integration_policy(sim, gap_threshold=None, gap_dt=P_INNER / 100)
    integrate_to(sim, GAP)

    assert sim.particles[1].x == plain.particles[1].x


def test_copy_sim_stamps_the_policy_on_the_copy():
    """Every simulation the model hands out for integration must carry the policy."""
    tm = PhotoDynamicalModel(2, [1, 1], 0.0, with_gr=False,
                             gap_threshold=10.0, gap_steps_per_orbit=100)
    tm.build_system(*build_system_args(TWO_PLANET))

    sim = tm.copy_sim()
    assert sim._gap_threshold == 10.0
    assert sim._gap_dt == pytest.approx(P_INNER / 100)


def test_gap_dt_follows_the_shortest_period():
    """The gap timestep is a fraction of the shortest orbit, not a fixed number of days."""
    tm = PhotoDynamicalModel(2, [1, 1], 0.0, with_gr=False, gap_steps_per_orbit=50)
    tm.build_system(*build_system_args(TWO_PLANET))

    assert tm.copy_sim()._gap_dt == pytest.approx(min(TWO_PLANET['p']) / 50)


# The RV path reaches its epochs through the same long jumps as the light curves, so it
# has to go through the policy as well. `gap_steps_per_orbit` is the lever both tests use:
# an absurdly coarse gap step must visibly move the answer if the policy is really in
# effect, and the shipped one must not move it at all.
RV_GAP_TIMES = np.array([600.0, 600.5, 601.0, 601.5])


def test_rv_path_goes_through_the_integration_policy(lpf_and_truth):
    """A deliberately coarse gap step must change predicted RVs across a gap."""
    lpf, pv = lpf_and_truth[:2]
    times = lpf.tm.tref + RV_GAP_TIMES

    lpf.tm.gap_threshold = None
    reference = lpf.predict_rvs(pv, times)

    lpf.tm.gap_threshold, lpf.tm.gap_steps_per_orbit = 10.0, 2
    assert np.abs(lpf.predict_rvs(pv, times) - reference).max() > 0.01     # measures ~0.08


def test_predicted_rvs_across_a_gap_match_ias15(lpf_and_truth):
    """With the shipped gap step the RV path must agree with pure IAS15."""
    lpf, pv = lpf_and_truth[:2]
    times = lpf.tm.tref + RV_GAP_TIMES

    lpf.tm.gap_threshold = None
    reference = lpf.predict_rvs(pv, times)

    lpf.tm.gap_threshold, lpf.tm.gap_steps_per_orbit = 10.0, 100
    hybrid = lpf.predict_rvs(pv, times)

    assert np.abs(hybrid - reference).max() < 0.001     # m/s; measures ~4e-6


# --------------------------------------------------------------------------------------
# IAS15 accuracy
# --------------------------------------------------------------------------------------
def transit_time_far_out(**model_kwargs):
    """Transit centre a long way from the epoch, with the gap policy out of the way."""
    tm = PhotoDynamicalModel(2, [1, 1], 0.0, with_gr=False, gap_threshold=None,
                             **model_kwargs)
    tm.build_system(*build_system_args(TWO_PLANET))
    return find_first_transit_center(tm.copy_sim(), GAP, P_INNER, 0, 100)[0]


def test_ias15_epsilon_reaches_the_integrated_simulation():
    """The tolerance has to survive `copy_sim`, since that is what gets integrated."""
    tm = PhotoDynamicalModel(2, [1, 1], 0.0, with_gr=False, ias15_epsilon=1e-6)
    tm.build_system(*build_system_args(TWO_PLANET))

    assert tm.copy_sim().ri_ias15.epsilon == 1e-6


# What these two guard, and what they do not. IAS15 adapts its own timestep, so over the
# smooth configurations this model integrates the tolerance decides how much work it does
# and not how accurate it ends up: transit times here are unchanged from rebound's 1e-9
# default all the way out to eps=1e-2 (measured; the deviation first clears a second at
# eps=1, at 55 s). So these are guard rails against a catastrophically wrong tolerance,
# not a fine-grained accuracy check -- do not read a pass as evidence that some newly
# loosened value is safe on a system it was not measured on.
def test_loosened_epsilon_preserves_transit_times():
    """The shipped tolerance must not move transit times against rebound's default."""
    reference = transit_time_far_out(ias15_epsilon=1e-9)
    loosened = transit_time_far_out(ias15_epsilon=1e-6)

    assert abs(loosened - reference) * SEC < 1.0


def test_default_epsilon_preserves_transit_times():
    """Whatever the default is, it has to clear the same bar as the value it ships with."""
    reference = transit_time_far_out(ias15_epsilon=1e-9)
    default = transit_time_far_out()

    assert abs(default - reference) * SEC < 1.0
