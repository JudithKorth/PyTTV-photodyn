"""Tests for the njit flux-model kernels quadratic_model_s and eclipse_model_s."""
import numpy as np
from meepmeep.numba3d import sep_c
from pytransit.models.numba.ma_quadratic_nb import eval_quad_z_s
from meepmeep.backends.numba.utils import eclipse_time_offset

from helpers import TWO_BODY
from src.pdmodel import (quadratic_model_s, eclipse_model_s, expand_orbit,
                         calculate_center_and_orbit)

P = TWO_BODY
K = P['k'][0]
LDC = np.array([0.4, 0.2])


def _transit_setup(sim):
    tc, coeffs = calculate_center_and_orbit(sim, P['t0'][0], 0)
    t = np.arange(tc - 0.15, tc + 0.15, 2.0 / 60 / 24)
    return tc, coeffs, t


def test_quadratic_model_depth(two_body_sim):
    """With nsamples=1 and zero exptime the minimum flux must equal the
    pytransit quadratic model evaluated at the minimum separation."""
    tc, coeffs, t = _transit_setup(two_body_sim)
    flux = quadratic_model_s(t, K, LDC, 0, 1, 0.0, 1, tc, coeffs, np.ones(t.size))
    z_min = sep_c(t[np.argmin(flux)] - tc, coeffs)
    assert abs(flux.min() - eval_quad_z_s(z_min, K, LDC)) < 1e-12
    assert 0.005 < 1 - flux.min() < 0.02


def test_quadratic_model_out_of_transit_unity(two_body_sim):
    tc, coeffs, _ = _transit_setup(two_body_sim)
    t = tc + np.linspace(0.5, 1.0, 20)  # far outside the window
    flux = quadratic_model_s(t, K, LDC, 0, 1, 0.0, 1, tc, coeffs, np.ones(t.size))
    assert np.all(flux == 1.0)


def test_quadratic_model_nan_k_is_noop(two_body_sim):
    tc, coeffs, t = _transit_setup(two_body_sim)
    flux = quadratic_model_s(t, np.nan, LDC, 0, 1, 0.0, 1, tc, coeffs, np.ones(t.size))
    assert np.all(flux == 1.0)


def test_quadratic_model_multi_passband_slicing(two_body_sim):
    """With npb=2 the model must select the requested passband's ldc pair."""
    tc, coeffs, t = _transit_setup(two_body_sim)
    ldc2 = np.array([0.1, 0.05, 0.7, 0.2])
    fl_a = quadratic_model_s(t, K, ldc2, 1, 1, 0.0, 2, tc, coeffs, np.ones(t.size))
    fl_b = quadratic_model_s(t, K, ldc2[2:], 0, 1, 0.0, 1, tc, coeffs, np.ones(t.size))
    assert np.abs(fl_a - fl_b).max() < 1e-14
    fl_c = quadratic_model_s(t, K, ldc2, 0, 1, 0.0, 2, tc, coeffs, np.ones(t.size))
    assert np.abs(fl_a - fl_c).max() > 1e-4  # different limb darkening -> different shape


def test_eclipse_model_depth_is_fr_k2(two_body_sim):
    fr = 0.1
    et = eclipse_time_offset(P['p'][0], P['inc'][0], P['e'][0], P['w'][0])
    tc, p, coeffs = expand_orbit(two_body_sim, 1, te=et)
    ec = tc + et
    t = np.arange(ec - 0.15, ec + 0.15, 2.0 / 60 / 24)
    flux = eclipse_model_s(t, K, fr, 1, 0.0, ec, coeffs, np.ones(t.size))
    # full eclipse (b << 1 - k): the planet disappears completely
    assert abs((1.0 - flux.min()) - fr * K ** 2) < 1e-12
    assert np.all(flux <= 1.0) and np.all(flux >= 1.0 - fr * K ** 2)


def test_eclipse_model_nan_fr_is_noop(two_body_sim):
    et = eclipse_time_offset(P['p'][0], P['inc'][0], P['e'][0], P['w'][0])
    tc, p, coeffs = expand_orbit(two_body_sim, 1, te=et)
    ec = tc + et
    t = np.arange(ec - 0.15, ec + 0.15, 2.0 / 60 / 24)
    flux = eclipse_model_s(t, K, np.nan, 1, 0.0, ec, coeffs, np.ones(t.size))
    assert np.all(flux == 1.0)


def test_supersampling_averages_over_exposure(two_body_sim):
    """Long exposures must smear the ingress: supersampled flux differs from
    instantaneous flux during ingress/egress but not at mid-transit."""
    tc, coeffs, t = _transit_setup(two_body_sim)
    exptime = 30.0 / 60 / 24  # 30 min
    f_inst = quadratic_model_s(t, K, LDC, 0, 1, 0.0, 1, tc, coeffs, np.ones(t.size))
    f_ss = quadratic_model_s(t, K, LDC, 0, 10, exptime, 1, tc, coeffs, np.ones(t.size))
    assert np.abs(f_ss - f_inst).max() > 1e-4
    assert abs(f_ss[np.argmin(f_inst)] - f_inst.min()) < 1e-4
