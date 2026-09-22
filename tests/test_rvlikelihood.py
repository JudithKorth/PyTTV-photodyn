"""Tests for the pluggable RV noise models in src/rvlikelihood.py."""
import numpy as np
import pytest
from scipy.stats import norm

from helpers import EXPTIME  # noqa: F401  (sets sys.path)
from pytransit.param import ParameterSet, GParameter, NormalPrior as NP
from numpy import inf
from src.rvlikelihood import RVLikelihood, WNRVLikelihood, lnlike_normal
from src.rvlikelihood import QPGPRVLikelihood, with_george
import src.rvlikelihood as rvlikelihood_module


class StubLPF:
    """Minimal stand-in exposing only the attributes an RVLikelihood touches."""

    def __init__(self, errors, ids, nrvsets, orvtimes=None):
        self.nrvsets = nrvsets
        self._orverrors = errors
        self._orvids = ids
        self._orvtimes = orvtimes


def make_wn(nrvsets=2, npt=12, seed=0):
    """Build a WNRVLikelihood bound to a stub LPF, with its parameters declared."""
    rng = np.random.default_rng(seed)
    errors = rng.uniform(0.5, 2.0, npt)
    ids = np.sort(rng.integers(0, nrvsets, npt))
    lpf = StubLPF(errors, ids, nrvsets)

    lnl = WNRVLikelihood()
    ps = ParameterSet()
    lnl.init_parameters(lpf, ps)
    ps.freeze()
    lnl.setup(lpf)
    return lnl, ps, lpf, rng


def test_lnlike_normal_accepts_scalar_model(rng):
    """The plugin interface passes residuals, so the model argument is a scalar 0."""
    r, e = rng.normal(0, 1, 25), rng.uniform(0.5, 2, 25)
    assert np.isclose(lnlike_normal(r, 0.0, e), norm.logpdf(r, 0.0, e).sum())


def test_base_class_is_abstract():
    base = RVLikelihood()
    with pytest.raises(NotImplementedError):
        base.init_parameters(None, None)
    with pytest.raises(NotImplementedError):
        base(None, None)


def test_wn_declares_one_jitter_parameter_per_set():
    lnl, ps, _, _ = make_wn(nrvsets=3)
    assert [p.name for p in ps] == ['log10rvj_0', 'log10rvj_1', 'log10rvj_2']
    assert lnl.start == 0
    assert lnl.slice == slice(0, 3)


def test_wn_matches_scipy():
    lnl, ps, lpf, rng = make_wn(nrvsets=2, npt=12)
    pv = np.array([-1.0, -0.5])
    resid = rng.normal(0, 1, lpf._orverrors.size)

    jitter = 10 ** pv[lpf._orvids]
    expected = norm.logpdf(resid, 0.0, np.sqrt(lpf._orverrors ** 2 + jitter ** 2)).sum()
    assert np.isclose(lnl(pv, resid), expected)


def test_wn_jitter_reduces_likelihood_of_small_residuals():
    """Adding jitter widens the errors, so tight residuals become less likely."""
    lnl, _, lpf, rng = make_wn(nrvsets=1, npt=20)
    resid = rng.normal(0, 0.1, lpf._orverrors.size)
    assert lnl(np.array([-3.0]), resid) > lnl(np.array([1.0]), resid)


def test_second_bind_raises_value_error():
    """A shared plugin instance must not silently rebind to a second LPF.

    Reusing one RVLikelihood instance across two LPFs would otherwise let the
    second LPF's init_parameters overwrite the first's start/slice (and any data
    cached in setup), corrupting the first LPF -- invisibly, if the two LPFs
    happen to share the same parameter layout.
    """
    lnl = WNRVLikelihood()
    ps1 = ParameterSet()
    lnl.init_parameters(StubLPF(np.array([1.0]), np.array([0]), 1), ps1)

    ps2 = ParameterSet()
    with pytest.raises(ValueError):
        lnl.init_parameters(StubLPF(np.array([1.0, 2.0]), np.array([0, 0]), 1), ps2)


def test_wn_zero_rvsets_gives_empty_block():
    """No RV data: the block is empty but still present, keeping later block starts valid."""
    ps = ParameterSet()
    ps.add_global_block('filler', [GParameter('filler', 'filler', '', NP(0, 1), [-inf, inf])])

    lnl = WNRVLikelihood()
    lnl.init_parameters(StubLPF(np.array([]), np.array([], int), 0), ps)
    ps.freeze()

    assert lnl.slice == slice(1, 1)
    assert lnl.start == 1
    assert len(ps) == 1


georgeonly = pytest.mark.skipif(not with_george, reason='george is not installed')

GP_PV = np.array([2.0,      # gp_ap_std
                  30.0,     # gp_ap_scale
                  5.0,      # gp_std
                  -0.5,     # gp_log10_gamma
                  20.95,    # gp_period
                  100.0])   # gp_coherence


@georgeonly
def test_gp_declares_named_physical_parameters():
    lpf = StubLPF(np.full(12, 1.0), np.zeros(12, int), 1)
    lnl = QPGPRVLikelihood()
    ps = ParameterSet()
    lnl.init_parameters(lpf, ps)
    ps.freeze()
    assert [p.name for p in ps] == ['gp_ap_std', 'gp_ap_scale', 'gp_std',
                                    'gp_log10_gamma', 'gp_period', 'gp_coherence']


@georgeonly
def test_gp_with_jitter_appends_one_parameter_per_set():
    lpf = StubLPF(np.full(12, 1.0), np.zeros(12, int), 2)
    lnl = QPGPRVLikelihood(with_jitter=True)
    ps = ParameterSet()
    lnl.init_parameters(lpf, ps)
    ps.freeze()
    assert [p.name for p in ps][-2:] == ['log10rvj_0', 'log10rvj_1']


@georgeonly
def test_gp_matches_hand_built_george():
    """The physical-to-kernel-vector mapping is the easiest thing to get silently wrong."""
    from numpy import log
    from george import GP
    from george.kernels import ExpSquaredKernel, ExpSine2Kernel, Matern32Kernel

    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 200, 30))
    e = np.full(30, 1.5)
    resid = rng.normal(0, 3, 30)

    lpf = StubLPF(e, np.zeros(30, int), 1, orvtimes=t)
    lnl = QPGPRVLikelihood()
    ps = ParameterSet()
    lnl.init_parameters(lpf, ps)
    ps.freeze()
    lnl.setup(lpf)

    s_ap, l_ap, s, g, p, l = GP_PV
    kernel = (s_ap ** 2 * Matern32Kernel(l_ap ** 2)
              + s ** 2 * ExpSine2Kernel(10.0 ** g, log(p)) * ExpSquaredKernel(l ** 2))
    ref = GP(kernel)
    ref.compute(t, yerr=e)

    assert np.isclose(lnl(GP_PV, resid), ref.log_likelihood(resid))


@georgeonly
def test_gp_rejects_bad_hyperparameters():
    """A non-finite kernel vector must return -inf rather than raising."""
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0, 200, 20))
    lpf = StubLPF(np.full(20, 1.0), np.zeros(20, int), 1, orvtimes=t)
    lnl = QPGPRVLikelihood()
    ps = ParameterSet()
    lnl.init_parameters(lpf, ps)
    ps.freeze()
    lnl.setup(lpf)

    bad = GP_PV.copy()
    bad[0] = np.nan
    assert lnl(bad, rng.normal(0, 1, 20)) == -np.inf


@georgeonly
def test_gp_rejects_non_finite_residuals():
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 200, 20))
    lpf = StubLPF(np.full(20, 1.0), np.zeros(20, int), 1, orvtimes=t)
    lnl = QPGPRVLikelihood()
    ps = ParameterSet()
    lnl.init_parameters(lpf, ps)
    ps.freeze()
    lnl.setup(lpf)

    resid = rng.normal(0, 1, 20)
    resid[3] = np.inf
    assert lnl(GP_PV, resid) == -np.inf


@georgeonly
def test_gp_survives_pickling():
    """sample_mcmc may run under a multiprocessing pool."""
    import pickle
    rng = np.random.default_rng(6)
    t = np.sort(rng.uniform(0, 200, 20))
    lpf = StubLPF(np.full(20, 1.0), np.zeros(20, int), 1, orvtimes=t)
    lnl = QPGPRVLikelihood()
    ps = ParameterSet()
    lnl.init_parameters(lpf, ps)
    ps.freeze()
    lnl.setup(lpf)

    resid = rng.normal(0, 1, 20)
    assert np.isclose(pickle.loads(pickle.dumps(lnl))(GP_PV, resid), lnl(GP_PV, resid))


def test_qpgp_raises_import_error_without_george(monkeypatch):
    """QPGPRVLikelihood must fail loudly, not import george lazily, when unavailable."""
    monkeypatch.setattr(rvlikelihood_module, 'with_george', False)
    with pytest.raises(ImportError):
        rvlikelihood_module.QPGPRVLikelihood()
