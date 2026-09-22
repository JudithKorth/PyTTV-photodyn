"""Tests for pdsimulation.py.

The tests drive a stub LPF rather than a real `PhotoDynamicalLPF`, following the pattern in
`test_log_posterior_and_grad.py`: `PDSimulation` is a driver, so what needs testing is the
sequencing, the file naming, and the pool lifetime, none of which need a 105 ms model.
"""
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xa

from src.pdsimulation import PDSimulation, _lnposterior


class StubPS:
    """Parameter set exposing what PDSimulation uses."""

    def __init__(self, npar=3):
        self.npar = npar
        self.names = [f'p_{i}' for i in range(npar)]
        self.draws = 0

    def __len__(self):
        return self.npar

    def sample_from_prior(self, n):
        # Deterministic but varying, so that roughly half the draws are rejected by the
        # stub likelihood below -- which is the loop `sample_from_prior` has to get past.
        self.draws += 1
        return np.random.default_rng(self.draws).uniform(-1, 1, (n, self.npar))


class StubLPF:
    """Records what it was asked to do, and the name it carried at the time."""

    def __init__(self, name='stub', npar=3):
        self.name = name
        self.ps = StubPS(npar)
        self.de = None
        self.sampler = None
        self.calls = []
        self.optimize_raises = False
        self.figure = None

    def lnposterior(self, pv):
        return -0.5 * float(np.sum(np.asarray(pv) ** 2))

    def lnlikelihood(self, pv):
        return -np.inf if pv[0] < 0.0 else self.lnposterior(pv)

    def optimize_global(self, niter=200, npop=100, population=None, pool=None, lnpost=None,
                        return_cplot=False, **kwargs):
        self.calls.append(('optimize', niter, npop, self.name))
        if self.optimize_raises:
            raise RuntimeError('optimisation blew up')
        self.de = SimpleNamespace(population=np.zeros((npop, len(self.ps))))
        return self.figure

    def sample_mcmc(self, niter=500, thin=5, repeats=1, population=None, pool=None,
                    lnpost=None, **kwargs):
        self.calls.append(('sample', niter, thin, repeats, self.name))

    def save(self, save_path='.'):
        Path(save_path).joinpath(f'{self.name}.nc').touch()


@pytest.fixture
def sim(tmp_path):
    lpf = StubLPF()
    s = PDSimulation(lpf, nproc=2, result_dir=tmp_path)
    yield s
    s.close()


# --------------------------------------------------------------------------------------
# The constraint the whole design exists for
# --------------------------------------------------------------------------------------
def test_the_worker_function_pickles_by_reference():
    """The pool callable must not drag the LPF across the pipe on every task batch.

    A module-level function pickles as a module-and-name reference; a bound method pickles
    the instance behind it, which for a real LPF is 4.8 MB per task batch.
    """
    assert len(pickle.dumps(_lnposterior)) < 1024


def test_the_worker_evaluates_the_published_lpf(tmp_path):
    """Publishing happens in the constructor, before the pool is forked."""
    lpf = StubLPF()
    sim = PDSimulation(lpf, nproc=2, result_dir=tmp_path)
    try:
        pv = np.array([1.0, 2.0, 3.0])
        assert _lnposterior(pv) == lpf.lnposterior(pv)
    finally:
        sim.close()


def test_a_second_simulation_does_not_trip_the_start_method(tmp_path):
    """`set_start_method` raises if called twice, which a notebook will do."""
    first = PDSimulation(StubLPF(), nproc=2, result_dir=tmp_path)
    first.close()
    second = PDSimulation(StubLPF(), nproc=2, result_dir=tmp_path)
    second.close()


# --------------------------------------------------------------------------------------
# Pool lifetime
# --------------------------------------------------------------------------------------
def test_close_is_idempotent(tmp_path):
    sim = PDSimulation(StubLPF(), nproc=2, result_dir=tmp_path)
    sim.close()
    sim.close()


def test_context_manager_closes_the_pool(tmp_path):
    with PDSimulation(StubLPF(), nproc=2, result_dir=tmp_path) as sim:
        assert sim.pool is not None
    assert sim.pool is None


# --------------------------------------------------------------------------------------
# The DE name suffix
# --------------------------------------------------------------------------------------
def test_optimisation_runs_under_the_de_name(sim):
    """`save` derives its filename from `lpf.name`, so DE has to run under the suffix."""
    sim.optimize(niter=10, npop=4, plot_every=10)

    assert [c for c in sim.lpf.calls if c[0] == 'optimize'][0][3] == 'stub_de'


def test_optimisation_restores_the_name(sim):
    sim.optimize(niter=10, npop=4, plot_every=10)

    assert sim.lpf.name == 'stub'


def test_optimisation_restores_the_name_after_a_failure(sim):
    """The current script leaks the `_de` suffix when DE raises; this must not."""
    sim.lpf.optimize_raises = True

    with pytest.raises(RuntimeError):
        sim.optimize(niter=10, npop=4, plot_every=10)

    assert sim.lpf.name == 'stub'


# --------------------------------------------------------------------------------------
# Chunking, saving, and returned population
# --------------------------------------------------------------------------------------
def test_optimisation_is_chunked_by_plot_every(sim):
    """Splitting the run is what lets a convergence plot be emitted along the way."""
    sim.optimize(niter=100, npop=4, plot_every=25)

    assert len([c for c in sim.lpf.calls if c[0] == 'optimize']) == 4


def test_optimisation_saves_under_the_de_name(sim, tmp_path):
    sim.optimize(niter=10, npop=4, plot_every=10)

    assert (tmp_path / 'stub_de.nc').exists()
    assert not (tmp_path / 'stub.nc').exists()


def test_optimisation_returns_the_evolved_population(sim):
    population = sim.optimize(niter=10, npop=4, plot_every=10)

    assert population.shape == (4, len(sim.lpf.ps))


def test_optimisation_writes_a_convergence_plot_when_one_is_returned(sim, tmp_path):
    saved = []
    sim.lpf.figure = SimpleNamespace(savefig=lambda p: saved.append(Path(p)))

    sim.optimize(niter=20, npop=4, plot_every=10)

    assert [p.name for p in saved] == ['stub_de_convergence_1.pdf',
                                       'stub_de_convergence_2.pdf']


def test_sampling_runs_once_per_repeat_and_saves(sim, tmp_path):
    sim.sample_mcmc(niter=10, thin=2, repeats=3)

    assert len([c for c in sim.lpf.calls if c[0] == 'sample']) == 3
    assert (tmp_path / 'stub.nc').exists()


def test_sampling_can_skip_saving(sim, tmp_path):
    sim.sample_mcmc(niter=10, thin=2, repeats=1, save=False)

    assert not (tmp_path / 'stub.nc').exists()


def test_only_the_first_repeat_gets_the_starting_population(sim):
    """emcee continues from its own chain state; re-seeding it would discard the burn-in."""
    seen = []
    sim.lpf.sample_mcmc = lambda population=None, **kw: seen.append(population)

    sim.sample_mcmc(niter=10, thin=2, repeats=3, population=np.zeros((4, 3)), save=False)

    assert seen[0] is not None
    assert seen[1] is None and seen[2] is None


# --------------------------------------------------------------------------------------
# The full run
# --------------------------------------------------------------------------------------
def _store_chain(path, name='stub'):
    chain = np.arange(60, dtype=float).reshape(4, 5, 3)     # walkers, steps, parameters
    xa.Dataset({'mcmc_samples': xa.DataArray(chain, dims=['pvector', 'step', 'parameter'])}
               ).to_netcdf(path / f'{name}.nc')


def test_a_fresh_run_optimizes_and_then_samples(sim):
    sim.run(npop=4, de_niter=10, mc_niter=10, mc_repeats=1, plot_every=10)

    assert [c[0] for c in sim.lpf.calls] == ['optimize', 'sample']


def test_a_run_with_an_mcmc_result_skips_the_optimisation(sim, tmp_path):
    _store_chain(tmp_path)

    sim.run(npop=4, de_niter=10, mc_niter=10, mc_repeats=1, plot_every=10)

    assert [c[0] for c in sim.lpf.calls] == ['sample']


def test_a_restarted_run_ignores_the_mcmc_result(sim, tmp_path):
    _store_chain(tmp_path)

    sim.run(npop=4, de_niter=10, mc_niter=10, mc_repeats=1, plot_every=10, restart=True)

    assert [c[0] for c in sim.lpf.calls] == ['optimize', 'sample']


def test_continuing_de_from_mc_seeds_the_optimisation_from_the_chain(sim, tmp_path):
    _store_chain(tmp_path)
    seeds = []

    def record_optimize(niter, npop, population=None, **kwargs):
        seeds.append(population)
        return population

    sim.optimize = record_optimize

    sim.run(npop=4, de_niter=10, mc_niter=10, mc_repeats=1, plot_every=10,
            continue_de_from_mc=True)

    assert seeds[0] is not None and seeds[0].shape == (4, 3)


def test_continuing_de_from_mc_fails_loudly_without_a_chain(sim):
    with pytest.raises(ValueError, match='MCMC result'):
        sim.run(npop=4, continue_de_from_mc=True)

    assert sim.lpf.calls == []


def test_run_accepts_a_splatted_argument_parser_namespace(sim):
    """`run(**vars(args))` is the whole command-line interface, so the keyword names must
    track the `dest` names the parser produces, and the constructor-only keys
    (scenario, nproc, result_dir) must be swallowed rather than raise."""
    ap = PDSimulation.create_argument_parser(('1a',))
    args = ap.parse_args(['1a', '--npop', '4', '--de-niter', '10', '--mc-niter', '10',
                          '--mc-repeats', '1', '--plot-every', '10'])

    sim.run(**vars(args))

    assert [c[0] for c in sim.lpf.calls] == ['optimize', 'sample']


# --------------------------------------------------------------------------------------
# Initial populations
# --------------------------------------------------------------------------------------
def test_prior_sample_keeps_only_finite_likelihood_vectors(sim):
    population = sim.sample_from_prior(5)

    assert population.shape == (5, len(sim.lpf.ps))
    assert all(np.isfinite(sim.lpf.lnlikelihood(pv)) for pv in population)


def test_prior_sampling_gives_up_rather_than_looping_forever(sim):
    """A prior region where the model never evaluates must fail loudly, not hang.

    The loop is unbounded by nature -- it draws until it has enough usable vectors -- so
    without a cap a badly specified prior silently wedges the run with no output at all.
    """
    sim.lpf.lnlikelihood = lambda pv: -np.inf

    with pytest.raises(ValueError, match='finite'):
        sim.sample_from_prior(2, max_attempts=10, use_tqdm=False)


def test_population_loaders_return_none_when_no_file_exists(sim):
    assert sim.load_de_population() is None
    assert sim.load_mcmc_population(4) is None


def test_de_population_is_loaded_from_the_de_file(sim, tmp_path):
    expected = np.arange(12, dtype=float).reshape(4, 3)
    xa.Dataset({'de_population': xa.DataArray(expected, dims=['pvector', 'parameter'])}
               ).to_netcdf(tmp_path / 'stub_de.nc')

    np.testing.assert_allclose(sim.load_de_population(), expected)


def test_mcmc_population_is_drawn_from_the_stored_chain(sim, tmp_path):
    chain = np.arange(60, dtype=float).reshape(4, 5, 3)     # walkers, steps, parameters
    xa.Dataset({'mcmc_samples': xa.DataArray(chain, dims=['pvector', 'step', 'parameter'])}
               ).to_netcdf(tmp_path / 'stub.nc')

    population = sim.load_mcmc_population(6)

    assert population.shape == (6, 3)
    flat = chain.reshape(-1, 3)
    assert all(any(np.allclose(pv, row) for row in flat) for pv in population)
