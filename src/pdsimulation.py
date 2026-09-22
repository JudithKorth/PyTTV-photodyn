"""Driver for photodynamical optimisation and MCMC sampling over a process pool.

`PDSimulation` holds a log posterior function and the pool its samplers evaluate it on,
and exposes the steps of a fitting run -- building an initial population, evolving it with
differential evolution, sampling it with emcee -- as methods rather than as a script. The
same object drives a command-line run and a notebook session.

The one thing that is not obvious from the outside is why the pool callable lives at module
level instead of being a method. `multiprocessing.Pool.map` pickles the callable along with
every batch of tasks it dispatches. A module-level function pickles as a module-and-name
reference, forty bytes, which the forked worker resolves against its own copy of this
module; a bound method pickles the instance behind it, which for a real photodynamical LPF
is 4.8 MB of light curves shipped down the pipe on every emcee iteration. So the LPF is
published to the module before the pool is forked, the workers inherit it, and
:func:`_lnposterior` reaches it without carrying it.
"""
import multiprocessing
import signal
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xa
from numpy import ndarray
from numpy.random.mtrand import permutation
from tqdm.auto import tqdm

__all__ = ['PDSimulation']

# The log posterior function the pool workers evaluate; see the module docstring.
_LPF = None


def _lnposterior(pv):
    """Evaluate the published log posterior function in a pool worker."""
    try:
        return _LPF.lnposterior(pv)
    except KeyboardInterrupt:
        # REBOUND installs its own C-level SIGINT handler for every integration,
        # overriding the SIG_IGN from `_ignore_sigint`, and turns a caught Ctrl+C into
        # a KeyboardInterrupt -- a BaseException that the pool's worker loop does not
        # catch. The worker would die, taking its chunk of the map with it and leaving
        # the main process waiting for results that can never arrive. Swallow it,
        # restore SIG_IGN, and report the vector as rejected; shutting the run down is
        # the main process's job.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        return -np.inf


def _ignore_sigint():
    """Make a pool worker ignore SIGINT.

    Ctrl+C sends SIGINT to the whole foreground process group, workers included. A worker
    killed mid-queue-operation dies holding the pool's shared locks, and the terminate that
    follows in the main process then hangs on them. With SIGINT ignored in the workers, the
    interrupt is handled by the main process alone, which shuts the pool down with
    `terminate` -- a SIGTERM the workers do respond to.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDSimulation:
    """A photodynamical fitting run: an LPF, a process pool, and the steps between them.

    Parameters
    ----------
    lpf
        Log posterior function to optimise and sample, such as a
        :class:`PhotoDynamicalLPF`.
    nproc
        Number of worker processes.
    result_dir
        Directory the NetCDF results and convergence plots are written to.

    Notes
    -----
    Constructing a `PDSimulation` publishes `lpf` as the one its workers evaluate, so the
    most recently constructed instance owns the workers. Drive one run at a time, and close
    the pool with :meth:`close` or by using the instance as a context manager.
    """

    def __init__(self, lpf, nproc: int = 12, result_dir='.'):
        global _LPF
        self.lpf = lpf
        self.nproc = nproc
        self.result_dir = Path(result_dir)

        # `set_start_method` raises if the context has already been set, which happens as
        # soon as a second simulation is built in the same session.
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('fork')

        _LPF = lpf                      # published before the fork, so workers inherit it
        self.pool = multiprocessing.Pool(processes=nproc, initializer=_ignore_sigint)

    @staticmethod
    def create_argument_parser(scenarios: tuple[str]):
        ap = ArgumentParser()
        ap.add_argument('scenario', type=str, choices=scenarios, help='Simulation scenario.')
        ap.add_argument('--npop', type=int, default=80)
        ap.add_argument('--nproc', type=int, default=12)
        ap.add_argument('--de-niter', type=int, default=25000, help='Number of first optimizer iterations')
        ap.add_argument('--mc-niter', type=int, default=5000, help='Number of MCMC steps per run')
        ap.add_argument('--mc-repeats', type=int, default=2, help='Number of MCMC runs')
        ap.add_argument('--mc-thin', type=int, default=10, help='MCMC thinning factor')
        ap.add_argument('--plot-every', type=int, default=500, help='Create a plot every n iterations')
        ap.add_argument('--result-dir', type=Path, default=Path('.'))
        ap.add_argument('--dont-save', dest='save', action='store_false', default=True)
        ap.add_argument('--restart', action='store_true', default=False)
        ap.add_argument('--continue-de', action='store_true', default=False)
        ap.add_argument('--continue-de-from-mc', action='store_true', default=False)
        ap.add_argument('--tqdm-mininterval', type=float, default=0.1,
                        help='Minimum time between progress-bar updates [s]; raise to minutes when logging to a file')
        return ap

    # ----------------------------------------------------------------------------------
    # Pool lifetime
    # ----------------------------------------------------------------------------------
    def close(self) -> None:
        """Terminate the worker pool. Safe to call more than once."""
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    # ----------------------------------------------------------------------------------
    # File names
    # ----------------------------------------------------------------------------------
    @property
    def de_name(self) -> str:
        """Name the optimisation results are stored under."""
        return f'{self.lpf.name}_de'

    @property
    def de_file(self) -> Path:
        return self.result_dir / f'{self.de_name}.nc'

    @property
    def mcmc_file(self) -> Path:
        return self.result_dir / f'{self.lpf.name}.nc'

    # ----------------------------------------------------------------------------------
    # Initial populations
    # ----------------------------------------------------------------------------------
    def sample_from_prior(self, npop: int, max_attempts: Optional[int] = None,
                          use_tqdm: bool = True) -> ndarray:
        """Draw `npop` parameter vectors from the priors, keeping the ones that evaluate.

        A vector drawn from the priors can still put the system somewhere the model cannot
        be evaluated -- an escape or a collision during the integration -- so the draws are
        filtered on a finite likelihood rather than taken as they come.

        The loop is bounded because it has to be: priors loose enough that the model almost
        never evaluates would otherwise wedge the run silently, with no output and no error.
        `max_attempts` defaults to a hundred draws per vector wanted.
        """
        max_attempts = max_attempts if max_attempts is not None else 100 * npop
        population = []
        with tqdm(total=npop, desc='Sampling from prior', disable=not use_tqdm) as pb:
            for _ in range(max_attempts):
                if len(population) == npop:
                    break
                pv = self.lpf.ps.sample_from_prior(1)[0]
                if np.isfinite(self.lpf.lnlikelihood(pv)):
                    population.append(pv)
                    pb.update()
        if len(population) < npop:
            raise ValueError(
                f'Only {len(population)} of {npop} draws had a finite likelihood in '
                f'{max_attempts} attempts. The priors are probably too wide, or too far '
                f'from a configuration the model can integrate.')
        return np.array(population)

    def load_de_population(self) -> Optional[ndarray]:
        """The stored optimisation population, or ``None`` if there is no result yet."""
        if not self.de_file.exists():
            return None
        with xa.load_dataset(self.de_file) as ds:
            return ds.de_population.values.copy()

    def load_mcmc_population(self, npop: int) -> Optional[ndarray]:
        """`npop` vectors drawn at random from the stored chains, or ``None`` if absent.

        The samples are permuted rather than taken from the end of the chains so that the
        population spans the posterior instead of one corner of it.
        """
        if not self.mcmc_file.exists():
            return None
        with xa.load_dataset(self.mcmc_file) as ds:
            samples = ds.mcmc_samples.data.reshape([-1, ds.parameter.size])
        return permutation(samples)[:npop]

    # ----------------------------------------------------------------------------------
    # The run
    # ----------------------------------------------------------------------------------
    def run(self, npop: int = 80, de_niter: int = 25000, mc_niter: int = 5000,
            mc_thin: int = 10, mc_repeats: int = 2, plot_every: int = 500,
            save: bool = True, restart: bool = False, continue_de: bool = False,
            continue_de_from_mc: bool = False, tqdm_mininterval: float = 0.1, **ignored) -> None:
        """Run the whole fit: resolve a starting population, optimize, then sample.

        Defaults mirror :meth:`create_argument_parser`, and extra keyword arguments are
        ignored so a parsed command-line namespace can be splatted in directly::

            sim.run(**vars(args))

        The starting population is resolved in priority order: ``continue_de`` restarts
        the optimisation from the stored DE population; ``continue_de_from_mc`` restarts
        it from a draw over the stored chains (an error if there are none); an existing
        MCMC result (unless ``restart``) skips the optimisation and continues sampling;
        otherwise the run starts fresh from the prior.
        """
        if continue_de:
            print('Continuing DE')
            skip_de, population = False, self.load_de_population()
        elif continue_de_from_mc:
            population = self.load_mcmc_population(npop)
            if population is None:
                raise ValueError('Cannot continue from an MCMC result')
            print('Continuing DE from an MCMC result')
            skip_de = False
        elif self.mcmc_file.exists() and not restart:
            print('Continuing MCMC')
            skip_de, population = True, self.load_mcmc_population(npop)
        else:
            skip_de, population = False, self.sample_from_prior(npop)

        # The workers ignore SIGINT, so a Ctrl+C lands here alone; terminate the pool
        # before re-raising so the run dies promptly instead of waiting on its workers.
        try:
            if not skip_de:
                population = self.optimize(de_niter, npop, population=population,
                                           plot_every=plot_every, save=save,
                                           tqdm_mininterval=tqdm_mininterval)

            self.sample_mcmc(mc_niter, thin=mc_thin, repeats=mc_repeats,
                             population=population, save=save,
                             tqdm_mininterval=tqdm_mininterval)
        except KeyboardInterrupt:
            self.close()
            raise

    def optimize(self, niter: int, npop: int, population: Optional[ndarray] = None,
                 plot_every: int = 500, save: bool = True, tqdm_mininterval: float = 0.1) -> ndarray:
        """Evolve a population with differential evolution, saving as it goes.

        The run is split into chunks of `plot_every` generations so that intermediate
        results are saved and a convergence plot can be written along the way; a long
        optimisation that dies partway through then still leaves something usable behind.

        Returns the evolved population, ready to seed :meth:`sample_mcmc`.
        """
        nruns = max(1, int(niter / plot_every))
        niter_per_run = int(ceil(niter / nruns))

        # `save` names its files after `lpf.name`, so the optimisation has to run under the
        # DE name for its results to land beside, rather than on top of, the MCMC ones.
        # Resolve the DE name before mutating, since `de_name` derives from `lpf.name`
        # and would otherwise compound the suffix into `<name>_de_de`.
        original_name = self.lpf.name
        de_name = self.de_name
        self.lpf.name = de_name
        try:
            for irun in range(nruns):
                figure = self.lpf.optimize_global(niter_per_run, npop, population=population,
                                                  pool=self.pool, lnpost=_lnposterior,
                                                  return_cplot=True, tqdm_mininterval=tqdm_mininterval)
                if save:
                    self.lpf.save(self.result_dir)
                population = self.lpf.de.population
                if figure:
                    figure.savefig(self.result_dir / f'{de_name}_convergence_{irun + 1}.pdf')
        finally:
            self.lpf.name = original_name
        return population

    def sample_mcmc(self, niter: int, thin: int = 10, repeats: int = 1,
                    population: Optional[ndarray] = None, save: bool = True,
                    tqdm_mininterval: float = 0.1) -> None:
        """Sample the posterior with emcee, saving after each repeat.

        Only the first repeat is given `population`: afterwards the sampler continues from
        its own chain state, and re-seeding it would throw away the burn-in.
        """
        for i in range(repeats):
            self.lpf.sample_mcmc(niter=niter, thin=thin, repeats=1,
                                 population=(population if i == 0 else None),
                                 pool=self.pool, lnpost=_lnposterior,
                                 tqdm_mininterval=tqdm_mininterval)
            if save:
                self.lpf.save(self.result_dir)
