"""Photodynamical log posterior function for transiting exoplanet systems.

This module defines :class:`PhotoDynamicalLPF`, the user-facing fitting class. It
extends :class:`pytransit.BaseLPF` to model photometry, radial velocities, and
transit-timing data jointly by N-body integrating the whole planetary system (see
:mod:`pyttv_photodyn.pdmodel`) rather than treating each planet's transits independently.

The class ingests the data, defines the fit parameters and priors, maps the flat
parameter vector onto the physical quantities the N-body engine expects, scores the
resulting model against the data, and drives the optimisation and MCMC sampling.
"""
import signal
import threading

from pathlib import Path
from typing import List, Iterable, Tuple, Union, Optional, Literal

import astropy.constants as cn
import astropy.units as u
import matplotlib.pyplot as plt
from emcee import EnsembleSampler
from meepmeep.backends.numba.utils import d_from_pkaiews
from numba import njit
from numpy import atleast_2d, zeros_like, sqrt, array, zeros, ones, ones_like, where, unique, concatenate, inf, pi, \
    arctan2, squeeze, \
    clip, isfinite, diag, full, ndarray, argsort, asarray, full_like, log, sum, any, poly1d, polyfit, percentile, \
    floor, median, arange
from numpy.random.mtrand import multivariate_normal

from pytransit.orbits.orbits_py import epoch
from pytransit import BaseLPF
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP, GParameter, PParameter
from pytransit.orbits.orbits_py import as_from_rhop, i_from_baew
from meepmeep.numba3d import t14
from pytransit.utils import downsample_time_1d
from pytransit.utils.de import DiffEvol
from pytransit.utils.io import LCData, LCDataGroup, RVData, RVDataGroup
from pytransit import LSTSQBaseline
from tqdm.auto import tqdm
import seaborn as sb

from .wnloglikelihood import WNLogLikelihood
# lnlike_normal is unused in this module; it is re-exported so that
# `from src.pdlpf import lnlike_normal` keeps working for existing callers/tests.
from .rvlikelihood import RVLikelihood, WNRVLikelihood, _rv_labels, lnlike_normal  # noqa: F401
from .pdmodel import (PhotoDynamicalModel, find_first_transit_center, calculate_center_and_orbit,
                      integrate_to)

c = (cn.c).to(u.AU / u.day).value  # Speed of light in AU / day
RV_CONVERSION = (u.AU / u.day).to(u.m / u.s)


def _restore_sigint_handler() -> None:
    """Reclaim SIGINT from REBOUND before a long wait on the worker pool.

    REBOUND replaces the process's SIGINT disposition with its own C-level handler on
    every `Simulation.integrate` call and never restores it. While an integration runs
    that handler stops it and surfaces as a KeyboardInterrupt, but between integrations
    it swallows the signal outright -- and this process spends the optimisation and
    sampling loops exactly there, waiting on the pool, which made Ctrl+C do nothing.
    Reinstalling Python's default handler right before the loops makes Ctrl+C raise
    KeyboardInterrupt in the main process again. Python only allows this from the main
    thread; anywhere else the disposition is left as it is.
    """
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.default_int_handler)


@njit(fastmath=True, cache=False)
def map_ldc(ldc):
    """Map triangular-sampling limb darkening coefficients to quadratic ones.

    Converts the ``(q1, q2)`` coefficients used for sampling (Kipping 2013) into the
    physical ``(u1, u2)`` quadratic limb darkening coefficients, doing so for every
    passband at once. The ``q`` parametrisation confines a physically valid quadratic
    law to the unit square, which makes it convenient to sample with uniform priors.

    Parameters
    ----------
    ldc : array_like
        Limb darkening coefficients with the ``q1``/``q2`` pairs alternating along the
        last axis: ``[q1_pb0, q2_pb0, q1_pb1, q2_pb1, ...]``. Promoted to 2D so that a
        single coefficient set or a stack of sets can be passed.

    Returns
    -------
    numpy.ndarray
        Array of the same shape holding the ``(u1, u2)`` quadratic coefficients in the
        same interleaved layout.
    """
    ldc = atleast_2d(ldc)
    uv = zeros_like(ldc)
    a, b = sqrt(ldc[:, 0::2]), 2. * ldc[:, 1::2]
    uv[:, 0::2] = a * b
    uv[:, 1::2] = a * (1. - b)
    return uv


@njit(cache=False)
def nan_lnlike_normal(o, m, e):
    """Gaussian log likelihood that tolerates non-finite model values.

    Like :func:`lnlike_normal`, but model entries that are not finite (e.g. a transit
    centre that could not be computed because the planet did not transit for a given
    parameter set) are excluded from the sum and each contributes a large fixed
    penalty instead. This keeps the likelihood finite while strongly disfavouring
    parameter vectors that fail to reproduce an observed event.

    Parameters
    ----------
    o : numpy.ndarray
        Observed values.
    m : numpy.ndarray
        Model values, same shape as ``o``; non-finite entries are masked out.
    e : numpy.ndarray
        Per-point uncertainties, same shape as ``o``.

    Returns
    -------
    float
        The log likelihood over the finite model entries, minus ``1e6`` per masked
        entry.
    """
    mask = isfinite(m)
    return -sum(log(e[mask])) - 0.5 * o[mask].size * log(2. * pi) - 0.5 * sum(
        (o[mask] - m[mask]) ** 2 / e[mask] ** 2) - (~mask).sum() * 1e6


# Configuration returned for a fit without photometry. A single dummy passband is
# needed because `_init_parameters` always emits a `2 * npb` limb-darkening block.
_NO_PHOTOMETRY = dict(passbands=('white',), times=None, fluxes=None, errors=None,
                      pbids=None, covariates=None, wnids=None,
                      nsamples=1, exptimes=0.0, pids=None)


# Configuration returned for a fit without radial velocities.
_NO_RVS = dict(times=None, values=None, errors=None, instruments=None)


def _unpack_rvs(rvdata) -> dict:
    """Expand a radial velocity group into the per-set lists the LPF works with.

    Parameters
    ----------
    rvdata : RVDataGroup, RVData, or None
        The radial velocities. A single dataset is wrapped into a group. ``None`` selects
        the no-RV configuration.

    Returns
    -------
    dict
        The keys ``times``, ``values``, ``errors`` and ``instruments``, each a list with
        one entry per dataset, or all ``None`` when there are no radial velocities.

    Raises
    ------
    TypeError
        If ``rvdata`` is neither a dataset, a dataset group, nor ``None``.
    ValueError
        If the group is empty.

    Notes
    -----
    The instrument labels come from ``instruments`` rather than from ``rvis``. The latter
    raises when two datasets share a label, because `pytransit.lpf.rvlpf.RVLPF` names one
    parameter per label; this class names its systemic velocities and jitters by set
    index instead, so duplicate and empty labels are harmless here.

    There is no counterpart to the photometry helper's ``has_errors`` gate: `RVData`
    requires its uncertainties and raises without them.
    """
    if rvdata is None:
        return dict(_NO_RVS)

    if isinstance(rvdata, RVData):
        rvdata = RVDataGroup(rvdata)
    if not isinstance(rvdata, RVDataGroup):
        raise TypeError(f"rvdata must be an RVData or an RVDataGroup, got "
                        f"{type(rvdata).__name__}.")

    if rvdata.size == 0:
        raise ValueError("rvdata is empty. Pass rvdata=None to fit without radial "
                         "velocities.")

    return dict(times=rvdata.times, values=rvdata.rvs, errors=rvdata.errors,
                instruments=rvdata.instruments)


# Configuration returned for a fit without transit-centre measurements.
_NO_CENTERS = dict(times=None, errors=None)


def _unpack_centers(ctdata, nplanets: int) -> dict:
    """Expand the transit-centre measurements into the per-planet lists the LPF uses.

    Parameters
    ----------
    ctdata : tuple, or None
        A ``(centers, errors)`` pair, each a sequence with one entry per planet holding
        that planet's measured mid-transit times and their uncertainties. A planet with
        no measurements gets an empty array. ``None`` selects the no-centres
        configuration.
    nplanets : int
        Number of planets in the system, used to check the sequence lengths.

    Returns
    -------
    dict
        The keys ``times`` and ``errors``, each a list with one entry per planet, or both
        ``None`` when there are no measurements.

    Raises
    ------
    ValueError
        If ``ctdata`` is not a pair, if the two sequences differ in length, if they name
        more planets than the system has, or if a planet's centres and uncertainties
        differ in length.

    Notes
    -----
    The planet a measurement belongs to is the index into the outer sequence, which is
    why the centres cannot simply be pooled into one flat array: the index becomes
    ``tcipl`` on the N-body model and decides which planet each centre event tracks.
    """
    if ctdata is None:
        return dict(_NO_CENTERS)

    if len(ctdata) != 2:
        raise ValueError(f"ctdata must be a (centers, errors) pair, got {len(ctdata)} "
                         f"elements.")
    times, errors = ctdata

    if len(times) != len(errors):
        raise ValueError(f"ctdata gives centres for {len(times)} planets but "
                         f"uncertainties for {len(errors)}.")
    if len(times) > nplanets:
        raise ValueError(f"ctdata gives centres for {len(times)} planets, but the system "
                         f"has {nplanets}.")

    times = [asarray(t, 'd') for t in times]
    errors = [asarray(e, 'd') for e in errors]

    # A flat `(centers, errors)` pair of 1D float arrays iterates into scalars here, which
    # would silently be read as one planet per measurement. Catch it rather than fit it.
    # Note `any` is numpy's in this module, so it must not be given a generator.
    ragged = [a for a in times + errors if a.ndim != 1]
    if ragged:
        raise ValueError("Each element of ctdata must be a sequence with one array per "
                         "planet, so ctdata[0][i] holds planet i's centres. Wrap a "
                         "single planet's measurements in a list: ([centers], [errors]).")

    bad = [i for i, (t, e) in enumerate(zip(times, errors)) if t.size != e.size]
    if bad:
        raise ValueError(f"The centres and their uncertainties differ in length for "
                         f"planets {bad}.")

    return dict(times=times, errors=errors)


def _unpack_lightcurves(lcdata, nplanets: int) -> dict:
    """Expand a light curve group into the arguments :class:`pytransit.BaseLPF` expects.

    Parameters
    ----------
    lcdata : LCDataGroup, LCData, or None
        The photometry. A single light curve is wrapped into a group. ``None`` selects
        the no-photometry configuration used by RV/TTV-only fits.
    nplanets : int
        Number of planets in the system, used to check the planet ids.

    Returns
    -------
    dict
        The keys ``passbands``, ``times``, ``fluxes``, ``errors``, ``pbids``,
        ``covariates``, ``wnids``, ``nsamples`` and ``exptimes`` to forward to the base
        class, plus ``pids`` which the base class does not take.

    Raises
    ------
    TypeError
        If ``lcdata`` is neither a light curve, a light curve group, nor ``None``.
    ValueError
        If the group is empty, if any light curve fails to declare the planets
        contributing to it, or if a planet id is out of range.
    """
    if lcdata is None:
        return dict(_NO_PHOTOMETRY)

    if isinstance(lcdata, LCData):
        lcdata = LCDataGroup(lcdata)
    if not isinstance(lcdata, LCDataGroup):
        raise TypeError(f"lcdata must be a LCData or a LCDataGroup, got "
                        f"{type(lcdata).__name__}.")

    if lcdata.size == 0:
        raise ValueError("lcdata is empty. Pass lcdata=None to fit without photometry "
                         "(an RV/TTV-only fit).")

    missing = [i for i, p in enumerate(lcdata.pids) if p is None]
    if missing:
        raise ValueError(f"Every light curve must declare the planets contributing to it, "
                         f"but light curves {missing} have no pids.")

    if lcdata.n_planets > nplanets:
        raise ValueError(f"lcdata refers to planet {lcdata.n_planets - 1}, but the system has "
                         f"{nplanets} planets (valid ids are 0-{nplanets - 1}).")

    return dict(passbands=lcdata.passband_names,
                times=lcdata.times,
                fluxes=lcdata.fluxes,
                # `errors` silently falls back to a repeated point-to-point noise estimate
                # when a light curve carries no explicit uncertainties, so only pass it on
                # when every light curve really has them.
                errors=lcdata.errors if lcdata.has_errors else None,
                pbids=lcdata.pbids,
                # Always forwarded, including the (npt, 0) matrices of light curves with
                # no covariates: LSTSQBaseline needs a design matrix for every light
                # curve, and a zero-column one gives it an intercept-only baseline.
                covariates=lcdata.covariates,
                wnids=lcdata.wnids,
                nsamples=lcdata.nsamples,
                exptimes=lcdata.exptimes,
                pids=lcdata.pids)


class PhotoDynamicalLPF(BaseLPF):
    """Photodynamical log posterior function for a transiting multi-planet system.

    Subclass of :class:`pytransit.BaseLPF` that fits photometry, radial velocities,
    and prior transit-centre measurements simultaneously. Instead of an analytic
    transit model with free transit times, the observables are produced by N-body
    integrating the whole system with :class:`pyttv_photodyn.pdmodel.PhotoDynamicalModel`, so
    transit-timing variations arise self-consistently from the planet-planet
    interactions.

    The flat parameter vector is laid out in blocks: ``star`` (mstar, rstar),
    ``ldc`` (q1/q2 per passband), ``planets`` (eight parameters per planet),
    ``rv`` (the polynomial trend coefficients, then one systemic velocity per RV
    set), and the RV noise model's own block. The ordering is
    load-bearing: :meth:`transit_model` and :meth:`eccentricity_prior` index the
    vector positionally with strided slices.

    A typical workflow is to instantiate the class with the data, call
    :meth:`optimize_global` to find a good starting population via differential
    evolution, and then :meth:`sample_mcmc` to sample the posterior with emcee.
    """

    def __init__(self, name: str, nplanets: int,
                 zero_epochs: List, periods: List,
                 lcdata: LCDataGroup | LCData | None = None,
                 rvdata: RVDataGroup | RVData | None = None,
                 ctdata: Tuple | None = None,
                 is_transiting: List | None = None,
                 result_dir: Path = None,
                 tref: float = None, lnlikelihood: str = 'wn', use_lstsq_baseline: bool = True,
                 use_grazing_parameter: bool = False,
                 rv_slope_order: int = 0, rv_trend_priors: Optional[List] = None,
                 rv_tref: Optional[float] = None,
                 rv_lnlikelihood: Optional[RVLikelihood] = None):
        """Set up the photodynamical LPF with its data and configuration.

        Parameters
        ----------
        name : str
            Name of the fit, used by the base class for result bookkeeping.
        nplanets : int
            Number of planets in the system.
        lcdata : LCDataGroup or LCData, optional
            The photometry, as a :class:`pytransit.utils.io.LCDataGroup`. A
            single :class:`~pytransit.utils.io.LCData` is wrapped into a group
            automatically. The group supplies the passbands, times, fluxes,
            uncertainties, covariates, passband ids, white-noise group ids,
            supersampling counts and exposure times, and every light curve must declare
            through its ``pids`` which planets contribute to it. Pass ``None`` to fit
            without photometry (see Notes).
        rvdata : RVDataGroup or RVData, optional
            The radial velocities, as a :class:`pytransit.utils.io.RVDataGroup`. A single
            :class:`~pytransit.utils.io.RVData` is wrapped into a group automatically. The
            group supplies the times, velocities, uncertainties and instrument names, one
            entry per dataset. Pass ``None`` to fit without radial velocities.
        ctdata : tuple, optional
            Prior mid-transit-time measurements, as a ``(centers, errors)`` pair. Each
            element is a sequence with one entry per planet, holding that planet's
            measured centres and their uncertainties; a planet with no measurements gets
            an empty array. Pass ``None`` to fit without transit-centre measurements.
        zero_epochs : list
            Reference transit epoch (T0) for each planet, used to assign epoch
            numbers to the photometry.
        periods : list
            Orbital period for each planet, used to assign epoch numbers.
        is_transiting : list, optional
            Boolean per planet. Transiting planets are parametrised with ``t0`` and an
            impact/grazing parameter; non-transiting planets with a mean anomaly and
            inclination, and they are excluded from the photometric model.
        result_dir : pathlib.Path, optional
            Directory for storing results.
        tref : float
            Reference time about which the N-body system is integrated forward and
            backward.
        lnlikelihood : str, optional
            Name of the photometric likelihood model to use. Defaults to ``'wn'``
            (white noise).
        use_lstsq_baseline : bool, optional
            If ``True`` (the default), model each light curve's baseline with a
            :class:`pytransit.LSTSQBaseline`: an intercept plus one linear term per
            covariate, solved by least squares at every likelihood evaluation rather
            than sampled. Light curves without covariates get an intercept alone, i.e. a
            free normalisation. Set to ``False`` to fit the light curves as they are.
        use_grazing_parameter : bool, optional
            If ``True``, transiting planets use a grazing parameter (scaled by
            ``1 + k``) instead of the impact parameter.
        rv_slope_order : {0, 1, 2}, optional
            Polynomial order of the systemic RV trend. ``0`` (the default) fits no
            trend and emits no trend parameters. A value greater than ``0`` requires
            RV data (``rvdata`` not ``None``); otherwise the trend parameters would
            be declared but never applied, since ``transit_model`` only applies the
            RV trend when RV data is present.
        rv_trend_priors : list, optional
            One prior per trend coefficient, so ``len(rv_trend_priors)`` must equal
            ``rv_slope_order``. Defaults to ``[UP(-1, 1), UP(-0.1, 0.1)]`` truncated
            to the requested order; the units are m/s/d and m/s/d**2.
        rv_tref : float, optional
            Reference time of the RV trend, which is evaluated as
            ``sum(c_j * (t - rv_tref) ** (j + 1))``. Defaults to ``tref``. It is fixed at
            construction rather than derived from the RV epochs, so the trend and
            systemic-velocity parameters keep their meaning when RV data is added or
            removed.

        Notes
        -----
        When ``lcdata`` is ``None`` the fit runs without photometry (RV/TTV-only). No
        light-curve events are built, the photometric data arrays are left empty, and
        no photometric likelihood is registered, so the photometric contribution to the
        log likelihood is simply zero.
        """

        self.nplanets = nplanets
        self.is_transiting = array(is_transiting)
        self.use_grazing_parameter = use_grazing_parameter
        self.use_lstsq_baseline = use_lstsq_baseline
        self.with_gr = True

        # Photometry
        # ----------
        # Expand the light curve group into the arguments the base class takes. `pids`
        # is the one piece the base class knows nothing about, so it is set here, before
        # `super().__init__` runs `_init_data` and `_init_parameters`.
        lc = _unpack_lightcurves(lcdata, nplanets)
        self.lcdata = lcdata
        self.pids = lc['pids']
        self.with_photometry = lc['times'] is not None

        # Radial velocities
        # -----------------
        # The group's order fixes the set indices, so it is never sorted here; only the
        # concatenated arrays are, since the N-body model walks its events in time order.
        rv = _unpack_rvs(rvdata)
        self.rvdata = rvdata
        self.rv_times = rv['times']
        self.rv_values = rv['values']
        self.rv_errors = rv['errors']
        self.rv_instruments = rv['instruments']

        if self.rv_times is not None:
            self._orvtimes = squeeze(concatenate(self.rv_times))
            sids = argsort(self._orvtimes)
            self._orvtimes = self._orvtimes[sids]
            self._orvvalues = squeeze(concatenate(self.rv_values))[sids]
            self._orverrors = squeeze(concatenate(self.rv_errors))[sids]
            self._orvids = \
            squeeze(concatenate([full_like(rvt, i) for i, rvt in enumerate(self.rv_times)]).astype('int'))[sids]
            self.nrvsets = len(self.rv_times)
            self.nrvs = self._orvtimes.size
        else:
            self._orvtimes = None
            self._orvvalues = None
            self._orverrors = None
            self._orvids = None
            self.nrvsets = 0
            self.nrvs = 0

        self.rv_slope_order = rv_slope_order
        if rv_slope_order not in (0, 1, 2):
            raise ValueError('rv_slope_order must be 0, 1, or 2')
        if rv_slope_order > 0 and self.nrvsets == 0:
            raise ValueError('rv_slope_order > 0 requires RV data (rvdata is None): '
                             'the trend parameters would be declared but never applied, '
                             'since transit_model only applies the RV trend when nrvsets > 0.')
        if rv_trend_priors is None:
            rv_trend_priors = [UP(-1, 1), UP(-0.1, 0.1)][:rv_slope_order]
        elif len(rv_trend_priors) != rv_slope_order:
            raise ValueError('rv_trend_priors must have one prior per trend order')
        self.rv_trend_priors = list(rv_trend_priors)
        self.rv_lnl = rv_lnlikelihood if rv_lnlikelihood is not None else WNRVLikelihood()
        tref = float(tref)
        self.rv_tref = tref if rv_tref is None else float(rv_tref)

        # Transit centres
        # ---------------
        # The per-planet lists are concatenated and time-sorted, and the planet index is
        # carried alongside as `_center_planet_ids`, which becomes `tcipl` on the model.
        ct = _unpack_centers(ctdata, nplanets)
        self.ctdata = ctdata
        self.center_times = ct['times']
        self.center_time_errors = ct['errors']

        if self.center_times is not None:
            self._center_array = concatenate(self.center_times)
            self._center_planet_ids = concatenate([full(t.size, i)
                                                   for i, t in enumerate(self.center_times)])
            sids = argsort(self._center_array)
            self._center_array = self._center_array[sids]
            self._center_planet_ids = self._center_planet_ids[sids]
            self._center_error_array = concatenate(self.center_time_errors)[sids]
            self.ncenters = self._center_array.size
        else:
            self._center_array = None
            self._center_error_array = None
            self._center_planet_ids = None
            self.ncenters = 0

        super().__init__(name, lc['passbands'], lc['times'], lc['fluxes'], lc['errors'],
                         lc['pbids'], lc['covariates'], lc['wnids'], None,
                         lc['nsamples'], lc['exptimes'], True, result_dir, tref, lnlikelihood)

        self.zero_epochs = array(zero_epochs)
        self.periods = array(periods)

        self.epoch_ids = zeros((nplanets, self.timea.size), int)
        self.n_epochs = zeros(nplanets, int)
        for ipl in range(nplanets):
            if self.is_transiting[ipl]:
                self.epoch_ids[ipl] = epoch(self.timea, self.zero_epochs[ipl], periods[ipl])
                self.n_epochs[ipl] = where(unique(self.epoch_ids[ipl]) >= 0)[0].size

        self.tm = PhotoDynamicalModel(self.nplanets, self.is_transiting, self._tref,
                                      lctimes=self.timea, pids=self.pids, lcids=self.lcids, pbids=self.pbids,
                                      exptimes=self.exptimes, nsamples=self.nsamples,
                                      rvtimes=self._orvtimes,
                                      tcs=self._center_array, tcipl=self._center_planet_ids,
                                      with_gr=self.with_gr)

        self.ecc_priors = [UP(0., 1.) for i in range(self.nplanets)]

        if self.nrvsets > 0:
            self.rv_lnl.setup(self)

    def _init_data(self, times, fluxes, pbids=None, covariates=None, errors=None,
                   wnids=None, nsamples=1, exptimes=0.):
        """Ingest the photometry, or set up empty arrays for RV/TTV-only fits.

        When ``times`` is given this simply delegates to the base-class
        implementation. When ``times`` is ``None`` (no photometry) the photometric
        attributes the base class would normally create are populated as empty arrays,
        so the rest of the machinery works with zero light-curve points rather than a
        fabricated dummy point.
        """
        if times is not None:
            super()._init_data(times, fluxes, pbids, covariates, errors, wnids, nsamples, exptimes)
        else:
            self.nlc = 0
            self.times = self.fluxes = self.errors = []
            self.wn = []
            self.lcslices = []
            self.pbids = array([], int)
            self.lcids = array([], int)
            self.timea = self.ofluxa = self.mfluxa = self.errora = array([], float)
            self.nsamples = array([], int)
            self.exptimes = array([], float)
            self.noise_ids = array([], int)
            self.n_noise_blocks = 0

    def _init_parameters(self):
        """Define the fit parameters and their priors.

        Builds the :class:`pytransit.param.ParameterSet` in the fixed block order
        ``star`` -> ``ldc`` -> ``planets`` -> ``rv`` -> the RV noise plugin's own
        block (named ``rv_noise`` by default; see ``self.rv_lnl``). The first four
        blocks' slice/start are cached as ``_sl_*``/``_start_*`` attributes; the noise
        block's slice lives on the plugin itself, at ``self.rv_lnl.slice`` (and its
        start at ``self.rv_lnl.start``), not as a ``_sl_*``/``_start_*`` attribute on
        the LPF. The per-planet block
        holds eight parameters and switches between the transiting parametrisation
        (``t0`` and impact/grazing parameter) and the non-transiting one (mean anomaly
        ``M`` and inclination) according to ``is_transiting``. Called by the base-class
        constructor; the resulting ordering is relied upon throughout the class.
        """
        self.ps = ParameterSet()

        # Star
        # ----
        pst = [GParameter('mstar', 'stellar_mass', 'm_sun', UP(0.1, 20), [0.1, 20]),
               GParameter('rstar', 'stellar_radius', 'r_sun', UP(0.1, 3.0), [0, inf])]
        self.ps.add_global_block('star', pst)
        self._sl_star = self.ps.blocks[-1].slice
        self._start_star = self.ps.blocks[-1].start

        # Limb darkening
        # --------------
        pld = concatenate([
            [PParameter(f'q1_{pb}', 'q1 coefficient {pb}', '', UP(0, 1), bounds=(0, 1)),
             PParameter(f'q2_{pb}', 'q2 coefficient {pb}', '', UP(0, 1), bounds=(0, 1))]
            for i, pb in enumerate(self.passbands)])
        self.ps.add_passband_block('ldc', 2, self.npb, pld)
        self._sl_ld = self.ps.blocks[-1].slice
        self._start_ld = self.ps.blocks[-1].start

        # Planets
        # -------
        ppl = []

        if self.use_grazing_parameter:
            gorb = 'g'
            grazing_or_impact = 'grazing_parameter'
        else:
            gorb = 'b'
            grazing_or_impact = 'impact_parameter'

        for i in range(self.nplanets):
            ppl.append(
                GParameter(f'log10mplanet_{i:d}', f'log10_planetary_mass_{i:d}', 'm_sun', UP(-6.5, -2), [-10, 0]))
            ppl.append(GParameter(f'k_{i:d}', f'radius_ratio_{i:d}', 'd', UP(0.01, 0.2), [0, inf]))
            if self.is_transiting[i]:
                ppl.append(GParameter(f't0_{i:d}', f'zero_epoch_{i:d}', 'd', NP(1, 0.1), [-inf, inf]))
            else:
                ppl.append(GParameter(f'M_{i:d}', f'mean_anomaly_{i:d}', 'rad', UP(0, 2 * pi), [0, 2 * pi]))

            ppl.append(GParameter(f'p_{i:d}', f'period_{i:d}', 'd', NP(1, 0.1), [0, inf]))

            if self.is_transiting[i]:
                ppl.append(GParameter(f'{gorb}_{i:d}', f'{grazing_or_impact}_{i:d}', '', UP(-1, 1), [-inf, inf]))
            else:
                ppl.append(GParameter(f'inc_{i:d}', f'inclination_{i:d}', '', UP(0.0, pi), [0.0, pi]))

            ppl.extend([GParameter(f'secosw_{i:d}', f'sqrt_e_cosw{i:d}', '', UP(-1, 1), [-1, 1]),
                        GParameter(f'sesinw_{i:d}', f'sqrt_e_sinw_{i:d}', '', UP(-1, 1), [-1, 1]),
                        GParameter(f'omega_{i:d}', f'longitude_of_ascending_node_{i:d}', 'rad', UP(0.5 * pi, 1.5 * pi),
                                   [-inf, inf])])
        self.ps.add_global_block('planets', ppl)
        self._sl_planets = self.ps.blocks[-1].slice
        self._start_planets = self.ps.blocks[-1].start

        # RVs: polynomial trend and per-instrument systemic velocities
        # ------------------------------------------------------------
        prv = [GParameter('rv_trend' if i == 0 else f'rv_trend_{i + 1:d}',
                          'rv_trend' if i == 0 else f'rv_trend_{i + 1:d}', '',
                          self.rv_trend_priors[i], [-inf, inf])
               for i in range(self.rv_slope_order)]
        for i, label in enumerate(_rv_labels(self)):
            rvm, rvs = self.rv_values[i].mean(), self.rv_values[i].std()
            prv.append(GParameter(f'srv_{i:d}', f'systemic_rv_{label}', '', NP(rvm, rvs), [-inf, inf]))
        self.ps.add_global_block('rv', prv)
        self._sl_rvs = self.ps.blocks[-1].slice
        self._start_rvs = self.ps.blocks[-1].start
        self.rv_lnl.init_parameters(self, self.ps)
        self.ps.freeze()

    def _init_lnlikelihood(self):
        """Register the photometric likelihood model (white noise by default).

        Skipped for RV/TTV-only fits (no photometry): with no model registered the
        photometric contribution to the log likelihood is zero.
        """
        if self.with_photometry:
            self._add_lnlikelihood_model(WNLogLikelihood(self))

    def _init_baseline(self):
        """Set up the least-squares baseline model.

        Called by the base-class constructor after the data, parameters and likelihood
        models are in place, which is when the covariates and the light-curve slices
        :class:`~pytransit.LSTSQBaseline` needs exist. Sets ``self.lstsq_baseline`` to
        ``None`` when the baseline is disabled or there is no photometry to detrend.

        Notes
        -----
        The model is deliberately not registered with ``_add_baseline_model``. The
        models in that chain are called as ``blm(pv, bl)`` and evaluated *before* the
        transit model, but a least-squares baseline is fitted against the model flux and
        must run after it. :meth:`baseline` is therefore overridden to return the fitted
        baseline directly rather than walking that chain. :meth:`apply_baseline` is the
        cheap variant used on the likelihood path, where the model flux is already at
        hand and the baseline is consumed immediately.
        """
        self.lstsq_baseline = None
        if self.use_lstsq_baseline and self.with_photometry:
            self.lstsq_baseline = LSTSQBaseline(self)

    def apply_baseline(self, mflux):
        """Multiply a model flux by the baseline fitted against the observed flux.

        Parameters
        ----------
        mflux : numpy.ndarray
            Model flux over :attr:`timea`, either a 1D ``(npt,)`` array or a 2D
            ``(npv, npt)`` stack.

        Returns
        -------
        numpy.ndarray
            The model flux times its baseline, shaped like ``mflux``. Returned
            unchanged when the LPF has no baseline model.
        """
        if self.lstsq_baseline is None:
            return mflux
        return mflux * self.lstsq_baseline(mflux)

    def baseline(self, pv, mflux=None):
        """Return the fitted baseline on its own, without the transit model.

        Overrides the base-class implementation, which returns a bare ``1.`` here because
        the least-squares baseline is not part of the ``_baseline_models`` chain. This is
        the accessor to use for visualisation: dividing the observed flux by it gives the
        detrended light curves.

        Parameters
        ----------
        pv : numpy.ndarray
            Parameter vector, or a 2D stack of parameter vectors.
        mflux : numpy.ndarray, optional
            Model flux to fit the baseline against. Computed from ``pv`` if not given,
            which costs a full N-body integration; pass it when you already have it, for
            instance alongside a :meth:`flux_model` call.

        Returns
        -------
        numpy.ndarray
            The multiplicative baseline, ``(npt,)`` for a single parameter vector and
            ``(npv, npt)`` for a stack. All ones when the LPF has no baseline model.

        Notes
        -----
        The returned array is a copy, so it stays valid across further calls;
        :class:`~pytransit.LSTSQBaseline` itself hands back a reused internal buffer.

        Not to be confused with :meth:`CeleriteLogLikelihood.predict_baseline`, which
        predicts a Gaussian-process baseline and is a different quantity entirely.
        """
        if self.lstsq_baseline is None:
            if mflux is not None:
                return ones_like(mflux)
            return squeeze(ones((atleast_2d(pv).shape[0], self.timea.size)))
        if mflux is None:
            mflux, _, _ = self.transit_model(pv)
        return self.lstsq_baseline(mflux).copy()

    def flux_model(self, pv, mflux=None):
        """Full photometric model: the transit model times its baseline.

        Overrides the base-class implementation, which assumes :meth:`transit_model`
        returns just the flux rather than the ``(flux, rvs, centres)`` triple this class
        produces.

        Parameters
        ----------
        pv : numpy.ndarray
            Parameter vector, or a 2D stack of parameter vectors.
        mflux : numpy.ndarray, optional
            Model flux to use instead of evaluating :meth:`transit_model` again. Lets a
            caller that wants both the model and its baseline pay for a single N-body
            integration.

        Returns
        -------
        numpy.ndarray
            The photometric model over :attr:`timea`, baseline included.
        """
        if mflux is None:
            mflux, _, _ = self.transit_model(pv)
        return self.apply_baseline(mflux)

    def set_gp_hyperparameters(self, hps):
        """Set Gaussian-process noise hyperparameters (currently a no-op).

        Placeholder for swapping the white-noise likelihood for a celerite GP noise
        model. The implementation is stubbed out; the commented body shows the
        intended wiring.

        Parameters
        ----------
        hps : array_like
            GP hyperparameters.
        """
        pass
        # self._lnlikelihood_models = []
        # self._add_lnlikelihood_model(CeleriteLogLikelihood(self, array(hps)))

    def eccentricity_prior(self, pv):
        """Evaluate the per-planet eccentricity prior.

        The eccentricity of each planet is reconstructed from its ``secosw`` and
        ``sesinw`` parameters (``e = secosw**2 + sesinw**2``) and scored against the
        corresponding prior in ``self.ecc_priors``.

        Parameters
        ----------
        pv : numpy.ndarray
            Parameter vector, or a 2D stack of parameter vectors.

        Returns
        -------
        float or numpy.ndarray
            Summed log prior, scalar for a single vector or one value per vector.
        """
        pvp = atleast_2d(pv)
        pli, rvi = self._start_planets, self._start_rvs
        eccs = pvp[:, pli + 5:rvi:8] ** 2 + pvp[:, pli + 6:rvi:8] ** 2
        lnprior = zeros(pvp.shape[0])
        for e, ep in zip(eccs.T, self.ecc_priors):
            lnprior += ep.logpdf(e)
        return squeeze(lnprior)

    def lnposterior(self, pv):
        """Log posterior probability of a parameter vector.

        Returns the sum of the log prior and log likelihood, short-circuiting to
        ``-inf`` when the prior is not finite. This is the objective maximised by
        :meth:`optimize_global` and sampled by :meth:`sample_mcmc`.

        Parameters
        ----------
        pv : numpy.ndarray
            Parameter vector.

        Returns
        -------
        float
            Log posterior, or ``-inf`` for parameters outside the prior support.
        """
        lnprior = self.lnprior(pv)
        if not isfinite(lnprior):
            return -inf
        else:
            return lnprior + self.lnlikelihood(pv)

    def lnprior(self, pv: ndarray) -> Union[Iterable, float]:
        """Total log prior: the base-class priors plus the eccentricity prior.

        Parameters
        ----------
        pv : numpy.ndarray
            Parameter vector, or a 2D stack of parameter vectors.

        Returns
        -------
        float or numpy.ndarray
            Log prior probability.
        """
        return super().lnprior(pv) + self.eccentricity_prior(pv)

    def create_pv_population(self, npop=50):
        """Draw an initial parameter-vector population from the priors.

        Parameters
        ----------
        npop : int, optional
            Number of parameter vectors to draw.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(npop, npar)`` sampled from the parameter priors.
        """
        return self.ps.sample_from_prior(npop)

    def lnlikelihood(self, pvp):
        """Combined log likelihood of photometry, RVs, and transit centres.

        Runs :meth:`transit_model` to obtain the model flux, radial velocities, and
        transit centres, then adds the photometric likelihood (from the registered
        likelihood models), the RV likelihood (delegated to ``self.rv_lnl``), and the
        transit-centre likelihood, as applicable to the data provided.

        Parameters
        ----------
        pvp : numpy.ndarray
            A single parameter vector. A 2D population is not supported and raises
            :class:`NotImplementedError`.

        Returns
        -------
        float
            Total log likelihood, or ``-inf`` if it is not finite.
        """
        mflux, mrvs, mcenters = self.transit_model(pvp)

        if pvp.ndim == 1:
            lnl = 0.
        else:
            raise NotImplementedError

        fmodel = self.apply_baseline(mflux)
        for lnlm in self._lnlikelihood_models:
            lnl += lnlm(pvp, fmodel)

        if self.nrvsets > 0:
            lnl += self.rv_lnl(pvp, self._orvvalues - mrvs)

        if self.ncenters > 0:
            lnl += nan_lnlike_normal(self._center_array, mcenters, self._center_error_array)

        if isfinite(lnl):
            return lnl
        else:
            return -inf

    def lnlikelihood_separated(self, pvp):
        """Log likelihood split into its photometry, RV, and transit-centre parts.

        Same computation as :meth:`lnlikelihood` but returns the three contributions
        individually, which is useful for diagnosing which dataset drives the fit.
        Contributions for datasets that were not provided are returned as ``0``.

        Parameters
        ----------
        pvp : numpy.ndarray
            A single parameter vector. A 2D population raises
            :class:`NotImplementedError`.

        Returns
        -------
        tuple of float
            ``(lnla, lnlb, lnlc)`` for the photometric, RV, and transit-centre log
            likelihoods.
        """
        mflux, mrvs, mcenters = self.transit_model(pvp)

        if pvp.ndim == 1:
            lnlb = 0.
            lnlc = 0.
        else:
            raise NotImplementedError
            # lnlb = zeros(pvp.shape[0])
            # lnlc = zeros(pvp.shape[0])

        lnla = 0.
        if self.with_photometry:
            lnla = self._lnlikelihood_models[0](pvp, self.apply_baseline(mflux))
        if self.nrvsets > 0:
            lnlb = self.rv_lnl(pvp, self._orvvalues - mrvs)
        if self.ncenters > 0:
            lnlc = nan_lnlike_normal(self._center_array, mcenters, self._center_error_array)
        return lnla, lnlb, lnlc

    def transit_model(self, pvp, build_system_only=False, planets=None):
        """Map parameter vectors onto model flux, radial velocities, and centres.

        This is the bridge between the flat parameter vector and the N-body engine.
        For each parameter vector it unpacks the strided per-planet slices, converts
        the sampling parameters into the physical quantities the simulation needs
        (stellar density from mass and radius, planetary masses from their base-10
        logs, eccentricity and argument of periastron from ``secosw``/``sesinw``,
        inclination from the impact/grazing parameter), and calls
        ``self.tm`` to integrate the system. The systemic RV offset per instrument,
        and the polynomial trend are then added on top of the N-body radial
        velocities.

        Parameter vectors that fall outside the bounds, or that lead to an unstable
        configuration (a caught :class:`ValueError`/:class:`ZeroDivisionError`), yield
        ``inf`` model values so the likelihood rejects them.

        Parameters
        ----------
        pvp : numpy.ndarray
            A parameter vector or a 2D stack of them; promoted to 2D internally.
        build_system_only : bool, optional
            If ``True``, only build the rebound simulation (leaving it on ``self.tm``)
            without evaluating the observables. Used by the analysis helpers.
        planets : iterable of int, optional
            Subset of planet indices to include when building the system; defaults to
            all planets.

        Returns
        -------
        tuple of numpy.ndarray
            ``(fluxes, rvs, centers)``, squeezed. Each is empty/degenerate for data
            types that were not provided.
        """
        pvp = atleast_2d(pvp)
        npv = pvp.shape[0]

        ldcs = map_ldc(pvp[:, self._sl_ld])
        # The planet block is read with a stride of eight from wherever it starts. Its
        # start depends on the width of the limb-darkening block, i.e. on the number of
        # passbands, so it must not be hard-coded.
        pli, rvi = self._start_planets, self._start_rvs

        im = is_transiting_mask = self.is_transiting.astype(bool)

        fluxes = zeros((npv, self.timea.size))
        rvs = zeros((npv, self.nrvs))
        centers = zeros((npv, self.ncenters))
        for ipv in range(npv):
            pv = pvp[ipv]
            ldc = ldcs[ipv]
            if any(ldc > 1.0) or any(pv < self.ps.lbounds) or any(pv > self.ps.ubounds):
                fluxes[ipv] = inf
                rvs[ipv] = inf
                centers[ipv] = inf
            else:
                try:
                    mstar = pv[0]
                    rstar = pv[1]
                    rho = ((mstar * u.M_sun).to(u.g) / (4. / 3. * pi * (rstar * u.R_sun).to(u.cm) ** 3)).value
                    ldc = ldcs[ipv]
                    mp = 10 ** pv[pli:rvi:8]
                    k = pv[pli + 1:rvi:8]
                    t0 = pv[pli + 2:rvi:8] - self.is_transiting * self.tm.tref
                    p = pv[pli + 3:rvi:8]

                    b_or_g_or_inc = pv[pli + 4:rvi:8].copy()
                    if self.use_grazing_parameter:
                        b_or_g_or_inc[im] *= (1. + k)

                    e = pv[pli + 5:rvi:8] ** 2 + pv[pli + 6:rvi:8] ** 2
                    w = arctan2(pv[pli + 6:rvi:8], pv[pli + 5:rvi:8])
                    a = as_from_rhop(rho, p)

                    inc = b_or_g_or_inc.copy()
                    inc[im] = i_from_baew(b_or_g_or_inc[im], a[im], e[im], w[im])

                    omega = pv[pli + 7:rvi:8]
                    fluxes[ipv], rvs[ipv], centers[ipv] = self.tm(mstar, rstar, ldc, mp, k, t0, p, inc, e, w, omega,
                                                                  build_only=build_system_only, planets=planets)
                except (ZeroDivisionError, ValueError) as e:
                    fluxes[ipv], rvs[ipv], centers[ipv] = inf, inf, inf

            if self.nrvsets > 0:
                rv_pars = pv[self._sl_rvs]
                n = self.rv_slope_order
                rvs[ipv] += rv_pars[n:][self._orvids]
                if n:
                    rvt = self._orvtimes - self.rv_tref
                    for j in range(n):
                        rvs[ipv] += rv_pars[j] * rvt ** (j + 1)

        return squeeze(fluxes), squeeze(rvs), squeeze(centers)

    def predict_rvs(self, pv, times, pid=None):
        """Predict the dynamical (N-body) radial velocity signal at arbitrary times.

        Builds the Rebound simulation for the parameter vector, integrates a copy of it
        to each time stamp, and returns the stellar reflex velocity. Unlike
        :meth:`transit_model`, which evaluates the model only at the observed RV epochs,
        this accepts any times, which is what plotting a model curve needs.

        The systemic velocity offsets and the polynomial trend are **not** included: the
        return value is the bare dynamical signal, centred on zero. Add ``srv_i`` (and
        the trend, if fitted) yourself to compare against a given instrument's data.

        Parameters
        ----------
        pv : numpy.ndarray
            A single parameter vector.
        times : array_like
            Times at which to evaluate the signal. They need not be sorted; the result
            follows the order given here.
        pid : int, optional
            If given, split the signal into the RV induced by planet ``pid`` and the RV
            induced by all the other planets, by integrating two modified copies of the
            simulation: one containing only planet ``pid``, and one with planet ``pid``
            removed.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            Stellar reflex velocity in m/s, of the same shape as ``times``. When ``pid``
            is given, the pair ``(rv_planet, rv_others)``, whose sum equals the full
            signal only in the limit of non-interacting planets.
        """
        self.transit_model(pv, build_system_only=True)
        times = asarray(times)
        st = times - self.tm.tref
        sids = argsort(st)

        def integrate_rvs(base_sim):
            # Integrate backward from tref for the times before it and forward for the
            # rest, each from its own copy of the base simulation, mirroring how
            # PhotoDynamicalModel.__call__ walks its event list.
            rvs = zeros(times.size)
            ib = sids[st[sids] < 0.0]
            sim_backward = self.tm.copy_sim(base_sim)
            for i in ib[::-1]:
                integrate_to(sim_backward, st[i])
                rvs[i] = -sim_backward.particles[0].vz * RV_CONVERSION
            ifw = sids[st[sids] >= 0.0]
            sim_forward = self.tm.copy_sim(base_sim)
            for i in ifw:
                integrate_to(sim_forward, st[i])
                rvs[i] = -sim_forward.particles[0].vz * RV_CONVERSION
            return rvs

        if pid is None:
            return integrate_rvs(self.tm.sim)

        # The reduced systems are templates handed to `integrate_rvs`, which copies them
        # again through `copy_sim`, so they are plain copies and carry no forces here.
        sim_planet = self.tm.sim.copy()
        for ipl in range(self.nplanets):
            if ipl != pid:
                sim_planet.remove(hash=f'planet_{ipl + 1}')
        # The original centre-of-mass frame was computed with the full system, so the
        # reduced systems carry net momentum that would otherwise show up as a constant
        # drift in the stellar velocity.
        sim_planet.move_to_com()

        sim_others = self.tm.sim.copy()
        sim_others.remove(hash=f'planet_{pid + 1}')
        sim_others.move_to_com()

        return integrate_rvs(sim_planet), integrate_rvs(sim_others)

    def get_transit_times_within_range(self, pv, ipl, tstart, tend, planets=None):
        """Calculate the transit times of a planetary body within the specified time range.

        This function computes the times of transits for the specified planetary body (`ipl`)
        in a planetary system defined by input parameter `pv`. It calculates all transit
        centers that fall between the `tstart` and `tend` time range. The function utilizes
        the orbital period of the planetary body to estimate subsequent transits iteratively,
        adding the transit times to a list.

        Parameters
        ----------
        pv : array_like
            Parameter vector defining the planetary system.
        ipl : int
            Index of the planetary body for which transit times are computed.
        tstart : float
            Start time for calculating transit times.
        tend : float
            End time for calculating transit times.

        Returns
        -------
        numpy.ndarray
            Array of computed transit times within the specified time interval.
        """
        self.transit_model(pv, build_system_only=True, planets=planets)
        tref = self.tm.tref
        sim = self.tm.copy_sim()
        times = []
        tnext = find_first_transit_center(sim, tstart - tref, self.periods[ipl], ipl, 100)[0]
        while tnext < tend - tref:
            times.append(calculate_center_and_orbit(sim, tnext, ipl)[0])
            tnext = times[-1] + sim.particles[ipl + 1].P
        return array(times) + tref

    def get_transit_durations_within_range(self, pv, ipl, tstart, tend, planets=None,
                                           kind: Literal['analytical', 'numerical'] = 'analytical'):
        """Calculate the transit durations of a planet within the specified time range.

        This function computes the transit durations for the specified planet (`ipl`)
        in a planetary system defined by input parameter `pv`.

        Parameters
        ----------
        pv : array_like
            Parameter vector defining the planetary system.
        ipl : int
            Index of the planet for which transit times are computed.
        tstart : float
            Start time for calculating transit times.
        tend : float
            End time for calculating transit times.

        Returns
        -------
        numpy.ndarray
            Array of computed transit durations within the specified time interval.
        """
        self.transit_model(pv, build_system_only=True, planets=planets)
        tref = self.tm.tref
        sim = self.tm.copy_sim()
        k = pv[5 + 8 * ipl]
        tcs = []
        durations = []
        tnext = find_first_transit_center(sim, tstart - tref, self.periods[ipl], ipl, 100)[0]
        while tnext < tend - tref:
            tcnew, vajs = calculate_center_and_orbit(sim, tnext, ipl)
            tcs.append(tcnew + tref)
            if kind == 'numerical':
                durations.append(t14(k, vajs))
            else:
                o = sim.particles[ipl + 1].orbit(sim.particles[0])
                durations.append(d_from_pkaiews(o.P, k, o.a / sim.particles[0].r, o.inc, o.e, o.omega, 1))
            tnext = tcnew + sim.particles[ipl + 1].P
        return array(tcs), array(durations)

    def fold_times(self, pv: ndarray, ipl: int,
                   mflux: Optional[ndarray] = None) -> tuple[list[ndarray], list[ndarray]]:
        """Fold the times of observed signals for a specific planet.

        This method calculates folded times for the given planet's data by subtracting
        the calculated transit center from the observed times. The calculations involve
        simulating the transit model and aligning the times accordingly.

        Parameters
        ----------
        pv : ndarray
            The parameter vector for the transit model. Contains the parameters needed
            to simulate the transit system.
        ipl : int
            The planet identifier for which the times need to be folded. Corresponds to
            the index of the planet in the dataset.
        mflux : ndarray, optional
            Model flux to fit the baseline against, which the returned fluxes are then
            divided by. Computed here if not given. It should come from the *full* model:
            fitting the baseline against a model with a planet suppressed would absorb
            that planet's transits into the baseline. Ignored when the LPF has no
            baseline model.

        Returns
        -------
        list of ndarray
            A list where each element is an array of folded times for the specific planet
            in each lightcurve, and a list of the corresponding observed fluxes divided
            by their fitted baseline. Only lightcurves that contain data for the given
            planet identifier are considered.
        """
        # Either way the system has to be built for `pv` before `tm.sim` is taken. A full
        # evaluation leaves `tm.sim` in the same state a build-only one does, since the
        # model integrates copies of it, so when the flux is needed anyway one call serves
        # both purposes.
        if self.lstsq_baseline is not None and mflux is None:
            mflux, _, _ = self.transit_model(pv)
        else:
            self.transit_model(pv, build_system_only=True)
        baseline = None if self.lstsq_baseline is None else self.baseline(pv, mflux)

        sim = self.tm.copy_sim()
        folded_times, fluxes = [], []
        for ilc in range(self.nlc):
            if ipl in self.pids[ilc]:
                oflux = self.fluxes[ilc]
                if baseline is not None:
                    oflux = oflux / baseline[self.lcslices[ilc]]
                fluxes.append(oflux)
                folded_times.append(self.times[ilc] - (
                            calculate_center_and_orbit(sim, self.times[ilc].mean() - self.tm.tref, ipl)[
                                0] + self.tm.tref))
        return folded_times, fluxes

    def optimize_global(self, niter=200, npop=100, population=None, label='Global optimisation', leave=False,
                        plot_convergence: bool = True, use_tqdm: bool = True, pool=None, lnpost=None,
                        fbounds=(0.25, 1.15), return_cplot=False, tqdm_mininterval: float = 0.1):
        """Run global optimisation of the posterior with differential evolution.

        Wraps :class:`pytransit.utils.de.DiffEvol`, creating the optimiser (and its
        initial population, drawn from the priors unless ``population`` is given) on
        the first call and continuing from the current state on subsequent calls. The
        evolved population is intended as the starting point for :meth:`sample_mcmc`.

        Parameters
        ----------
        niter : int, optional
            Number of differential-evolution generations to run.
        npop : int, optional
            Population size, used when the optimiser is first created.
        population : numpy.ndarray, optional
            Explicit initial population; if omitted it is sampled from the priors.
        label : str, optional
            Progress-bar label.
        leave : bool, optional
            Whether to leave the progress bar after completion.
        plot_convergence : bool, optional
            If ``True``, draw a diagnostic figure of the posterior distribution and a
            few parameters against the log posterior.
        use_tqdm : bool, optional
            Toggle the progress bar.
        pool : optional
            Parallel map pool passed to the optimiser.
        lnpost : callable, optional
            Posterior function to maximise; defaults to :meth:`lnposterior`.
        fbounds : tuple of float, optional
            Differential-evolution scaling-factor bounds.
        return_cplot : bool, optional
            If ``True`` (and ``plot_convergence`` is set), return the diagnostic
            figure.
        tqdm_mininterval : float, optional
            Minimum time between progress-bar updates, in seconds. Raise it (to
            minutes) when the output goes to a log file rather than a terminal.
            Time-based throttling is used because tqdm's monitor thread silently
            resets an explicit ``miniters`` back to one whenever an update gap
            exceeds ``maxinterval``, which slow iterations always do.

        Returns
        -------
        matplotlib.figure.Figure or None
            The convergence figure when ``return_cplot`` and ``plot_convergence`` are
            both ``True``; otherwise ``None``.
        """

        if self.with_photometry and self._lnlikelihood_models == []:
            raise ValueError('Need to set the GP hyperparameters first')

        _restore_sigint_handler()
        lnpost = lnpost or self.lnposterior
        if self.de is None:
            self.de = DiffEvol(lnpost, clip(self.ps.bounds, -1, 1), npop, maximize=True, vectorize=False, pool=pool,
                               fbounds=fbounds)
            if population is None:
                self.de._population[:, :] = self.create_pv_population(npop)
            else:
                self.de._population[:, :] = population
        for _ in tqdm(self.de(niter), total=niter, desc=label, leave=leave, disable=(not use_tqdm),
                      mininterval=tqdm_mininterval):
            # Any REBOUND integration run in this process takes SIGINT back -- and
            # DiffEvol evaluates the whole initial population in the main process
            # before its first yield -- so the handler has to be re-asserted inside
            # the loop, not just before it, for Ctrl+C to keep working.
            _restore_sigint_handler()

        fig = None
        if plot_convergence:
            try:
                fig, axs = plt.subplots(1, 5, figsize=(13, 2), constrained_layout=True)
                rfit = self.de._fitness
                mfit = isfinite(rfit)

                if hasattr(self, '_old_de_fitness') and self._old_de_fitness is not None:
                    m = isfinite(self._old_de_fitness)
                    axs[0].hist(-self._old_de_fitness[m], facecolor='midnightblue', bins=25, alpha=0.25)
                axs[0].hist(-rfit[mfit], facecolor='midnightblue', bins=25)

                for i, ax in zip([4, 7, 12, 15], axs[1:]):
                    if hasattr(self, '_old_de_fitness') and self._old_de_fitness is not None:
                        m = isfinite(self._old_de_fitness)
                        ax.plot(self._old_de_population[m, i], -self._old_de_fitness[m], 'kx', alpha=0.25)
                    ax.plot(self.de.population[mfit, i], -rfit[mfit], 'k.')
                    ax.set_xlabel(self.ps.descriptions[i])
                plt.setp(axs, yticks=[])
                plt.setp(axs[1], ylabel='Log posterior')
                plt.setp(axs[0], xlabel='Log posterior')
                sb.despine(fig, offset=5)
            except IndexError:
                print(self.de._fitness)

        self._old_de_population = self.de.population.copy()
        self._old_de_fitness = self.de._fitness.copy()

        if plot_convergence and return_cplot:
            return fig

    def sample_mcmc(self, niter: int = 500, thin: int = 5, repeats: int = 1, population=None,
                    label='MCMC sampling', reset=True, leave=True, use_tqdm: bool = True, pool=None, lnpost=None,
                    tqdm_mininterval: float = 0.1):
        """Sample the posterior with the emcee affine-invariant ensemble sampler.

        Creates an :class:`emcee.EnsembleSampler` on the first call, seeding it from
        (in order of preference) an explicit ``population``, a stored local
        minimisation result, or the differential-evolution population from
        :meth:`optimize_global`. Subsequent calls continue from the last chain state.

        Parameters
        ----------
        niter : int, optional
            Number of iterations per repeat.
        thin : int, optional
            Thinning factor applied while sampling.
        repeats : int, optional
            Number of sampling runs to perform in sequence.
        population : numpy.ndarray, optional
            Explicit initial walker population.
        label : str, optional
            Progress-bar label.
        reset : bool, optional
            If ``True``, reset the sampler before the first run (later runs are always
            reset).
        leave : bool, optional
            Whether to leave the progress bar after completion.
        use_tqdm : bool, optional
            Toggle the progress bar.
        pool : optional
            Parallel map pool passed to the sampler.
        lnpost : callable, optional
            Posterior function to sample; defaults to :meth:`lnposterior`.
        tqdm_mininterval : float, optional
            Minimum time between progress-bar updates, in seconds. Raise it (to
            minutes) when the output goes to a log file rather than a terminal;
            see :meth:`optimize_global` for why the throttle is time-based.

        Raises
        ------
        ValueError
            If no initial population is available from any source.
        """

        if self.with_photometry and self._lnlikelihood_models == []:
            raise ValueError('Need to set the GP hyperparameters first')

        _restore_sigint_handler()
        lnpost = lnpost or self.lnposterior
        if self.sampler is None:
            if population is not None:
                pop0 = population
            elif hasattr(self, '_local_minimization') and self._local_minimization is not None:
                pop0 = multivariate_normal(self._local_minimization.x, diag(full(len(self.ps), 0.001 ** 2)),
                                           size=self.npop)
            elif self.de is not None:
                pop0 = self.de.population.copy()
            else:
                raise ValueError('Sample MCMC needs an initial population.')
            self.sampler = EnsembleSampler(pop0.shape[0], pop0.shape[1], lnpost, vectorize=False, pool=pool)
        else:
            pop0 = self.sampler.chain[:, -1, :].copy()

        for i in tqdm(range(repeats), desc='MCMC sampling', disable=(not use_tqdm)):
            if reset or i > 0:
                self.sampler.reset()
            for _ in tqdm(self.sampler.sample(pop0, iterations=niter, thin=thin), total=niter,
                          desc='Run {:d}/{:d}'.format(i + 1, repeats), leave=False, disable=(not use_tqdm),
                          mininterval=tqdm_mininterval):
                # Re-assert per iteration: a REBOUND integration run in this process
                # anywhere along the way takes SIGINT back (see optimize_global).
                _restore_sigint_handler()

            pop0 = self.sampler.chain[:, -1, :].copy()

    def plot_folded(self, pv, ipl, ax=None, figsize=None, bwidth=None, pmax=0.25, roffset=0.01):
        """Plot the phase-folded transit of one planet with the model and residuals.

        Folds the observed flux of light curves containing planet ``ipl`` onto the
        transit centre computed from ``pv``, divides out the contribution of the other
        planets, overplots the model transit, and shows the residuals offset below.

        Parameters
        ----------
        pv : pandas.Series or array_like
            Parameter vector with a ``.values`` attribute (e.g. a posterior summary
            row), used both to fold the data and to evaluate the model.
        ipl : int
            Index of the planet to fold on.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created if omitted.
        figsize : tuple, optional
            Figure size when creating a new figure.
        bwidth : float, optional
            If given, bin the folded data to this width (in days) before plotting.
        pmax : float, optional
            Half-width (in days) of the folded window to keep around the transit.
        roffset : float, optional
            Vertical offset applied to the residuals in the unbinned case.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure

        others = set(arange(self.nplanets))
        others.remove(ipl)

        pid = self.ps.find_pid(f'k_{ipl}')
        pids_o = [self.ps.find_pid(f'k_{i}') for i in others]

        pv1 = pv.values.copy()
        pv0 = pv.values.copy()

        pv0[pids_o] = 0.0001
        pv1[pid] = 0.0001

        mflux0 = self.transit_model(pv0)[0]
        mflux1 = self.transit_model(pv1)[0]
        # The baseline is fitted against the full model: using one of the
        # planet-suppressed variants above would absorb a real transit into it.
        mflux = self.transit_model(pv.values)[0] if self.lstsq_baseline is not None else None
        times, oflux = map(concatenate, self.fold_times(pv, ipl, mflux=mflux))

        fmflux0, fmflux1 = [], []
        for ilc in range(self.nlc):
            if ipl in self.pids[ilc]:
                sl = self.lcslices[ilc]
                fmflux0.append(mflux0[sl])
                fmflux1.append(mflux1[sl])
        fmflux0 = concatenate(fmflux0)
        fmflux1 = concatenate(fmflux1)

        m = abs(times) < pmax
        times, oflux, fmflux0, fmflux1 = map(lambda a: a[m], [times, oflux, fmflux0, fmflux1])
        sids = argsort(times)

        if bwidth is not None:
            pb, fb, eb = downsample_time_1d(times[sids], (oflux / fmflux1)[sids], bwidth)
            mpb, mfb, _ = downsample_time_1d(times[sids], fmflux0[sids], bwidth)
            if self.lstsq_baseline is None:
                # Stand-in normalisation for a fit without a baseline model; with one,
                # the fitted intercept has already normalised the flux.
                fb /= median(fb[abs(pb) > 0.1])
            res = fb - mfb

            ax.errorbar(pb, fb, eb, fmt='.', alpha=0.5)
            ax.plot(mpb, mfb, 'k')
            ax.errorbar(pb, res + 1 - 1.2 * (-(fb - 1).min() + res.max()), eb, c='C00', fmt='.', alpha=0.5)
        else:
            if self.lstsq_baseline is None:
                oflux = oflux / median(oflux[abs(times) > 0.08])
            res = oflux - fmflux0
            ax.plot(times[sids], (oflux / fmflux1)[sids], '.', alpha=0.5)
            ax.plot(times[sids], fmflux0[sids], 'k')
            ax.plot(times[sids], (res + 1 - roffset)[sids], '.', c='C00', alpha=0.5)

        plt.setp(ax, xlim=(-0.2, 0.2), xlabel='Time - T$_c$ [d]', ylabel='Normalized Flux')
        return fig

    def plot_ttvs_within_range(self, pvp, ipl: int, tstart: float, tend: float, figsize=None, planets=None, ax=None):
        """Plot the transit-timing variations of one planet over a time range.

        For each parameter vector in ``pvp`` the transit times of planet ``ipl``
        between ``tstart`` and ``tend`` are computed, a linear ephemeris is removed,
        and the residuals (in minutes) are summarised as a median curve with a
        16th-84th percentile band. Epochs that were actually observed are marked.

        Parameters
        ----------
        pvp : numpy.ndarray
            A parameter vector or a 2D stack of posterior samples; promoted to 2D.
        ipl : int
            Index of the planet whose TTVs are plotted.
        tstart : float
            Start of the time range.
        tend : float
            End of the time range.
        figsize : tuple, optional
            Figure size.
        planets : iterable of int, optional
            Subset of planets to include when building the system.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the TTV plot.
        """
        pvp = atleast_2d(pvp)
        nsamples = pvp.shape[0]
        times = []
        for pv in tqdm(pvp, total=nsamples):
            times.append(self.get_transit_times_within_range(pv, ipl, tstart, tend, planets=planets))

        if nsamples > 2:
            target = int(median([t.size for t in times]))
            times = array([t for t in times if t.size == target])

        all_epochs = epoch(times[0, :], self.zero_epochs[ipl], self.periods[ipl])

        observed_epochs = []
        for ilc in range(self.nlc):
            if ipl in self.pids[ilc]:
                observed_epochs.append(epoch(self.times[ilc].mean(), self.zero_epochs[ipl], self.periods[ipl]))
        observed_epochs = array(observed_epochs)

        was_observed = zeros(times.shape[1], dtype=bool)
        for i in range(times.shape[1]):
            if all_epochs[i] in observed_epochs:
                was_observed[i] = True

        ttvs = zeros((times.shape[0], times.shape[1]))
        for i in range(times.shape[0]):
            ttvs[i] = 24 * 60 * (times[i] - poly1d(polyfit(all_epochs, times[i], 1))(all_epochs))

        ttvp = percentile(ttvs, [50, 16, 84], axis=0)
        tref = floor(times.min())

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure
        ax.fill_between(times[0, :] - tref, ttvp[1], ttvp[2], alpha=0.2)
        ax.plot(times[0, :] - tref, ttvp[0])
        ax.plot(times[0, was_observed] - tref, ttvp[0][was_observed], 'k.')
        plt.setp(ax, xlabel=f'Time - {tref:.0f} [d]', ylabel="TTV [min]")
        return fig