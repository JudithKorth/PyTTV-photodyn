"""Photodynamical log posterior function for transiting exoplanet systems.

This module defines :class:`PhotoDynamicalLPF`, the user-facing fitting class. It
extends :class:`pytransit.BaseLPF` to model photometry, radial velocities, and
transit-timing data jointly by N-body integrating the whole planetary system (see
:mod:`pyttv_photodyn.pdmodel`) rather than treating each planet's transits independently.

The class ingests the data, defines the fit parameters and priors, maps the flat
parameter vector onto the physical quantities the N-body engine expects, scores the
resulting model against the data, and drives the optimisation and MCMC sampling.
"""
from pathlib import Path
from typing import List, Iterable, Union, Optional, Literal

import astropy.constants as cn
import astropy.units as u
import matplotlib.pyplot as plt
from emcee import EnsembleSampler
from meepmeep.backends.numba.utils import d_from_pkaiews
from numba import njit
from numpy import atleast_2d, zeros_like, sqrt, array, zeros, where, unique, concatenate, inf, pi, arctan2, squeeze, \
    clip, isfinite, diag, full, ndarray, argsort, full_like, log, sum, any, sin, poly1d, polyfit, percentile, \
    floor, median, arange
from numpy.random.mtrand import multivariate_normal

from pytransit.orbits.orbits_py import epoch
from pytransit import BaseLPF
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP, GParameter, PParameter
from pytransit.orbits.orbits_py import as_from_rhop, i_from_baew
from meepmeep.numba3d import t14
from pytransit.utils import downsample_time_1d
from pytransit.utils.de import DiffEvol
from tqdm.auto import tqdm
import seaborn as sb

from .wnloglikelihood import WNLogLikelihood
from .pdmodel import PhotoDynamicalModel, find_first_transit_center, calculate_center_and_orbit

c = (cn.c).to(u.AU / u.day).value  # Speed of light in AU / day
RV_CONVERSION = (u.AU / u.day).to(u.m / u.s)


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
def lnlike_normal(o, m, e):
    """Gaussian (white-noise) log likelihood.

    Parameters
    ----------
    o : numpy.ndarray
        Observed values.
    m : numpy.ndarray
        Model values, same shape as ``o``.
    e : numpy.ndarray
        Per-point uncertainties, same shape as ``o``.

    Returns
    -------
    float
        The summed log likelihood assuming independent normal errors.
    """
    return -sum(log(e)) - 0.5 * o.size * log(2. * pi) - 0.5 * sum((o - m) ** 2 / e ** 2)


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
    ``rv`` (trend, optional second-order trend, a sinusoid, then one systemic
    velocity per RV set), and ``rv_jitter`` (one per RV set). The ordering is
    load-bearing: :meth:`transit_model` and :meth:`eccentricity_prior` index the
    vector positionally with strided slices.

    A typical workflow is to instantiate the class with the data, call
    :meth:`optimize_global` to find a good starting population via differential
    evolution, and then :meth:`sample_mcmc` to sample the posterior with emcee.
    """

    def __init__(self, name: str, nplanets: int, passbands: List, zero_epochs: List, periods: List,
                 times: List = None, fluxes: Iterable = None, errors: List = None,
                 pids: Iterable[Iterable[int]] | None = None,
                 is_transiting: List | None= None,
                 rv_times: Optional[List] = None, rv_values: Optional[List] = None, rv_errors: Optional[List] = None,
                 center_times: Optional[List] = None, center_time_errors: Optional[List] = None,
                 pbids: List = None, covariates: List = None, wnids: List = None,
                 nsamples: tuple | int = 1, exptimes: tuple | float = 0., result_dir: Path = None,
                 tref: float = None, lnlikelihood: str = 'wn', use_grazing_parameter: bool = False,
                 rv_slope_order: int = 1):
        """Set up the photodynamical LPF with its data and configuration.

        Parameters
        ----------
        name : str
            Name of the fit, used by the base class for result bookkeeping.
        nplanets : int
            Number of planets in the system.
        passbands : list
            Photometric passband labels.
        zero_epochs : list
            Reference transit epoch (T0) for each planet, used to assign epoch
            numbers to the photometry.
        periods : list
            Orbital period for each planet, used to assign epoch numbers.
        times, fluxes, errors : list, optional
            Per-light-curve photometric time stamps, normalised fluxes, and
            uncertainties. If ``times`` is ``None`` the fit runs without photometry
            (see Notes).
        pids : iterable of iterable of int, optional
            For each light curve, the indices of the planets that contribute to it.
        is_transiting : list, optional
            Boolean per planet. Transiting planets are parametrised with ``t0`` and an
            impact/grazing parameter; non-transiting planets with a mean anomaly and
            inclination, and they are excluded from the photometric model.
        rv_times, rv_values, rv_errors : list, optional
            Radial-velocity data, optionally split into several instrument sets.
            A single set may be passed as plain arrays.
        center_times, center_time_errors : list, optional
            Prior mid-transit-time measurements and their uncertainties, given per
            planet.
        pbids, covariates, wnids : list, optional
            Passband ids, covariate matrices, and white-noise group ids per light
            curve, forwarded to the base class.
        nsamples, exptimes : tuple or scalar, optional
            Supersampling count and exposure time per light curve, for binning the
            model to finite integration times.
        result_dir : pathlib.Path, optional
            Directory for storing results.
        tref : float
            Reference time about which the N-body system is integrated forward and
            backward.
        lnlikelihood : str, optional
            Name of the photometric likelihood model to use. Defaults to ``'wn'``
            (white noise).
        use_grazing_parameter : bool, optional
            If ``True``, transiting planets use a grazing parameter (scaled by
            ``1 + k``) instead of the impact parameter.
        rv_slope_order : {1, 2}, optional
            Polynomial order of the systemic RV trend.

        Notes
        -----
        When ``times`` is ``None`` the fit runs without photometry (RV/TTV-only). No
        light-curve events are built, the photometric data arrays are left empty, and
        no photometric likelihood is registered, so the photometric contribution to the
        log likelihood is simply zero.
        """

        self.nplanets = nplanets
        self.is_transiting = array(is_transiting)
        self.use_grazing_parameter = use_grazing_parameter
        self.with_gr = True
        self.with_photometry = times is not None
        self.rv_slope_order = rv_slope_order
        if rv_slope_order not in (1, 2,):
            raise ValueError('rv_slope_order must be either 1 or 2')
        tref = float(tref)

        # RV setup
        # --------
        if rv_times is not None:
            if isinstance(rv_times, ndarray):
                self.rv_times = [rv_times.copy()]
                self.rv_values = [rv_values.copy()]
                self.rv_errors = [rv_errors.copy()]
            else:
                self.rv_times = [array(rvt) for rvt in rv_times]
                self.rv_values = [array(rvv) for rvv in rv_values]
                self.rv_errors = [array(rve) for rve in rv_errors]
            self._orvtimes = squeeze(concatenate(self.rv_times))
            sids = argsort(self._orvtimes)
            self._orvtmean = self._orvtimes.mean()
            self._orvtimes = self._orvtimes[sids]
            self._orvvalues = squeeze(concatenate(self.rv_values))[sids]
            self._orverrors = squeeze(concatenate(self.rv_errors))[sids]
            self._orvids = \
            squeeze(concatenate([full_like(rvt, i) for i, rvt in enumerate(self.rv_times)]).astype('int'))[sids]
            self.nrvsets = len(self.rv_times)
            self.nrvs = self._orvtimes.size
        else:
            self.rv_times = None
            self.rv_values = None
            self.rv_errors = None
            self._orvtimes = None
            self._orvtmean = None
            self._orvvalues = None
            self._orverrors = None
            self._orvids = None
            self.nrvsets = 0
            self.nrvs = 0

        if center_times is not None:
            self.center_times = center_times
            self.center_time_errors = center_time_errors

            self._center_array = concatenate(center_times)
            self._center_planet_ids = concatenate([full(len(center_times[i]), i) for i in range(len(center_times))])
            sids = argsort(self._center_array)
            self._center_array = self._center_array[sids]
            self._center_planet_ids = self._center_planet_ids[sids]
            self._center_error_array = concatenate(self.center_time_errors)[sids]
            self.ncenters = self._center_array.size
        else:
            self.center_times = None
            self.center_time_errors = None
            self._center_array = None
            self._center_error_array = None
            self._center_planet_ids = None
            self.ncenters = 0

        super().__init__(name, passbands, times, fluxes, errors, pbids, covariates, wnids, None,
                         nsamples, exptimes, True, result_dir, tref, lnlikelihood)
        self.pids = pids

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
        ``star`` -> ``ldc`` -> ``planets`` -> ``rv`` -> ``rv_jitter`` and caches each
        block's slice/start as ``_sl_*``/``_start_*`` attributes. The per-planet block
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

        # RVs
        # ---
        prv = []
        prv.append(GParameter(f'rv_trend', f'rv_trend', '', NP(0, 1e-15), [-inf, inf]))
        if self.rv_slope_order == 2:
            prv.append(GParameter('rv_trend_2', 'rv_trend_2', '', NP(0, 1e-15), [-inf, inf]))
        prv.append(GParameter(f'rv_sine_amplitude', f'rv_sine_amplitude', '', NP(1e-12, 1e-15), [0.0, inf]))
        prv.append(GParameter(f'rv_sine_period', f'rv_sine_period', '', NP(1, 1e-15), [0., inf]))
        prv.append(GParameter(f'rv_sine_phase', f'rv_sine_phase', '', NP(0, 1e-15), [-inf, inf]))
        for i in range(self.nrvsets):
            rvm, rvs = self.rv_values[i].mean(), self.rv_values[i].std()
            prv.append(GParameter(f'srv_{i:d}', f'systemic_rv_{i:d}', '', NP(rvm, rvs), [-inf, inf]))
        self._start_rvs = len(self.ps)
        self.ps.add_global_block('rv', prv)
        self._sl_rvs = self.ps.blocks[-1].slice
        self._start_rvs = self.ps.blocks[-1].start
        prv = []
        for i in range(self.nrvsets):
            prv.append(GParameter(f'log10rvj_{i:d}', f'log10_rv_jitter_{i:d}', '', NP(0, 0.5), [-inf, inf]))
        self._start_rvj = len(self.ps)
        self.ps.add_global_block('rv_jitter', prv)
        self._sl_rvj = self.ps.blocks[-1].slice
        self._start_rvj = self.ps.blocks[-1].start
        self.ps.freeze()

    def _init_lnlikelihood(self):
        """Register the photometric likelihood model (white noise by default).

        Skipped for RV/TTV-only fits (no photometry): with no model registered the
        photometric contribution to the log likelihood is zero.
        """
        if self.with_photometry:
            self._add_lnlikelihood_model(WNLogLikelihood(self))

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
        rvi = self._start_rvs
        eccs = pvp[:, 9:rvi:8] ** 2 + pvp[:, 10:rvi:8] ** 2
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
        likelihood models), the RV likelihood (including per-set jitter), and the
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

        for lnlm in self._lnlikelihood_models:
            lnl += lnlm(pvp, mflux)

        if self.rv_times is not None:
            rv_jitter = 10 ** pvp[self._sl_rvj][self._orvids]
            lnl += lnlike_normal(self._orvvalues, mrvs, sqrt(self._orverrors ** 2 + rv_jitter ** 2))

        if self.center_times is not None:
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
            lnla = self._lnlikelihood_models[0](pvp, mflux)
        if self.rv_times is not None:
            rv_jitter = 10 ** pvp[self._sl_rvj][self._orvids]
            lnlb = lnlike_normal(self._orvvalues, mrvs, sqrt(self._orverrors ** 2 + rv_jitter ** 2))
        if self.center_times is not None:
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
        the polynomial trend, and the sinusoid are then added on top of the N-body
        radial velocities.

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
        rvi = self._start_rvs

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
                    mp = 10 ** pv[4:rvi:8]
                    k = pv[5:rvi:8]
                    t0 = pv[6:rvi:8] - self.is_transiting * self.tm.tref
                    p = pv[7:rvi:8]

                    b_or_g_or_inc = pv[8:rvi:8].copy()
                    if self.use_grazing_parameter:
                        b_or_g_or_inc[im] *= (1. + k)

                    e = pv[9:rvi:8] ** 2 + pv[10:rvi:8] ** 2
                    w = arctan2(pv[10:rvi:8], pv[9:rvi:8])
                    a = as_from_rhop(rho, p)

                    inc = b_or_g_or_inc.copy()
                    inc[im] = i_from_baew(b_or_g_or_inc[im], a[im], e[im], w[im])

                    omega = pv[11:rvi:8]
                    fluxes[ipv], rvs[ipv], centers[ipv] = self.tm(mstar, rstar, ldc, mp, k, t0, p, inc, e, w, omega,
                                                                  build_only=build_system_only, planets=planets)
                except (ZeroDivisionError, ValueError) as e:
                    fluxes[ipv], rvs[ipv], centers[ipv] = inf, inf, inf

            if self.rv_times is not None:
                if self.nrvsets > 0:
                    rv_pars = pv[self._sl_rvs]
                    if self.rv_slope_order == 1:
                        slope, samp, sper, sphase = rv_pars[:4]
                        rv_shifts = rv_pars[4:]
                    elif self.rv_slope_order == 2:
                        slope, slope2, samp, sper, sphase = rv_pars[:5]
                        rv_shifts = rv_pars[5:]
                    else:
                        raise ValueError()

                    rvs += rv_shifts[self._orvids]
                    rvt = (self._orvtimes - self._orvtmean)
                    rvs += slope * rvt
                    if self.rv_slope_order == 2:
                        rvs += slope2 * rvt ** 2
                    rvs += samp * sin(2 * pi * (self._orvtimes) / sper - sphase)

        return squeeze(fluxes), squeeze(rvs), squeeze(centers)

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

    def fold_times(self, pv: ndarray, ipl: int) -> tuple[list[ndarray], list[ndarray]]:
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

        Returns
        -------
        list of ndarray
            A list where each element is an array of folded times for the specific planet
            in each lightcurve. Only lightcurves that contain data for the given planet
            identifier are considered.
        """
        self.transit_model(pv, build_system_only=True)
        sim = self.tm.copy_sim()
        folded_times, fluxes = [], []
        for ilc in range(self.nlc):
            if ipl in self.pids[ilc]:
                fluxes.append(self.fluxes[ilc])
                folded_times.append(self.times[ilc] - (
                            calculate_center_and_orbit(sim, self.times[ilc].mean() - self.tm.tref, ipl)[
                                0] + self.tm.tref))
        return folded_times, fluxes

    def optimize_global(self, niter=200, npop=100, population=None, label='Global optimisation', leave=False,
                        plot_convergence: bool = True, use_tqdm: bool = True, pool=None, lnpost=None,
                        fbounds=(0.25, 1.15), return_cplot=False):
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

        Returns
        -------
        matplotlib.figure.Figure or None
            The convergence figure when ``return_cplot`` and ``plot_convergence`` are
            both ``True``; otherwise ``None``.
        """

        if self.with_photometry and self._lnlikelihood_models == []:
            raise ValueError('Need to set the GP hyperparameters first')

        lnpost = lnpost or self.lnposterior
        if self.de is None:
            self.de = DiffEvol(lnpost, clip(self.ps.bounds, -1, 1), npop, maximize=True, vectorize=False, pool=pool,
                               fbounds=fbounds)
            if population is None:
                self.de._population[:, :] = self.create_pv_population(npop)
            else:
                self.de._population[:, :] = population
        for _ in tqdm(self.de(niter), total=niter, desc=label, leave=leave, disable=(not use_tqdm)):
            pass

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
                    label='MCMC sampling', reset=True, leave=True, use_tqdm: bool = True, pool=None, lnpost=None):
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

        Raises
        ------
        ValueError
            If no initial population is available from any source.
        """

        if self.with_photometry and self._lnlikelihood_models == []:
            raise ValueError('Need to set the GP hyperparameters first')

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
                          desc='Run {:d}/{:d}'.format(i + 1, repeats), leave=False, disable=(not use_tqdm)):
                pass

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
        times, oflux = map(concatenate, self.fold_times(pv, ipl))

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
            fb /= median(fb[abs(pb) > 0.1])
            res = fb - mfb

            ax.errorbar(pb, fb, eb, fmt='.', alpha=0.5)
            ax.plot(mpb, mfb, 'k')
            ax.errorbar(pb, res + 1 - 1.2 * (-(fb - 1).min() + res.max()), eb, c='C00', fmt='.', alpha=0.5)
        else:
            oflux /= median(oflux[abs(times) > 0.08])
            res = oflux - fmflux0
            ax.plot(times[sids], (oflux / fmflux1)[sids], '.', alpha=0.5)
            ax.plot(times[sids], fmflux0[sids], 'k')
            ax.plot(times[sids], (res + 1 - roffset)[sids], '.', c='C00', alpha=0.5)

        plt.setp(ax, xlim=(-0.2, 0.2), xlabel='Time - T$_c$ [d]', ylabel='Normalized Flux')
        return fig

    def plot_ttvs_within_range(self, pvp, ipl: int, tstart: float, tend: float, figsize=None, planets=None):
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
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.fill_between(times[0, :] - tref, ttvp[1], ttvp[2], alpha=0.2)
        ax.plot(times[0, :] - tref, ttvp[0])
        ax.plot(times[0, was_observed] - tref, ttvp[0][was_observed], 'k.')
        plt.setp(ax, xlabel=f'Time - {tref:.0f} [d]', ylabel="TTV [min]")
        return fig