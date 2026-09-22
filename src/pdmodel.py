"""N-body photodynamical engine for transiting exoplanet systems.

This module turns a set of physical system parameters into the observable signals
(transit light curves, radial velocities, and mid-transit times) by integrating the
whole planetary system with :mod:`rebound`, optionally including a general-relativity
correction via :mod:`reboundx`.

The design is event-based and reference-time centred. Each observable is an
:class:`Event` (a :class:`TransitLC`, :class:`RVPoint`, or :class:`TransitCenter`)
held in a time-sorted :class:`EventList`. :class:`PhotoDynamicalModel` builds the
simulation once and integrates outward from a reference epoch -- backward for events
before it and forward for those after it -- so each event is reached with the
shortest integration and the split around the reference time stays clean.

The numerically delicate part is locating mid-transit times: the helper functions at
the top of the module integrate a short stencil around a candidate time, fit a Taylor
expansion to the sky-projected star-planet separation, and minimise it.

Most of the wall time of a likelihood evaluation goes not into the observations but into
the months of empty sky between them, which the integrator still has to cross to reach the
next visit with the right orbital phases. Every integration therefore goes through
:func:`integrate_to`, which crosses those gaps with a cheap fixed-step integrator and
keeps IAS15 for the event-local work; see its docstring for the trade-off.
"""
from typing import List, Optional, Tuple, Iterable

import astropy.constants as cn
import astropy.units as u
import rebound
import reboundx
from numba import njit
from numba import TypingError
from numpy import zeros, array, asarray, atleast_1d, fmax, isnan, inf, ndarray, pi, argmin, ones_like, \
    zeros_like, floor, sqrt, full_like, nan, searchsorted, ones, unique, where, full, linspace
from meepmeep.numba3d import solve3d, solve3d_d, sep_c, t14, find_z_min
from meepmeep.backends.numba.utils import mean_anomaly_at_transit, eclipse_time_offset
from pytransit.models.numba.ma_quadratic_nb import eval_quad_z_s
from pytransit.models.numba.ma_uniform_nb import uniform_z_s
from pytransit.orbits.orbits_py import mean_anomaly_offset
from rebound import Simulation


c = (cn.c).to(u.AU/u.day).value  # Speed of light in AU / day
RV_CONVERSION = (u.AU/u.day).to(u.m/u.s)

rs2au = u.R_sun.to(u.AU)
me2ms = u.M_earth.to(u.M_sun)


def set_integration_policy(sim: Simulation, gap_threshold: Optional[float], gap_dt: float):
    """Record on `sim` how :func:`integrate_to` should cross long empty stretches.

    A `gap_threshold` of ``None`` disables the policy, leaving every step on IAS15.
    """
    sim._gap_threshold = gap_threshold
    sim._gap_dt = gap_dt


def integrate_to(sim: Simulation, t: float):
    """Integrate `sim` to time `t`, using a cheap integrator for long empty jumps.

    Photometric campaigns are clustered into a handful of visits separated by months of
    empty sky, and the model has to propagate the system across those gaps to reach the
    next visit with the right orbital phases. IAS15 keeps resolving every inner orbit at
    full tolerance while it does so, which is where most of a likelihood evaluation goes.
    Nothing is measured inside a gap, so it is crossed with fixed-step WHFast instead and
    IAS15 is restored for the event-local work, where transit times are actually extracted.

    Steps at or below the threshold are handed straight to `sim.integrate`, so the
    behaviour near the data is bit-for-bit what it was without a policy. Simulations with
    no policy attached (a bare `rebound.Simulation`) are integrated the same way.
    """
    threshold = getattr(sim, '_gap_threshold', None)
    if threshold is None or abs(t - sim.t) <= threshold:
        sim.integrate(t)
        return

    integrator, dt = sim.integrator, sim.dt
    sim.integrator = 'whfast'
    sim.dt = sim._gap_dt if t > sim.t else -sim._gap_dt
    try:
        sim.integrate(t)
    finally:
        sim.integrator, sim.dt = integrator, dt


def expand_orbit(sim: Simulation, pid: int, te: float = 0.0):
    """Expand the orbit of a single particle into a 3D Taylor series using MeepMeep.

    Creates a single-point 3D Taylor series expansion of the osculating orbit of
    particle `pid` around particle 0 (the star) using MeepMeep's numba backend.

    Parameters
    ----------
    sim : Simulation
        Rebound simulation containing the system. The osculating orbital elements
        are taken at the current simulation time, so integrate the simulation close
        to the event of interest before calling.
    pid : int
        Index of the particle to expand in the simulation's particle list.
    te : float, optional
        Expansion point offset from the transit centre [days]. The default 0.0
        expands at the transit centre; pass the transit-to-eclipse offset to
        expand at the secondary eclipse.

    Returns
    -------
    tc : float
        Osculating transit centre time in the simulation time frame.
    p : float
        Osculating orbital period [days].
    coeffs : ndarray
        MeepMeep (3, 5) Taylor coefficient matrix with the expansion point at
        ``tc + te``. Rows are (x, y, z) in stellar radii with z the line of sight
        (positive toward the observer, transit at z > 0); columns are the
        factorial-prescaled series terms. The coordinates match rebound's
        star-relative frame: ``lan = Omega + pi`` maps MeepMeep's sky frame onto
        rebound's.
    """
    pid = int(pid)  # rebound rejects numpy integers as particle indices
    o = sim.particles[pid].orbit(sim.particles[0])
    tc = o.T + mean_anomaly_at_transit(o.e, o.omega) / (2.0 * pi) * o.P
    coeffs = solve3d(te, o.P, o.a / sim.particles[0].r, o.inc, o.e, o.omega, o.Omega + pi)
    return tc, o.P, coeffs


def expand_orbit_d(sim: Simulation, pid: int, te: float = 0.0):
    """Expand the orbit of a single particle into a 3D Taylor series with derivatives.

    Identical to `expand_orbit`, but also returns the analytic derivatives of the
    Taylor coefficients with respect to the orbital parameters. Not used by
    `PhotoDynamicalModel`; available for future gradient-based fitting.

    Returns
    -------
    tc, p, coeffs
        As in `expand_orbit`.
    dcoeffs : ndarray
        A (7, 3, 5) derivative tensor in the ``(tc, p, a, i, e, w, lan)`` basis.
        The `lan` slot equals d/dOmega since ``lan = Omega + pi`` is a constant
        shift.
    """
    pid = int(pid)  # rebound rejects numpy integers as particle indices
    o = sim.particles[pid].orbit(sim.particles[0])
    tc = o.T + mean_anomaly_at_transit(o.e, o.omega) / (2.0 * pi) * o.P
    coeffs, dcoeffs = solve3d_d(te, o.P, o.a / sim.particles[0].r, o.inc, o.e, o.omega, o.Omega + pi)
    return tc, o.P, coeffs, dcoeffs


def find_separation_minimum(coeffs, max_iter=10):
    """Find the projected-separation minimum near the expansion point.

    Wraps MeepMeep's `find_z_min`, whose golden-section search brackets only
    +-0.01 d around its guess, re-centring the bracket until the minimum lies
    in its interior so that large conjunction-to-minimum offsets (very
    eccentric orbits) are not clipped. Returns the time of minimum projected
    separation as an offset from the expansion point.
    """
    guess = 0.0
    for _ in range(max_iter):
        t_min = find_z_min(guess, coeffs)[0]
        if abs(t_min - guess) < 0.0095:
            break
        guess = t_min
    return t_min


def calculate_center_and_orbit(sim, tc, ipl, max_iter=10, w=1.0):
    """Calculate the light-travel-corrected transit centre and orbit expansion.

    Integrates the simulation to the transit nearest to the guess `tc`, iterates
    the osculating transit centre until convergence, and nudges the result to the
    time of minimum projected separation. Returns the light-travel-corrected
    transit centre and the MeepMeep (3, 5) Taylor coefficient matrix expanded at
    the transit centre. The light-travel correction uses the planet's barycentric
    z: the transit modulation is imprinted where the stellar photons are blocked,
    at the planet's plane, which also makes the modelled transit-to-eclipse
    interval carry the full 2a/c Roemer delay. The `w` argument is unused and
    kept for backwards compatibility.
    """
    ipl = int(ipl)  # rebound rejects numpy integers as particle indices
    integrate_to(sim, tc)
    tcb, p, coeffs = expand_orbit(sim, ipl + 1)
    tcb += round((tc - tcb) / p) * p
    tca = inf
    i = 0
    while abs(tca - tcb) > 1e-8:
        if i > max_iter:
            tcb = nan
            break
        tca = tcb
        integrate_to(sim, tca)
        tcb, p, coeffs = expand_orbit(sim, ipl + 1)
        tcb += round((tca - tcb) / p) * p
        i += 1
    if not isnan(tcb):
        # Nudge tc from the osculating conjunction to the minimum projected
        # separation and re-expand there so that the expansion point coincides
        # with the reported transit centre.
        dt_min = find_separation_minimum(coeffs)
        coeffs = expand_orbit(sim, ipl + 1, te=dt_min)[2]
        tcb += dt_min
    return tcb - sim.particles[ipl + 1].z / c, coeffs


def calculate_eclipse_center_and_orbit(sim, tc, ipl, max_iter=10):
    """Calculate the light-travel-corrected secondary eclipse centre and orbit expansion.

    Integrates the simulation to the eclipse nearest to the guess `tc`, iterates
    the osculating eclipse centre until convergence, and nudges the result to the
    time of minimum projected separation. Returns the light-travel-corrected
    eclipse centre and the MeepMeep (3, 5) Taylor coefficient matrix expanded at
    the eclipse centre. The light-travel correction uses the planet's barycentric
    z since the light occulted in a secondary eclipse is emitted by the planet.
    """
    ipl = int(ipl)  # rebound rejects numpy integers as particle indices
    integrate_to(sim, tc)
    o = sim.particles[ipl + 1].orbit(sim.particles[0])
    et = eclipse_time_offset(o.P, o.inc, o.e, o.omega)
    tcb, p, coeffs = expand_orbit(sim, ipl + 1, te=et)
    ecb = tcb + et
    ecb += round((tc - ecb) / p) * p
    eca = inf
    i = 0
    while abs(eca - ecb) > 1e-8:
        if i > max_iter:
            ecb = nan
            break
        eca = ecb
        integrate_to(sim, eca)
        o = sim.particles[ipl + 1].orbit(sim.particles[0])
        et = eclipse_time_offset(o.P, o.inc, o.e, o.omega)
        tcb, p, coeffs = expand_orbit(sim, ipl + 1, te=et)
        ecb = tcb + et
        ecb += round((eca - ecb) / p) * p
        i += 1
    if not isnan(ecb):
        # Nudge the centre from the osculating conjunction to the minimum
        # projected separation and re-expand there so that the expansion point
        # coincides with the reported eclipse centre.
        dt_min = find_separation_minimum(coeffs)
        coeffs = expand_orbit(sim, ipl + 1, te=et + dt_min)[2]
        ecb += dt_min
    return ecb - sim.particles[ipl + 1].z / c, coeffs


def find_first_transit_center(sim: rebound.Simulation, tstart: float, period: float, ipl: int, nt: int = 100):
    """Find the center time and orbital parameters for the first transit occurring after tstart.

    This function integrates a simulation over a given time interval, calculates the star-planet
    distances at each integration step, and determines the time of the closest approach. It then
    calculates the transit center time and orbital parameters for the specified celestial body.

    Parameters
    ----------
    sim : object
        The simulation instance used for orbital integration.
    tstart : float
        The starting time of the integration.
    period : float
        The orbital period of the celestial body.
    ipl : int
        The index of the celestial body in the simulation's particle list.
    nt : int, optional
        The number of time steps to sample within the integration range. Default is 100.

    Returns
    -------
    tc : float
        The time of the first transit center.
    coeffs : ndarray
        MeepMeep (3, 5) Taylor coefficient matrix expanded at the transit center.
    """
    zs = full(nt, inf)
    times = linspace(tstart, tstart+period, nt)
    for i, t in enumerate(times):
        integrate_to(sim, t)
        if sim.particles[ipl+1].z - sim.particles[0].z > 0.0:
            zs[i] = sqrt((sim.particles[ipl+1].x - sim.particles[0].x)**2 + (sim.particles[ipl+1].y - sim.particles[0].y)**2)
    t0 = times[argmin(zs)]
    tc, coeffs = calculate_center_and_orbit(sim, t0, ipl)
    return tc, coeffs


@njit(fastmath=False)
def quadratic_model_s(t, k, ldc, ipb, nsamples, exptimes, npb, tc, coeff, flux):
    """Add one planet's quadratic-limb-darkening transit to a flux array.

    For each time within a window around the transit centre ``tc``, evaluates the
    quadratic limb-darkening transit model (via pytransit's ``eval_quad_z_s``) using
    the Taylor-series projected separation from :func:`z_taylor_st`, optionally
    supersampling over the exposure time, and accumulates the resulting dip into
    ``flux``. Points far from the transit, or a ``nan`` radius ratio or limb-darkening
    coefficient, are skipped (leaving the flux unchanged).

    Parameters
    ----------
    t : numpy.ndarray
        Observation times.
    k : float
        Planet-to-star radius ratio.
    ldc : numpy.ndarray
        Quadratic limb-darkening coefficients (two per passband).
    ipb : int
        Passband index selecting the coefficient pair from ``ldc``.
    nsamples : int
        Number of supersampling points per exposure.
    exptimes : float
        Exposure time over which to supersample.
    npb : int
        Number of passbands represented in ``ldc`` (for validation).
    tc : float
        Transit-centre time.
    coeff : numpy.ndarray
        The 10-element Taylor state vector at the transit centre.
    flux : numpy.ndarray
        Flux array to accumulate into, modified and returned.

    Returns
    -------
    numpy.ndarray
        The ``flux`` array with this planet's transit added.
    """
    ldc = atleast_1d(ldc)
    if ldc.size != 2*npb:
        raise ValueError("The quadratic model needs two limb darkening coefficients per passband")

    half_window_width = 0.025 + 0.5 * t14(k, coeff)
    npt = t.size
    ld = ldc[2*ipb:2*(ipb + 1)]
    if isnan(k) or isnan(ld[0]) or isnan(ld[1]):
        return flux

    for j in range(npt):
        tt = t[j] - tc
        if abs(tt) < 0.2+half_window_width:
            for isample in range(1, nsamples + 1):
                time_offset = exptimes*((isample - 0.5)/nsamples - 0.5)
                z = sep_c(tt + time_offset, coeff)
                if z < 1.0 + k:
                    flux[j] += (eval_quad_z_s(z, k, ld) - 1.0) / nsamples
    return flux


@njit(fastmath=False)
def eclipse_model_s(t, k, fr, nsamples, exptimes, tc, coeff, flux):
    """Secondary eclipse model for a planet with a uniform disk.

    Adds the eclipse of a uniform-disk planet with a planet-star surface
    brightness ratio `fr` to `flux`. The eclipse depth is fr*k^2 and the
    out-of-eclipse continuum is 1.0 (the planet flux is not added to the
    continuum). The coefficient matrix `coeff` must be expanded at the
    eclipse centre `tc`.
    """
    if isnan(k) or isnan(fr):
        return flux

    half_window_width = 0.025 + 0.5 * t14(k, coeff)
    npt = t.size
    for j in range(npt):
        tt = t[j] - tc
        if abs(tt) < 0.2 + half_window_width:
            for isample in range(1, nsamples + 1):
                time_offset = exptimes * ((isample - 0.5) / nsamples - 0.5)
                z = sep_c(tt + time_offset, coeff)
                if z < 1.0 + k:
                    flux[j] += (uniform_z_s(z, k, 1.0) - 1.0) * fr / nsamples
    return flux


class Event:
    """A timed observable that knows how to evaluate itself against a simulation.

    Base class for the photodynamical observables. An event has a ``center`` time (in
    absolute units) and is reached by integrating a simulation to ``center - tref``,
    where ``tref`` is the model's reference epoch. Events are ordered by their centre
    time so they can be held in a time-sorted :class:`EventList`. Subclasses
    (:class:`TransitLC`, :class:`RVPoint`, :class:`TransitCenter`) override
    :meth:`compute` to extract the relevant observable.
    """

    def __init__(self, center: Optional[float] = None, bbox: Optional[Tuple[float, float]] = None, tref: Optional[float] = 0.0):
        """Create an event.

        Parameters
        ----------
        center : float, optional
            Absolute centre time of the event. If omitted, it is taken as the midpoint
            of ``bbox``.
        bbox : tuple of float, optional
            ``(start, end)`` time bounds of the event (used by light-curve segments).
        tref : float, optional
            Reference epoch subtracted before integrating the simulation.
        """
        self.bbox: Tuple[float, float] = bbox
        self.center: float = center if center is not None else 0.5*(bbox[0] + bbox[1])
        self.dirty: bool = True
        self.tref: float = tref
        self.result = None

    def compute(self, sim):
        """Advance the simulation to this event's centre time.

        Parameters
        ----------
        sim : rebound.Simulation
            Simulation to integrate to ``center - tref``; advanced in place.
        """
        integrate_to(sim, self.center - self.tref)

    def __eq__(self, other):
        """Two events are equal when their centre times match."""
        return self.center == other.center

    def __lt__(self, other):
        """Order by centre time; ``other`` may be another event or a bare float."""
        if isinstance(other, float):
            return self.center < other
        else:
            return self.center < other.center

    def __gt__(self, other):
        """Order by centre time; ``other`` may be another event or a bare float."""
        if isinstance(other, float):
            return self.center > other
        else:
            return self.center > other.center


class TransitLC(Event):
    """A light-curve segment covering one or more transits.

    Holds the time stamps and model-flux buffer for a contiguous light curve, the
    planets that contribute to it, and its passband and supersampling settings. On
    :meth:`compute` it fills the flux buffer with the combined transit signal of all
    its planets and records the fitted transit centres and Taylor coefficients.
    """

    def __init__(self, time: ndarray, tref: float, pids: Iterable[int], pbid: int, nsamples: int, exptime: float, model: ndarray):
        """Create a light-curve event.

        Parameters
        ----------
        time : numpy.ndarray
            Observation times of this light curve; its endpoints set the event bbox.
        tref : float
            Reference epoch.
        pids : iterable of int
            Indices of the planets contributing to this light curve.
        pbid : int
            Passband index.
        nsamples : int
            Supersampling count per exposure.
        exptime : float
            Exposure time.
        model : numpy.ndarray
            View into the shared flux model array that this event writes to.
        """
        super().__init__(None, time[[0, -1]], tref)
        self.time: ndarray = time
        self.flux: ndarray = model
        self.pids: Iterable[int] = pids
        self.pbid: int = pbid
        self.nsamples: int = nsamples
        self.exptime: float = exptime
        self.fitted_centers = []
        self.fitted_coeffs = []

    def __repr__(self):
        return f"LC({self.center:.3f}, pids={self.pids}, pbid={self.pbid})"

    def compute(self, sim, k, ldc, **kwargs):
        """Fill the flux buffer with the combined transit model of all planets.

        Resets the flux to unity, then for each contributing planet locates the
        transit centre with :func:`calculate_center_and_orbit` and adds its
        quadratic-limb-darkening transit with :func:`quadratic_model_s`, storing the
        fitted centres and Taylor coefficients on the event.

        Parameters
        ----------
        sim : rebound.Simulation
            Simulation integrated to this event's centre; advanced in place.
        k : array_like
            Radius ratio per planet.
        ldc : numpy.ndarray
            Quadratic limb-darkening coefficients.
        """
        super().compute(sim)
        ldc = atleast_1d(ldc)
        npb = ldc.size // 2
        self.flux[:] = 1.0
        self.fitted_centers=  []
        self.fitted_coeffs = []
        for ipl in self.pids:
            tc, tsv = calculate_center_and_orbit(sim, self.center - self.tref, ipl)
            self.flux = quadratic_model_s(self.time, k[ipl], ldc, self.pbid, self.nsamples, self.exptime, npb, tc + self.tref, tsv, self.flux)
            self.fitted_centers.append(tc + self.tref)
            self.fitted_coeffs.append(tsv)


class EclipseLC(Event):
    def __init__(self, time: ndarray, tref: float, pids: Iterable[int], pbid: int, nsamples: int, exptime: float, model: ndarray):
        super().__init__(None, time[[0, -1]], tref)
        self.time: ndarray = time
        self.flux: ndarray = model
        self.pids: Iterable[int] = pids
        self.pbid: int = pbid
        self.nsamples: int = nsamples
        self.exptime: float = exptime
        self.fitted_centers = []
        self.fitted_coeffs = []

    def __repr__(self):
        return f"ELC({self.center:.3f}, pids={self.pids}, pbid={self.pbid})"

    def compute(self, sim, k, fr=None, **kwargs):
        if fr is None:
            raise ValueError("Planet-star flux ratios (fr) are required to model eclipse light curves.")
        super().compute(sim)
        self.flux[:] = 1.0
        self.fitted_centers = []
        self.fitted_coeffs = []
        for ipl in self.pids:
            tc, tsv = calculate_eclipse_center_and_orbit(sim, self.center - self.tref, ipl)
            self.flux = eclipse_model_s(self.time, k[ipl], fr[ipl, self.pbid], self.nsamples, self.exptime, tc + self.tref, tsv, self.flux)
            self.fitted_centers.append(tc + self.tref)
            self.fitted_coeffs.append(tsv)


class RVPoint(Event):
    """A single radial-velocity measurement.

    On :meth:`compute` it records the star's line-of-sight velocity (converted to
    m/s) at the measurement time into its result buffer.
    """

    def __init__(self, center: float, tref: float, result: ndarray):
        """Create a radial-velocity event.

        Parameters
        ----------
        center : float
            Measurement time.
        tref : float
            Reference epoch.
        result : numpy.ndarray
            Length-1 view into the shared RV model array that this event writes to.
        """
        super().__init__(center, None, tref)
        self.result = result

    def __repr__(self):
        return f"RV({self.center:.3f})"

    def compute(self, sim, **kwargs):
        """Record the star's radial velocity (m/s) at the measurement time.

        Parameters
        ----------
        sim : rebound.Simulation
            Simulation integrated to this event's centre; advanced in place.
        **kwargs
            Ignored; present for a uniform event interface.
        """
        super().compute(sim)
        self.result[:] = -sim.particles[0].vz*RV_CONVERSION
        self.dirty = False


class TransitCenter(Event):
    """A modelled mid-transit time for one planet.

    On :meth:`compute` it refines the transit centre of planet ``ipl`` near its
    nominal time and records the absolute centre time into its result buffer, for
    comparison against an observed transit-timing measurement.
    """

    def __init__(self, center: float, tref: float, ipl: int, result: ndarray):
        """Create a transit-centre event.

        Parameters
        ----------
        center : float
            Nominal (observed) transit-centre time, used as the refinement guess.
        tref : float
            Reference epoch.
        ipl : int
            Planet index.
        result : numpy.ndarray
            Length-1 view into the shared transit-centre model array.
        """
        super().__init__(center, None, tref)
        self.ipl: int = ipl
        self.result = result
        self.window_width = 1.0

    def __repr__(self):
        return f"TC({self.center:.3f}, ipl={self.ipl})"

    def compute(self, sim, **kwargs):
        """Record the refined absolute mid-transit time of planet ``ipl``.

        Parameters
        ----------
        sim : rebound.Simulation
            Simulation integrated to this event's centre; advanced in place.
        **kwargs
            Ignored; present for a uniform event interface.
        """
        super().compute(sim)
        self.result[:] = calculate_center_and_orbit(sim, self.center - self.tref, self.ipl,
                                                    w=self.window_width)[0] + self.tref
        self.dirty = False


class EclipseCenter(Event):
    def __init__(self, center: float, tref: float, ipl: int, result: ndarray):
        super().__init__(center, None, tref)
        self.ipl: int = ipl
        self.result = result

    def __repr__(self):
        return f"EC({self.center:.3f}, ipl={self.ipl})"

    def compute(self, sim, **kwargs):
        super().compute(sim)
        self.result[:] = calculate_eclipse_center_and_orbit(sim, self.center - self.tref,
                                                            self.ipl)[0] + self.tref
        self.dirty = False


class EventList:
    """A time-sorted collection of :class:`Event` objects.

    Keeps the events ordered by centre time and tracks ``itref``, the index of the
    last event at or before the reference epoch ``tref``. That split lets
    :class:`PhotoDynamicalModel` integrate backward over events ``[:itref+1]`` and
    forward over events ``[itref+1:]`` from a single built simulation.
    """

    def __init__(self, tref: float, v: Optional[list] = None):
        """Create an event list.

        Parameters
        ----------
        tref : float
            Reference epoch that splits backward and forward integration.
        v : list, optional
            Initial events; sorted on construction.
        """
        self.tref: float = tref
        self.itref: Optional[int] = None
        self.events: list = sorted(v or [])
        self.update_itref()

    def update_itref(self):
        """Recompute ``itref``, the index of the last event at or before ``tref``."""
        if self.events is not None:
            self.itref = searchsorted(self.events, self.tref) - 1

    def __len__(self):
        """Number of events in the list."""
        return len(self.events)

    def __getitem__(self, key):
        """Index or slice into the sorted event list."""
        return self.events[key]

    def __add__(self, v):
        return EventList(self.tref, self.events + v)

    def __iadd__(self, v):
        """Add the events in ``v`` in place, re-sorting and updating ``itref``."""
        self.events = sorted(self.events + v)
        self.update_itref()
        return self

    def __repr__(self):
        s = []
        for i, e in enumerate(self.events):
            s.append(f"[{i:>3d}] {e}")
            if i == self.itref:
                s.append(f"--*--    {self.tref:.3f}    --*--")
        return "\n".join(s)

class PhotoDynamicalModel:
    """N-body model that produces photometry, RVs, and transit centres.

    Given the data layout at construction, this class assembles the corresponding
    :class:`Event` objects into a time-sorted :class:`EventList` keyed on a reference
    epoch. Calling the instance with a set of physical system parameters builds a
    :class:`rebound.Simulation` (optionally with a GR force) and evaluates every event
    by integrating outward from the reference epoch, filling preallocated model arrays
    for the flux, radial velocities, and transit centres.
    """

    def __init__(self, nplanets: int, is_transiting: ndarray, tref: float,
                 lctimes = None, pids = None, lcids = None, pbids: Optional[Iterable[Iterable[int]]] = None, exptimes = None, nsamples = None,
                 lctypes: Optional[Iterable[int]] = None,
                 rvtimes: Optional[ndarray] = None,
                 tcs: Optional[ndarray] = None, tcipl: Optional[ndarray] = None,
                 tctypes: Optional[Iterable[int]] = None,
                 with_gr: bool = True,
                 gap_threshold: Optional[float] = 10.0, gap_steps_per_orbit: int = 100,
                 ias15_epsilon: float = 1e-9):
        """
        The `lctypes` and `tctypes` arguments select the event type per light curve
        and per centre time: 0 (default) models a transit, 1 a secondary eclipse.

        Steps longer than `gap_threshold` days cross a stretch with no data in it and are
        integrated with fixed-step WHFast at `gap_steps_per_orbit` steps per innermost
        orbit rather than with IAS15; see :func:`integrate_to`. Pass
        ``gap_threshold=None`` to integrate everything with IAS15, which is what a system
        with close encounters or a strongly eccentric inner planet wants.
        """

        self.sim = None
        self.nplanets = nplanets
        self.is_transiting = is_transiting
        self.tref = tref
        self.events = EventList(self.tref)

        self.lctimes = lctimes
        self.flmodel = None
        if self.lctimes is not None:
            self.flmodel = ones(lctimes.size)
            self.lcids = lcids
            self.pbids = pbids
            self.exptimes = exptimes
            self.nsamples = nsamples

            self.ulcids = unique(lcids)
            self.lcslices = []
            for lcid in self.ulcids:
                start, end = where(lcids == lcid)[0][[0, -1]]
                self.lcslices.append(slice(start, end + 1))

            lce = []
            for lcid, sl in zip(self.ulcids, self.lcslices):
                lctype = 0 if lctypes is None else lctypes[lcid]
                lccls = EclipseLC if lctype == 1 else TransitLC
                lce.append(lccls(self.lctimes[sl], tref, pids[lcid], pbids[lcid], nsamples[lcid], exptimes[lcid], self.flmodel[sl]))
            self.events += lce

        self.rvtimes = rvtimes
        self.rvmodel = None
        if self.rvtimes is not None:
            self.rvmodel = zeros(rvtimes.size)
            self.events += [RVPoint(t, tref, self.rvmodel[i:i + 1]) for i, t in enumerate(self.rvtimes)]

        self.tcs = tcs
        self.tcmodel = None
        if self.tcs is not None:
            self.tcmodel = zeros(tcs.size)
            tctypes = zeros(tcs.size, dtype=int) if tctypes is None else tctypes
            tce = []
            for i, (t, ipl, tctype) in enumerate(zip(tcs, tcipl, tctypes)):
                tccls = EclipseCenter if tctype == 1 else TransitCenter
                tce.append(tccls(t, tref, ipl, self.tcmodel[i:i + 1]))
            self.events += tce

        self.gap_threshold = gap_threshold
        self.gap_steps_per_orbit = gap_steps_per_orbit
        self.ias15_epsilon = ias15_epsilon
        self._gap_dt = None

        self.with_tides = False
        self.with_gr = with_gr
        self.with_gh = False

        self._rx = None
        self._rx_tides = None
        self._rx_gr = None
        self._rx_gh = None

    def add_gr(self, sim):
        """Attach a general-relativity force to `sim` via reboundx.

        The returned `Extras` needs no keeping alive here: `reboundx.Extras.__init__`
        stores itself in `sim._extras_ref`, so it lives as long as the simulation does.
        """
        rx = reboundx.Extras(sim)
        gr = rx.load_force('gr')
        gr.params['c'] = cn.c.to('AU/day').value
        rx.add_force(gr)
        return rx

    def copy_sim(self, sim=None):
        """Return a copy of `sim` (`self.sim` by default) with the extra forces attached.
        """
        sim = (self.sim if sim is None else sim).copy()
        if self.with_gr:
            self.add_gr(sim)
        set_integration_policy(sim, self.gap_threshold, self._gap_dt)
        return sim

    def build_system(self, mstar, rstar, mp, t0, p, inc, e, w, omega, planets=None):
        """Build the rebound simulation for the given physical parameters.

        Creates a fresh simulation in (day, AU, Msun) units, adds the star and each
        planet (using ``T`` for transiting planets and the mean anomaly ``M`` for
        non-transiting ones), moves to the centre of mass, sets escape/encounter
        distance limits. The result is stored on ``self.sim``, which is a template that
        is never integrated directly: :meth:`copy_sim` copies it and attaches the extra
        forces to the copy.

        Parameters
        ----------
        mstar : float
            Stellar mass in solar masses.
        rstar : float
            Stellar radius in solar radii (converted to AU internally).
        mp : array_like
            Planetary masses in solar masses.
        t0 : array_like
            Transit epoch (transiting planets) or mean anomaly (non-transiting), per
            planet.
        p : array_like
            Orbital periods.
        inc : array_like
            Inclinations.
        e : array_like
            Eccentricities.
        w : array_like
            Arguments of periastron.
        omega : array_like
            Longitudes of the ascending node.
        planets : iterable of int, optional
            Subset of planet indices to add; defaults to all planets.
        """
        self.rstar = rs2au*rstar
        self.sim = rebound.Simulation()
        self.sim.units = ("day", 'AU', 'Msun')
        self.sim.add(m=mstar, r=self.rstar, hash="star")

        planets = planets if planets is not None else range(self.nplanets)
        for ipl in planets:
            if self.is_transiting[ipl]:
                tc_offset = mean_anomaly_offset(e[ipl], w[ipl])/(2*pi)*p[ipl]
                self.sim.add(m=mp[ipl], P=p[ipl], T=t0[ipl] - tc_offset, inc=inc[ipl],
                             e=e[ipl], omega=w[ipl], Omega=omega[ipl],
                             hash=f'planet_{ipl + 1}')
            else:
                self.sim.add(m=mp[ipl], P=p[ipl], M=t0[ipl], inc=inc[ipl],
                             e=e[ipl], omega=w[ipl], Omega=omega[ipl],
                             hash=f'planet_{ipl + 1}')
        self.sim.move_to_com()
        self.sim.ri_ias15.epsilon = self.ias15_epsilon
        self._gap_dt = min(p[ipl] for ipl in planets)/self.gap_steps_per_orbit
        self.sim.exit_min_distance = 0.01*array([p.rhill for p in self.sim.particles[1:]]).max()
        self.sim.exit_max_distance = 2.0*array([p.a for p in self.sim.particles[1:]]).max()
        #self.sim.ri_ias15.min_dt = self.periods.min()/80

    def __call__(self, mstar, rstar, ldc, mp, k, t0, p, inc, e, w, omega, fr = None, build_only: bool = False, planets = None):
        """
        The `fr` argument gives the planet-star surface brightness ratios as an
        array shaped (nplanets, npassbands); it is required when secondary
        eclipse light curves are modelled and gives an eclipse depth of fr*k^2.
        A flat array is interpreted planet-major and reshaped to
        (nplanets, npassbands).
        """
        self.build_system(mstar, rstar, mp, t0, p, inc, e, w, omega, planets=planets)
        if fr is not None:
            fr = asarray(fr)
            if fr.ndim == 1:
                fr = fr.reshape(self.nplanets, -1)
            if fr.shape[0] != self.nplanets:
                raise ValueError("fr must have a shape (nplanets, npassbands)")
        if build_only:
            return None, None, None
        else:
            try:
                sim_backward = self.copy_sim()
                for e in self.events[self.events.itref::-1]:
                    e.compute(sim_backward, k=k, ldc=ldc, fr=fr)
                sim_forward = self.copy_sim()
                for e in self.events[self.events.itref + 1:]:
                    e.compute(sim_forward, k=k, ldc=ldc, fr=fr)
                return self.flmodel, self.rvmodel, self.tcmodel
            except (rebound.Escape, rebound.Collision, rebound.GenericError, rebound.Encounter):
                raise ValueError
