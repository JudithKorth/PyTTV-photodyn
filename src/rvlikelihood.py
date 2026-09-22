"""Pluggable radial-velocity noise models for :class:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF`.

The LPF owns the RV *mean* model -- a polynomial trend plus one systemic velocity
per instrument -- and applies it inside ``transit_model``. The classes here model
the *noise*: each receives the residuals ``observed - model`` and returns their log
likelihood.

Because :class:`pytransit.BaseLPF` freezes its parameter set before the likelihood
models are normally constructed, a noise model that owns fit parameters cannot be
built the way the photometric plugins are. These classes therefore follow a
two-phase lifecycle:

1. :meth:`RVLikelihood.init_parameters` -- called from
   ``PhotoDynamicalLPF._init_parameters`` before ``ps.freeze()``. Adds exactly one
   global parameter block, which may be empty.
2. :meth:`RVLikelihood.setup` -- called at the end of ``PhotoDynamicalLPF.__init__``,
   once the RV data arrays and the N-body model exist.
"""
from numba import njit
from numpy import log, pi, sum, sqrt, inf, isfinite, all as npall

from pytransit.param import GParameter, NormalPrior as NP, UniformPrior as UP

try:
    from george import GP
    from george.kernels import ExpSquaredKernel, ExpSine2Kernel, Matern32Kernel

    with_george = True
except ImportError:
    with_george = False


def _rv_labels(lpf) -> list:
    """Instrument labels for the RV sets, falling back to the set index.

    Used only in parameter *descriptions*; the parameter names stay indexed by set, so a
    label need be neither unique nor set. Read defensively, because a likelihood model may
    be driven by anything exposing ``nrvsets`` and the test stubs carry no instruments.
    """
    instruments = getattr(lpf, 'rv_instruments', None) or []
    return [instruments[i] if i < len(instruments) and instruments[i] else f'{i:d}'
            for i in range(lpf.nrvsets)]


@njit(cache=False)
def lnlike_normal(o, m, e):
    """Gaussian (white-noise) log likelihood.

    Parameters
    ----------
    o : numpy.ndarray
        Observed values, or residuals when ``m`` is zero.
    m : numpy.ndarray or float
        Model values, broadcastable against ``o``.
    e : numpy.ndarray
        Per-point uncertainties, same shape as ``o``.

    Returns
    -------
    float
        The summed log likelihood assuming independent normal errors.
    """
    return -sum(log(e)) - 0.5 * o.size * log(2. * pi) - 0.5 * sum((o - m) ** 2 / e ** 2)


class RVLikelihood:
    """Abstract base for pluggable RV noise models.

    Subclasses must implement :meth:`init_parameters` and :meth:`__call__`, and
    should extend :meth:`setup` to cache whatever LPF data they need.

    One instance per LPF
    ---------------------
    An ``RVLikelihood`` instance is bound to exactly one LPF: :meth:`init_parameters`
    records the block's ``start``/``slice`` on ``self``, and :meth:`setup` caches that
    LPF's data arrays on ``self``. Passing the same instance to a second LPF would
    silently rebind these to the second LPF's layout and data, corrupting the first
    (or, if the two LPFs happen to share the same layout, corrupting it invisibly).
    Construct a separate instance per LPF.

    Attributes
    ----------
    name : str
        Name of the parameter block this model adds.
    lpf : pyttv_photodyn.pdlpf.PhotoDynamicalLPF or None
        The parent LPF, set by :meth:`setup`. Not read by this base class, but
        available to subclasses that need access to the LPF beyond the data arrays
        cached at setup time.
    start : int or None
        Index of this model's first parameter, set by :meth:`_add_block`.
    slice : slice or None
        Slice selecting this model's parameters from a parameter vector.
    """

    def __init__(self, name: str = 'rv_noise'):
        self.name = name
        self.lpf = None
        self.start = None
        self.slice = None

    def init_parameters(self, lpf, ps):
        """Declare this model's fit parameters (phase 1).

        Called from ``PhotoDynamicalLPF._init_parameters`` before the parameter set
        is frozen. Implementations must add exactly one global block via
        :meth:`_add_block`, even when it is empty, so that block indices stay
        predictable.

        Parameters
        ----------
        lpf : pyttv_photodyn.pdlpf.PhotoDynamicalLPF
            The parent LPF. Its RV data attributes are already populated, but
            ``lpf.tm`` does not exist yet.
        ps : pytransit.param.ParameterSet
            The parameter set under construction.
        """
        raise NotImplementedError

    def setup(self, lpf):
        """Bind to the LPF's RV data (phase 2).

        Called at the end of ``PhotoDynamicalLPF.__init__``, once the RV arrays and
        the N-body model exist. Subclasses should call ``super().setup(lpf)``.

        Not always called: ``PhotoDynamicalLPF`` only calls ``setup`` when it has RV
        data (``self.rv_times is not None``), while :meth:`init_parameters` always
        runs. A subclass must not assume ``setup`` has run by the time
        :meth:`__call__` executes for an RV-less LPF (``self.lpf`` stays ``None``, and
        any data cached here stays unset).

        Implementations read their data from ``lpf._orvtimes``, ``lpf._orvvalues``,
        ``lpf._orverrors``, and ``lpf._orvids``. These arrays are concatenated across
        instruments and sorted by time; ``_orvids`` gives each point's RV-set
        (instrument) index into that sorted order.
        """
        self.lpf = lpf

    def __call__(self, pv, residuals):
        """Log likelihood of the RV residuals.

        Parameters
        ----------
        pv : numpy.ndarray
            The full parameter vector, always a single 1-D array -- ``lnlikelihood``
            raises ``NotImplementedError`` for 2-D input, so implementations need not
            handle a population of vectors. Use ``pv[self.slice]`` for this model's
            own parameters.
        residuals : numpy.ndarray
            ``observed - model`` radial velocities, in the order of
            ``lpf._orvtimes``.

        Returns
        -------
        float
            The log likelihood, or ``-inf`` for an unusable configuration.
        """
        raise NotImplementedError

    def _add_block(self, ps, pars):
        """Add this model's parameter block and record its start index and slice.

        Raises
        ------
        ValueError
            If this instance is already bound to a parameter set, i.e. it is being
            reused across LPFs (see the one-instance-per-LPF note on the class).
        """
        if self.slice is not None:
            raise ValueError(f'This {type(self).__name__} instance is already bound to an LPF. '
                             'Construct a separate instance per LPF.')
        ps.add_global_block(self.name, pars)
        self.start = ps.blocks[-1].start
        self.slice = ps.blocks[-1].slice


class WNRVLikelihood(RVLikelihood):
    """White-noise RV likelihood with one jitter term per instrument.

    Adds a ``log10rvj_{i}`` parameter per RV set and evaluates a Gaussian likelihood
    with the jitter added in quadrature to the reported uncertainties. This is the
    default RV noise model and reproduces the behaviour that was hard-coded into
    ``PhotoDynamicalLPF`` before the noise model became pluggable.
    """

    def __init__(self, jitter_prior=None, name: str = 'rv_noise'):
        """
        Parameters
        ----------
        jitter_prior : pytransit.param.Prior, optional
            Prior on each ``log10rvj_{i}``. Defaults to ``NormalPrior(0, 0.5)``.
        name : str, optional
            Name of the parameter block.
        """
        super().__init__(name)
        self.jitter_prior = jitter_prior if jitter_prior is not None else NP(0, 0.5)
        self._orverrors = None
        self._orvids = None

    def init_parameters(self, lpf, ps):
        pars = [GParameter(f'log10rvj_{i:d}', f'log10_rv_jitter_{label}', '',
                           self.jitter_prior, [-inf, inf])
                for i, label in enumerate(_rv_labels(lpf))]
        self._add_block(ps, pars)

    def setup(self, lpf):
        super().setup(lpf)
        self._orverrors = lpf._orverrors
        self._orvids = lpf._orvids

    def __call__(self, pv, residuals):
        jitter = 10 ** pv[self.slice][self._orvids]
        return lnlike_normal(residuals, 0.0, sqrt(self._orverrors ** 2 + jitter ** 2))


class QPGPRVLikelihood(RVLikelihood):
    """Aperiodic plus quasi-periodic Gaussian-process RV likelihood, backed by george.

    Models the RV residuals with a fixed two-term kernel::

        ap_std**2 * Matern32(ap_scale**2)
        + std**2 * ExpSine2(10**log10_gamma, log(period)) * ExpSquared(coherence**2)

    The Matern-3/2 term absorbs aperiodic long-term variability that a polynomial
    trend would otherwise have to carry; the quasi-periodic term models rotationally
    modulated stellar activity, with ``coherence`` setting how quickly the active
    regions evolve.

    The kernel structure is fixed, but every prior is caller-configurable, so the
    period and coherence bounds can be matched to the star under study.

    Not thread-safe: :meth:`__call__` mutates ``self._gp`` in place on every
    evaluation. A multiprocessing pool is fine because each worker gets its own
    pickled copy (see the pickling test), but sharing one instance across a thread
    pool would race.

    Raises
    ------
    ImportError
        If george is not installed.
    """

    #: Parameter names, in the order they map onto george's kernel vector.
    parameter_names = ('gp_ap_std', 'gp_ap_scale', 'gp_std',
                       'gp_log10_gamma', 'gp_period', 'gp_coherence')

    def __init__(self, ap_std_prior=None, ap_scale_prior=None, std_prior=None,
                 log10_gamma_prior=None, period_prior=None, coherence_prior=None,
                 with_jitter: bool = False, jitter_prior=None, name: str = 'rv_noise'):
        """
        Parameters
        ----------
        ap_std_prior, ap_scale_prior : pytransit.param.Prior, optional
            Priors on the aperiodic Matern-3/2 amplitude [m/s] and time scale [d].
            Default to ``UP(0, 50)`` and ``UP(1, 300)``.
        std_prior : pytransit.param.Prior, optional
            Prior on the quasi-periodic amplitude [m/s]. Defaults to ``UP(0, 50)``.
        log10_gamma_prior : pytransit.param.Prior, optional
            Prior on the base-10 log of the ExpSine2 inverse-width. Defaults to
            ``UP(-2, 5)``.
        period_prior : pytransit.param.Prior, optional
            Prior on the rotation period [d]. Defaults to ``UP(1, 100)``; set this to
            the star's known rotation period range.
        coherence_prior : pytransit.param.Prior, optional
            Prior on the active-region evolution time scale [d]. Defaults to
            ``UP(20, 500)``.
        with_jitter : bool, optional
            If ``True``, add a ``log10rvj_{i}`` parameter per RV set, applied in
            quadrature to the reported uncertainties. The kernel has no white-noise
            term of its own, so enable this if the RV errors may be underestimated.
        jitter_prior : pytransit.param.Prior, optional
            Prior on each jitter term. Defaults to ``NormalPrior(0, 0.5)``.
        name : str, optional
            Name of the parameter block.
        """
        if not with_george:
            raise ImportError('QPGPRVLikelihood requires george.')
        super().__init__(name)

        self.priors = (ap_std_prior or UP(0, 50),
                       ap_scale_prior or UP(1, 300),
                       std_prior or UP(0, 50),
                       log10_gamma_prior or UP(-2, 5),
                       period_prior or UP(1, 100),
                       coherence_prior or UP(20, 500))
        self.with_jitter = with_jitter
        self.jitter_prior = jitter_prior if jitter_prior is not None else NP(0, 0.5)

        self._gp = None
        self._orvtimes = None
        self._orverrors = None
        self._orvids = None

    def init_parameters(self, lpf, ps):
        # A small positive lower bound keeps the 2*log(p) mapping finite.
        bounds = [(1e-6, inf), (1e-6, inf), (1e-6, inf),
                  (-inf, inf), (1e-6, inf), (1e-6, inf)]
        descriptions = ('rv_gp_ap_std', 'rv_gp_ap_time_scale', 'rv_gp_std',
                        'log10_rv_gp_gamma', 'rv_gp_period', 'rv_gp_coherence_scale')
        units = ('m/s', 'd', 'm/s', '', 'd', 'd')

        pars = [GParameter(n, d, u, prior, list(b))
                for n, d, u, prior, b in zip(self.parameter_names, descriptions, units,
                                             self.priors, bounds)]
        if self.with_jitter:
            pars += [GParameter(f'log10rvj_{i:d}', f'log10_rv_jitter_{label}', '',
                                self.jitter_prior, [-inf, inf])
                     for i, label in enumerate(_rv_labels(lpf))]
        self._add_block(ps, pars)

    def setup(self, lpf):
        super().setup(lpf)
        self._orvtimes = lpf._orvtimes
        self._orverrors = lpf._orverrors
        self._orvids = lpf._orvids

        kernel = (1.0 * Matern32Kernel(1.0)
                  + 1.0 * ExpSine2Kernel(1.0, 0.0) * ExpSquaredKernel(1.0))
        self._gp = GP(kernel)
        self._gp.compute(self._orvtimes, yerr=self._orverrors)

        # Validate the physical-to-kernel-vector mapping once, at construction time,
        # rather than discovering a length mismatch as a silent -inf from __call__'s
        # narrowed `except ValueError` around `_gp.compute`.
        n_expected = len(self._gp.get_parameter_vector())
        n_actual = len(self._kernel_vector([1.0] * len(self.parameter_names)))
        if n_actual != n_expected:
            raise ValueError(
                f'_kernel_vector returns {n_actual} values but the george kernel expects '
                f'{n_expected}.')

    def _kernel_vector(self, p):
        """Map the physical parameters onto george's kernel vector.

        george orders the vector as::

            [k1:k1:log_constant, k1:k2:metric:log_M_0_0,
             k2:k1:k1:log_constant, k2:k1:k2:gamma, k2:k1:k2:log_period,
             k2:k2:metric:log_M_0_0]

        The ``log_constant`` entries are log variances and the ``log_M`` entries are
        log squared length scales, hence the factors of two.
        """
        return [2.0 * log(p[0]), 2.0 * log(p[1]),
                2.0 * log(p[2]), 10.0 ** p[3], log(p[4]), 2.0 * log(p[5])]

    def __call__(self, pv, residuals):
        if not npall(isfinite(residuals)):
            return -inf

        p = pv[self.slice]
        yerr = self._orverrors
        if self.with_jitter:
            jitter = 10 ** p[6:][self._orvids]
            yerr = sqrt(self._orverrors ** 2 + jitter ** 2)

        self._gp.set_parameter_vector(self._kernel_vector(p))
        try:
            self._gp.compute(self._orvtimes, yerr=yerr)
        except ValueError:
            # Covers both a non-finite kernel vector and, via numpy.linalg.LinAlgError
            # (a ValueError subclass), a covariance matrix that is not positive
            # definite. Deliberately not narrowed further than ValueError for that
            # reason. `set_parameter_vector` above is intentionally outside this
            # try/except: it also raises ValueError, but only on a wrong-length
            # vector, which should be a loud bug rather than a silent -inf.
            return -inf

        lnl = self._gp.log_likelihood(residuals, quiet=True)
        return lnl if isfinite(lnl) else -inf
