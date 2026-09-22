"""Gaussian-process (celerite) photometric likelihood.

Defines :class:`CeleriteLogLikelihood`, an optional alternative to the white-noise
likelihood that models correlated (red) noise in the photometry with a celerite
Matern-3/2 Gaussian process. The :mod:`celerite` dependency is imported lazily; if it
is unavailable, constructing the class raises :class:`ImportError`.
"""
from numpy import asarray, unique, zeros, inf, squeeze, zeros_like, isfinite

try:
    from celerite import GP
    from celerite.terms import Matern32Term

    with_celerite = True
except ImportError:
    with_celerite = False


class CeleriteLogLikelihood:
    """Gaussian-process photometric likelihood backed by a celerite Matern-3/2 kernel.

    Builds a celerite GP over the (optionally noise-group-restricted) photometry with
    fixed hyperparameters and uses it to evaluate the marginal log likelihood of the
    transit-model residuals, accounting for correlated noise. It can also predict the
    GP baseline for plotting and detrending.
    """

    def __init__(self, lpf, hps, name: str = 'gp', noise_ids=None):
        """Set up the GP likelihood and compute the kernel over the data.

        Parameters
        ----------
        lpf : pyttv_photodyn.pdlpf.PhotoDynamicalLPF
            The parent LPF; its data (``lcids``, ``noise_ids``, ``timea``, ``ofluxa``,
            ``errora``) must already be initialised with per-point errors.
        hps : array_like
            Fixed GP hyperparameters for the Matern-3/2 term.
        name : str, optional
            Identifier for this likelihood model.
        noise_ids : array_like, optional
            White-noise group ids this likelihood applies to; defaults to all groups.
            When a strict subset is given, the GP is restricted to those points via a
            boolean mask.

        Raises
        ------
        ImportError
            If celerite is not installed.
        ValueError
            If the LPF data is uninitialised or lacks per-point errors.
        """
        if not with_celerite:
            raise ImportError("CeleriteLogLikelihood requires celerite.")

        self.name = name
        self.lpf = lpf
        self.hps = asarray(hps)

        if lpf.lcids is None:
            raise ValueError('The LPF data needs to be initialised before initialising CeleriteLogLikelihood.')

        if lpf.lcids.size != lpf.errora.size:
            raise ValueError('The LPF needs per-point white noise errors.')

        self.noise_ids = noise_ids if noise_ids is not None else unique(lpf.noise_ids)

        self.mask = m = zeros(lpf.lcids.size, bool)
        for lcid, nid in enumerate(lpf.noise_ids):
            if nid in self.noise_ids:
                m[lcid == lpf.lcids] = 1

        if m.sum() == lpf.lcids.size:
            self.times = lpf.timea
            self.fluxes = lpf.ofluxa
            self.errors = lpf.errora
        else:
            self.times = lpf.timea[m]
            self.fluxes = lpf.ofluxa[m]
            self.errors = lpf.errora[m]

        self.gp = GP(Matern32Term(0, 0))
        self.compute_gp()

    def compute_gp(self):
        """Set the GP hyperparameters and factorise the kernel over the data times."""
        self.gp.set_parameter_vector(self.hps)
        self.gp.compute(self.times, yerr=self.errors)

    def compute_gp_lnlikelihood(self, model):
        """Return the GP marginal log likelihood of the model residuals.

        Parameters
        ----------
        model : numpy.ndarray
            Full model flux array; the masked subset is subtracted from the observed
            fluxes to form the residuals scored by the GP.

        Returns
        -------
        float
            The celerite marginal log likelihood of the residuals.
        """
        return self.gp.log_likelihood(self.fluxes - model[self.mask])

    def predict_baseline(self, pv):
        """Predict the GP noise baseline for a parameter vector.

        Computes the transit-model residuals and uses the GP to predict the correlated
        baseline at the data times, returning it as a multiplicative ``1 + baseline``
        on the full LPF time grid (zero outside the masked points).

        Parameters
        ----------
        pv : numpy.ndarray
            Parameter vector passed to the LPF's transit model.

        Returns
        -------
        numpy.ndarray
            The predicted baseline, length equal to the full LPF time array.
        """
        residuals = self.fluxes - squeeze(self.lpf.flux_model(pv))[self.mask]
        bl = zeros_like(self.lpf.timea)
        bl[self.mask] = self.gp.predict(residuals, self.times, return_cov=False)
        return 1. + bl

    def __call__(self, pvp, model):
        """Evaluate the GP log likelihood for one model or a population of models.

        Models containing any non-finite flux are assigned ``-inf``.

        Parameters
        ----------
        pvp : numpy.ndarray
            Parameter vector(s); only the dimensionality is used, to decide between
            the single-model and population paths.
        model : numpy.ndarray
            Model flux for a single parameter vector, or a 2D stack for a population.

        Returns
        -------
        float or numpy.ndarray
            The log likelihood, scalar for a single model or one value per model.
        """
        if pvp.ndim == 1:
            if all(isfinite(model)):
                lnlike = self.compute_gp_lnlikelihood(model)
            else:
                lnlike = -inf
        else:
            npv = pvp.shape[0]
            lnlike = zeros(npv)
            for ipv in range(npv):
                if all(isfinite(model[ipv])):
                    lnlike[ipv] = self.compute_gp_lnlikelihood(model[ipv])
                else:
                    lnlike[ipv] = -inf
        return lnlike
