"""White-noise (per-point Gaussian) photometric likelihood.

Defines :class:`WNLogLikelihood`, the default photometric likelihood used by
:class:`pyttv_photodyn.pdlpf.PhotoDynamicalLPF`. It assumes independent normal errors on each
flux point, optionally restricted to a subset of white-noise groups, and is backed by
a numba-accelerated kernel that sums the log likelihood over per-light-curve slices.
"""
from numba import njit
from numpy import atleast_2d, zeros, log, pi, unique, array, inf, arange, squeeze, isnan


@njit
def lnlike_normal(o, m, e, slices, nids):
    """Gaussian log likelihood summed over light-curve slices, for a model population.

    Computes the white-noise log likelihood of each model in a (possibly singleton)
    population, accumulating only over the index ranges given in ``slices`` so that
    points outside the selected light curves are ignored. Any model whose running sum
    becomes ``nan`` is assigned ``-inf``.

    Parameters
    ----------
    o : numpy.ndarray
        Observed fluxes (1D, length equal to the total number of points).
    m : numpy.ndarray
        Model fluxes; promoted to 2D as ``(npv, npoints)`` for a population.
    e : numpy.ndarray
        Per-point flux uncertainties, aligned with ``o``.
    slices : numpy.ndarray
        ``(nslice, 2)`` array of ``[start, stop]`` index ranges to sum over.
    nids : numpy.ndarray
        Noise-group id per slice. Accepted for interface symmetry; not used in the
        sum (a single shared error array is assumed).

    Returns
    -------
    numpy.ndarray
        Length-``npv`` array of log-likelihood values, one per model.
    """
    m = atleast_2d(m)
    slices = atleast_2d(slices)
    npv = m.shape[0]
    nsl = slices.shape[0]
    lnl = zeros(npv)
    for i in range(npv):
        for isl in range(nsl):
            for j in range(slices[isl, 0], slices[isl, 1]):
                lnl[i] += -log(e[j]) - 0.5 * log(2 * pi) - 0.5 * ((o[j] - m[i, j]) / e[j]) ** 2
            if isnan(lnl[i]):
                lnl[i] = -inf
                break
    return lnl


class WNLogLikelihood:
    """White-noise photometric likelihood model for an LPF.

    Precomputes, from the parent LPF, the per-light-curve index slices and the mapping
    from global to local noise-group ids for the white-noise groups it covers, then
    evaluates the Gaussian log likelihood of a model flux array via
    :func:`lnlike_normal`. Instances are registered into the LPF's list of likelihood
    models and called with ``(pvp, model)``.
    """

    def __init__(self, lpf, name: str = 'wn', noise_ids=None):
        """Set up the white-noise likelihood from an initialised LPF.

        Parameters
        ----------
        lpf : pyttv_photodyn.pdlpf.PhotoDynamicalLPF
            The parent LPF; its data (``noise_ids``, ``lcslices``, ``timea``,
            ``ofluxa``, ``errora``) must already be initialised.
        name : str, optional
            Identifier for this likelihood model.
        noise_ids : array_like, optional
            White-noise group ids this likelihood applies to; defaults to all groups
            present in the LPF.

        Raises
        ------
        ValueError
            If the LPF data has not been initialised.
        """
        self.name = name
        self.lpf = lpf

        if lpf.noise_ids is None:
            raise ValueError('The LPF data needs to be initialised before initialising WNLogLikelihood.')

        self.global_noise_ids = noise_ids if noise_ids is not None else unique(lpf.noise_ids)
        self.mapping = {g:l for g,l in zip(self.global_noise_ids, arange(self.global_noise_ids.size))}

        slices, lnids = [], []
        for nid, sl in zip(lpf.noise_ids, lpf.lcslices):
            if nid in self.global_noise_ids:
                slices.append([sl.start, sl.stop])
                lnids.append(self.mapping[nid])
        self.lcslices = array(slices)
        self.local_pv_noise_ids = array(lnids)

        self.times = lpf.timea
        self.fluxes = lpf.ofluxa
        self.ferrs = lpf.errora

    def __call__(self, pvp, model):
        """Evaluate the white-noise log likelihood of a model flux array.

        Parameters
        ----------
        pvp : numpy.ndarray
            Parameter vector(s). Unused here (the white-noise model has no free
            parameters of its own) but kept for the common likelihood interface.
        model : numpy.ndarray
            Model flux to compare against the observed fluxes.

        Returns
        -------
        float or numpy.ndarray
            The log likelihood, squeezed to a scalar for a single model.
        """
        return squeeze(lnlike_normal(self.fluxes, model, self.ferrs, self.lcslices, self.local_pv_noise_ids))
