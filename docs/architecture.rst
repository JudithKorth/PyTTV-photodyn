Architecture
============

PyTTV-photodyn is organised in three layers, plus pluggable likelihoods.

1. The fitting front-end
------------------------

:class:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF` (in ``src/pdlpf.py``) is the
class users instantiate. It subclasses ``pytransit.BaseLPF`` and is responsible
for:

* **Data ingestion** — photometry (times, fluxes, errors), radial velocities
  from multiple instruments (concatenated and time-sorted, with a mapping from
  each point to its instrument set), and prior transit-center measurements.
  Radial velocities and transit centers are each optional.
* **Parameter definition and priors** — see :doc:`parameters` for the positional,
  load-bearing layout of the parameter vector.
* **The model bridge** —
  :meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.transit_model` unpacks the flat
  parameter vector, converts it to physical quantities, and calls the N-body
  model. It also adds the RV systemic offsets, a polynomial trend (order 1 or 2),
  and a sinusoid on top of the N-body radial velocities.
* **Likelihood and posterior** — combining the photometric, RV, and
  transit-center contributions.
* **Inference drivers** —
  :meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.optimize_global` (differential
  evolution) and :meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.sample_mcmc`
  (emcee).
* **Analysis and plotting helpers** — transit times and durations within a
  range, TTV plots, and phase-folded photometry.

2. The N-body engine
--------------------

:class:`~pyttv_photodyn.pdmodel.PhotoDynamicalModel` (in ``src/pdmodel.py``)
builds a ``rebound`` simulation (units: day, AU, solar mass) from the physical
parameters, optionally adding general relativity through ``reboundx``.

Evaluation is **event-based and reference-time centered**. Every observable is an
``Event`` object held in a time-sorted ``EventList``. The subtypes are
:class:`~pyttv_photodyn.pdmodel.TransitLC` (a light-curve segment),
:class:`~pyttv_photodyn.pdmodel.RVPoint`, and
:class:`~pyttv_photodyn.pdmodel.TransitCenter`; each knows how to ``compute``
itself by integrating the simulation to its time.

Calling the model copies the built simulation and integrates **backward** from
the reference epoch for earlier events and **forward** for later ones, filling
preallocated model arrays. This minimises integration distance and keeps a clean
split around the reference time. rebound exceptions (escape, collision,
encounter, generic errors) are caught and re-raised as ``ValueError``, which the
front-end turns into infinite model values (and therefore ``-inf`` likelihood),
so unstable configurations are rejected during sampling.

Transit-center finding is the numerically delicate part: a 7-point stencil
around a candidate time yields position, velocity, and acceleration plus
finite-difference jerk and snap; a Taylor expansion of the sky-projected
separation is minimised by a golden-section search, iterated to convergence,
with a light-travel-time correction applied. Light curves are evaluated with a
numba quadratic-limb-darkening model, oversampled according to the per-light-curve
sampling and exposure times.

3. Pluggable likelihoods
------------------------

The photometric likelihood is registered as a pluggable model:

* :class:`~pyttv_photodyn.wnloglikelihood.WNLogLikelihood` — a white-noise
  (per-point Gaussian) likelihood, numba-accelerated and evaluated per light
  curve. This is the default.
* :class:`~pyttv_photodyn.celeriteloglikelihood.CeleriteLogLikelihood` — an
  optional Matérn-3/2 Gaussian-process noise model. It imports ``celerite``
  lazily and raises if the package is unavailable. The GP path is wired but not
  active by default.

The RV and transit-center likelihoods are computed directly in the front-end's
likelihood method (the transit-center term masks NaN model centers and penalises
them).
