Quickstart
==========

This page walks through the typical workflow: build a
:class:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF` from your data, find a good
starting point with global optimisation, and sample the posterior with MCMC.

Because the package exposes no top-level shortcuts, import the classes you need
from their submodules:

.. code-block:: python

   from pyttv_photodyn.pdlpf import PhotoDynamicalLPF

Building the model
------------------

The constructor ingests photometry, and optionally radial velocities and prior
transit-center measurements. At minimum you provide the number of planets, the
photometric passbands, and per-planet reference epochs and periods (used to
assign epoch numbers), plus the light curves themselves:

.. code-block:: python

   lpf = PhotoDynamicalLPF(
       name="my_system",
       nplanets=2,
       passbands=["TESS"],
       zero_epochs=[t0_b, t0_c],       # reference transit epoch per planet
       periods=[period_b, period_c],   # orbital period per planet
       times=times,                    # list of per-light-curve time arrays
       fluxes=fluxes,                  # matching normalised fluxes
       errors=errors,                  # matching uncertainties
       is_transiting=[True, True],     # parametrisation switch per planet
   )

Radial velocities (``rv_times`` / ``rv_values`` / ``rv_errors``) and prior
transit centers (``center_times`` / ``center_time_errors``) are optional; supply
them to fit those data jointly with the photometry. See
:class:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF` for the full constructor
signature and the layout of the parameter vector.

Global optimisation
-------------------

Use differential evolution to evolve a population of parameter vectors toward
the posterior mode. This gives MCMC a well-placed starting population:

.. code-block:: python

   lpf.optimize_global(niter=200, npop=100)

Sampling the posterior
----------------------

Run emcee starting from the optimised population:

.. code-block:: python

   lpf.sample_mcmc(niter=500, thin=5, repeats=1)

Inspecting the results
----------------------

Several helpers turn a parameter vector into physical quantities and plots:

* :meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.get_transit_times_within_range`
  and
  :meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.get_transit_durations_within_range`
  — predicted transit times and durations for a planet over a time span.
* :meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.plot_ttvs_within_range` — the
  transit-timing-variation diagram.
* :meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.fold_times` and
  :meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.plot_folded` — phase-folded
  photometry for a planet.

.. note::

   PyTTV-photodyn also supports fitting without photometry (RV/TTV-only). Pass
   ``times=None`` and provide radial velocities and/or transit centers instead;
   the photometric likelihood is then disabled internally.
