PyTTV-photodyn
==============

**PyTTV-photodyn** (import package ``pyttv_photodyn``, distribution
``pyttv-photodyn``) performs **photodynamical modeling of transiting exoplanet
systems**. It fits photometry, radial velocities, and transit-timing data
simultaneously by N-body integrating the whole planetary system, rather than
treating each planet's transits independently.

This is what makes it *photodynamical*: transit times, durations, and shapes
emerge from the gravitational interactions between the planets (transit-timing
variations, TTVs) instead of being free parameters.

A typical analysis is interactive (Jupyter or a script): instantiate a
:class:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF`, call
:meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.optimize_global` (differential
evolution) to find a starting population, then
:meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.sample_mcmc` (emcee) to sample
the posterior.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   parameters
   architecture

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
