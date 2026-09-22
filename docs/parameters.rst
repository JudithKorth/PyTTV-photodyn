Parameter vector layout
=======================

Inference in PyTTV-photodyn operates on a **flat parameter vector** ``pv``. This
vector is indexed *positionally* throughout
:meth:`~pyttv_photodyn.pdlpf.PhotoDynamicalLPF.transit_model` using strided
slices, so the block order and the per-planet stride are load-bearing: if you
add or reorder parameters you must update every strided index in
``transit_model`` (and ``eccentricity_prior``).

The blocks appear in the following fixed order.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Block
     - Size
     - Contents
   * - **star**
     - 2
     - ``mstar``, ``rstar`` (indices 0, 1).
   * - **ldc**
     - 2 per passband
     - Quadratic limb-darkening coefficients ``q1_<pb>``, ``q2_<pb>`` for each
       passband, starting at index 2.
   * - **planets**
     - 8 per planet
     - One block of eight parameters per planet, stride 8 (see below).
   * - **rv**
     - variable
     - ``rv_trend`` (plus ``rv_trend_2`` if ``rv_slope_order == 2``),
       ``rv_sine_amplitude``, ``rv_sine_period``, ``rv_sine_phase``, then one
       systemic velocity ``srv_<i>`` per RV set.
   * - **rv_jitter**
     - 1 per RV set
     - ``log10rvj_<i>`` for each RV set.

The per-planet block (8 parameters, stride 8)
---------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Meaning
   * - ``log10mplanet``
     - Planet mass, base-10 logarithm.
   * - ``k``
     - Radius ratio (planet / star).
   * - ``t0`` *or* ``M``
     - ``t0`` (transit center) for a transiting planet, or ``M`` (mean anomaly)
       for a non-transiting planet.
   * - ``p``
     - Orbital period.
   * - ``b`` *or* ``g`` *or* ``inc``
     - Impact parameter ``b`` (or grazing parameter ``g`` when
       ``use_grazing_parameter=True``) for a transiting planet, or inclination
       ``inc`` for a non-transiting planet.
   * - ``secosw``
     - :math:`\sqrt{e}\,\cos\omega`.
   * - ``sesinw``
     - :math:`\sqrt{e}\,\sin\omega`.
   * - ``omega``
     - Argument of periastron :math:`\omega`.

Transiting vs. non-transiting
-----------------------------

The per-planet ``is_transiting`` flag switches **both** the parametrisation and
how the particle is added to the rebound simulation:

* **Transiting** — parametrised by transit center ``t0`` and impact/grazing
  parameter; added to the N-body simulation with a transit time (``T=`` in
  ``build_system``).
* **Non-transiting** — parametrised by mean anomaly ``M`` and inclination
  ``inc``; added with a mean anomaly (``M=`` in ``build_system``).

Strided indexing
----------------

Because each block has a fixed size, the code addresses parameters with strided
slices. With a single passband the planet block starts at index 4, so for
example the planet masses are ``pv[4:rvi:8]``, the radius ratios ``pv[5:rvi:8]``,
and so on, where ``rvi = self._start_rvs`` marks the start of the RV block. The
block starts and slices are cached on the LPF as ``_start_*`` and ``_sl_*``
attributes.

.. note::

   Physical quantities are derived from these raw parameters inside
   ``transit_model``: stellar density from mass and radius, planet masses from
   their base-10 logarithms, eccentricity and :math:`\omega` from ``secosw`` and
   ``sesinw``, and inclination from the impact/grazing parameter.
