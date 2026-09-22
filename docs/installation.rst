Installation
============

PyTTV-photodyn is a pure-Python library with no build step. Install it in
editable mode from a checkout of the repository:

.. code-block:: bash

   pip install -e .

The import name is ``pyttv_photodyn`` while the distribution is
``pyttv-photodyn``. The package source lives in ``src/``, which is mapped to the
import name via ``[tool.setuptools] package-dir`` in ``pyproject.toml``. So even
though the file is ``src/pdlpf.py``, you import it as:

.. code-block:: python

   from pyttv_photodyn.pdlpf import PhotoDynamicalLPF

Dependencies
------------

PyTTV-photodyn depends on a stack of scientific packages, installed
automatically with the command above:

* `pytransit <https://github.com/hpparvi/PyTransit>`_ — provides the ``BaseLPF``
  base class and orbit utilities.
* `rebound <https://rebound.readthedocs.io>`_ and
  `reboundx <https://reboundx.readthedocs.io>`_ — N-body integration and the
  general-relativity force.
* `numba <https://numba.pydata.org>`_ — JIT-compiled numerical kernels.
* `emcee <https://emcee.readthedocs.io>`_ — MCMC sampling.
* `celerite <https://celerite.readthedocs.io>`_ — optional Gaussian-process
  noise model.
* NumPy, SciPy, pandas, xarray, PyTables, and uncertainties.

Building the documentation
---------------------------

The documentation is built with Sphinx. Install the package together with its
``docs`` extra, then run the build:

.. code-block:: bash

   pip install -e '.[docs]'
   cd docs
   make html

The rendered HTML lands in ``docs/_build/html/``. Autodoc imports the real
package during the build, so the full dependency stack must be installed.
