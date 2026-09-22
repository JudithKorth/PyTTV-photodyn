# PyTTV-photodyn

Photodynamical modelling of transiting multi-planet systems.

PyTTV-photodyn fits photometry, radial velocities, and prior transit-time
measurements *jointly* by N-body integrating the whole planetary system with
[REBOUND](https://rebound.readthedocs.io). 

## Features

- **Three data types, all optional.** Photometry, radial velocities from any number
  of instruments, and prior mid-transit times can be fitted in any combination,
  including transit centres alone.
- **Transiting and non-transiting planets.** Transiting planets are parametrised by
  their transit centre and impact parameter, non-transiting ones by their mean
  anomaly and inclination.

## Installation

PyTTV-photodyn is not (yet) on PyPI; install it from GitHub. The package is pure Python,
and Python 3.12 or newer is required.

### Directly with pip

```bash
pip install git+https://github.com/JudithKorth/PyTTV-photodyn.git
```

### From a clone, for development

Clone the repository and install it in editable mode, so changes to the source take
effect without reinstalling:

```bash
git clone https://github.com/JudithKorth/PyTTV-photodyn.git
cd PyTTV-photodyn
pip install -e .
```
The distribution is called `pyttv-photodyn` and the import package
`pyttv_photodyn`:

```python
from pyttv_photodyn import PhotoDynamicalLPF
```

## Citation

The first papers using the pyttv-photodyn code are Korth et al. (2023 & 2024).

<!-- TODO: add the reference to the paper describing the method once available. -->

If you use PyTTV-photodyn in your research, please cite the accompanying paper
and the underlying tools: REBOUND (Rein & Liu 2012), REBOUNDx (Tamayo et al.
2020), and PyTransit (Parviainen 2015, Parviainen & Korth 2020).

