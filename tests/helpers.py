"""Shared builders and canonical test systems for the pdmodel/pdlpf test suite."""
import sys
from pathlib import Path

import numpy as np
import rebound

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import pdmodel as pdm  # noqa: E402
from pytransit.orbits.orbits_py import mean_anomaly_offset  # noqa: E402

EXPTIME = 2.0 / 60 / 24  # 2 min in days

# Canonical eccentric single-planet system: transit depth ~0.012, e large
# enough that conjunction, minimum separation, and p/2 eclipse phase all differ.
TWO_BODY = dict(mstar=1.0, rstar=1.0, mp=(3e-6,), k=(0.1,), t0=(1.0,), p=(3.4,),
                inc=(1.55,), e=(0.15,), w=(0.4,), om=(0.0,))

# Perturbed pair: an inner transiting planet and a massive outer perturber.
TWO_PLANET = dict(mstar=1.0, rstar=1.0,
                  mp=(3e-6, 5e-5), k=(0.1, 0.08), t0=(1.0, 2.5), p=(3.4, 7.1),
                  inc=(1.545, 1.55), e=(0.05, 0.03), w=(0.4, 1.2), om=(0.0, 0.05))


def build_sim(pars):
    """Build a rebound simulation the same way PhotoDynamicalModel.build_system does."""
    sim = rebound.Simulation()
    sim.units = ("day", "AU", "Msun")
    sim.add(m=pars['mstar'], r=pars['rstar'] * pdm.rs2au)
    for i in range(len(pars['p'])):
        toff = mean_anomaly_offset(pars['e'][i], pars['w'][i]) / (2 * np.pi) * pars['p'][i]
        sim.add(m=pars['mp'][i], P=pars['p'][i], T=pars['t0'][i] - toff, inc=pars['inc'][i],
                e=pars['e'][i], omega=pars['w'][i], Omega=pars['om'][i])
    sim.move_to_com()
    return sim


def window(t_center, half_width=0.15, exptime=EXPTIME):
    """Regularly sampled light-curve window around an event."""
    return np.arange(t_center - half_width, t_center + half_width, exptime)


def model_arrays(pars):
    """Convert a parameter dict into PhotoDynamicalModel.__call__ positional args."""
    return (pars['mstar'], pars['rstar'], np.array([0.4, 0.2]), np.array(pars['mp']),
            np.array(pars['k']), np.array(pars['t0']), np.array(pars['p']),
            np.array(pars['inc']), np.array(pars['e']), np.array(pars['w']),
            np.array(pars['om']))


def nbody_minimum_separation(sim, guess, halfwidth=0.05):
    """Time of minimum projected star-planet separation from direct N-body integration."""
    from scipy.optimize import minimize_scalar

    def sep(t):
        sim.integrate(t)
        pl, st = sim.particles[1], sim.particles[0]
        return np.hypot(pl.x - st.x, pl.y - st.y)

    res = minimize_scalar(sep, bounds=(guess - halfwidth, guess + halfwidth),
                          method="bounded", options={"xatol": 1e-10})
    return res.x
