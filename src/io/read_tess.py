"""TESS light curve reader returning PyTransit `LCData` objects.

A port of `snowleopards.read_tess` from `snowleopards.io.LCData` onto
`pytransit.utils.io.lcdata.LCData`, and a sibling of `read_cheops`.

The two containers express per-light-curve metadata differently. `LCData` carries
precomputed index arrays (`nids` for the white noise blocks, `pbids` for the passbands),
while `LCData` carries the observation metadata itself (`instrument`, `sector`,
`segment`, `passband`) and lets `LCDataGroup` derive the indices. The old
`nids = 0`, meaning "one white noise parameter shared by all of TESS", becomes "give each
window the sector it came from", so every sector gets its own white noise parameter, and
`pbids = 0` becomes `passband='TESS'`.

The port also keeps what `LCData` threw away. A TESS light curve is cut into windows around
the transits, and each window is now told **which planets transit in it**, which is what
`PhotoDynamicalLPF` needs and refuses to run without.

Each window keeps its median normalisation, and carries no covariates. Handed to an LPF
using `LSTSQBaseline`, an empty covariate matrix gives a design matrix of a single
intercept column, so the window's normalisation is fitted jointly with the transit model
rather than frozen up front::

    lcs = read_tess('data/tess', tic, zero_epochs, periods, wbaseline=24)
    lcs.pids            # -> [(0,), (0, 1), (1,), ...] one entry per window
    lpf = PhotoDynamicalLPF('toi1237', 2, lcs, zero_epochs=zero_epochs, periods=periods,
                            is_transiting=[True, True], tref=tref)
"""

import warnings

from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import lightkurve as lk

from pytransit.utils.io.lcdata import LCData, LCDataGroup

from . import _common
from ._common import (_check_ephemerides, _exposure_time, _extract_windows,
                      _transiting_planets)

__all__ = ['read_tess', 'read_lc']

# TESS observes through a single broad red passband, so every light curve shares one
# passband name and one instrument name.
PASSBAND = 'TESS'
INSTRUMENT = 'TESS'

# Flux columns to try for each type, in order of preference. The SPOC products carry
# `sap_flux` and `pdcsap_flux`, the QLP ones `kspsap_flux` only.
_FLUX_COLUMNS = {'pdc': ('pdcsap_flux', 'kspsap_flux'),
                 'sap': ('sap_flux',)}

# Exposure times above this, in days (~5.8 min), count as long cadence. The 20 s and
# 2 min target stamps and the 200 s full-frame images fall below it, the 10 and 30 min
# full-frame images above it. TESS files carry no OBSMODE, so the exposure time is all
# there is to classify on.
LONG_CADENCE_THRESHOLD = 0.004


def _resolve_nsamples(meta: dict, exptime: float,
                      nsamples_short: int, nsamples_long: int) -> int:
    """Number of supersamples for a light curve, decided from its cadence.

    Classifies the exposure time against `LONG_CADENCE_THRESHOLD`; see
    `_common._resolve_nsamples` for the details.
    """
    return _common._resolve_nsamples(meta, exptime, nsamples_short, nsamples_long,
                                     LONG_CADENCE_THRESHOLD)


def _flux_columns(lc, type: str = 'pdc') -> tuple:
    """Flux and uncertainty columns of the requested type.

    'pdc' selects the detrended flux, falling back to the QLP `kspsap_flux`, and 'sap'
    the simple aperture photometry; see `_common._flux_columns` for the details.
    """
    return _common._flux_columns(lc, _FLUX_COLUMNS, type, 'TESS')


def _finite_mask(lc, type: str = 'pdc') -> tuple:
    """The light curve's times, fluxes and uncertainties, with the unusable points dropped.

    See `_common._finite_mask` for the details.
    """
    return _common._finite_mask(lc, _FLUX_COLUMNS, type, 'TESS')


def read_lc(lc,
            zero_epochs: Sequence[float],
            periods: Sequence[float],
            wbaseline: float,
            min_window: float = 1.0,
            type: Literal['sap', 'pdc'] = 'pdc',
            padding: float = 0.0,
            drop_empty: bool = False,
            nsamples_short: int = 1,
            nsamples_long: int = 10,
            min_points: int = 1) -> LCDataGroup:
    """Cut a single TESS light curve into a `LCDataGroup` of transit windows.

    Parameters
    ----------
    lc
        A `lightkurve.LightCurve`.
    zero_epochs
        Zero epoch of each planet, in the same time system as the data (BJD_TDB).
    periods
        Orbital period of each planet, in days.
    wbaseline
        Width of the window kept around each transit centre, **in hours**.
    min_window
        Drop the windows spanning less than this, **in hours**.
    type
        Flux to read: 'pdc' (default) for the detrended flux, falling back to the QLP
        `kspsap_flux`, or 'sap' for the simple aperture photometry.
    padding
        Margin **in hours** by which a window's time bounding box is widened before testing
        whether a transit centre falls inside it. Defaults to zero, meaning the centre must
        lie within the window. Raise it to also catch the windows a data gap has cut back
        to an ingress or an egress.
    drop_empty
        Drop the windows in which no planet transits. Defaults to `False`, which keeps them
        with an empty `pids`; they still constrain the baseline and the white noise.
    nsamples_short, nsamples_long
        Number of supersamples used to integrate the model over the exposure, for a
        short-cadence and a long-cadence light curve respectively. The cadence is
        classified from the exposure time against `LONG_CADENCE_THRESHOLD`, so the 20 s
        and 2 min target stamps count as short and the 10 and 30 min full-frame images
        as long.
    min_points
        Drop the windows left with fewer than this many points. Defaults to one, which
        removes the empty light curves that would otherwise give a NaN noise estimate.

    Returns
    -------
    LCDataGroup
        The transit windows, each a `LCData` carrying its times, fluxes, errors,
        transiting planets, and observation metadata.
    """
    _check_ephemerides(zero_epochs, periods)

    time, flux, ferr = _finite_mask(lc, type)
    meta = getattr(lc, 'meta', None) or {}
    sector = meta.get('SECTOR', -1)
    exptime = _exposure_time(time, meta.get('TIMEDEL'), 1.0,
                             f'sector {sector}' if sector != -1 else 'the light curve')
    nsamples = _resolve_nsamples(meta, exptime, nsamples_short, nsamples_long)
    padding_d = padding / 24.0

    out = []
    for t, f, e in _extract_windows(time, flux, ferr, zero_epochs, periods,
                                    wbaseline, min_window):
        if t.size < min_points:
            continue
        pids = _transiting_planets(t, zero_epochs, periods, padding_d)
        if drop_empty and not pids:
            continue
        # `segment` stays at zero so that `LCDataGroup.wnids` groups the windows by
        # sector, giving one white noise parameter per sector rather than per window.
        out.append(LCData(t, f, e, passband=PASSBAND, pids=pids,
                                  instrument=INSTRUMENT, sector=int(sector), segment=0,
                                  exptime=exptime, nsamples=nsamples))
    return LCDataGroup(out)


def read_tess(datadir: Union[str, Path],
              tic: int,
              zero_epochs: Sequence[float],
              periods: Sequence[float],
              wbaseline: float,
              min_window: float = 1.0,
              type: Literal['sap', 'pdc'] = 'pdc',
              padding: float = 0.0,
              drop_empty: bool = False,
              quality_bitmask: Optional[Union[str, int]] = 'default',
              nsamples_short: int = 1,
              nsamples_long: int = 10,
              min_points: int = 1) -> LCDataGroup:
    """Read TESS light curve data into a `LCDataGroup`.

    Reads every TESS light curve file found under `datadir` and cuts each into the windows
    around its transits, keeping the sector each window came from so that every sector gets
    its own white noise parameter.

    Each window is also told which planets transit in it. A planet counts as transiting if
    its transit centre nearest the middle of the window falls inside the window's time
    bounding box, widened at both ends by `padding`. Windows of different planets that
    overlap are merged, and the merged window names both planets.

    Parameters
    ----------
    datadir
        Directory searched recursively for TESS light curve files.
    tic
        TIC number of the target.
    zero_epochs
        Zero epoch of each planet, in the same time system as the data (BJD_TDB).
    periods
        Orbital period of each planet, in days.
    wbaseline
        Width of the window kept around each transit centre, **in hours**.
    min_window
        Drop the windows spanning less than this, **in hours**.
    type
        Flux to read, 'pdc' (default) or 'sap'.
    padding
        Margin **in hours** widening a window's bounding box before the transit-centre test.
    drop_empty
        Drop the windows in which no planet transits. Defaults to `False`.
    quality_bitmask
        Quality bitmask handed to `lightkurve.read`, which drops the cadences flagged by
        the pipeline. Accepts 'none', 'default', 'hard', 'hardest', or an integer mask.
    nsamples_short, nsamples_long
        Number of supersamples used to integrate the model over the exposure, per cadence.
        Each file is classified on its own, so target-stamp and full-frame-image light
        curves of the same target mix freely in one group.
    min_points
        Drop the windows left with fewer than this many points.

    Returns
    -------
    LCDataGroup
        The transit windows of every file, ordered by time.

    Raises
    ------
    ValueError
        If `zero_epochs` and `periods` differ in length, if any period is not finite and
        positive, or if a file's flux format is not recognised.

    Notes
    -----
    Files are matched with the glob ``**/*{tic}*fits``, which is a **substring** test on the
    whole path. A short TIC number can therefore pull in a longer one that contains it, so
    give the full TIC and keep one target per directory.

    `LCDataGroup.n_planets` is one past the largest planet index actually seen, so
    it under-reports the system if the outermost planet never transits in any window. Give
    the planet count to the LPF explicitly rather than reading it off the group.
    """
    _check_ephemerides(zero_epochs, periods)

    files = sorted(Path(datadir).glob(f'**/*{tic}*fits'))
    if not files:
        warnings.warn(f'No TESS light curve files matching TIC {tic} found under {datadir}.')
        return LCDataGroup()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lcs = [lk.read(f, quality_bitmask=quality_bitmask) for f in files]

    out = []
    for lc in lcs:
        out.extend(read_lc(lc, zero_epochs, periods, wbaseline, min_window, type,
                           padding, drop_empty, nsamples_short, nsamples_long,
                           min_points).data)
    return LCDataGroup(out).sorted_by('time')
