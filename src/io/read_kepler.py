"""Kepler light curve reader returning PyTransit `LCData` objects.

A sibling of `read_tess`, differing only where the missions differ. Kepler files carry a
`QUARTER` instead of a `SECTOR`, and the quarter goes into the `sector` metadata slot so
that `LCDataGroup.wnids` gives every quarter its own white noise parameter. The flux
columns are the SPOC names, `pdcsap_flux` and `sap_flux`, with no QLP fallback.

Kepler mixes two cadences: long cadence at 29.4 minutes and short cadence at 58.8 seconds.
A long-cadence exposure smears the transit shape enough that the model has to be integrated
over it, so each file's cadence sets the supersampling of its windows: `nsamples_short`
(default 1) for short cadence and `nsamples_long` (default 10) for long cadence. The
cadence is read from the `OBSMODE` header when present, and from the exposure time
otherwise. `LCDataGroup` keeps `nsamples` per light curve, so short- and long-cadence
windows mix freely in one group::

    lcs = read_kepler('data/kepler', kic, zero_epochs, periods, wbaseline=24)
    lcs.nsamples        # -> [10, 10, ...] for long-cadence windows
    lpf = PhotoDynamicalLPF('toi1237', 2, zero_epochs, periods, lcs,
                            is_transiting=[True, True], tref=tref)
"""

import warnings

from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import lightkurve as lk

from pytransit.utils.io.lcdata import LCData, LCDataGroup

from . import _common
from ._common import (_check_ephemerides, _exposure_time, _extract_windows, _finite_mask,
                      _transiting_planets)

__all__ = ['read_kepler', 'read_lc']

# Kepler observes through a single broad optical passband, so every light curve shares one
# passband name and one instrument name.
PASSBAND = 'Kepler'
INSTRUMENT = 'Kepler'

# Flux columns to try for each type, in order of preference. Kepler products carry the
# SPOC names only.
_FLUX_COLUMNS = {'pdc': ('pdcsap_flux',),
                 'sap': ('sap_flux',)}

# Exposure times above this, in days (~14.4 min), count as long cadence when the OBSMODE
# header is missing. Kepler long cadence is 29.4 minutes, short cadence 58.8 seconds.
LONG_CADENCE_THRESHOLD = 0.01


def _resolve_nsamples(meta: dict, exptime: float,
                      nsamples_short: int, nsamples_long: int) -> int:
    """Number of supersamples for a light curve, decided from its cadence.

    Trusts the `OBSMODE` header when it says 'long cadence' or 'short cadence', and falls
    back to testing the exposure time against `LONG_CADENCE_THRESHOLD`; see
    `_common._resolve_nsamples` for the details.
    """
    return _common._resolve_nsamples(meta, exptime, nsamples_short, nsamples_long,
                                     LONG_CADENCE_THRESHOLD)


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
    """Cut a single Kepler light curve into a `LCDataGroup` of transit windows.

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
        Flux to read: 'pdc' (default) for the detrended flux, or 'sap' for the simple
        aperture photometry.
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
        short-cadence and a long-cadence light curve respectively. The cadence is read from
        the `OBSMODE` header, falling back to the exposure time against
        `LONG_CADENCE_THRESHOLD`.
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

    time, flux, ferr = _finite_mask(lc, _FLUX_COLUMNS, type, 'Kepler')
    meta = getattr(lc, 'meta', None) or {}
    quarter = meta.get('QUARTER', -1)
    exptime = _exposure_time(time, meta.get('TIMEDEL'), 1.0,
                             f'quarter {quarter}' if quarter != -1 else 'the light curve')
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
        # quarter, giving one white noise parameter per quarter rather than per window.
        out.append(LCData(t, f, e, passband=PASSBAND, pids=pids,
                          instrument=INSTRUMENT, sector=int(quarter), segment=0,
                          exptime=exptime, nsamples=nsamples))
    return LCDataGroup(out)


def read_kepler(datadir: Union[str, Path],
                kic: int,
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
    """Read Kepler light curve data into a `LCDataGroup`.

    Reads every Kepler light curve file found under `datadir` and cuts each into the
    windows around its transits, keeping the quarter each window came from so that every
    quarter gets its own white noise parameter, and setting each file's supersampling from
    its cadence: `nsamples_short` for short cadence, `nsamples_long` for long cadence.

    Each window is also told which planets transit in it. A planet counts as transiting if
    its transit centre nearest the middle of the window falls inside the window's time
    bounding box, widened at both ends by `padding`. Windows of different planets that
    overlap are merged, and the merged window names both planets.

    Parameters
    ----------
    datadir
        Directory searched recursively for Kepler light curve files.
    kic
        KIC number of the target.
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
    Files are matched with the glob ``**/kplr{kic:09d}*.fits``, the zero-padded KIC form
    of the archive filenames (``kplr004349452-2013098041711_llc.fits``), which matches the
    long- and short-cadence products alike.

    `LCDataGroup.n_planets` is one past the largest planet index actually seen, so
    it under-reports the system if the outermost planet never transits in any window. Give
    the planet count to the LPF explicitly rather than reading it off the group.
    """
    _check_ephemerides(zero_epochs, periods)

    files = sorted(Path(datadir).glob(f'**/kplr{int(kic):09d}*.fits'))
    if not files:
        warnings.warn(f'No Kepler light curve files matching KIC {kic} found under {datadir}.')
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
