"""Helpers shared by the instrument-specific light curve readers.

The readers under `src.io` all end up doing the same things: picking a flux column,
dropping the unusable points, cutting the data into transit windows, normalising a
covariate matrix, working out which planets transit in a light curve, and settling on an
exposure time. The details differ per instrument -- CHEOPS keeps its exposure time in
seconds under `EXPTIME`, TESS and Kepler in days under `TIMEDEL`, and each mission has its
own flux column names -- so the parts that do differ are parameters rather than separate
implementations.
"""

import warnings

from typing import Sequence

from numpy import asarray, diff, isfinite, median, ndarray, ones, where, zeros
from scipy.ndimage import label

from pytransit.orbits import epoch, fold

__all__ = ['_check_ephemerides', '_exposure_time', '_extract_windows', '_finite_mask',
           '_flux_columns', '_normalise', '_resolve_nsamples', '_transiting_planets']


def _resolve_nsamples(meta: dict, exptime: float, nsamples_short: int, nsamples_long: int,
                      threshold: float) -> int:
    """Number of supersamples for a light curve, decided from its cadence.

    Parameters
    ----------
    meta
        The light curve's metadata dictionary. Kepler files carry the cadence type in
        `OBSMODE`, as the strings 'long cadence' or 'short cadence', and the header is
        trusted when it says either (compared case-insensitively -- lightkurve upper-cases
        the *keys*, not the values). TESS files carry no such keyword and simply fall
        through to the exposure-time test.
    exptime
        Exposure time in days, as returned by `_exposure_time`. Zero means the light curve
        was too short to even estimate a cadence, and disables supersampling -- `LCData`
        warns that it has no effect with a zero exposure time.
    nsamples_short, nsamples_long
        The supersampling counts to return for short and long cadence.
    threshold
        Exposure times above this, in days, count as long cadence when the metadata does
        not settle it.

    Returns
    -------
    The number of supersamples for every window cut from this light curve.
    """
    if exptime == 0.0:
        return 1
    obsmode = str(meta.get('OBSMODE', '')).lower()
    if obsmode == 'long cadence':
        return nsamples_long
    if obsmode == 'short cadence':
        return nsamples_short
    return nsamples_long if exptime > threshold else nsamples_short


def _flux_columns(lc, columns: dict, type: str, mission: str) -> tuple:
    """Flux and uncertainty columns of the requested type.

    Parameters
    ----------
    lc
        A `lightkurve.LightCurve`.
    columns
        Mapping from a type name to the flux columns accepted for it, in order of
        preference. Every reader uses the keys 'pdc' and 'sap', which the invalid-type
        error message assumes.
    type
        Key into `columns`.
    mission
        Mission name used in the unknown-format error message.

    Returns
    -------
    The flux and flux uncertainty columns, as they are stored, which may be masked arrays.

    Raises
    ------
    ValueError
        If `type` is not a key of `columns`, or if the light curve carries none of the
        columns that type accepts.
    """
    if type not in columns:
        raise ValueError(f"Invalid light curve type {type!r}, use 'sap' or 'pdc'.")
    for name in columns[type]:
        if name in lc.colnames:
            return getattr(lc, name).value, getattr(lc, f'{name}_err').value
    raise ValueError(f'Unknown {mission} light curve format.')


def _finite_mask(lc, columns: dict, type: str, mission: str) -> tuple:
    """The light curve's times, fluxes and uncertainties, with the unusable points dropped.

    Drops the points the flux column masks out, and any point whose time, flux or
    uncertainty is not finite. `LCData` rejects non-finite times outright and warns about
    non-finite fluxes, so the points have to go before the container sees them.

    `columns`, `type` and `mission` are passed through to `_flux_columns`.
    """
    time = asarray(lc.time.jd, 'd')
    flux, ferr = _flux_columns(lc, columns, type, mission)

    m = ~flux.mask if hasattr(flux, 'mask') else ones(time.size, bool)
    flux = asarray(flux, 'd')
    ferr = asarray(ferr, 'd')
    m &= isfinite(time) & isfinite(flux) & isfinite(ferr)
    return time[m], flux[m], ferr[m]


def _extract_windows(time: ndarray, flux: ndarray, ferr: ndarray,
                     zero_epochs: Sequence[float], periods: Sequence[float],
                     wbaseline: float, min_window: float = 1.0) -> list:
    """Cut a light curve into the windows around its transits.

    Every planet contributes the points within half a `wbaseline` of one of its transit
    centres. The windows of different planets are merged where they overlap, so a window is
    a contiguous run of selected points rather than a single planet's transit.

    Parameters
    ----------
    time, flux, ferr
        The light curve, with the unusable points already dropped.
    zero_epochs
        Zero epoch of each planet, in the same time system as `time`.
    periods
        Orbital period of each planet, in days.
    wbaseline
        Width of the window around each transit centre, **in hours**.
    min_window
        Drop the windows spanning less than this, **in hours**.

    Returns
    -------
    A list of ``(time, flux, error)`` tuples, one per window, in time order. The flux and
    the uncertainties of each window are divided by the window's own median flux, so the
    relative uncertainties are preserved.
    """
    if time.size < 2:
        return []

    ww = wbaseline / 24.0
    in_window = zeros(time.size, bool)
    for t0, p in zip(zero_epochs, periods):
        in_window |= abs(fold(time, p, t0)) < 0.5 * ww

    cadence = float(median(diff(time)))
    labels, nl = label(in_window)

    windows = []
    for i in range(1, nl + 1):
        m = in_window & (labels == i)
        if m.sum() * cadence > min_window / 24.0:
            fn = median(flux[m])
            windows.append((time[m], flux[m] / fn, ferr[m] / fn))
    return windows


def _check_ephemerides(zero_epochs: Sequence[float], periods: Sequence[float]) -> None:
    """Validate a set of planet ephemerides.

    Raises
    ------
    ValueError
        If the two sequences differ in length, or if any period is not finite and positive.
    """
    if len(zero_epochs) != len(periods):
        raise ValueError(f'Got {len(zero_epochs)} zero epochs but {len(periods)} periods.')
    if not all(isfinite(pr) and pr > 0.0 for pr in periods):
        raise ValueError(f'The periods must all be finite and positive, got {list(periods)}.')


def _normalise(covs: ndarray) -> ndarray:
    """Zero-mean, unit-variance normalisation of every covariate column.

    Columns with no variance are centred but not scaled: dividing them by a zero standard
    deviation would turn them into NaNs, which `LCData` would happily accept and
    the baseline fit would then silently poison.

    The covariate matrices carry no constant column. `LSTSQBaseline` prepends its own
    intercept in `init_data`, and its `_check_conditioning` rejects the resulting design
    matrix outright if we hand it a second one.
    """
    s = covs.std(0)
    return (covs - covs.mean(0)) / where(s > 0.0, s, 1.0)


def _exposure_time(time: ndarray, value=None, scale: float = 1.0, source: str = '') -> float:
    """Exposure time in days, from a header value with a median-cadence fallback.

    Parameters
    ----------
    time
        Mid-exposure times of the light curve.
    value
        Raw header value, or `None` if the keyword was missing or unreadable.
    scale
        Factor converting `value` into days: ``1 / 86400`` for a value in seconds, ``1.0``
        for one already in days.
    source
        Name used in the warning emitted when the fallback is taken.

    Returns
    -------
    The exposure time in days, or zero for a light curve too short to estimate a cadence.
    """
    if value is not None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = None
        if v is not None and isfinite(v) and v > 0.0:
            return v * scale
    warnings.warn(f"No usable exposure time in {source}, using the median cadence instead.")
    return float(median(diff(time))) if time.size > 1 else 0.0


def _transiting_planets(time: ndarray, zero_epochs: Sequence[float], periods: Sequence[float],
                        padding: float) -> tuple:
    """Indices of the planets whose nearest transit falls inside a light curve.

    `pytransit.orbits.epoch` rounds to the nearest integer, so
    ``t0 + epoch(tmean, t0, p) * p`` is the transit centre closest to the middle of the
    light curve. That is the only centre that can plausibly fall inside a light curve that
    is hours long against periods of days. A planet counts as transiting if its centre lies
    within the light curve's time bounding box widened by `padding`.

    Parameters
    ----------
    time
        Mid-exposure times of the light curve, after any filtering.
    zero_epochs
        Zero epoch of each planet, in the same time system as `time`.
    periods
        Orbital period of each planet, in days.
    padding
        Margin **in days** by which the bounding box is widened at both ends.

    Returns
    -------
    The indices of the transiting planets, in ascending order. Empty if none transit.
    """
    if time.size == 0:
        return ()
    tmean, tmin, tmax = time.mean(), time.min() - padding, time.max() + padding
    return tuple(i for i, (t0, p) in enumerate(zip(zero_epochs, periods))
                 if tmin <= t0 + epoch(tmean, t0, p) * p <= tmax)
