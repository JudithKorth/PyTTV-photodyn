"""CHEOPS light curve reader returning PyTransit `LCData` objects.

A port of `snowleopards.read_cheops` from `snowleopards.io.LCData` onto
`pytransit.utils.io.lcdata.LCData`.

The two containers express per-light-curve metadata differently. `LCData` carries
precomputed index arrays (`nids` for the white noise blocks, `pbids` for the passbands),
while `LCData` carries the observation metadata itself (`instrument`, `sector`,
`segment`, `passband`) and lets `LCDataGroup` derive the indices. The old
`nids = [1, 2, ..., n]`, meaning "one white noise parameter per visit", therefore becomes
"give each visit its own `sector`", and `pbids = n * [1]` becomes `passband='CHEOPS'`.

The per-file readers return a single `LCData`; `read_cheops` returns the
`LCDataGroup` collecting them, which exposes the lists and arrays a `BaseLPF`
expects::

    lcs = read_cheops('data', zero_epochs, periods, type='DRP', drp_radius=25)
    lcs.pids            # -> [(0,), (), (1,), (0,), ...] one entry per visit
    lpf = BaseLPF('toi4495', passbands=lcs.passband_names, times=lcs.times,
                  fluxes=lcs.fluxes, errors=lcs.errors, pbids=lcs.pbids,
                  covariates=lcs.covariates, wnids=lcs.wnids,
                  exptimes=lcs.exptimes, nsamples=lcs.nsamples)
"""

import warnings

from pathlib import Path
from typing import Literal, Optional, Sequence, Union

from astropy.io import fits
from astropy.table import Table
from numpy import argsort, array, cos, inf, median, ndarray, ones, radians, sin

from pytransit.utils.io.lcdata import LCData, LCDataGroup

from ._common import _check_ephemerides, _exposure_time, _normalise, _transiting_planets

__all__ = ['read_cheops', 'read_cheops_drp', 'read_cheops_pipe']

# CHEOPS observes through a single broad optical passband, so every light curve shares one
# passband name and one instrument name.
PASSBAND = 'CHEOPS'
INSTRUMENT = 'CHEOPS'


def _exptime(fname: Union[str, Path], time: ndarray) -> float:
    """Exposure time in days.

    Read from the `EXPTIME` header keyword, which both the DRP and PIPE products carry in
    seconds, and falls back to the median cadence if the keyword is missing.
    """
    try:
        exptime = fits.getheader(fname, 1).get('EXPTIME')
    except OSError:
        exptime = None
    return _exposure_time(time, exptime, 1.0 / 86400.0, Path(fname).name)


def _build(fname, time, flux, ferr, covs, mask, sector, segment, pids, nsamples) -> LCData:
    """Assemble a `LCData` from the columns read out of a single visit."""
    return LCData(time[mask], flux[mask], ferr[mask], _normalise(covs)[mask],
                          passband=PASSBAND, pids=pids, instrument=INSTRUMENT,
                          sector=sector, segment=segment,
                          exptime=_exptime(fname, time), nsamples=nsamples)


def read_cheops_pipe(fname, flag_val: Optional[int] = None, sector: int = -1, segment: int = 0,
                     pids=None, nsamples: int = 1) -> LCData:
    """Read a single PIPE-reduced CHEOPS visit into a `LCData`.

    Parameters
    ----------
    fname
        Path to a PIPE `*_sa.fits` file.
    flag_val
        Keep only the points whose `FLAG` equals this value. Keeps everything if `None`.
    sector
        Visit id, stored as the `sector`. Together with `instrument` and `segment` it
        decides which light curves share a white noise parameter in `LCDataGroup`.
    segment
        Segment id within the visit.
    pids
        Indices of the planets transiting in this visit.
    nsamples
        Number of supersamples used to integrate the model over the exposure.
    """
    tb = Table.read(fname)
    time = tb['BJD_TIME'].value.astype('d')
    flux = tb['FLUX'].value.astype('d')
    fn = median(flux)
    flux = flux / fn
    ferr = tb['FLUXERR'].value.astype('d') / fn
    flag = tb['FLAG'].value.astype(int)
    roll = tb['ROLL'].value.astype('d')
    xc = tb['XC'].value.astype('d')
    yc = tb['YC'].value.astype('d')
    tf2 = tb['thermFront_2'].value.astype('d')

    m = flag == flag_val if flag_val is not None else ones(time.shape, bool)
    dx = xc - xc.mean()
    dy = yc - yc.mean()

    covs = array(
        [time, tf2, dx, dy, dx ** 2, dy ** 2, dx * dy,
         sin(roll), sin(2 * roll), sin(3 * roll),
         cos(roll), cos(2 * roll), cos(3 * roll)]).T

    return _build(fname, time, flux, ferr, covs, m, sector, segment, pids, nsamples)


def read_cheops_drp(fname, flag_val: Optional[int] = None, sector: int = -1, segment: int = 0,
                    pids=None, nsamples: int = 1) -> LCData:
    """Read a single DRP-reduced CHEOPS visit into a `LCData`.

    Parameters
    ----------
    fname
        Path to a DRP `*SCI_COR_Lightcurve*.fits` file.
    flag_val
        Keep only the points whose `STATUS` equals this value. Keeps everything if `None`.
    sector
        Visit id, stored as the `sector`. Together with `instrument` and `segment` it
        decides which light curves share a white noise parameter in `LCDataGroup`.
    segment
        Segment id within the visit.
    pids
        Indices of the planets transiting in this visit.
    nsamples
        Number of supersamples used to integrate the model over the exposure.
    """
    tb = Table.read(fname)
    time = tb['BJD_TIME'].value.astype('d')
    flux = tb['FLUX'].value.astype('d')
    fn = median(flux)
    flux = flux / fn
    ferr = tb['FLUXERR'].value.astype('d') / fn
    status = tb['STATUS'].value.astype(int)
    background = tb['BACKGROUND'].value.astype('d')
    roll = radians(tb['ROLL_ANGLE'].value.astype('d'))
    cx = tb['CENTROID_X'].value.astype('d')
    cy = tb['CENTROID_Y'].value.astype('d')

    m = status == flag_val if flag_val is not None else ones(time.shape, bool)
    dx = cx - cx.mean()
    dy = cy - cy.mean()

    covs = array(
        [time, background, dx, dy, dx ** 2, dy ** 2, dx * dy,
         sin(roll), sin(2 * roll), sin(3 * roll),
         cos(roll), cos(2 * roll), cos(3 * roll)]).T

    return _build(fname, time, flux, ferr, covs, m, sector, segment, pids, nsamples)


def read_cheops(datadir: str,
                zero_epochs: Sequence[float],
                periods: Sequence[float],
                type: Literal['DRP', 'PIPE'] = 'DRP',
                drp_radius: Optional[int] = None,
                minf: float = -inf, maxf: float = inf,
                padding: float = 0.0,
                drop_empty: bool = False,
                flag: Optional[int] = None,
                nsamples: int = 1,
                min_points: int = 1) -> LCDataGroup:
    """Read CHEOPS light curve data into a `LCDataGroup`.

    Reads every CHEOPS visit found under `datadir` and filters it by flux range. The visits
    are ordered by time, and each is given its own `sector` so that every visit gets its own
    white noise parameter, reproducing the `nids = [1, 2, ..., n]` of the `LCData` version.

    Each visit is also told which planets transit in it. A planet counts as transiting if
    its transit centre nearest the middle of the visit falls inside the visit's time
    bounding box, widened at both ends by `padding`. Because a CHEOPS visit is hours long
    against periods of days, this is at most one or two planets, and it differs from visit
    to visit.

    Parameters
    ----------
    datadir
        Directory searched recursively for CHEOPS light curve files.
    zero_epochs
        Zero epoch of each planet, in the same time system as the data (BJD_TDB).
    periods
        Orbital period of each planet, in days.
    type
        Type of CHEOPS data to read. Accepts 'DRP' (default) or 'PIPE'.
    drp_radius
        Aperture radius in pixels selecting which DRP light curve to read. Required for
        'DRP', ignored for 'PIPE'.
    minf
        Minimum flux threshold for filtering.
    maxf
        Maximum flux threshold for filtering.
    padding
        Margin **in hours** by which a visit's time bounding box is widened before testing
        whether a transit centre falls inside it. Defaults to zero, meaning the centre must
        lie within the observed span. Raise it to also catch the visits that cover only an
        ingress or an egress.
    drop_empty
        Drop the visits in which no planet transits. Defaults to `False`, which keeps them
        with an empty `pids`; they still constrain the baseline and the white noise.
    flag
        Keep only the points whose quality flag (`STATUS` for DRP, `FLAG` for PIPE) equals
        this value. Keeps everything if `None`.
    nsamples
        Number of supersamples used to integrate the model over the exposure.
    min_points
        Drop visits left with fewer than this many points after filtering. Defaults to one,
        which removes the empty light curves that would otherwise give a NaN noise estimate.

    Returns
    -------
    LCDataGroup
        The CHEOPS visits ordered by time, each a `LCData` carrying its times,
        fluxes, errors, covariates, transiting planets, and observation metadata.

    Raises
    ------
    ValueError
        If `type` is not one of 'DRP' or 'PIPE'; if `drp_radius` is missing for 'DRP'; if
        `zero_epochs` and `periods` differ in length; or if any period is not finite and
        positive.

    Notes
    -----
    `LCDataGroup.n_planets` is one past the largest planet index actually seen, so
    it under-reports the system if the outermost planet never transits in any visit. Give
    the planet count to the LPF explicitly rather than reading it off the group.
    """
    _check_ephemerides(zero_epochs, periods)

    if type == 'DRP':
        if drp_radius is None:
            raise ValueError('DRP radius must be specified.')
        files = sorted(Path(datadir).glob(f'**/*SCI_COR_*R{drp_radius:02d}*.fits'))
        read = read_cheops_drp
    elif type == 'PIPE':
        files = sorted(Path(datadir).glob('**/*_sa.fits'))
        read = read_cheops_pipe
    else:
        raise ValueError('Unknown type of Cheops light curve.')

    padding_d = padding / 24.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lcs = [read(f, flag, nsamples=nsamples) for f in files]

    # Order the visits by time. The sectors are assigned afterwards so that they run in
    # time order rather than in the order the files happened to be globbed.
    lcs = [lcs[i] for i in argsort([lc.time.mean() if lc.size else inf for lc in lcs])]

    out = []
    for lc in lcs:
        m = (lc.flux > minf) & (lc.flux < maxf)
        if m.sum() < min_points:
            continue
        # The transiting planets are worked out from the cropped times: the flux cut can
        # trim the ends of a visit, and a transit centre that sat just inside the raw span
        # may fall outside the cropped one.
        pids = _transiting_planets(lc.time[m], zero_epochs, periods, padding_d)
        if drop_empty and not pids:
            continue
        out.append(LCData(lc.time[m], lc.flux[m], lc.error[m], lc.covariates[m],
                                  passband=lc.passband, pids=pids, instrument=lc.instrument,
                                  sector=len(out), segment=lc.segment,
                                  exptime=lc.exptime, nsamples=lc.nsamples))
    return LCDataGroup(out)
