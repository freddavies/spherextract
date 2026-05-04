#!/usr/bin/env python
"""
spherextract.py

Download SPHEREx cutouts from IRSA and extract photometry via analogy to
optimal extraction (Horne 1986) using the official SPHEREx PSF model.

Essentially a merger of spherex-tools/get_cutouts_spherex.py
and spiff/single_fit.py, aided by Claude Sonnet 4.6

The PSF model is assumed to be correct with a perfect position. 
The flux is estimated by the inverse-variance-weighted matched filter:

    f_hat = sum(P * ivar * D) / sum(P^2 * ivar)
    var(f_hat) = 1 / sum(P^2 * ivar)

where P is the (flux-conserved) downsampled PSF profile,
D is the background-subtracted data, and ivar = 1/variance.

Outlier rejection iterates the above, masking pixels where

    |D - f_hat * P| > kappa * sigma

until convergence or a maximum number of iterations.

Usage examples
--------------
# Single target:
python spherextract.py --ra 129.1827 --dec 0.914806 --name J0836p0054

# Read targets from file (name ra dec columns):
python spherextract.py --input targets.txt

# Skip download if cutouts already exist:
python spherextract.py --input targets.txt --fits-dir my_cutouts/ --skip-download

# Save results to specific directory:
python spherextract.py --input targets.txt --results-dir my_results/

# Tune extraction:
python spherextract.py --ra 129.1827 --dec 0.914806 \\
    --name J0836p0054 --fit-radius 4.0 --kappa 4.0 --max-iter 10 \\
    --cutout-size 0.1 --search-radius 5 \\
    --fits-dir cutouts_J0836/ --results-dir results_J0836/
"""
from __future__ import print_function, division

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path
from typing import Dict, List, Optional, Any
from http.client import IncompleteRead
from urllib.error import HTTPError
from urllib.request import urlopen

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from scipy import ndimage

from IPython import embed


# ---------------------------------------------------------------------------
# Data class for extraction results
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """All outputs from optimal extraction of one SPHEREx cutout file."""

    # Identification
    name: str
    fits_path: str
    input_ra_deg: float
    input_dec_deg: float

    # File metadata (from FITS header)
    obsid: Optional[str]
    detector_id: Optional[int]
    bandpass: Optional[str]
    expid: Optional[int]
    mjd_avg: Optional[float]
    psf_index: Optional[int]
    omega_sr: Optional[float]
    px_scale_arcsec: Optional[float]

    # Spectral WCS
    wv_um: Optional[float]
    wv_width_um: Optional[float]

    # Quality flags
    near_detector_edge: bool
    n_pix_total: int
    n_pix_flagged: int
    n_pix_outlier: int
    n_pix_used: int
    n_iter: int
    converged: bool

    # Background
    bkg_MJysr: Optional[float]
    bkg_npix: int

    # Optimal extraction results
    opt_flux_MJysr: Optional[float]       # fitted amplitude in MJy/sr
    opt_flux_MJysr_err: Optional[float]
    opt_flux_uJy: Optional[float]         # integrated flux in µJy
    opt_flux_uJy_err: Optional[float]
    opt_snr: Optional[float]
    opt_chi2: Optional[float]
    opt_dof: Optional[int]

    # Target pixel position
    xpix_fulldet: Optional[float]
    ypix_fulldet: Optional[float]
    xpix_cutout: Optional[float]
    ypix_cutout: Optional[float]


# ---------------------------------------------------------------------------
# Helpers: IRSA download
# ---------------------------------------------------------------------------

def _query_spherex(ra, dec, radius_arcsec=3.0):
    """Return a list of SPHEREx access URLs covering (ra, dec)."""
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    try:
        rslt = Irsa.query_sia(
            pos=(coord.ra, coord.dec, radius_arcsec * u.arcsec),
            instrument="SPHEREx",
        )
    except Exception as exc:
        print(f"  [WARN] IRSA query failed: {exc}")
        return []
    if len(rslt) == 0:
        return []
    return list(rslt["access_url"])


def _build_cutout_url(access_url, ra, dec, cutout_size_deg):
    """Convert an IRSA access URL to a cutout service URL."""
    base = "https://irsa.ipac.caltech.edu/"
    url = access_url
    if url.startswith(base):
        data_path = url[len(base):]
    elif url.startswith("/"):
        data_path = url[1:]
    else:
        data_path = url
    # Strip leading ibe/data/ if present
    for prefix in ("ibe/data/",):
        if data_path.startswith(prefix):
            data_path = data_path[len(prefix):]
            break
    return (
        f"{base}ibe/cutout?ra={ra}&dec={dec}"
        f"&size={cutout_size_deg}&path={data_path}"
    )

def _download_file(url, out_path, max_retries = 10, retry_delay = 2.0):
    """
    Download *url* to *out_path*. Returns True on success.

    On failure the download is retried up to *max_retries* times, with a
    pause of *retry_delay* seconds between attempts.
    """
    import time

    for attempt in range(1, max_retries + 1):
        try:
            response = urlopen(url)
            data = response.read()

            if len(data) < 100:
                snippet = data.decode("utf-8", errors="ignore")
                if any(k in snippet.lower() for k in ("error", "not found", "404")):
                    # Treat as a hard failure — no point retrying an error page
                    print(f"  [WARN] Response looks like an error page: {snippet[:120]}")
                    return False

            with open(out_path, "wb") as fh:
                fh.write(data)
            return True

        except (IncompleteRead, HTTPError, OSError) as exc:
            if attempt < max_retries:
                print(
                    f"  [WARN] Download attempt {attempt}/{max_retries} failed: {exc}. "
                    f"Retrying in {retry_delay:.0f}s..."
                )
                time.sleep(retry_delay)
            else:
                print(
                    f"  [ERROR] Download failed after {max_retries} attempt(s): {exc}"
                )
                return False

    return False  # unreachable, but satisfies type checkers

from concurrent.futures import ThreadPoolExecutor, as_completed

def download_cutouts(name, ra, dec, output_dir, search_radius_arcsec = 3.0,
                     cutout_size_deg = 0.05, overwrite = False,
                     max_workers = 8, max_retries = 10, retry_delay = 2.0):
    """
    Query IRSA for SPHEREx files covering (ra, dec) and download cutouts
    in parallel.

    Returns a list of local file paths that were successfully downloaded
    (or already existed).

    Parameters
    ----------
    name                 : source label used as a filename prefix
    ra, dec              : target position in decimal degrees
    output_dir           : directory in which to save cutout FITS files
    search_radius_arcsec : IRSA cone-search radius in arcseconds
    cutout_size_deg      : side length of the requested cutout in degrees
    overwrite            : re-download files that already exist on disk
    max_workers          : number of simultaneous download threads
    max_retries          : passed through to _download_file
    retry_delay          : seconds between retries, passed to _download_file
    """
    os.makedirs(output_dir, exist_ok=True)

    access_urls = _query_spherex(ra, dec, search_radius_arcsec)
    if not access_urls:
        print(f"  No SPHEREx data found for {name}.")
        return []

    print(f"  Found {len(access_urls)} file(s) for {name}; "
          f"downloading with up to {max_workers} parallel thread(s).")

    # ----------------------------------------------------------------
    # Build the work list: (cutout_url, local_path) for every file
    # that still needs to be downloaded.
    # ----------------------------------------------------------------
    work: List[tuple] = []
    already_have: List[str] = []

    for url in access_urls:
        orig_name = url.rstrip("/").split("/")[-1].split("?")[0]
        stem, ext = os.path.splitext(orig_name)
        out_name  = f"{name}_cutout_{stem}{ext}"
        cutout_dir= os.path.join(output_dir, name)
        out_path  = os.path.join(cutout_dir, out_name)
        os.makedirs(cutout_dir, exist_ok=True)
        if os.path.exists(out_path) and not overwrite:
            print(f"  Skipping (exists): {out_name}")
            already_have.append(out_path)
        else:
            cutout_url = _build_cutout_url(url, ra, dec, cutout_size_deg)
            work.append((cutout_url, out_path, out_name))

    if not work:
        return already_have

    # ----------------------------------------------------------------
    # Download in parallel.
    # ----------------------------------------------------------------
    local_paths: List[str] = list(already_have)  # seed with files we already had

    # Each future maps back to its output path so we can report results
    # in a human-readable order regardless of completion order.
    future_to_meta: Dict = {}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(work))) as pool:
        for cutout_url, out_path, out_name in work:
            future = pool.submit(
                _download_file,
                cutout_url,
                out_path,
                max_retries,
                retry_delay,
            )
            future_to_meta[future] = (out_path, out_name)

        for future in as_completed(future_to_meta):
            out_path, out_name = future_to_meta[future]
            try:
                success = future.result()
            except Exception as exc:
                # Unexpected exception inside _download_file — treat as failure
                print(f"  [ERROR] Unexpected error downloading {out_name}: {exc}")
                success = False

            if success:
                print(f"  Downloaded: {out_name}")
                local_paths.append(out_path)
            else:
                print(f"  [WARN] Could not download {out_name} (skipping).")

    return local_paths

# ---------------------------------------------------------------------------
# Helpers: spectral WCS (lookup-table interpolation)
# ---------------------------------------------------------------------------

def _load_wcswave(hdul):
    """
    Load the WCS-WAVE lookup table from extension 6.

    Returns a dict with keys 'X_grid', 'Y_grid', 'V_lam', 'V_wid',
    or an empty dict if the extension is missing / malformed.
    """
    try:
        tab = hdul[6].data
        X_grid = np.asarray(tab["X"][0], dtype=float).ravel()
        Y_grid = np.asarray(tab["Y"][0], dtype=float).ravel()
        A = np.asarray(tab["VALUES"][0])

        # Identify the two-layer axis
        if A.ndim == 3:
            if A.shape[-1] == 2:
                Vlam_raw, Vwid_raw = A[..., 0], A[..., 1]
            elif A.shape[0] == 2:
                Vlam_raw, Vwid_raw = A[0], A[1]
            else:
                return {}
        else:
            return {}

        def _orient(V):
            V = np.squeeze(np.asarray(V, dtype=float))
            if V.shape == (len(Y_grid), len(X_grid)):
                return V
            if V.shape == (len(X_grid), len(Y_grid)):
                return V.T
            return V  # best-effort

        return dict(
            X_grid=X_grid,
            Y_grid=Y_grid,
            V_lam=_orient(Vlam_raw),
            V_wid=_orient(Vwid_raw),
        )
    except Exception:
        return {}


def _wcswave_eval(x_fulldet, y_fulldet, ww):
    """
    Evaluate the wavelength/width lookup table at a full-detector pixel.

    Parameters
    ----------
    x_fulldet, y_fulldet : 0-based pixel coordinates in the *full detector*
        array (i.e. as if the entire detector image were in memory).
        These are obtained by asking the cutout's WCS for the sky position
        and then asking the full-image WCS for the corresponding pixel.
    ww : dict returned by _load_wcswave

    Returns
    -------
    (lambda_um, width_um) — both NaN if the table is unavailable.

    """
    if not ww:
        return np.nan, np.nan

    # Convert 0-based astropy pixel → 1-based FITS pixel used by the table,
    # adding the sub-array offset so the coordinate is in the focal-plane frame
    # that the lookup table was built against.
    x_raw = float(x_fulldet) + 1.0
    y_raw = float(y_fulldet) + 1.0

    Xg, Yg = ww["X_grid"], ww["Y_grid"]
    x_raw = np.clip(x_raw, Xg[0], Xg[-1])
    y_raw = np.clip(y_raw, Yg[0], Yg[-1])

    i = int(np.clip(np.searchsorted(Xg, x_raw) - 1, 0, len(Xg) - 2))
    j = int(np.clip(np.searchsorted(Yg, y_raw) - 1, 0, len(Yg) - 2))

    tx = (x_raw - Xg[i]) / (Xg[i + 1] - Xg[i]) if Xg[i + 1] != Xg[i] else 0.0
    ty = (y_raw - Yg[j]) / (Yg[j + 1] - Yg[j]) if Yg[j + 1] != Yg[j] else 0.0

    def _bilerp(V):
        return (
            (1 - tx) * (1 - ty) * V[j,     i    ]
            + tx       * (1 - ty) * V[j,     i + 1]
            + (1 - tx) * ty       * V[j + 1, i    ]
            + tx       * ty       * V[j + 1, i + 1]
        )

    return float(_bilerp(ww["V_lam"])), float(_bilerp(ww["V_wid"]))


# ---------------------------------------------------------------------------
# Helpers: PSF downsampling
# ---------------------------------------------------------------------------

def _make_psf_detgrid(psf_img, oversamp, cutout_shape, xcut, ycut):
    """
    Place the high-resolution PSF model on the detector pixel grid.

    Strategy
    --------
    1. Crop the PSF to a square whose size is divisible by *oversamp*.
    2. Embed it in a high-res canvas whose size matches the cutout × oversamp.
    3. Apply a subpixel shift (in high-res units) so the PSF centre lands at
       (xcut, ycut) in the detector cutout frame.
    4. Block-sum (flux-conserving) to the detector grid.

    Parameters
    ----------
    psf_img   : high-resolution PSF array (shape ny_hr × nx_hr)
    oversamp  : oversampling factor
    cutout_shape : (H, W) of the detector cutout
    xcut, ycut   : target pixel position within the cutout (0-based)

    Returns
    -------
    det : (H, W) PSF evaluated on the detector grid
    """
    H, W = cutout_shape

    # 1. Square crop divisible by oversamp
    cy_hr = psf_img.shape[0] // 2
    cx_hr = psf_img.shape[1] // 2
    m = (min(psf_img.shape) // oversamp) * oversamp
    hr = psf_img[cy_hr - m // 2: cy_hr + m // 2,
                 cx_hr - m // 2: cx_hr + m // 2]

    # 2. High-res canvas
    Hr, Wr = H * oversamp, W * oversamp
    canvas = np.zeros((Hr, Wr), dtype=hr.dtype)
    # Centre of canvas in high-res pixels
    cyc = Hr // 2
    cxc = Wr // 2
    y0c = cyc - m // 2
    x0c = cxc - m // 2
    # Clip to canvas bounds
    ys0 = max(0, -y0c)
    xs0 = max(0, -x0c)
    ye1 = m - max(0, y0c + m - Hr)
    xe1 = m - max(0, x0c + m - Wr)
    canvas[max(0, y0c): min(Hr, y0c + m),
           max(0, x0c): min(Wr, x0c + m)] = hr[ys0:ye1, xs0:xe1]

    # 3. Subpixel shift: move PSF centre from canvas centre to (xcut, ycut)
    dX_det = xcut - (W - 1) / 2.0
    dY_det = ycut - (H - 1) / 2.0
    shifted = ndimage.shift(canvas, shift=(dY_det * oversamp, dX_det * oversamp),
                            order=3, mode="constant", prefilter=True)

    # 4. Flux-conserving block-sum onto detector pixels
    det = shifted.reshape(H, oversamp, W, oversamp).sum(axis=(1, 3))
    return det


# ---------------------------------------------------------------------------
# SPHEREx flag-bit definitions
# ---------------------------------------------------------------------------

_MP = {
    "TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2,
    "NONFUNC": 6, "DICHROIC": 7, "MISSING_DATA": 9,
    "HOT": 10, "COLD": 11, "FULLSAMPLE": 12,
    "PHANMISS": 14, "NONLINEAR": 15, "PERSIST": 17,
    "OUTLIER": 19, "SOURCE": 21,
}

def _bit(n: int) -> int:
    return 1 << int(n)

# Pixels to exclude from the photometric fit
_BAD_BITS = (
    _bit(_MP["SUR_ERROR"])
    | _bit(_MP["NONFUNC"])
    | _bit(_MP["MISSING_DATA"])
    | _bit(_MP["HOT"])
    | _bit(_MP["COLD"])
    | _bit(_MP["NONLINEAR"])
    | _bit(_MP["PERSIST"])
)

# Additional bits to exclude when estimating the background
_BAD_BITS_BCG = (
    _BAD_BITS
    | _bit(_MP["OVERFLOW"])
    | _bit(_MP["SOURCE"])
    | _bit(_MP["OUTLIER"])
    | _bit(_MP["TRANSIENT"])
)


# ---------------------------------------------------------------------------
# Core: optimal extraction from a single FITS file
# ---------------------------------------------------------------------------

def optimal_extract(fits_path, name, ra, dec, extract_size = (21, 21), fit_radius_px = 3.0,
                    kappa = 4.0, max_iter = 10, debug = False, show_figs = False,
                    save_figs = False, results_dir = None, no_masking = False):
    """
    Run 2D optimal extraction on one SPHEREx cutout FITS file.

    The method
    ----------
    Given a fixed (assumed-correct) PSF model P and data D with pixel
    variance σ²:

        f_hat  = Σ_i [P_i σ_i⁻² D_i] / Σ_i [P_i² σ_i⁻²]
        var(f) = 1 / Σ_i [P_i² σ_i⁻²]

    Outlier rejection is iterated:
        flag pixel i if |D_i − f_hat P_i| > kappa σ_i
    until convergence or *max_iter* iterations.

    Parameters
    ----------
    fits_path     : local path to the SPHEREx cutout FITS
    name          : source label (used in output filenames)
    ra, dec       : target position (degrees)
    extract_size  : (ny, nx) cutout to extract from FITS in detector pixels
    fit_radius_px : pixels beyond this radius (from target) are excluded
    kappa         : outlier rejection threshold in units of pixel σ
    max_iter      : maximum outlier-rejection iterations
    debug         : print diagnostic information
    show_figs     : display matplotlib figures interactively
    save_figs     : write figures
    results_dir   : base directory for saved figures
    no_masking    : ignore flag-based masking in the fit (not background)

    Returns
    -------
    ExtractionResult dataclass
    """
    def dprint(*a, **kw):
        if debug:
            print(*a, **kw)

    # ---- helpers for figures ----
    def _save_or_show(fig, stub):
        import matplotlib.pyplot as plt
        if show_figs:
            plt.show()
        elif save_figs and results_dir:
            figs_dir = os.path.join(results_dir, f"{name}_figs")
            Path(figs_dir).mkdir(parents=True, exist_ok=True)
            obs = hdr["OBSID"]
            det = hdr["DETECTOR"]
            fig.savefig(Path(figs_dir) / f"{name}_{obs}D{det}_{stub}.png", dpi=130)
        plt.close(fig)

    def _nan_result(**overrides):
        """Return a fully-NaN ExtractionResult for early-abort cases."""
        base = dict(
            name=name, fits_path=fits_path,
            input_ra_deg=float(ra), input_dec_deg=float(dec),
            obsid=None, detector_id=None, bandpass=None, expid=None,
            mjd_avg=None, psf_index=None, omega_sr=None,
            px_scale_arcsec=None, wv_um=None, wv_width_um=None,
            near_detector_edge=False,
            n_pix_total=0, n_pix_flagged=0, n_pix_outlier=0,
            n_pix_used=0, n_iter=0, converged=False,
            bkg_MJysr=None, bkg_npix=0,
            opt_flux_MJysr=None, opt_flux_MJysr_err=None,
            opt_flux_uJy=None, opt_flux_uJy_err=None,
            opt_snr=None, opt_chi2=None, opt_dof=None,
            xpix_fulldet=None, ypix_fulldet=None,
            xpix_cutout=None, ypix_cutout=None,
        )
        base.update(overrides)
        return ExtractionResult(**base)

    # ================================================================
    # 1. Load FITS
    # ================================================================
    try:
        hdul = fits.open(fits_path)
    except Exception as exc:
        print(f"  [ERROR] Cannot open {fits_path}: {exc}")
        return _nan_result()

    hdr = hdul[1].header

    img        = hdul[1].data   # science image [MJy/sr]
    flags      = hdul[2].data   # flag map
    var        = hdul[3].data   # variance [MJy/sr]²
    psf_cube   = hdul[5].data   # PSF cube (n_zones, ny_hr, nx_hr)
    hdr_psf    = hdul[5].header

    wcs_img  = WCS(hdr)
    oversamp = hdr_psf["OVERSAMP"]
    cdelt    = hdr_psf["CDELT1"]   # arcsec per high-res px
    px_arcsec = cdelt * oversamp   # arcsec per detector px

    omega_arcsec2 = hdr["HIERARCH OMEGA_MEDIAN"]
    arcsec2_to_sr = (np.pi / (180.0 * 3600.0)) ** 2
    omega_sr = omega_arcsec2 * arcsec2_to_sr

    detector_id = hdr["DETECTOR"]
    det_id_int  = int(detector_id)
    bandpass    = f"D{det_id_int}"
    obsid       = hdr["OBSID"]
    expid_val   = hdr["EXPIDN"]
    mjd_avg_val = hdr["MJD-AVG"]

    det_origin_x = -1*float(hdr["CRPIX1A"])
    det_origin_y = -1*float(hdr["CRPIX2A"])
    
    det_w        = img.shape[1]
    det_h        = img.shape[0]

    # Spectral WCS lookup table
    ww = _load_wcswave(hdul)

    dprint(f"Opened {fits_path}")
    dprint(f"  Image shape: {img.shape}, PSF cube: {psf_cube.shape}")
    dprint(f"  oversamp={oversamp}, px_arcsec={px_arcsec:.3f}, omega_sr={omega_sr:.3e}")

    # ================================================================
    # 2. Target pixel position
    #
    # We need two distinct pixel coordinates for the target:
    #
    #  (a) xcut, ycut  — position within the cutout array (H × W).
    #      Used for PSF placement and the extraction itself.
    #      Derived from the *cutout's own WCS* so it is correct
    #      regardless of how the cutout was produced.
    #
    #  (b) xpix_fulldet, ypix_fulldet  — position in the original
    #      full-detector array.  Used only for PSF zone selection and
    #      spectral WCS evaluation, both of which were calibrated
    #      against the full-detector coordinate system.
    #      Derived from the *full-image WCS* loaded from the header.
    #
    # When the file was delivered by the IRSA cutout service its header
    # carries an updated CRPIX, so the file's own WCS IS the cutout WCS.
    # The full-detector coordinate is recovered by asking that same WCS
    # for the pixel that corresponds to the target sky position and then
    # subtracting the sub-array offset stored in CRPIX1A / CRPIX1A.
    # ================================================================
    sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # (a) Cutout pixel coordinate — always from the file's own WCS
    xcut, ycut = wcs_img.world_to_pixel(sky)

    # (b) Full-detector pixel coordinate.
    # CRPIX1A / CRPIX1A record how many detector pixels separate the
    # [0, 0] corner of this file's array from the [0, 0] corner of the
    # full detector.  Subtracting them converts the cutout-local pixel back
    # to the full-detector frame (still 0-based).
    xpix_fulldet = float(xcut) + det_origin_x
    ypix_fulldet = float(ycut) + det_origin_y

    dprint(f"  Target cutout pixel : x={xcut:.3f}, y={ycut:.3f}")
    dprint(f"  Target full-det pixel: x={xpix_fulldet:.3f}, y={ypix_fulldet:.3f}")

    h_img, w_img = img.shape
    near_edge = bool(
        xcut < 10 or ycut < 10
        or (w_img - xcut) < 10
        or (h_img - ycut) < 10
    )

    # ================================================================
    # 3. Cutout
    # ================================================================
    try:
        cut_img   = Cutout2D(img,   (xcut, ycut), extract_size, wcs=wcs_img,
                             mode="partial", fill_value=np.nan)
        cut_flags = Cutout2D(flags, (xcut, ycut), extract_size, wcs=wcs_img,
                             mode="partial", fill_value=0).data
        cut_var   = Cutout2D(var,   (xcut, ycut), extract_size, wcs=wcs_img,
                             mode="partial", fill_value=np.nan).data
    except NoOverlapError:
        print("  [WARN] Target outside image footprint; skipping.")
        return _nan_result(near_detector_edge=near_edge)

    # Re-derive the target position within the Cutout2D sub-array using
    # the cutout's own (updated) WCS.  This absorbs any rounding that
    # Cutout2D applied when choosing the sub-array centre.
    wcs_cut = cut_img.wcs
    xcut, ycut = wcs_cut.world_to_pixel(sky)

    data_raw = cut_img.data.copy()
    H, W = data_raw.shape
    dprint(f"  Cutout shape: {H}×{W}, target at xcut={xcut:.3f}, ycut={ycut:.3f}")
    
    # ================================================================
    # 4. Background subtraction
    # ================================================================
    mask_bcg = (cut_flags.astype(np.uint32) & _BAD_BITS_BCG) != 0
    good_bcg = (~mask_bcg) & np.isfinite(data_raw)
    bkg_npix = int(np.sum(good_bcg))

    if bkg_npix >= 3:
        bkg = float(np.median(data_raw[good_bcg]))
    else:
        # fallback: ignore flags, use any finite pixel
        good_any = np.isfinite(data_raw)
        bkg = float(np.median(data_raw[good_any])) if np.any(good_any) else 0.0
        bkg_npix = int(np.sum(good_any))
        dprint("  [WARN] Few good BCG pixels; using unmasked background estimate.")

    data = data_raw - bkg
    dprint(f"  Background: {bkg:.4g} MJy/sr (from {bkg_npix} pixels)")

    # ================================================================
    # 5. Select PSF zone
    # ================================================================
    xctr_items = sorted(
        [(int(k.split("_")[1]), hdr_psf[k])
         for k in hdr_psf if k.startswith("XCTR_")]
    )
    yctr_items = sorted(
        [(int(k.split("_")[1]), hdr_psf[k])
         for k in hdr_psf if k.startswith("YCTR_")]
    )
    nzone = min(len(xctr_items), len(yctr_items))
    if nzone == 0:
        # No zone info: just use the single PSF
        idx_psf = 0
    else:
        xctrs = np.array([v for _, v in xctr_items[:nzone]])
        yctrs = np.array([v for _, v in yctr_items[:nzone]])
        dists = np.hypot(xctrs - xpix_fulldet, yctrs - ypix_fulldet)
        idx_psf = int(np.argmin(dists))

    psf_hr = psf_cube[idx_psf]
    dprint(f"  PSF zone index: {idx_psf}/{psf_cube.shape[0]-1}")

    # ================================================================
    # 6. Build detector-grid PSF (block-sum from oversampled model)
    # ================================================================
    P = _make_psf_detgrid(psf_hr, oversamp, (H, W), xcut, ycut)
    psf_sum = float(np.nansum(P))
    dprint(f"  PSF detector-grid sum: {psf_sum:.6g} (≈1 if unit-normalised)")

    # ================================================================
    # 7. Build pixel masks and weights
    # ================================================================
    YY, XX = np.indices((H, W))
    r2 = (XX - xcut) ** 2 + (YY - ycut) ** 2
    radmask = r2 <= fit_radius_px ** 2

    if no_masking:
        flag_mask = np.zeros((H, W), dtype=bool)
        dprint("  [INFO] --no-masking: flag-based masking disabled in fit.")
    else:
        flag_mask = (cut_flags.astype(np.uint32) & _BAD_BITS) != 0

    ivar = np.where((cut_var > 0) & np.isfinite(cut_var),1.0/cut_var,0.0)

    # Base good-pixel mask (will be updated per iteration)
    base_good = radmask & (~flag_mask) & np.isfinite(data) & (ivar > 0)

    if not np.any(base_good):
        # Auto-fallback: try without flag masking
        dprint("  [WARN] No usable pixels with flag masking; retrying without flags.")
        flag_mask = np.zeros((H, W), dtype=bool)
        base_good = radmask & np.isfinite(data) & (ivar > 0)

    n_total = int(np.sum(radmask))
    n_flagged = int(np.sum(flag_mask & radmask))

    # Zero out NaN/bad pixels so they don't corrupt array ops
    data_safe = np.where(np.isfinite(data), data, 0.0)

    # ================================================================
    # 8. Iterative optimal extraction with outlier rejection
    # ================================================================
    #
    # Algorithm (Horne 1986 §3, adapted to 2D):
    #
    #   Step A: Estimate flux using current good-pixel mask
    #       f_hat = Σ_i [P_i ivar_i D_i] / Σ_i [P_i² ivar_i]
    #
    #   Step B: Compute residuals and flag outliers
    #       outlier_i ← |D_i − f_hat P_i| > kappa * sigma_i
    #
    #   Repeat until no new outliers are flagged or max_iter reached.
    #
    good = base_good.copy()
    outlier_mask = np.zeros((H, W), dtype=bool)
    f_hat = 0.0
    var_f  = np.nan
    converged = False
    n_iter = 0

    for iteration in range(max_iter + 1):
        n_iter = iteration

        # --- Step A: linear optimal estimator ---
        num = np.sum(P[good] * ivar[good] * data_safe[good])
        den = np.sum(P[good] ** 2 * ivar[good])

        if den <= 0 or not np.isfinite(num) or not np.isfinite(den):
            dprint(f"  [WARN] Degenerate system at iteration {iteration}; stopping.")
            break

        f_hat = num / den
        var_f  = 1.0 / den

        dprint(
            f"  iter {iteration}: f_hat={f_hat:.5g} MJy/sr, "
            f"σ={np.sqrt(var_f):.3g}, good_px={int(np.sum(good))}"
        )

        if iteration == max_iter:
            converged = True   # reached limit; accept current solution
            break

        # --- Step B: residual-based outlier detection ---
        model_px = f_hat * P
        resid     = data_safe - model_px          # (H, W)
        sigma_px  = np.sqrt(np.where(cut_var > 0, cut_var, np.inf))
        outlier_new = (np.abs(resid) > kappa * sigma_px) & radmask

        # Only mark pixels as outliers if they were in the fit
        newly_flagged = outlier_new & good & ~outlier_mask

        if not np.any(newly_flagged):
            converged = True
            dprint(f"  Converged after {iteration} iteration(s).")
            break

        outlier_mask |= newly_flagged
        good = base_good & ~outlier_mask

        if not np.any(good):
            dprint("  [WARN] All pixels rejected by outlier masking; reverting.")
            outlier_mask[:] = False
            good = base_good.copy()
            f_hat = num / den   # keep last valid estimate
            converged = False
            break

    n_outlier = int(np.sum(outlier_mask & radmask))
    n_used    = int(np.sum(good))

    # ================================================================
    # 9. Chi² of the final model
    # ================================================================
    model_final = f_hat * P
    resid_final = data_safe - model_final
    chi2_val = float(np.sum((resid_final[good] ** 2) * ivar[good]))
    dof_val  = max(n_used - 1, 1)    # 1 free parameter (flux)

    dprint(f"  chi²={chi2_val:.3f}, dof={dof_val}, chi²/dof={chi2_val/dof_val:.3f}")

    # ================================================================
    # 10. Convert amplitude to integrated flux [µJy]
    # ================================================================
    # f_hat is the surface-brightness amplitude [MJy/sr].
    # Integrated flux = f_hat × Ω_pix × Σ P_det
    # (Σ P_det ≈ 1 for a unit-normalised PSF)
    f_hat_err = float(np.sqrt(var_f)) if np.isfinite(var_f) else np.nan

    if np.isfinite(omega_sr) and psf_sum > 0:
        flux_uJy     = f_hat * omega_sr * psf_sum * 1e12
        flux_uJy_err = f_hat_err * omega_sr * psf_sum * 1e12
    else:
        flux_uJy = flux_uJy_err = np.nan

    snr = f_hat / f_hat_err if (np.isfinite(f_hat_err) and f_hat_err > 0) else np.nan

    dprint(
        f"  Optimal flux: {f_hat:.4g} ± {f_hat_err:.3g} MJy/sr  "
        f"≈ {flux_uJy:.4g} ± {flux_uJy_err:.3g} µJy  S/N={snr:.2f}"
    )

    # ================================================================
    # 11. Spectral WCS at target position
    # ================================================================
    wv_um, wv_width_um = _wcswave_eval(xpix_fulldet, ypix_fulldet, ww)
    dprint(f"  Spectral WCS: λ={wv_um:.5f} µm, Δλ={wv_width_um:.5f} µm")

    # ================================================================
    # 12. Optional diagnostic figure
    # ================================================================
    if show_figs or save_figs:
        _plot_extraction(
            data=data,
            model=model_final,
            psf=P,
            ivar=ivar,
            good=good,
            outlier_mask=outlier_mask,
            radmask=radmask,
            xcut=xcut, ycut=ycut,
            fit_radius_px=fit_radius_px,
            f_hat=f_hat, f_hat_err=f_hat_err,
            flux_uJy=flux_uJy, snr=snr,
            chi2=chi2_val, dof=dof_val,
            name=name,
            save_or_show_fn=_save_or_show,
        )

    hdul.close()

    return ExtractionResult(
        name=name,
        fits_path=fits_path,
        input_ra_deg=float(ra),
        input_dec_deg=float(dec),
        obsid=obsid,
        detector_id=det_id_int,
        bandpass=bandpass,
        expid=int(expid_val),
        mjd_avg=float(mjd_avg_val),
        psf_index=int(idx_psf),
        omega_sr=float(omega_sr),
        px_scale_arcsec=float(px_arcsec),
        wv_um=float(wv_um),
        wv_width_um=float(wv_width_um),
        near_detector_edge=near_edge,
        n_pix_total=n_total,
        n_pix_flagged=n_flagged,
        n_pix_outlier=n_outlier,
        n_pix_used=n_used,
        n_iter=n_iter,
        converged=converged,
        bkg_MJysr=bkg,
        bkg_npix=bkg_npix,
        opt_flux_MJysr=float(f_hat),
        opt_flux_MJysr_err=float(f_hat_err),
        opt_flux_uJy=float(flux_uJy),
        opt_flux_uJy_err=float(flux_uJy_err),
        opt_snr=float(snr),
        opt_chi2=float(chi2_val),
        opt_dof=int(dof_val),
        xpix_fulldet=float(xpix_fulldet),
        ypix_fulldet=float(ypix_fulldet),
        xpix_cutout=float(xcut),
        ypix_cutout=float(ycut),
    )


# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------

def _plot_extraction(data, model, psf, ivar, good, outlier_mask, radmask,
                     xcut, ycut, fit_radius_px, f_hat, f_hat_err, flux_uJy, snr,
                     chi2, dof, name, save_or_show_fn):
    """Four-panel diagnostic: data, model, residual, and pixel classification."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f"{name}  |  flux = {f_hat:.4g} ± {f_hat_err:.3g} MJy/sr"
        f"  ≈ {flux_uJy:.4g} µJy  |  S/N = {snr:.1f}"
        f"  |  χ²/dof = {chi2:.1f}/{dof}",
        fontsize=12,
    )

    resid = data - model

    # Shared colour scale for data / model
    vmax = np.nanpercentile(np.abs(data[radmask]), 99.5)
    vmin = np.nanpercentile(np.abs(data[radmask]),  0.5)

    kw_img  = dict(origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
    kw_res  = dict(origin="lower",
                   vmin=-3 * np.nanstd(resid[good]) if np.any(good) else -1,
                   vmax=3 * np.nanstd(resid[good]) if np.any(good) else 1,
                   cmap='RdBu_r')

    axs[0].imshow(data,   **kw_img); axs[0].set_title("Data (bkg-subtracted)")
    axs[1].imshow(model,  **kw_img); axs[1].set_title("Optimal model  (f·P)")
    axs[2].imshow(resid,  **kw_res); axs[2].set_title("Residual  (data − model)")

    # Panel 4: pixel classification map
    classification = np.zeros(data.shape, dtype=int)  # 0 = outside radius
    classification[radmask]                            = 1  # in radius, used
    classification[radmask & ~good & ~outlier_mask]    = 2  # flagged
    classification[outlier_mask & radmask]             = 3  # outlier-rejected

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", "steelblue", "orange", "red"])
    axs[3].imshow(classification, origin="lower", cmap=cmap, vmin=0, vmax=3)
    axs[3].set_title("Pixel classification")
    legend_elements = [
        mpatches.Patch(facecolor="white",     edgecolor="k", label="outside radius"),
        mpatches.Patch(facecolor="steelblue", label="used in fit"),
        mpatches.Patch(facecolor="orange",    label="flag-masked"),
        mpatches.Patch(facecolor="red",       label="outlier-rejected"),
    ]
    axs[3].legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Overlay: target position and fit radius on every panel
    for ax in axs:
        ax.scatter([xcut], [ycut], marker="+", s=60, color="white", zorder=5)
        circle = plt.Circle(
            (xcut, ycut), fit_radius_px,
            fill=False, linestyle="--", color="white", linewidth=1.2,
        )
        ax.add_patch(circle)

    plt.tight_layout()
    save_or_show_fn(fig, "optimal_extraction")

# ---------------------------------------------------------------------------
# Input file reader
# ---------------------------------------------------------------------------

def _read_input_file(filename):
    """
    Read a whitespace/comma-delimited file with columns name, ra, dec.
    Returns an astropy Table.
    """
    for delim in [None, " ", "\t", ","]:
        try:
            data = ascii.read(filename, delimiter=delim)
            cols_lower = [c.lower() for c in data.colnames]
            # Rename to canonical names
            for canon, alternatives in [
                ("name", ["name", "object", "source", "id", "target"]),
                ("ra",   ["ra", "right_ascension", "alpha"]),
                ("dec",  ["dec", "declination", "delta"]),
            ]:
                for alt in alternatives:
                    if alt in cols_lower:
                        actual = data.colnames[cols_lower.index(alt)]
                        if actual != canon:
                            data.rename_column(actual, canon)
                        break
            if all(c in data.colnames for c in ("name", "ra", "dec")):
                return data
        except Exception:
            continue
    raise ValueError(
        f"Cannot parse {filename}. "
        "Expected columns: name  ra  dec  (decimal degrees)."
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Download SPHEREx cutouts from IRSA and extract photometry "
            "via 2D optimal extraction (Horne 1986)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Target specification (mutually exclusive: single target or file)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", "-i", metavar="FILE",
                     help="Input file with columns: name ra dec")
    grp.add_argument("--ra", type=float,
                     help="Target RA in decimal degrees (single target)")

    p.add_argument("--dec", type=float,
                   help="Target Dec in decimal degrees (required with --ra)")
    p.add_argument("--name", type=str, default="target",
                   help="Source name label (used with --ra/--dec)")

    # Download options
    p.add_argument("--fits-dir", default="spherex_cutouts",
                   help="Directory for downloaded FITS cutouts (default: spherex_cutouts/)")
    p.add_argument("--skip-download", action="store_true",
                   help="Do not query IRSA; use FITS files already in --fits-dir")
    p.add_argument("--search-radius", type=float, default=3.0, metavar="ARCSEC",
                   help="IRSA search radius in arcsec (default: 3.0)")
    p.add_argument("--cutout-size", type=float, default=0.05, metavar="DEG",
                   help="Cutout size in degrees (default: 0.01 = 36 arcsec)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download existing cutout files")

    # Extraction options
    p.add_argument("--fit-radius", type=float, default=4.0, metavar="PX",
                   help="Extraction radius in detector pixels (default: 4.0)")
    p.add_argument("--kappa", type=float, default=4.0,
                   help="Outlier rejection threshold in σ (default: 4.0)")
    p.add_argument("--max-iter", type=int, default=10,
                   help="Maximum outlier-rejection iterations (default: 10)")
    p.add_argument("--no-masking", action="store_true",
                   help="Ignore flag-based pixel masking in the fit")

    # Output / display
    p.add_argument("--debug", action="store_true",
                   help="Print diagnostic messages")
    p.add_argument("--show-figs", action="store_true",
                   help="Show matplotlib figures interactively")
    p.add_argument("--save-figs", action="store_true",
                   help="Save diagnostic figures")
    p.add_argument("--results-dir", default="spherex_results",
                   help="Directory for result files (default: spherex_results/)")

    return p

import csv

def _results_to_csv(results, out_path):
    """
    Write all ExtractionResult objects to a single CSV file.

    The column order follows the field declaration order in the dataclass,
    which groups related quantities together (identification, metadata,
    quality flags, background, photometry).
    """
    if not results:
        print("  [WARN] No results to write.")
        return

    rows = [asdict(r) for r in results]
    fieldnames = list(rows[0].keys())

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Results written to: {out_path}")
    
def _spec_to_txt(results, out_path):
    """
    Write out spectrophotometry to a simple text file.
    """
    if not results:
        print("  [WARN] No results to write.")
        return

    wv = np.array([results[ii].wv_um for ii in range(len(results))])
    fx = np.array([results[ii].opt_flux_uJy for ii in range(len(results))])
    er = np.array([results[ii].opt_flux_uJy_err for ii in range(len(results))])
        
    # We usually want the spectrum file to be sorted from short to long wavelength
    idx = np.argsort(wv)
    wv = wv[idx]; fx = fx[idx]; er = er[idx]
    
    np.savetxt(out_path,np.vstack([wv,fx,er]).T)
    
    print(f"  Spectrum written to {out_path}")

def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate single-target mode
    if args.ra is not None and args.dec is None:
        parser.error("--dec is required when --ra is given")

    # Build target list
    if args.input:
        table = _read_input_file(args.input)
        targets = [
            (str(row["name"]).strip(), float(row["ra"]), float(row["dec"]))
            for row in table
        ]
    else:
        targets = [(args.name, args.ra, args.dec)]

    # Cutout size in detector pixels for extraction
    extract_px = 21

    for name, ra, dec in targets:
        all_results: List[ExtractionResult] = []
        os.makedirs(args.results_dir, exist_ok=True)
    
        print(f"\n{'='*60}")
        print(f"Target: {name}  RA={ra:.6f}  Dec={dec:.6f}")

        # ------------------------------------------------------------------
        # Step 1: download cutouts (unless --skip-download)
        # ------------------------------------------------------------------
        if args.skip_download:
            import glob
            pattern = os.path.join(args.fits_dir, name, f"{name}_cutout_*.fits")
            fits_paths = sorted(glob.glob(pattern))
            if not fits_paths:
                fits_paths = sorted(glob.glob(os.path.join(args.fits_dir, name, "*.fits")))
            print(f"  [skip-download] Found {len(fits_paths)} local file(s).")
        else:
            fits_paths = download_cutouts(
                name=name,
                ra=ra,
                dec=dec,
                output_dir=args.fits_dir,
                search_radius_arcsec=args.search_radius,
                cutout_size_deg=args.cutout_size,
                overwrite=args.overwrite,
            )

        if not fits_paths:
            print(f"  No FITS files available for {name}; skipping extraction.")
            continue

        # ------------------------------------------------------------------
        # Step 2: optimal extraction from each cutout
        # ------------------------------------------------------------------
        for fits_path in fits_paths:
            print(f"  Extracting: {os.path.basename(fits_path)}")
            result = optimal_extract(
                fits_path=fits_path,
                name=name,
                ra=ra,
                dec=dec,
                extract_size=(extract_px, extract_px),
                fit_radius_px=args.fit_radius,
                kappa=args.kappa,
                max_iter=args.max_iter,
                debug=args.debug,
                show_figs=args.show_figs,
                save_figs=args.save_figs,
                results_dir=args.results_dir,
                no_masking=args.no_masking,
            )
            all_results.append(result)

            # Print compact one-line summary
            print(
                f"    λ={result.wv_um or 'N/A':.4f} µm  "
                f"flux={result.opt_flux_MJysr or 'N/A'} MJy/sr"
                f"  ({result.opt_flux_uJy or 'N/A'} µJy)"
                f"  S/N={result.opt_snr or 'N/A':.2f}"
                f"  n_used={result.n_pix_used}"
                f"  n_outlier={result.n_pix_outlier}"
                f"  converged={result.converged}"
            )

        # ------------------------------------------------------------------
        # Combined CSV + TXT (always written if there are any results)
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        if all_results:
            csv_path = os.path.join(args.results_dir, f"{name}_spherex_photometry.csv")
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            _results_to_csv(all_results, csv_path)
            txt_path = os.path.join(args.results_dir, f"{name}_spherex_spectrum.txt")
            _spec_to_txt(all_results, txt_path)
        else:
            print("  No results to write.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
