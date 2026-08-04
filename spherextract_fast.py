#!/usr/bin/env python
"""
spherextract_fast.py

Download SPHEREx pixels with talltable and extract photometry via analogy to
optimal extraction (Horne 1986) using the official SPHEREx PSF model.

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
python spherextract_fast.py --ra 129.1827 --dec 0.914806 --name J0836p0054

# Read targets from file (name ra dec columns):
python spherextract_fast.py --input targets.txt

# Save results to specific directory:
python spherextract_fast.py --input targets.txt --results-dir my_results/

# Tune extraction:
python spherextract_fast.py --ra 129.1827 --dec 0.914806 \\
    --name J0836p0054 --fit-radius 4.0 --kappa 4.0 --max-iter 10 \\
    --cutout-size 0.1 \\
    --results-dir results_J0836/
"""
import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from astropy.io import ascii, fits
from scipy import ndimage

import talltable
import healpy
import pyarrow
import pyarrow.parquet
from time import sleep

from astropy.coordinates import SkyCoord
import astropy.units as u

from IPython import embed


# ---------------------------------------------------------------------------
# Data class for extraction results
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """All outputs from optimal extraction of one SPHEREx cutout file."""

    # Identification
    name: str
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
    
    # Aperture extraction
    aper_flux_uJy: Optional[float]        # integrated flux in µJy
    aper_flux_uJy_err: Optional[float]

    # Target pixel position
    xpix_fulldet: Optional[float]
    ypix_fulldet: Optional[float]
    xpix_cutout: Optional[float]
    ypix_cutout: Optional[float]


# ---------------------------------------------------------------------------
# Helpers: Talltable processing
# ---------------------------------------------------------------------------

def download_cutout_pixels(ra,dec,cutout_size_deg=0.05,wvbounds=None):
    """
    Load the pixels corresponding to a small region around the target source.
    """
    radius = 60.0*cutout_size_deg/2+(6.15/60) # in arcmin, with one pixel buffer
    custom_mask = ~(_BAD_BITS | _BAD_BITS_BKG | _BAD_BITS_MISC) # we actually want all the pixels, and will mask later on
    if wvbounds is None:
        query = (
                 talltable.PixelQuery(web=True)
                 .disc(ra, dec, radius)
                 .flags(custom_mask=custom_mask)
                 .with_wavelengths()
                 .with_rowcoldet()
                )
    else:
        wvmin, wvmax = wvbounds
        query = (
                 talltable.PixelQuery(web=True)
                 .disc(ra, dec, radius)
                 .flags(custom_mask=custom_mask)
                 .wavelength(wvmin, wvmax)
                 .with_wavelengths()
                 .with_rowcoldet()
                )
    print("Executing talltable query...")
    pixels = query.execute()
    print("Done.")
    return pixels
    
def cutout_pixels_to_images(pixels, image_tab, ra, dec, cutout_size, nodup=True):
    """
    Reconstruction of individual images from a big bucket of pixels.
    """
    # --- Extract pyarrow.table columns up front ---
    pix_ids   = np.asarray(pixels['imageid'])
    all_hp    = np.asarray(pixels['hphigh'])
    if nodup:
        undup_idx = np.arange(len(pix_ids))
    else:
        # Remove duplicate pixels (in case multiple pixel queries are combined ---
        print("   Removing duplicate pixels (if any)")
        _,undup_idx = np.unique(np.vstack([pix_ids,all_hp]).T,axis=0,return_index=True)
    pix_ids   = pix_ids[undup_idx]
    all_hp    = all_hp[undup_idx]
    all_row   = np.asarray(pixels['row'])[undup_idx]
    all_col   = np.asarray(pixels['col'])[undup_idx]
    all_flux  = np.asarray(pixels['flux'])[undup_idx]
    all_var   = np.asarray(pixels['variance'])[undup_idx]
    all_flag  = np.asarray(pixels['flags'])[undup_idx]
    all_wave  = np.asarray(pixels['wavelength'])[undup_idx]
    all_dwave = np.asarray(pixels['bandwidth'])[undup_idx]
    
    all_det   = np.asarray(pixels['det'])[undup_idx]
    
    # --- Sort once by imageid to make groups contiguous ---
    sort_idx    = np.argsort(pix_ids, kind='stable')
    sorted_ids  = pix_ids[sort_idx]
    unique_ids, start_indices = np.unique(sorted_ids, return_index=True)

    # --- Precompute metadata lookup ---
    tab_meta = {
        img_id: (t_beg, t_end, obsid)
        for img_id, t_beg, t_end, obsid in zip(
            np.asarray(image_tab['imageid']),
            np.asarray(image_tab['t_beg']),
            np.asarray(image_tab['t_end']),
            np.asarray(image_tab['obsid']),
        )
    }

    # --- Compute RA/Dec for ALL pixels in one healpy call ---
    # healpy.pix2ang vectorises efficiently over the full array
    all_ra, all_dec = healpy.pix2ang(
        2**22, all_hp[sort_idx], lonlat=True, nest=True
    )
    # all_ra / all_dec are now aligned with sort_idx order

    images = []
    n_imgs = len(unique_ids)
    print("Sorting pixels into images...")

    for i in range(n_imgs):
        start = start_indices[i]
        end   = start_indices[i + 1] if i < n_imgs - 1 else len(pix_ids)

        idx = sort_idx[start:end]       # original positions
        sl  = slice(start, end)         # positions in sorted arrays

        # Coordinates & bounding box
        xind = all_row[idx]
        yind = all_col[idx]
        xmin, xmax = xind.min(), xind.max()
        ymin, ymax = yind.min(), yind.max()
        h, w  = xmax - xmin + 1, ymax - ymin + 1
        off_x = xind - xmin
        off_y = yind - ymin

        # Preallocate output arrays
        fluximg  = np.zeros((h, w))
        varimg   = np.zeros((h, w))
        flagimg  = np.zeros((h, w))
        waveimg  = np.zeros((h, w))
        dwaveimg = np.zeros((h, w))
        rowimg   = np.zeros((h, w))
        colimg   = np.zeros((h, w))
        raimg    = np.empty((h, w))
        decimg   = np.empty((h, w))

        # Vectorised scatter into 2-D arrays
        fluximg [off_x, off_y] = all_flux [idx]
        varimg  [off_x, off_y] = all_var  [idx]
        flagimg [off_x, off_y] = all_flag [idx]
        waveimg [off_x, off_y] = all_wave [idx]
        dwaveimg[off_x, off_y] = all_dwave[idx]
        rowimg  [off_x, off_y] = xind
        colimg  [off_x, off_y] = yind
        raimg   [off_x, off_y] = all_ra [sl]   # sl works because sorted
        decimg  [off_x, off_y] = all_dec[sl]

        img_id = unique_ids[i]
        try:
            t_beg, t_end, obsid_val = tab_meta[img_id]
        except:
            print("Image ID not found in image.parquet. Try a 'git pull' or update yourself from flatiron site.")
            continue

        images.append({
            'ra': raimg,  'dec': decimg,
            'row': rowimg, 'col': colimg,
            'flux': fluximg, 'var': varimg, 'flags': flagimg,
            'wave': waveimg, 'dwave': dwaveimg,
            'detector_id': all_det[idx[0]],
            'obsid': obsid_val, 'imageid': img_id,
            'mjd_avg': 0.5 * (t_beg + t_end),
        })

    print("Done.")
    return images
    
# Helper: WCS determination from constructed images
def fit_affine_wcs(image, mask=None):
    """
    Fit a local affine WCS using tangent-plane projection.
    Returns a callable that maps (ra, dec) -> (col_offset, row_offset).
    """
    ra_flat  = image['ra'][mask].flatten() * (np.pi / 180.0) # convert to radians
    dec_flat = image['dec'][mask].flatten() * (np.pi / 180.0)
    col_flat = image['col'][mask].flatten() - image['col'][mask].min()
    row_flat = image['row'][mask].flatten() - image['row'][mask].min()

    # 1. Reference sky coordinates (field center)
    sin_ra_mean = np.mean(np.sin(ra_flat))
    cos_ra_mean = np.mean(np.cos(ra_flat))
    ra0 = np.arctan2(sin_ra_mean, cos_ra_mean)
    dec0 = np.mean(dec_flat)
    
    # 2. Convert pixels to unit vectors on the celestial sphere
    cos_ra, sin_ra = np.cos(ra_flat), np.sin(ra_flat)
    cos_dec, sin_dec = np.cos(dec_flat), np.sin(dec_flat)
    Vx, Vy, Vz = cos_ra*cos_dec, sin_ra*cos_dec, sin_dec

    # 3. Reference direction & local tangent plane basis (East, North)
    cos_ra0, sin_ra0 = np.cos(ra0), np.sin(ra0)
    cos_dec0, sin_dec0 = np.cos(dec0), np.sin(dec0)
    
    # Reference unit vector
    Nx, Ny, Nz = cos_ra0*cos_dec0, sin_ra0*cos_dec0, sin_dec0
    # East basis (points along increasing RA at constant Dec)
    Ex, Ey, Ez = -sin_ra0, cos_ra0, 0.0
    # North basis (points along increasing Dec at constant RA)
    Bx, By, Bz = -cos_ra0*sin_dec0, -sin_ra0*sin_dec0, cos_dec0

    # 4. Gnomonic projection to tangent plane coordinates (xp, yp) in radians
    c_dot = Vx*Nx + Vy*Ny + Vz*Nz  # cosine of angular distance from reference
    
    # Safeguard against division by zero (only triggers for FoV > ~90° or exact pole centering)
    c_safe = np.where(np.abs(c_dot) < 1e-6, 1e-6 * np.sign(c_dot), c_dot)
    
    xp = (Vx*Ex + Vy*Ey + Vz*Ez) / c_safe
    yp = (Vx*Bx + Vy*By + Vz*Bz) / c_safe

    # 5. Linear fit in tangent plane: [xp, yp, 1] -> [col, row]
    A = np.column_stack([xp, yp, np.ones_like(xp)])
    coeffs, *_ = np.linalg.lstsq(A, np.column_stack([col_flat, row_flat]), rcond=None)

    # Cache constants for closure to avoid recomputation during evaluation
    _Nx, _Ny, _Nz = Nx, Ny, Nz
    _Ex, _Ey, _Ez = Ex, Ey, Ez
    _Bx, _By, _Bz = Bx, By, Bz

    def evaluate(ra, dec):
        ra *= np.pi / 180.0
        dec *= np.pi / 180.0
        cos_ra_q, sin_ra_q = np.cos(ra), np.sin(ra)
        cos_dec_q, sin_dec_q = np.cos(dec), np.sin(dec)
        Vx_q = cos_ra_q*cos_dec_q
        Vy_q = sin_ra_q*cos_dec_q
        Vz_q = sin_dec_q
        
        c_dot_q = Vx_q*_Nx + Vy_q*_Ny + Vz_q*_Nz
        c_safe_q = np.where(np.abs(c_dot_q) < 1e-6, 1e-6 * np.sign(c_dot_q), c_dot_q)
        
        xp_q = (Vx_q*_Ex + Vy_q*_Ey + Vz_q*_Ez) / c_safe_q
        yp_q = (Vx_q*_Bx + Vy_q*_By + Vz_q*_Bz) / c_safe_q
        
        col_offset = coeffs[0, 0]*xp_q + coeffs[1, 0]*yp_q + coeffs[2, 0]
        row_offset = coeffs[0, 1]*xp_q + coeffs[1, 1]*yp_q + coeffs[2, 1]
        return col_offset, row_offset

    return evaluate

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
    3. Apply a subpixel shift (in high-res units) so the PSF center lands at
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
    # Center of canvas in high-res pixels
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

    # 3. Subpixel shift: move PSF center from canvas center to (xcut, ycut)
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
    # New flags 2026 Jun
    "GHOST": 22, "GHOST_EXT": 24, "BLOOM": 26,
    "SNOWBALL": 27, "HALO": 28, "SATELLITE_HALO": 29
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
    | _bit(_MP["HALO"])
    | _bit(_MP["SATELLITE_HALO"])
)

# Additional bits to exclude when estimating the background
_BAD_BITS_BKG = (
    _BAD_BITS
    | _bit(_MP["OVERFLOW"])
    | _bit(_MP["SOURCE"])
    | _bit(_MP["OUTLIER"])
    | _bit(_MP["TRANSIENT"])
)

# All other bits
_BAD_BITS_MISC = (
    _bit(_MP["DICHROIC"])
    | _bit(_MP["FULLSAMPLE"])
    | _bit(_MP["PHANMISS"])
    | _bit(_MP["GHOST"])
    | _bit(_MP["GHOST_EXT"])
    | _bit(_MP["BLOOM"])
    | _bit(_MP["SNOWBALL"])
)

# Helper: Get nearby objects via astroquery
def find_nearby_objects(ra, dec,
                        catalog='catwise',
                        deblend_radius=20.0, maglim=None):
    """
    Locate nearby objects for deblending purposes.
    """
    coord = SkyCoord(ra,dec,unit='deg',frame='icrs')
    if catalog == 'catwise':
        if maglim is None:
            maglim = 17.0 # reasonable default for W1
        print(f"    Querying CatWISE2020 for nearby objects brighter than W1={maglim+2.7} AB mag within {deblend_radius} arcsec")
        
        from astroquery.ipac.irsa import Irsa
        
        all_objs = Irsa.query_region(coordinates=coord,spatial='Cone',catalog='catwise_2020',
                                 radius=deblend_radius*u.arcsec)
                                 
        # Trim faint objects
        keep = (np.ma.getdata(all_objs['w1mpro']) < maglim)
                                 
        # Remove target object if present
        obj_coords = [SkyCoord(np.ma.getdata(all_objs['ra'])[ii], np.ma.getdata(all_objs['dec'])[ii],
                               unit='deg', frame='icrs') for ii in range(len(all_objs))]
        seps = np.array([coord.separation(obj_coords[ii]).arcsec for ii in range(len(all_objs))])
        if np.min(seps) < 2.0:                # this minimum might be a bit too big, but closer than that
            target = np.argmin(seps)          # and I wouldn't expect deblending to work well anyway, right?
            keep = keep & (np.arange(len(all_objs)) != target)
            
        if np.sum(keep) > 0:
            deblend_list = np.vstack([np.ma.getdata(all_objs['ra'])[keep],np.ma.getdata(all_objs['dec'])[keep]]).T
        else:
            deblend_list = None
    elif catalog == 'gaia':
        if maglim is None:
            maglim = 19.7 # reasonable default for GRP
        print(f"    Querying Gaia for nearby objects brighter than G_RP={maglim} mag within {deblend_radius} arcsec")
        
        from astroquery.gaia import Gaia
        
        ref_epoch = 57388.0 # 2016.0, Gaia DR3
        
        j = Gaia.cone_search_async(coord, radius=u.Quantity(deblend_radius, u.arcsec))
        r = j.get_results()
        
        embed()

        # Trim faint objects
        keep = (np.ma.getdata(r['phot_rp_mean_mag']) < maglim)
                                 
        # Remove target object if present
        obj_coords = [SkyCoord(np.ma.getdata(r['ra'])[ii], np.ma.getdata(r['dec'])[ii],
                               unit='deg', frame='icrs') for ii in range(len(r))]
        seps = np.array([coord.separation(obj_coords[ii]).arcsec for ii in range(len(r))])
        if np.min(seps) < 2.0:                # this minimum might be a bit too big, but closer than that
            target = np.argmin(seps)          # and I wouldn't expect deblending to work well anyway, right?
            keep = keep & (np.arange(len(r)) != target)

        if np.sum(keep) > 0:
            deblend_list = np.vstack([np.ma.getdata(r['ra'])[keep],np.ma.getdata(r['dec'])[keep],
                                      np.ma.getdata(r['pmra'])[keep],
                                      np.ma.getdata(r['pmdec'])[keep],
                                      ref_epoch*np.ones(np.sum(keep))]).T
        else:
            deblend_list = None
    else:
        print("???? what. no such catalog supported, no deblending for you")
        deblend_list = None
    if deblend_list is not None:
        print(f"    {np.sum(keep)} found.")
    else:
        print("    No nearby objects to deblend.")
        
    return deblend_list

# ---------------------------------------------------------------------------
# Core: optimal extraction from a single image
# ---------------------------------------------------------------------------

def optimal_extract(image, psf_cube_fits, name, ra, dec,
                    pm_ra=None, pm_dec=None, ref_epoch=None,
                    deblend_list = None, sapm_fits = None,
                    fit_radius_px = 3.0, kappa = 4.0, max_iter = 10,
                    linear_bkg = False, debug = False, show_figs = False,
                    save_figs = False, results_dir = None, no_masking = False):
    """
    Run 2D optimal extraction on one SPHEREx cutout image.

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
    image         : cutout image
    name          : source label (used in output filenames)
    ra, dec       : target position (degrees)
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
            obs = image['obsid']
            det = image['detector_id']
            fig.savefig(Path(figs_dir) / f"{name}_{obs}D{det}_{stub}.png", dpi=130)
        plt.close(fig)

    def _nan_result(**overrides):
        """Return a fully-NaN ExtractionResult for early-abort cases."""
        base = dict(
            name=name,
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
            aper_flux_uJy=None, aper_flux_uJy_err=None
        )
        base.update(overrides)
        return ExtractionResult(**base)

    # ================================================================
    # 1. Extract information from cutout image
    # ================================================================

    img        = image['flux']
    flags      = image['flags']
    var        = image['var']
    wave       = image['wave']
    dwave      = image['dwave']
    
    detector_id = image['detector_id']
    det_id_int  = int(detector_id)
    bandpass    = f"D{det_id_int}"
    
    mjd_avg_val = image['mjd_avg']
    
    # Correct RA/Dec to MJD of current image
    pm_conv = (1e-3/3600/31557600)*86400 # convert from mas/yr to deg/day
    if pm_ra is not None:
        ra += (pm_conv*pm_ra/np.cos(dec*np.pi/180.0))*(mjd_avg_val-ref_epoch)
        dec += pm_conv*pm_dec*(mjd_avg_val-ref_epoch)
    
    # Load PSF cube
    hdul       = psf_cube_fits
    psf_cube   = hdul[1].data
    hdr_psf    = hdul[1].header

    oversamp = hdr_psf["OVERSAMP"]
    cdelt    = hdr_psf["CDELT1"]   # arcsec per high-res px
    px_arcsec = cdelt * oversamp   # arcsec per detector px

    obsid       = image['obsid']
    det_w        = img.shape[1]
    det_h        = img.shape[0]
    
    unused = (wave == 0)
    gpm = ~unused

    omega_arcsec2 = 6.15*6.15*np.ones_like(img)
    if sapm_fits is not None:
        omega_arcsec2[gpm] = sapm_fits[1].data[image['row'].astype(int)[gpm],image['col'].astype(int)[gpm]]
    arcsec2_to_sr = (np.pi / (180.0 * 3600.0)) ** 2
    omega_sr = omega_arcsec2 * arcsec2_to_sr
    
    nblend = len(deblend_list) if deblend_list is not None else 0

#    # ================================================================
#    # 2. Target pixel position
#    #
#    # We need two distinct pixel coordinates for the target:
#    #
#    #  (a) xcut, ycut  — position within the cutout array (H × W).
#    #      Used for PSF placement and the extraction itself.
#    #
#    #  (b) xpix_fulldet, ypix_fulldet  — position in the original
#    #      full-detector array.  Used only for PSF zone selection
#    #
#    # ================================================================

#    # (a) Cutout pixel coordinate
    # First we reconstruct th
    wcs = fit_affine_wcs(image,gpm)
    xcut, ycut = wcs(ra, dec)
    # Now check whether the nearest pixel is actually on the detector
    try:
        badflux = img[int(np.round(ycut)),int(np.round(xcut))] == 0
    except:
        dprint("  [WARN] Target outside image footprint; skipping.")
        return _nan_result(near_detector_edge=True)
    if xcut < 0 or xcut > det_w-1 or ycut < 0 or ycut > det_h-1 or badflux:
        dprint("  [WARN] Target outside image footprint; skipping.")
        return _nan_result(near_detector_edge=True)
        
    if deblend_list is not None:
        xcut_deblend = np.zeros(nblend)
        ycut_deblend = np.zeros(nblend)
        for ii in range(nblend):
            rad, decd = deblend_list[ii,0], deblend_list[ii,1]
            if deblend_list.shape[1] > 2:
                pm_rad, pm_decd, ref_epochd = deblend_list[ii,2], deblend_list[ii,3], deblend_list[ii,4]
                rad += (pm_conv*pm_decd/np.cos(decd*np.pi/180.0))*(mjd_avg_val-ref_epochd)
                decd += pm_conv*pm_rad*(mjd_avg_val-ref_epochd)
            xcut_deblend[ii], ycut_deblend[ii] = wcs(rad, decd)

#    # (b) Full-detector pixel coordinate.
    xpix_fulldet = xcut+image['col'][gpm].min()
    ypix_fulldet = ycut+image['row'][gpm].min()

    dprint(f"  Target cutout pixel : x={xcut:.3f}, y={ycut:.3f}")
    dprint(f"  Target full-det pixel: x={xpix_fulldet:.3f}, y={ypix_fulldet:.3f}")

    H,W = img.shape
    dprint(f"  Cutout shape: {H}×{W}, target at xcut={xcut:.3f}, ycut={ycut:.3f}")
    
    # ================================================================
    # 3. Background subtraction
    # ================================================================
    mask_bkg = (flags.astype(np.uint32) & _BAD_BITS_BKG) != 0
    good_bkg = (~mask_bkg) & np.isfinite(img) & gpm
    bkg_npix = int(np.sum(good_bkg))

    if bkg_npix >= 3:
        bkg = float(np.median(img[good_bkg]))
    else:
        # fallback: ignore flags, use any finite pixel
        good_any = np.isfinite(img)
        bkg = float(np.median(img[good_any])) if np.any(good_any) else 0.0
        bkg_npix = int(np.sum(good_any))
        dprint("  [WARN] Few good BCG pixels; using unmasked background estimate.")
        
    # Linear model for the background
    # Basic weighted linear least squares for efficiency
    if linear_bkg and np.sum(good_bkg) >= 10:
        bkg_rows = image['row'][good_bkg]-np.median(image['row'])
        bkg_flux = img[good_bkg]
        bkg_wgts = 1.0/var[good_bkg]
        X = np.column_stack([bkg_rows, np.ones_like(bkg_rows)])
        WX = X * bkg_wgts[:, None]
        try:
            result = np.linalg.solve(WX.T @ X, WX.T @ bkg_flux)
            bkg = result[1] + result[0]*(image['row']-np.median(image['row']))
            
            # Now do one round of clipping on the background pixels for robustness
            outlier_bkg = (np.abs((bkg-img)/np.sqrt(var+(var==0))) > 5.0)
            good_bkg2 = good_bkg & ~outlier_bkg
            # And repeat the background estimate
            bkg_rows = image['row'][good_bkg2]-np.median(image['row'])
            bkg_flux = img[good_bkg2]
            bkg_wgts = 1.0/var[good_bkg2]
            X = np.column_stack([bkg_rows, np.ones_like(bkg_rows)])
            WX = X * bkg_wgts[:, None]
            result = np.linalg.solve(WX.T @ X, WX.T @ bkg_flux)
            bkg = result[1] + result[0]*(image['row']-np.median(image['row']))
        except: # fallback to median
            dprint("Linear background fit failed; falling back to median")
        if np.sum(~np.isfinite(bkg)) > 0:
            dprint("Linear background fit failed; falling back to median")
            bkg = float(np.median(img[good_bkg]))
            
    data = img - bkg
    dprint(f"  Background: {np.mean(bkg):.4g} MJy/sr (from {bkg_npix} pixels)")

    # ================================================================
    # 4. Select PSF zone
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
    # 5. Build detector-grid PSF (block-sum from oversampled model)
    # ================================================================
    P = _make_psf_detgrid(psf_hr, oversamp, (H, W), xcut, ycut)
    if deblend_list is not None:
        P = [P]
        for ii in range(nblend):
            P.append(_make_psf_detgrid(psf_hr, oversamp, (H, W),
                                       xcut_deblend[ii], ycut_deblend[ii]))
        P = np.array(P)
        psf_sum = np.nansum(P,axis=(1,2))
    else:
        psf_sum = float(np.nansum(P))
        dprint(f"  PSF detector-grid sum: {psf_sum:.6g} (≈1 if unit-normalised)")

    # ================================================================
    # 6. Build pixel masks and weights
    # ================================================================
    YY, XX = np.indices((H, W))
    r2 = (XX - xcut) ** 2 + (YY - ycut) ** 2
    radmask = r2 <= fit_radius_px ** 2
    
    if deblend_list is not None:
        keep = np.ones(nblend,dtype=bool)
        for ii in range(nblend):
            r2 = (XX-xcut_deblend[ii])**2 + (YY-ycut_deblend[ii])**2
            radmask_deblend = r2 <= fit_radius_px ** 2
            if np.sum(radmask_deblend) == 0:
                keep[ii] = False
            else:
                radmask = radmask | radmask_deblend
        deblend_list = deblend_list[keep]
        P[1:] = P[1:][keep]
        xcut_deblend = xcut_deblend[keep]
        ycut_deblend = ycut_deblend[keep]
        nblend = np.sum(keep)

    if no_masking:
        flag_mask = np.zeros((H, W), dtype=bool)
        dprint("  [INFO] --no-masking: flag-based masking disabled in fit.")
    else:
        flag_mask = (flags.astype(np.uint32) & _BAD_BITS) != 0

    ivar = np.where((var > 0) & np.isfinite(var),1.0/(var+(var==0)),0.0)

    # Base good-pixel mask (will be updated per iteration)
    base_good = radmask & (~flag_mask) & np.isfinite(data) & (ivar > 0) & gpm

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
    # 7. Iterative optimal extraction with outlier rejection
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
    if deblend_list is not None:
        f_hat = np.zeros(nblend+1)
        cov = np.full((nblend+1,nblend+1),np.nan)
        cond = np.nan
    else:
        f_hat = 0.0
        var_f  = np.nan
    converged = False
    n_iter = 0

    for iteration in range(max_iter + 1):
        n_iter = iteration

        # --- Step A: linear optimal estimator ---
        if deblend_list is not None:
            Phi = P.reshape((nblend+1,H*W))
            w = ivar * good.astype(float)
            A = (Phi * w.flatten()) @ Phi.T
            b = np.array([np.sum(P[ii]*w*data_safe) for ii in range(nblend+1)])
            cond = np.linalg.cond(A)
            
            if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
                dprint(f"  [WARN] Non-finite normal equations at iter {iteration}.")
                break
            
            if cond > 1e10:
                # Some of the PSF profiles are nearly co-linear: the problem is
                # ill-conditioned (sources too close / too similar).
                dprint(f"  [WARN] Ill-conditioned system (cond={cond:.2e}); "
                       "sources may be indistinguishable at this resolution.")

            try:
                cov = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                dprint("  [WARN] Singular normal-equation matrix; aborting.")
                break

            f_hat = np.linalg.solve(A,b)

        else:
            num = np.sum(P[good] * ivar[good] * data_safe[good])
            den = np.sum(P[good] ** 2 * ivar[good])

            if den <= 0 or not np.isfinite(num) or not np.isfinite(den):
                dprint(f"  [WARN] Degenerate system at iteration {iteration}; stopping.")
                break

            f_hat = num / den
            var_f  = 1.0 / den
            

        dprint(
            f"  iter {iteration}: f_hat={f_hat[0] if deblend_list is not None else f_hat:.5g} MJy/sr, "
            f"σ={np.sqrt(cov[0,0]) if deblend_list is not None else np.sqrt(var_f):.3g}, good_px={int(np.sum(good))}"
        )

        if iteration == max_iter:
            converged = True   # reached limit; accept current solution
            break

        # --- Step B: residual-based outlier detection ---
        model_px = f_hat * P if deblend_list is None else np.sum((f_hat*P.T).T,axis=0)
        resid     = data_safe - model_px          # (H, W)
        sigma_px  = np.sqrt(np.where(var > 0, var, np.inf))
        newly_bad = (np.abs(resid) > kappa * sigma_px) & radmask & good

        if not np.any(newly_bad):
            converged = True
            dprint(f"  Converged after {iteration} iteration(s).")
            break

        outlier_mask |= newly_bad
        good = base_good & ~outlier_mask

        if not np.any(good):
            dprint("  [WARN] All pixels rejected by outlier masking; reverting.")
            outlier_mask[:] = False
            good = base_good.copy()
            converged = False
            break

    n_outlier = int(np.sum(outlier_mask & radmask))
    n_used    = int(np.sum(good))

    # ================================================================
    # 8. Chi² of the final model
    # ================================================================
    model_final = f_hat * P if deblend_list is None else np.sum((f_hat*P.T).T,axis=0)
    resid_final = data_safe - model_final
    w_final     = ivar * good.astype(float)
    chi2_val    = float(np.sum(resid_final ** 2 * w_final))
    dof_val  = max(n_used - (1+nblend), 1)    # N free parameters (fluxes)

    dprint(f"  chi²={chi2_val:.3f}, dof={dof_val}, chi²/dof={chi2_val/dof_val:.3f}")

    # ================================================================
    # 9. Convert amplitude to integrated flux [µJy]
    # ================================================================
    # f_hat is the surface-brightness amplitude [MJy/sr].
    # Integrated flux = f_hat × Ω_pix × Σ P_det
    # (Σ P_det ≈ 1 for a unit-normalised PSF)
    if deblend_list is not None:
        # nuke the rest of the objects and just keep the target
        f_hat_err = float(np.sqrt(cov[0,0])) if np.isfinite(cov[0,0]) else np.nan
        f_hat = f_hat[0]
        psf_sum = psf_sum[0]
    else:
        f_hat_err = float(np.sqrt(var_f)) if np.isfinite(var_f) else np.nan

    if psf_sum > 0:
        flux_uJy     = f_hat * omega_sr[int(ycut),int(xcut)] * psf_sum * 1e12
        flux_uJy_err = f_hat_err * omega_sr[int(ycut),int(xcut)] * psf_sum * 1e12
    else:
        flux_uJy = flux_uJy_err = np.nan

    snr = f_hat / f_hat_err if (np.isfinite(f_hat_err) and f_hat_err > 0) else np.nan

    dprint(
        f"  Optimal flux: {f_hat:.4g} ± {f_hat_err:.3g} MJy/sr  "
        f"≈ {flux_uJy:.4g} ± {flux_uJy_err:.3g} µJy  S/N={snr:.2f}"
    )
    
    # Let's also get the aperture flux for comparison
    aper_radius_px = 2.0 # This should get most of the flux
    apermask = r2 <= aper_radius_px ** 2
    aper_flux = np.sum(data_safe[apermask])
    aper_flux_uJy = aper_flux * omega_sr[int(ycut),int(xcut)] * 1e12
    aper_flux_uJy_err = np.sqrt(np.sum(var[apermask & (data_safe != 0)]))

    # ================================================================
    # 10. Spectral WCS at target position
    # ================================================================
    
    # Use xcut/ycut and bilinearly interpolate from nearby pixels
    # this is quite dangerous, so lots of checks to make sure nothing bad creeps in
    x1 = int(xcut); x2 = x1+1; y1 = int(ycut); y2 = y1+1
    try:
        if wave[y1,x1] == 0 or wave[y2,x1] == 0 or wave[y1,x2] == 0 or wave[y2,x2] == 0:
            wv_um = np.nan
            wv_width_um = np.nan
        else:
            wv_um = wave[y1,x1]*(x2-xcut)*(y2-ycut)+wave[y2,x1]*(x2-xcut)*(ycut-y1)+wave[y1,x2]*(xcut-x1)*(y2-ycut)+wave[y2,x2]*(xcut-x1)*(ycut-y1)
            wv_width_um = dwave[y1,x1]*(x2-xcut)*(y2-ycut)+dwave[y2,x1]*(x2-xcut)*(ycut-y1)+dwave[y1,x2]*(xcut-x1)*(y2-ycut)+dwave[y2,x2]*(xcut-x1)*(ycut-y1)
    except:
        wv_um = np.nan
        wv_width_um = np.nan
   
    dprint(f"  Spectral WCS: λ={wv_um:.5f} µm, Δλ={wv_width_um:.5f} µm")

    # ================================================================
    # 11. Optional diagnostic figure
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

    return ExtractionResult(
        name=name,
        input_ra_deg=float(ra),
        input_dec_deg=float(dec),
        obsid=obsid,
        detector_id=det_id_int,
        bandpass=bandpass,
        expid=int(image['imageid']),
        mjd_avg=float(mjd_avg_val),
        psf_index=int(idx_psf),
        omega_sr=float(omega_sr[int(ycut),int(xcut)]),
        px_scale_arcsec=float(px_arcsec),
        wv_um=float(wv_um),
        wv_width_um=float(wv_width_um),
        near_detector_edge=False,
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
        aper_flux_uJy=float(aper_flux_uJy),
        aper_flux_uJy_err=float(aper_flux_uJy_err),
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
                ("pm_ra", ["pm_ra", "pmra"]),
                ("pm_dec", ["pm_dec", "pmdec"])
            ]:
                for alt in alternatives:
                    if alt in cols_lower:
                        actual = data.colnames[cols_lower.index(alt)]
                        if actual != canon:
                            data.rename_column(actual, canon)
                        break
            if all(c in data.colnames for c in ("name", "ra", "dec")):
                if "pm_ra" in data.colnames:
                    return data, True
                else:
                    return data, False
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
            "Download SPHEREx cutouts with talltable and extract photometry "
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
    # proper motion
    p.add_argument("--pm-ra", type=float, default=None,
                   help="Proper motion in the RA direction (mas/yr)")
    p.add_argument("--pm-dec", type=float, default=None,
                   help="Proper motion in the Dec direction (mas/yr)")
    p.add_argument("--ref-epoch", type=float, default=57388.0,
                   help="Reference epoch for proper motion (MJD; default=57388.0 [2016.0] used by Gaia DR3)")
    p.add_argument("--input-pm", action="store_true",
                   help="Read proper motions from input table")
                   
    # Download options
    p.add_argument("--cutout-size", type=float, default=0.05, metavar="DEG",
                   help="Cutout size in degrees (default: 0.01 = 36 arcsec)")
    # TODO: Reintroduce threading for talltable queries
#    p.add_argument("--dl-threads", type=int, default=8,
#                   help="Number of simultaneous downloads (default = 8)")
    p.add_argument("--nchunks", type=int, default=2,
                   help="Number of wavelength chunks to divide detectors in case of PixelQuery failure")

    # Extraction options
    p.add_argument("--fit-radius", type=float, default=4.0, metavar="PX",
                   help="Extraction radius in detector pixels (default: 4.0)")
    p.add_argument("--kappa", type=float, default=4.0,
                   help="Outlier rejection threshold in σ (default: 4.0)")
    p.add_argument("--max-iter", type=int, default=10,
                   help="Maximum outlier-rejection iterations (default: 10)")
    p.add_argument("--no-masking", action="store_true",
                   help="Ignore flag-based pixel masking in the fit")
    p.add_argument("--linear-bkg", action="store_true",
                   help="Fit a linear model to the background instead of local median")
    p.add_argument("--auto-deblend", type=str, default=None,
                   help="Automatically deblend target from nearby sources. Use 'catwise' or 'gaia")
    p.add_argument("--deblend-radius", type=float, default=20.0,
                   help="Maximum distance to search for deblending objects in arcsec")
    p.add_argument("--deblend-maglim", type=float, default=None,
                   help="Minimum magnitude of deblended objects. W1 for catwise (Vega mag) or GRP mag for gaia")
    

    # Output / display
    p.add_argument("--debug", action="store_true",
                   help="Print diagnostic messages")
    p.add_argument("--show-figs", action="store_true",
                   help="Show matplotlib figures interactively")
    p.add_argument("--save-figs", action="store_true",
                   help="Save diagnostic figures")
    p.add_argument("--results-dir", default="spherex_results",
                   help="Directory for result files (default: spherex_results/)")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce output to terminal")
                   
    # Data file options
    p.add_argument("--image-tab-path", default="spherex_calibs",
                   help="Directory to look for the image.parquet talltable data file.")
    p.add_argument("--psf-path", default="spherex_calibs/psf",
                   help="Directory to look for oversampled PSF model cubes.")
    p.add_argument("--sapm-path", default=None,
                   help="Directory to look for solid angle maps. Not used by default.")

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
        table,has_pm = _read_input_file(args.input)
        if has_pm:
            targets = [
                (str(row["name"]).strip(), float(row["ra"]), float(row["dec"]),
                 float(row["pm_ra"]), float(row["pm_dec"]), args.ref_epoch)
                for row in table
            ]
        else:
            targets = [
                (str(row["name"]).strip(), float(row["ra"]), float(row["dec"]),
                 0.0, 0.0, args.ref_epoch)
                for row in table
            ]
    else:
        targets = [(args.name, args.ra, args.dec, args.pm_ra, args.pm_dec, args.ref_epoch)]

    # Load in table of all SPHEREx image metadata
    image_tab = pyarrow.parquet.read_table(os.path.join(args.image_tab_path,"image.parquet"))
    # Load in PSF model cubes
    psf_cubes = [fits.open(os.path.join(args.psf_path,f'average_psf_D{ii+1}_spx_cal-psf-v5-2026-082.fits')) for ii in range(6)]

    # Load in solid angle maps
    if args.sapm_path is not None:
        sapm_images = [fits.open(os.path.join(args.sapm_path,r'solid_angle_pixel_map_D1_spx_cal-sapm-v2-2025-164.fits')) for ii in range(6)]
    else:
        sapm_images = None

    failed = []

    for name, ra, dec, pm_ra, pm_dec, ref_epoch in targets:
        all_results: List[ExtractionResult] = []
        os.makedirs(args.results_dir, exist_ok=True)
    
        print(f"\n{'='*60}")
        print(f"Target: {name}  RA={ra:.6f}  Dec={dec:.6f}")

        # ------------------------------------------------------------------
        # Step 1: download cutout pixels
        # ------------------------------------------------------------------
        
        csv_path = os.path.join(args.results_dir, f"{name}_spherex_photometry.csv")
        if len(targets) > 1 and os.path.isfile(csv_path):
            print(f"  Already extracted this object, skipping to the next one.")
            continue
        
        try:
            cutout_pixels = download_cutout_pixels(ra=ra,dec=dec,cutout_size_deg=args.cutout_size)
            nodup = True
        except:
            print(f"   PixelQuery failed. Trying again in {6*args.nchunks} wavelength chunks...")
            sleep(3)
            try:
                nwv1234 = 4*args.nchunks
                wvmin1234 = 0.733
                wvmax1234 = 3.811
                
                nwv56 = 2*args.nchunks
                wvmin56 = 3.809
                wvmax56 = 5.015
                
                cutout_pixels_wv = [None for ii in range(nwv1234+nwv56)]
                
                wvbins = 10**np.linspace(np.log10(wvmin1234),np.log10(wvmax1234),nwv1234+1)
                for ii in range(nwv1234):
                    print(f"   Query {ii+1}/{nwv1234+nwv56}")
                    cutout_pixels_wv[ii] = download_cutout_pixels(ra=ra,dec=dec,cutout_size_deg=args.cutout_size,
                                                                  wvbounds=[wvbins[ii],wvbins[ii+1]])
                 
                wvbins = 10**np.linspace(np.log10(wvmin56),np.log10(wvmax56),nwv56+1)
                for ii in range(nwv56):
                    print(f"   Query {nwv1234+ii+1}/{nwv1234+nwv56}")
                    cutout_pixels_wv[nwv1234+ii] = download_cutout_pixels(ra=ra,dec=dec,cutout_size_deg=args.cutout_size,
                                                                  wvbounds=[wvbins[ii],wvbins[ii+1]])
                                                                  
                cutout_pixels = pyarrow.concat_tables(cutout_pixels_wv)
                nodup = False
            except:
                print("   PixelQuery failed yet again. Try increasing --nchunks or reducing the primary cutout size.")
                sleep(5)
                failed.append(name)
                continue
        
        cutout_images = cutout_pixels_to_images(cutout_pixels,image_tab,ra,dec,args.cutout_size,nodup=nodup)
        
        if args.auto_deblend is not None:
            deblend_list = find_nearby_objects(ra, dec, catalog=args.auto_deblend,
                                               deblend_radius=args.deblend_radius, maglim=args.deblend_maglim)
        else:
            deblend_list = None

        # ------------------------------------------------------------------
        # Step 2: optimal extraction from each cutout
        # ------------------------------------------------------------------
        print("Extracting spectrophotometry...")
        for image in cutout_images:
            det = image['detector_id']
            result = optimal_extract(
                image=image,
                psf_cube_fits=psf_cubes[det-1],
                name=name,
                ra=ra,
                dec=dec,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                ref_epoch=ref_epoch,
                deblend_list=deblend_list,
                sapm_fits=sapm_images[det-1] if sapm_images is not None else None,
                fit_radius_px=args.fit_radius,
                kappa=args.kappa,
                max_iter=args.max_iter,
                linear_bkg=args.linear_bkg,
                debug=args.debug,
                show_figs=args.show_figs,
                save_figs=args.save_figs,
                results_dir=args.results_dir,
                no_masking=args.no_masking,
            )
            if result.wv_um is not None and result.wv_um is not np.nan:
                all_results.append(result)
                if not args.quiet:
                    print(
                            f"    λ={result.wv_um or 'N/A'} µm  "
                            f"flux={result.opt_flux_MJysr or 'N/A'} MJy/sr"
                            f"  ({result.opt_flux_uJy or 'N/A'} µJy)"
                            f"  S/N={result.opt_snr or 'N/A'}"
                            f"  n_used={result.n_pix_used}"
                            f"  n_outlier={result.n_pix_outlier}"
                            f"  converged={result.converged}"
                         )
        print("Done.")
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
            
    if len(failed) > 0:
        print("The following targets failed to download: ",end="")
        for name in failed:
            print(name+" ",end="")
        print(".")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
