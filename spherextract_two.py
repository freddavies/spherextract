#!/usr/bin/env python
"""
spherextract_two.py

Joint two-source flux extraction for SPHEREx cutouts.

When a target has a nearby contaminant that falls within the PSF footprint,
a single-source optimal extraction is biased because the contaminating flux
is folded into the weight profile.  This script instead fits for the
amplitudes of *two* PSF components simultaneously via a linear least-squares
solve, with the same iterative outlier rejection as the single-source case.

The model for pixel i is:

    M_i = f1 * P1_i + f2 * P2_i

where P1, P2 are the (fixed) PSF models centred at the two sky positions
and f1, f2 are the unknown flux amplitudes [MJy/sr].  The system is linear
in (f1, f2), so the inverse-variance-weighted least-squares solution is
exact and closed-form:

    A = [[sum(P1^2 * w),  sum(P1*P2 * w)],
         [sum(P1*P2 * w), sum(P2^2 * w) ]]

    b = [sum(P1 * w * D),
         sum(P2 * w * D)]

    [f1, f2] = A^{-1} b
    Cov(f1,f2) = A^{-1}

Outlier rejection follows the same kappa-sigma scheme as
spxtract.py: pixels where

    |D_i - M_i| > kappa * sigma_i

are masked and the solve is repeated until convergence.

Usage examples
--------------
# Single cutout already on disk:
python spherextract_two.py \\
    --fits path/to/cutout.fits \\
    --ra1 83.633  --dec1 22.014  --name1 Crab \\
    --ra2 83.640  --dec2 22.019  --name2 Contaminant
    
# Download cutouts then extract (searches within --search-radius of ra1/dec1):
python spherextract_two.py \\
    --ra1 83.633  --dec1 22.014  --name1 Crab \\
    --ra2 83.640  --dec2 22.019  --name2 Contaminant \\
    --download --search-radius 5 --cutout-size 0.1

# Batch mode: text file with columns name1 ra1 dec1 name2 ra2 dec2
python spherextract_two.py --input pairs.txt --download \\
    --results-dir joint_results/

"""
from __future__ import annotations

import argparse
import csv
import os
import glob
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.wcs import WCS
import astropy.units as u

# ---------------------------------------------------------------------------
# Re-use helpers from the single-source script
# ---------------------------------------------------------------------------
from spxtract import (
    _BAD_BITS,
    _BAD_BITS_BCG,
    _build_cutout_url,
    _download_file,
    _load_wcswave,
    _make_psf_detgrid,
    _query_spherex,
    _read_input_file,
    _wcswave_eval,
    download_cutouts,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class JointExtractionResult:
    """Outputs from a joint two-source extraction on one SPHEREx cutout."""

    # --- identification ---
    name1: str
    name2: str
    fits_path: str
    ra1_deg: float
    dec1_deg: float
    ra2_deg: float
    dec2_deg: float
    separation_arcsec: float

    # --- file metadata ---
    obsid: Optional[str]
    detector_id: Optional[int]
    bandpass: Optional[str]
    expid: Optional[int]
    mjd_avg: Optional[float]
    psf_index1: Optional[int]       # PSF zone chosen for source 1
    psf_index2: Optional[int]       # PSF zone chosen for source 2
    omega_sr: Optional[float]
    px_scale_arcsec: Optional[float]

    # --- spectral WCS (evaluated at source 1 position) ---
    wv_um: Optional[float]
    wv_width_um: Optional[float]

    # --- quality ---
    near_detector_edge: bool
    n_pix_total: int
    n_pix_flagged: int
    n_pix_outlier: int
    n_pix_used: int
    n_iter: int
    converged: bool
    condition_number: Optional[float]   # of the 2×2 normal-equation matrix

    # --- background ---
    bkg_MJysr: Optional[float]
    bkg_npix: int

    # --- source 1 ---
    flux1_MJysr: Optional[float]
    flux1_MJysr_err: Optional[float]
    flux1_uJy: Optional[float]
    flux1_uJy_err: Optional[float]
    snr1: Optional[float]

    # --- source 2 ---
    flux2_MJysr: Optional[float]
    flux2_MJysr_err: Optional[float]
    flux2_uJy: Optional[float]
    flux2_uJy_err: Optional[float]
    snr2: Optional[float]

    # --- flux covariance ---
    cov12_MJysr2: Optional[float]       # off-diagonal of Cov(f1, f2)
    corr12: Optional[float]             # normalised correlation

    # --- goodness of fit ---
    chi2: Optional[float]
    dof: Optional[int]

    # --- pixel positions ---
    xpix_cutout1: Optional[float]
    ypix_cutout1: Optional[float]
    xpix_cutout2: Optional[float]
    ypix_cutout2: Optional[float]
    xpix_fulldet1: Optional[float]
    ypix_fulldet1: Optional[float]
    xpix_fulldet2: Optional[float]
    ypix_fulldet2: Optional[float]


# ---------------------------------------------------------------------------
# Core joint extraction
# ---------------------------------------------------------------------------

def joint_extract(fits_path, name1, ra1, dec1, name2, ra2, dec2,
                  extract_size = (21, 21), fit_radius_px = 4.0,
                  kappa = 4.0, max_iter = 10, debug = False,
                  show_figs = False, save_figs = False, results_dir = None,
                  no_masking = False):
    """
    Jointly fit two PSF components to a single SPHEREx cutout.

    The fit is a closed-form inverse-variance-weighted least squares solve
    for the amplitudes (f1, f2) of two fixed PSF profiles (P1, P2).
    Outlier rejection iterates the solve, masking pixels where the
    two-component residual exceeds kappa * sigma.

    Parameters
    ----------
    fits_path     : local path to the SPHEREx cutout FITS file
    name1/2       : labels for the two sources
    ra1/dec1      : sky position of source 1 (primary target) in degrees
    ra2/dec2      : sky position of source 2 (contaminant) in degrees
    extract_size  : (ny, nx) of the sub-array to extract, in detector pixels.
                    Should be large enough to encompass both PSF footprints.
    fit_radius_px : only pixels within this radius of *either* source
                    are included in the fit
    kappa         : outlier-rejection threshold in units of pixel sigma
    max_iter      : maximum outlier-rejection iterations
    debug         : print diagnostics
    show_figs     : display matplotlib figures
    save_figs     : write figures to figs_dir
    results_dir   : base directory for saved figures
    no_masking    : disable flag-based pixel masking in the fit

    Returns
    -------
    JointExtractionResult dataclass
    """

    def dprint(*a, **kw):
        if debug:
            print(*a, **kw)

    sky1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    sky2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    separation_as = float(sky1.separation(sky2).arcsec)

    def _nan_result(**overrides) -> JointExtractionResult:
        base = dict(
            name1=name1, name2=name2, fits_path=fits_path,
            ra1_deg=ra1, dec1_deg=dec1, ra2_deg=ra2, dec2_deg=dec2,
            separation_arcsec=separation_as,
            obsid=None, detector_id=None, bandpass=None,
            expid=None, mjd_avg=None,
            psf_index1=None, psf_index2=None,
            omega_sr=None, px_scale_arcsec=None,
            wv_um=None, wv_width_um=None,
            near_detector_edge=False,
            n_pix_total=0, n_pix_flagged=0,
            n_pix_outlier=0, n_pix_used=0,
            n_iter=0, converged=False, condition_number=None,
            bkg_MJysr=None, bkg_npix=0,
            flux1_MJysr=None, flux1_MJysr_err=None,
            flux1_uJy=None, flux1_uJy_err=None, snr1=None,
            flux2_MJysr=None, flux2_MJysr_err=None,
            flux2_uJy=None, flux2_uJy_err=None, snr2=None,
            cov12_MJysr2=None, corr12=None,
            chi2=None, dof=None,
            xpix_cutout1=None, ypix_cutout1=None,
            xpix_cutout2=None, ypix_cutout2=None,
            xpix_fulldet1=None, ypix_fulldet1=None,
            xpix_fulldet2=None, ypix_fulldet2=None,
        )
        base.update(overrides)
        return JointExtractionResult(**base)

    # ================================================================
    # 1. Load FITS
    # ================================================================
    try:
        hdul = fits.open(fits_path)
    except Exception as exc:
        print(f"  [ERROR] Cannot open {fits_path}: {exc}")
        return _nan_result()

    hdr      = hdul[1].header
    img      = hdul[1].data
    flags    = hdul[2].data
    var      = hdul[3].data
    psf_cube = hdul[5].data
    hdr_psf  = hdul[5].header

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
    
    ww = _load_wcswave(hdul)

    dprint(f"Opened {fits_path}")
    dprint(f"  oversamp={oversamp}, px_arcsec={px_arcsec:.3f}, "
           f"sep={separation_as:.2f}\"")

    # ================================================================
    # 2. Pixel coordinates
    #
    # Cutout pixel: derived from the file's own (possibly shifted) WCS.
    # Full-detector pixel: cutout pixel - CRPIX1A/2A offset, used for
    # PSF zone selection and the spectral WCS lookup table.
    # ================================================================
    xcut1, ycut1 = wcs_img.world_to_pixel(sky1)
    xcut2, ycut2 = wcs_img.world_to_pixel(sky2)

    xdet1 = float(xcut1) + det_origin_x
    ydet1 = float(ycut1) + det_origin_y
    xdet2 = float(xcut2) + det_origin_x
    ydet2 = float(ycut2) + det_origin_y

    h_img, w_img = img.shape
    near_edge = bool(
        xcut1 < 10 or ycut1 < 10
        or (w_img - xcut1) < 10
        or (h_img - ycut1) < 10
    )

    dprint(f"  Source 1 cutout px: ({xcut1:.3f}, {ycut1:.3f})")
    dprint(f"  Source 2 cutout px: ({xcut2:.3f}, {ycut2:.3f})")

    # ================================================================
    # 3. Cutout centred on the midpoint of the two sources
    #    so that both PSF footprints are included
    # ================================================================
    xmid = 0.5 * (xcut1 + xcut2)
    ymid = 0.5 * (ycut1 + ycut2)

    try:
        cut_img   = Cutout2D(img,   (xmid, ymid), extract_size, wcs=wcs_img,
                             mode="partial", fill_value=np.nan)
        cut_flags = Cutout2D(flags, (xmid, ymid), extract_size, wcs=wcs_img,
                             mode="partial", fill_value=0).data
        cut_var   = Cutout2D(var,   (xmid, ymid), extract_size, wcs=wcs_img,
                             mode="partial", fill_value=np.nan).data
    except NoOverlapError:
        print("  [WARN] Both sources outside image footprint; skipping.")
        return _nan_result(near_detector_edge=near_edge)

    # Refine both positions using the cutout's own WCS
    wcs_cut = cut_img.wcs
    xcut1, ycut1 = wcs_cut.world_to_pixel(sky1)
    xcut2, ycut2 = wcs_cut.world_to_pixel(sky2)

    data_raw = cut_img.data.copy()
    H, W = data_raw.shape
    dprint(f"  Cutout shape: {H}×{W}")
    dprint(f"  Source 1 in cutout: ({xcut1:.3f}, {ycut1:.3f})")
    dprint(f"  Source 2 in cutout: ({xcut2:.3f}, {ycut2:.3f})")

    # ================================================================
    # 4. Background subtraction
    # ================================================================
    mask_bcg = (cut_flags.astype(np.uint32) & _BAD_BITS_BCG) != 0
    good_bcg = (~mask_bcg) & np.isfinite(data_raw)
    bkg_npix = int(np.sum(good_bcg))

    if bkg_npix >= 3:
        bkg = float(np.median(data_raw[good_bcg]))
    else:
        good_any = np.isfinite(data_raw)
        bkg = float(np.median(data_raw[good_any])) if np.any(good_any) else 0.0
        bkg_npix = int(np.sum(good_any))
        dprint("  [WARN] Few BCG pixels; using unmasked background estimate.")

    data = data_raw - bkg
    dprint(f"  Background: {bkg:.4g} MJy/sr ({bkg_npix} pixels)")

    # ================================================================
    # 5. PSF zone selection (independently for each source)
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
        idx_psf1 = idx_psf2 = 0
    else:
        xctrs = np.array([v for _, v in xctr_items[:nzone]])
        yctrs = np.array([v for _, v in yctr_items[:nzone]])
        idx_psf1 = int(np.argmin(np.hypot(xctrs - xdet1, yctrs - ydet1)))
        idx_psf2 = int(np.argmin(np.hypot(xctrs - xdet2, yctrs - ydet2)))

    dprint(f"  PSF zone: source 1 → {idx_psf1}, source 2 → {idx_psf2}")

    # ================================================================
    # 6. Build detector-grid PSF profiles for each source
    # ================================================================
    P1 = _make_psf_detgrid(psf_cube[idx_psf1], oversamp, (H, W), xcut1, ycut1)
    P2 = _make_psf_detgrid(psf_cube[idx_psf2], oversamp, (H, W), xcut2, ycut2)

    psf_sum1 = float(np.nansum(P1))
    psf_sum2 = float(np.nansum(P2))
    dprint(f"  PSF sums: P1={psf_sum1:.4g}, P2={psf_sum2:.4g}")

    # ================================================================
    # 7. Pixel masks and inverse-variance weights
    # ================================================================
    YY, XX = np.indices((H, W))
    r2_1 = (XX - xcut1) ** 2 + (YY - ycut1) ** 2
    r2_2 = (XX - xcut2) ** 2 + (YY - ycut2) ** 2
    # Include pixels within fit_radius_px of *either* source
    radmask = (r2_1 <= fit_radius_px ** 2) | (r2_2 <= fit_radius_px ** 2)

    if no_masking:
        flag_mask = np.zeros((H, W), dtype=bool)
    else:
        flag_mask = (cut_flags.astype(np.uint32) & _BAD_BITS) != 0

    ivar = np.where(
        (cut_var > 0) & np.isfinite(cut_var),
        1.0 / cut_var,
        0.0,
    )

    base_good = radmask & (~flag_mask) & np.isfinite(data) & (ivar > 0)

    if not np.any(base_good):
        dprint("  [WARN] No usable pixels with flag masking; retrying without.")
        flag_mask = np.zeros((H, W), dtype=bool)
        base_good = radmask & np.isfinite(data) & (ivar > 0)

    n_total   = int(np.sum(radmask))
    n_flagged = int(np.sum(flag_mask & radmask))
    data_safe = np.where(np.isfinite(data), data, 0.0)

    # ================================================================
    # 8. Iterative joint least-squares with outlier rejection
    #
    # At each iteration we solve the 2×2 normal equations:
    #
    #   A [f1, f2]^T = b
    #
    # where
    #   A[0,0] = Σ_i P1_i² w_i
    #   A[1,1] = Σ_i P2_i² w_i
    #   A[0,1] = A[1,0] = Σ_i P1_i P2_i w_i
    #   b[0]   = Σ_i P1_i w_i D_i
    #   b[1]   = Σ_i P2_i w_i D_i
    #
    # The full 2×2 covariance matrix is A^{-1}.
    # ================================================================
    good          = base_good.copy()
    outlier_mask  = np.zeros((H, W), dtype=bool)
    f1 = f2       = 0.0
    cov           = np.full((2, 2), np.nan)
    converged     = False
    n_iter        = 0
    cond          = np.nan

    for iteration in range(max_iter + 1):
        n_iter = iteration
        w = ivar * good.astype(float)   # zero weight for excluded pixels

        A = np.array([
            [np.sum(P1 * P1 * w), np.sum(P1 * P2 * w)],
            [np.sum(P1 * P2 * w), np.sum(P2 * P2 * w)],
        ])
        b = np.array([
            np.sum(P1 * w * data_safe),
            np.sum(P2 * w * data_safe),
        ])

        cond = np.linalg.cond(A)

        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
            dprint(f"  [WARN] Non-finite normal equations at iter {iteration}.")
            break

        if cond > 1e10:
            # The two PSF profiles are nearly co-linear: the problem is
            # ill-conditioned (sources too close / too similar).
            dprint(f"  [WARN] Ill-conditioned system (cond={cond:.2e}); "
                   "sources may be indistinguishable at this resolution.")

        try:
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            dprint("  [WARN] Singular normal-equation matrix; aborting.")
            break

        params = cov @ b
        f1, f2 = float(params[0]), float(params[1])

        dprint(
            f"  iter {iteration}: f1={f1:.4g}, f2={f2:.4g} MJy/sr, "
            f"good_px={int(np.sum(good))}, cond={cond:.2e}"
        )

        if iteration == max_iter:
            converged = True
            break

        # Outlier detection against the two-component model
        model_px   = f1 * P1 + f2 * P2
        resid      = data_safe - model_px
        sigma_px   = np.sqrt(np.where(cut_var > 0, cut_var, np.inf))
        newly_bad  = (np.abs(resid) > kappa * sigma_px) & radmask & good

        if not np.any(newly_bad):
            converged = True
            dprint(f"  Converged after {iteration} iteration(s).")
            break

        outlier_mask |= newly_bad
        good = base_good & ~outlier_mask

        if not np.any(good):
            dprint("  [WARN] All pixels rejected; reverting outlier mask.")
            outlier_mask[:] = False
            good = base_good.copy()
            converged = False
            break

    n_outlier = int(np.sum(outlier_mask & radmask))
    n_used    = int(np.sum(good))

    # ================================================================
    # 9. Uncertainties, covariance, chi²
    # ================================================================
    sig1 = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
    sig2 = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan
    cov12 = float(cov[0, 1]) if np.isfinite(cov[0, 1]) else np.nan
    corr12 = (
        cov12 / (sig1 * sig2)
        if (np.isfinite(sig1) and np.isfinite(sig2) and sig1 > 0 and sig2 > 0)
        else np.nan
    )

    snr1 = f1 / sig1 if (np.isfinite(sig1) and sig1 > 0) else np.nan
    snr2 = f2 / sig2 if (np.isfinite(sig2) and sig2 > 0) else np.nan

    model_final = f1 * P1 + f2 * P2
    resid_final = data_safe - model_final
    w_final     = ivar * good.astype(float)
    chi2_val    = float(np.sum(resid_final ** 2 * w_final))
    dof_val     = max(n_used - 2, 1)   # 2 free parameters

    dprint(
        f"  f1={f1:.4g}±{sig1:.3g}, f2={f2:.4g}±{sig2:.3g} MJy/sr  "
        f"corr={corr12:.3f}  χ²/dof={chi2_val:.1f}/{dof_val}"
    )

    # ================================================================
    # 10. Convert amplitude → integrated flux [µJy]
    # ================================================================
    def _to_uJy(amp, psf_sum):
        return amp * omega_sr * psf_sum * 1e12 if (
            np.isfinite(omega_sr) and psf_sum > 0
        ) else np.nan

    flux1_uJy     = _to_uJy(f1,   psf_sum1)
    flux1_uJy_err = _to_uJy(sig1, psf_sum1)
    flux2_uJy     = _to_uJy(f2,   psf_sum2)
    flux2_uJy_err = _to_uJy(sig2, psf_sum2)

    dprint(
        f"  Source 1: {f1:.4g} ± {sig1:.3g} MJy/sr  "
        f"≈ {flux1_uJy:.4g} ± {flux1_uJy_err:.3g} µJy  S/N={snr1:.2f}"
    )
    dprint(
        f"  Source 2: {f2:.4g} ± {sig2:.3g} MJy/sr  "
        f"≈ {flux2_uJy:.4g} ± {flux2_uJy_err:.3g} µJy  S/N={snr2:.2f}"
    )

    # ================================================================
    # 11. Spectral WCS (evaluated at source 1 position)
    # ================================================================
    wv_um, wv_width_um = _wcswave_eval(xdet1, ydet1, ww)
    dprint(f"  Spectral WCS (source 1): λ={wv_um:.5f} µm, Δλ={wv_width_um:.5f} µm")

    # ================================================================
    # 12. Optional diagnostic figure
    # ================================================================
    if show_figs or save_figs:
        _plot_joint(
            data=data,
            model=model_final,
            P1=P1, P2=P2,
            good=good,
            outlier_mask=outlier_mask,
            radmask=radmask,
            xcut1=xcut1, ycut1=ycut1,
            xcut2=xcut2, ycut2=ycut2,
            fit_radius_px=fit_radius_px,
            f1=f1, sig1=sig1, flux1_uJy=flux1_uJy, snr1=snr1,
            f2=f2, sig2=sig2, flux2_uJy=flux2_uJy, snr2=snr2,
            chi2=chi2_val, dof=dof_val,
            name1=name1, name2=name2,
            show_figs=show_figs,
            save_figs=save_figs,
            results_dir=results_dir,
            fits_path=fits_path,
            obsid=obsid,
        )

    hdul.close()

    return JointExtractionResult(
        name1=name1, name2=name2,
        fits_path=fits_path,
        ra1_deg=ra1, dec1_deg=dec1,
        ra2_deg=ra2, dec2_deg=dec2,
        separation_arcsec=separation_as,
        obsid=obsid,
        detector_id=det_id_int,
        bandpass=bandpass,
        expid=int(expid_val) if expid_val is not None else None,
        mjd_avg=float(mjd_avg_val) if mjd_avg_val is not None else None,
        psf_index1=int(idx_psf1),
        psf_index2=int(idx_psf2),
        omega_sr=float(omega_sr) if np.isfinite(omega_sr) else None,
        px_scale_arcsec=float(px_arcsec),
        wv_um=float(wv_um) if np.isfinite(wv_um) else None,
        wv_width_um=float(wv_width_um) if np.isfinite(wv_width_um) else None,
        near_detector_edge=near_edge,
        n_pix_total=n_total,
        n_pix_flagged=n_flagged,
        n_pix_outlier=n_outlier,
        n_pix_used=n_used,
        n_iter=n_iter,
        converged=converged,
        condition_number=float(cond) if np.isfinite(cond) else None,
        bkg_MJysr=bkg,
        bkg_npix=bkg_npix,
        flux1_MJysr=float(f1),
        flux1_MJysr_err=float(sig1) if np.isfinite(sig1) else None,
        flux1_uJy=float(flux1_uJy) if np.isfinite(flux1_uJy) else None,
        flux1_uJy_err=float(flux1_uJy_err) if np.isfinite(flux1_uJy_err) else None,
        snr1=float(snr1) if np.isfinite(snr1) else None,
        flux2_MJysr=float(f2),
        flux2_MJysr_err=float(sig2) if np.isfinite(sig2) else None,
        flux2_uJy=float(flux2_uJy) if np.isfinite(flux2_uJy) else None,
        flux2_uJy_err=float(flux2_uJy_err) if np.isfinite(flux2_uJy_err) else None,
        snr2=float(snr2) if np.isfinite(snr2) else None,
        cov12_MJysr2=cov12 if np.isfinite(cov12) else None,
        corr12=float(corr12) if np.isfinite(corr12) else None,
        chi2=float(chi2_val),
        dof=int(dof_val),
        xpix_cutout1=float(xcut1),
        ypix_cutout1=float(ycut1),
        xpix_cutout2=float(xcut2),
        ypix_cutout2=float(ycut2),
        xpix_fulldet1=float(xdet1),
        ypix_fulldet1=float(ydet1),
        xpix_fulldet2=float(xdet2),
        ypix_fulldet2=float(ydet2),
    )


# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------

def _plot_joint(data, model, P1, P2, good, outlier_mask, radmask,
                xcut1, ycut1, xcut2, ycut2, fit_radius_px,
                f1, sig1, flux1_uJy, snr1, f2, sig2, flux2_uJy, snr2,
                chi2, dof, name1, name2, show_figs, save_figs, results_dir,
                fits_path, obsid):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    fig, axs = plt.subplots(1, 4, figsize=(15, 4.5))
    fig.suptitle(
        f"{name1}: {f1:.4g}±{sig1:.3g} MJy/sr ({flux1_uJy:.4g} µJy, "
        f"S/N={snr1:.1f})   |   "
        f"{name2}: {f2:.4g}±{sig2:.3g} MJy/sr ({flux2_uJy:.4g} µJy, "
        f"S/N={snr2:.1f})   |   χ²/dof={chi2:.1f}/{dof}",
        fontsize=12,
    )

    resid = data - model
    vmax  = np.nanpercentile(np.abs(data[radmask]), 99.5)
    vmin  = np.nanpercentile(np.abs(data[radmask]),  0.5)
    res_std = np.nanstd(resid[good]) if np.any(good) else 1.0

    axs[0].imshow(data,       origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
    axs[0].set_title("Data (bkg-subtracted)")
#    axs[1].imshow(f1 * P1,    origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
#    axs[1].set_title(f"Model: {name1}")
#    axs[2].imshow(f2 * P2,    origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
#    axs[2].set_title(f"Model: {name2}")
    axs[1].imshow(f1 * P1 + f2 * P2,    origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
    axs[1].set_title(f"Model: {name1}+{name2}")
    axs[2].imshow(resid,       origin="lower",
                  vmin=-3 * res_std, vmax=3 * res_std, cmap="RdBu_r")
    axs[2].set_title("Residual")

    # Pixel classification
    cls = np.zeros(data.shape, dtype=int)
    cls[radmask]                          = 1   # used
    cls[radmask & ~good & ~outlier_mask]  = 2   # flag-masked
    cls[outlier_mask & radmask]           = 3   # outlier-rejected
    cmap = ListedColormap(["white", "steelblue", "orange", "red"])
    axs[3].imshow(cls, origin="lower", cmap=cmap, vmin=0, vmax=3)
    axs[3].set_title("Pixel classification")
    legend_els = [
        mpatches.Patch(facecolor="white",     edgecolor="k", label="outside radius"),
        mpatches.Patch(facecolor="steelblue", label="used"),
        mpatches.Patch(facecolor="orange",    label="flag-masked"),
        mpatches.Patch(facecolor="red",       label="outlier-rejected"),
    ]
    axs[3].legend(handles=legend_els, loc="upper right", fontsize=9)

    markers = [
        (xcut1, ycut1, "+", "white",  name1),
        (xcut2, ycut2, "x", "yellow", name2),
    ]
    for ax in axs:
        for xc, yc, mk, col, lbl in markers:
            ax.scatter([xc], [yc], marker=mk, s=60, color=col,
                       zorder=5, label=lbl)
        for xc, yc in [(xcut1, ycut1), (xcut2, ycut2)]:
            ax.add_patch(plt.Circle(
                (xc, yc), fit_radius_px,
                fill=False, linestyle="--", color="white", linewidth=1.0,
            ))

    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles, labels, loc="lower right", fontsize=9)

    plt.tight_layout()

    if show_figs:
        plt.show()
    elif save_figs and results_dir:
        figs_dir = os.path.join(results_dir, f"{name1}_{name2}_figs")
        Path(figs_dir).mkdir(parents=True, exist_ok=True)
        tag = obsid or Path(fits_path).stem
        fig.savefig(
            Path(figs_dir) / f"{name1}_{name2}_{tag}_joint.png", dpi=130
        )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Input file reader for pairs
# ---------------------------------------------------------------------------

def _read_pairs_file(filename: str):
    """
    Read a file with columns: name1 ra1 dec1 name2 ra2 dec2
    Returns a list of (name1, ra1, dec1, name2, ra2, dec2) tuples.
    """
    for delim in [None, " ", "\t", ","]:
        try:
            data = ascii.read(filename, delimiter=delim)
            cols = [c.lower() for c in data.colnames]
            required = ["name1", "ra1", "dec1", "name2", "ra2", "dec2"]
            if not all(r in cols for r in required):
                continue
            pairs = []
            for row in data:
                pairs.append((
                    str(row["name1"]).strip(), float(row["ra1"]), float(row["dec1"]),
                    str(row["name2"]).strip(), float(row["ra2"]), float(row["dec2"]),
                ))
            return pairs
        except Exception:
            continue
    raise ValueError(
        f"Cannot parse {filename}. "
        "Expected columns: name1 ra1 dec1 name2 ra2 dec2"
    )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _results_to_csv(results: List[JointExtractionResult], out_path: str) -> None:
    if not results:
        print("  [WARN] No results to write.")
        return
    rows = [asdict(r) for r in results]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
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
    fx1 = np.array([results[ii].flux1_uJy for ii in range(len(results))])
    er1 = np.array([results[ii].flux1_uJy_err for ii in range(len(results))])
    fx2 = np.array([results[ii].flux2_uJy for ii in range(len(results))])
    er2 = np.array([results[ii].flux2_uJy_err for ii in range(len(results))])
    
    # We usually want the spectrum file to be sorted from short to long wavelength
    idx = np.argsort(wv)
    wv = wv[idx]; fx1 = fx1[idx]; er1 = er1[idx]; fx2 = fx2[idx]; er2 = er2[idx]
    
    np.savetxt(out_path,np.vstack([wv,fx1,er1,fx2,er2]).T)
    
    print(f"  Spectrum written to {out_path}")



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Joint two-source PSF extraction for SPHEREx cutouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Target specification
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", "-i", metavar="FILE",
                     help="Pairs file with columns: name1 ra1 dec1 name2 ra2 dec2")
    grp.add_argument("--ra1", type=float,
                     help="RA of source 1 in decimal degrees")

    p.add_argument("--dec1",  type=float)
    p.add_argument("--name1", type=str, default="source1")
    p.add_argument("--ra2",   type=float)
    p.add_argument("--dec2",  type=float)
    p.add_argument("--name2", type=str, default="source2")

    # FITS source
    p.add_argument("--fits", dest="fits_path", default=None,
                   help="Path to a single cutout FITS file (skips download)")
    p.add_argument("--fits-dir", default="spherex_cutouts",
                   help="Directory of downloaded cutout FITS files")
    p.add_argument("--download", action="store_true",
                   help="Query IRSA and download cutouts before extracting")
    p.add_argument("--search-radius", type=float, default=3.0, metavar="ARCSEC")
    p.add_argument("--cutout-size", type=float, default=0.05, metavar="DEG",
                   help="Cutout download size in degrees (default: 0.05)")
    p.add_argument("--overwrite", action="store_true")

    # Extraction
    p.add_argument("--fit-radius", type=float, default=4.0, metavar="PX",
                   help="Extraction radius around each source in pixels (default: 4.0)")
    p.add_argument("--kappa",    type=float, default=4.0)
    p.add_argument("--max-iter", type=int,   default=10)
    p.add_argument("--no-masking", action="store_true")

    # Output
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--show-figs",   action="store_true")
    p.add_argument("--save-figs",   action="store_true")
    p.add_argument("--results-dir",  default="joint_results")

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate single-pair mode
    if args.ra1 is not None:
        if args.dec1 is None or args.ra2 is None or args.dec2 is None:
            parser.error("--ra1/--dec1/--ra2/--dec2 must all be provided together.")
        pairs = [(args.name1, args.ra1, args.dec1, args.name2, args.ra2, args.dec2)]
    else:
        pairs = _read_pairs_file(args.input)

    for name1, ra1, dec1, name2, ra2, dec2 in pairs:
        all_results: List[JointExtractionResult] = []
    
        print(f"\n{'='*60}")
        print(f"Pair: {name1} ({ra1:.6f}, {dec1:.6f})  +  "
              f"{name2} ({ra2:.6f}, {dec2:.6f})")

        # ----------------------------------------------------------------
        # Collect FITS files to process
        # ----------------------------------------------------------------
        if args.fits_path:
            fits_paths = [args.fits_path]

        elif args.download:
            # Download centred on source 1; use a generous cutout so
            # source 2 is also captured
            fits_paths = download_cutouts(
                name=name1,
                ra=ra1,
                dec=dec1,
                output_dir=args.fits_dir,
                search_radius_arcsec=args.search_radius,
                cutout_size_deg=args.cutout_size,
                overwrite=args.overwrite,
            )
        else:
            pattern = os.path.join(args.fits_dir, f"{name1}_cutout_*.fits")
            fits_paths = sorted(glob.glob(pattern))
            if not fits_paths:
                fits_paths = sorted(glob.glob(
                    os.path.join(args.fits_dir, "*.fits")
                ))
            print(f"  Found {len(fits_paths)} local file(s).")

        if not fits_paths:
            print(f"  No FITS files for {name1}; skipping.")
            continue

        # ----------------------------------------------------------------
        # Joint extraction from each file
        # ----------------------------------------------------------------
        for fits_path in fits_paths:
            print(f"  Extracting: {os.path.basename(fits_path)}")
            result = joint_extract(
                fits_path=fits_path,
                name1=name1, ra1=ra1, dec1=dec1,
                name2=name2, ra2=ra2, dec2=dec2,
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

            print(
                f"    λ={result.wv_um or 'N/A':.4f} µm  "
                f"{name1}: {result.flux1_MJysr or 'N/A'} MJy/sr "
                f"({result.flux1_uJy or 'N/A'} µJy, "
                f"S/N={result.snr1 or 'N/A':.2f})  |  "
                f"{name2}: {result.flux2_MJysr or 'N/A'} MJy/sr "
                f"({result.flux2_uJy or 'N/A'} µJy, "
                f"S/N={result.snr2 or 'N/A':.2f})  |  "
                f"corr={result.corr12 or 'N/A':.3f}  "
                f"cond={result.condition_number or 'N/A':.2e}  "
                f"converged={result.converged}"
            )

        # ----------------------------------------------------------------
        # Combined CSV
        # ----------------------------------------------------------------
        print(f"\n{'='*60}")
        if all_results:
            os.makedirs(args.results_dir, exist_ok=True)
            csv_path = os.path.join(args.results_dir, f"{name1}_spherex_joint_photometry.csv")
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            _results_to_csv(all_results, csv_path)
            txt_path = os.path.join(args.results_dir, f"{name1}_spherex_joint_spectrum.txt")
            _spec_to_txt(all_results, txt_path)
        else:
            print("  No results to write.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
