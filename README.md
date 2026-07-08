# spherextract
Download SPHEREx images and "optimally" extract spectrophotometry of point sources.

AI-aided synthesis of the cutout download machinery from Eduardo Bañados' [spherex-tools](https://github.com/banados/spherex-tools/) with the PSF downsampling code from Jonathan Gagné's [SPIFF](https://github.com/jgagneastro/SPIFF) (see also: https://arxiv.org/abs/2604.22012). As I personally know very little about doing photometry on images, the flux extraction is performed using an analogy of optimal extraction ([Horne 1986](https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract)) from spectroscopy analysis, effectively a matched filter with outlier rejection. Provided that the PSF model and variance maps are accurate, and the target is a point source, it should produce results vaguely similar to the IRSA Spectrophotometry Tool.

Update 25.06.2026: now with experimental support for talltable, enabling much faster data downloads.

Four scripts are provided: 
- `spherextract_fast.py`, **NEW** a much faster version of the single-object tool that uses [talltable](https://github.com/cmhainje/talltable/) to download the data. Updates to the extraction or other parts of the code will generally be focused here, but may eventually trickle down to the rest.
- `spherextract.py`, a standalone single-object tool
- `spherextract_two.py`, which attempts to deblend two nearby sources
- `spherextract_three.py`, which attempts to deblend three nearby sources

The combined spectrum is written out in a results directory both as a csv file with all the associated metadata, and as a text file with only the wavelength, flux, and error columns.

*Highly experimental, use at your own risk!*

# Examples

- ```
  python spherextract.py --ra 129.1827 --dec 0.914806 --name J0836p0054 \
      --results-dir results_J0836p0054/ --save-figs
  ```

Download the cutouts for the quasar SDSS J0836+0054 to the default directory `spherex_cutouts/J0836p0054/`, and save the resulting single-object extractions to `results_J0836p0054/`. Diagnostic figures for the extraction from each spectral image will be saved to `results_J0836p0054/J0836p0054_figs/`.


- ```
  python spherextract_two.py --ra1 202.5219167 --dec1 -9.0843944 --name1 J1330_QSO \
    --ra2 202.52297676 --dec2 -9.086666152 --name2 J1330_STAR \
    --results-dir results_J1330m0905/ --download --kappa 25 --save-figs
  ```

Download the cutouts for the lensed quasar J1330-0905 to the default directory `spherex_cutouts/J1330_QSO/`, save the resulting two-object extractions to `results_J1330m0905/`, and save diagnostic plots to `results_J1330m0905/J1330_QSO_figs/`. This close blend (~10 arcsec) is tricky because both objects are fairly bright; this means that the default outlier rejection (kappa = 4) is too aggressive, and masks most of the useful pixels.

The `spherextract_three.py` tool has a similar syntax, but with an additional ra3, dec3, name3.

There are many other command line options, run any of the scripts with `-h` to take a look.

# Known issues

- The single-object tool assumes a point-source morphology for the target. Extended sources may be supported in the future.

- Support for more than three objects will be implemented in the future using a straightforward extrapolation of the least-squares method in the two- and three-object tools. I just have to come up with a reasonable data model, it is kind of a pain. Multi-object modes will be merged into the fast code in due time.

- The default median background is sometimes inaccurate, particularly in the vicinity of sky lines (He I 1.08 micron). A linear background model can be used in this case with `--linear-bkg`.

- The talltable queries can fail if your current internet connection is not fast enough to download the data in time. 
