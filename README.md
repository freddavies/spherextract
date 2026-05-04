# spherextract
Download SPHEREx images and "optimally" extract spectrophotometry of point sources.

AI-aided synthesis of the cutout download machinery from Eduardo Bañados' [spherex-tools](https://github.com/banados/spherex-tools/) with the PSF downsampling code from [SPIFF](https://github.com/jgagneastro/SPIFF). As I know very little about doing photometry on images, the flux extraction is performed using an analogy of optimal extraction ([Horne 1986](https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract)) from spectroscopy analysis. Provided that the PSF model and variance maps are accurate, and the target is a point source, it should produce results similar to the IRSA Spectrophotometry Tool.

Two scripts are provided: 
- `spherextract.py`, a standalone single-object tool
- `spherextract_two.py`, which attempts to deblend two nearby sources

*Highly experimental, use at your own risk!*
