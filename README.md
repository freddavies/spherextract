# spherextract
Download SPHEREx images and "optimally" extract spectrophotometry of point sources.

AI-aided synthesis of the cutout download machinery from `<https://github.com/banados/spherex-tools/>` with the PSF downsampling code from `<https://github.com/jgagneastro/SPIFF>`. As I know very little about doing photometry on images, the flux extraction is performed using an analogy of optimal extraction (Horne 1986) from spectroscopy analysis. Provided that the PSF model and variance maps are accurate, and the target is a point source, it should produce results similar to the IRSA Spectrophotometry Tool.

Two scripts are provided: `spherextract.py`, a standalone single-object tool, and `spherextract_two.py`, which attempts to deblend two nearby sources using a multi-object least-squares analogy to optimal extraction.

*Highly experimental, use at your own risk!*
