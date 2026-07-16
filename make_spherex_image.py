import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import talltable
import healpy
import pyarrow
import pyarrow.parquet
from scipy.signal import medfilt

from spherextract_fast import download_cutout_pixels, _BAD_BITS, _BAD_BITS_BKG, _BAD_BITS_MISC

def background_subtract_pixels(flux,imgids,flags):
    """
    Subtract background model from pixels
    """
    
    # Just doing a median background for now.
    # Linear background will require a lot more overhead...
    
    sort_idx    = np.argsort(imgids)
    sorted_ids  = imgids[sort_idx]
    unique_ids, start_indices = np.unique(sorted_ids, return_index=True)
    n_imgs = len(unique_ids)
    
    for ii in range(n_imgs):
        start = start_indices[ii]
        end = start_indices[ii+1] if ii<n_imgs-1 else len(imgids)
        idx = sort_idx[start:end]
        bm = (flags[idx].astype(np.uint32) & _BAD_BITS_BKG) != 0
        gd = np.isfinite(flux[idx]) & ~bm
        if np.sum(gd) > 0:
            flux[idx] -= np.median(flux[idx][gd])
        else:
            gd = np.isfinite(flux[idx])
            flux[idx] -= np.median(flux[idx][gd])
    return flux

ra0, dec0 = 273.8750417, 65.3483333 # EDF-N z = 5.4 faint quasar
#ra0, dec0 = 335.4703333, -18.4341472 # Luminous SkyMapper quasar
#ra0, dec0 = 345.7860500, 34.93910300 # QFC-I z = 5.43 BAL quasar
#ra0, dec0 = 255.2525833, 64.2025333 # HS1700+6416 field

# TODO: Scale nchunks based on number of image matches returned by IRSA
size = 0.1
nchunks = 10

nwv1234 = 4*nchunks
wvmin1234 = 0.733
wvmax1234 = 3.810

nwv56 = 2*nchunks
wvmin56 = 3.810
wvmax56 = 5.015

cutout_pixels_wv = [None for ii in range(nwv1234+nwv56)]

wvbins = 10**np.linspace(np.log10(wvmin1234),np.log10(wvmax1234),nwv1234+1)
for ii in range(nwv1234):
    print(f"   Query {ii+1}/{nwv1234+nwv56}")
    cutout_pixels_wv[ii] = download_cutout_pixels(ra=ra0,dec=dec0,cutout_size_deg=size,
                                                  wvbounds=[wvbins[ii],wvbins[ii+1]])
 
wvbins = 10**np.linspace(np.log10(wvmin56),np.log10(wvmax56),nwv56+1)
for ii in range(nwv56):
    print(f"   Query {nwv1234+ii+1}/{nwv1234+nwv56}")
    cutout_pixels_wv[nwv1234+ii] = download_cutout_pixels(ra=ra0,dec=dec0,cutout_size_deg=size,
                                                  wvbounds=[wvbins[ii],wvbins[ii+1]])
                                                  
pix = pyarrow.concat_tables(cutout_pixels_wv)

ra,dec = healpy.pix2ang(2**22,np.asarray(pix['hphigh']),lonlat=True,nest=True)
#keep = (ra > ra0-0.13/np.cos(dec0*np.pi/180)) & (ra < ra0+0.13/np.cos(dec0*np.pi/180)) & (dec > dec0-0.13) & (dec < dec0+0.13)
#ra = ra[keep]
#dec = dec[keep]

flux = np.asarray(pix['flux'])#[keep]
var = np.asarray(pix['variance'])#[keep]
wave = np.asarray(pix['wavelength'])#[keep]
imgids = np.asarray(pix['imageid'])#[keep]

flags = np.asarray(pix['flags'])#[keep]
bm = (flags.astype(np.uint32) & _BAD_BITS) != 0
bm = bm | ((flags.astype(np.uint32) & _BAD_BITS_MISC) != 0)

print("   Background subtraction...")
flux = background_subtract_pixels(flux,imgids,flags)
print("Done.")

dxdec = 2.0/3600.0 # final pixel scale in arcsec
dxra = dxdec/np.cos(dec0*np.pi/180) # correct for cos(dec) in RA
rabins = np.linspace(ra.min(),ra.max(),int(((ra.max()-ra.min())/dxra))+1)
decbins = np.linspace(dec.min(),dec.max(),int((dec.max()-dec.min())/dxdec)+1)

nwv = 6
wvbins = 10**np.linspace(np.log10(wave.min()),np.log10(wave.max()),nwv+1)

img = [None for ii in range(nwv)]
for ii in range(nwv):
    mw = (wave > wvbins[ii]) & (wave < wvbins[ii+1])
    # mw = mw & ~((wave > 1.06) & (wave < 1.10)) # optionally mask the He I 1.08 micron line
    img[ii],_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
    norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
    img[ii] /= norm
    gd = np.isfinite(img[ii])
    img[ii] -= np.median(img[ii][gd])

    # Re-centering the background globally
#    mid = (img[ii][gd] > np.percentile(img[ii][gd],15)) & (img[ii][gd] < np.percentile(img[ii][gd],85))
#    ph,pf = np.histogram(img[ii][gd][mid].flatten(),bins=200)
#    pfc = 0.5*(pf[:-1]+pf[1:])
#    img[ii] -= pfc[np.argmax(ph)]
#    
    # Re-centering the background locally
    # bg = ~((flags.astype(np.uint32) & _BAD_BITS_BKG) != 0)
    bkg = medfilt(img[ii],(25,25))
    img[ii] -= bkg

    # Outlier in-painting
    filt = medfilt(img[ii],(3,3))
    diff = np.abs(img[ii]-filt)
    gdd = np.isfinite(diff)
    fill = diff > np.percentile(diff[gdd],97.5)
    img[ii][fill] = filt[fill]
    
    # Renormalization
    img[ii] = np.asinh(img[ii]/0.01)

rac = (len(rabins)-1)//2
decc = (len(decbins)-1)//2
npix = np.min([len(rabins)-1,len(decbins)-1])
ramin = rabins[rac-npix//2]; ramax = rabins[rac+npix//2]
decmin = decbins[decc-npix//2]; decmax = decbins[decc+npix//2]

# Plot them up
fig,axg = plt.subplots(2,nwv//2,figsize=(20,6))
ax = axg.flatten()
for ii in range(nwv):
    gd = np.isfinite(img[ii])
    lo = np.percentile(img[ii][gd],2.5)
    hi = np.percentile(img[ii][gd],97.5)
    ax[ii].imshow(img[ii].T,vmin=lo,vmax=hi,extent=[rabins[0],rabins[-1],decbins[0],decbins[-1]],
                  origin='lower',cmap='bone_r',aspect=1/np.cos(dec0*np.pi/180))
    ax[ii].scatter([ra0],[dec0],c='darkorange',marker='x',lw=1)
    ax[ii].set_xlim(ramax,ramin)
    ax[ii].set_title(f"{wvbins[ii]:.2f}-{wvbins[ii+1]:.2f} micron")
plt.tight_layout()
plt.show()

# One band at a time

wv0 = 0.73
wv1 = 5.02

dxdec = 2.0/3600.0 # final pixel scale in arcsec
dxra = dxdec*np.cos(dec0*np.pi/180)
rabins = np.linspace(ra.min(),ra.max(),int(((ra.max()-ra.min())/dxdec))+1)
decbins = np.linspace(dec.min(),dec.max(),int((dec.max()-dec.min())/dxra)+1)

# Create the image
mw = (wave>wv0)&(wave<wv1) #&~((wave>1.02)&(wave<1.14))
img,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1.0/var[~bm&mw])
img /= norm
gd = np.isfinite(img)
img -= np.median(img[gd])

# Bias-correct the image by finding the peak of the pixel histogram
#mid = (img[gd] > np.percentile(img[gd],15)) & (img[gd] < np.percentile(img[gd],85))
#ph,pf = np.histogram(img[gd][mid].flatten(),bins=200)
#pfc = 0.5*(pf[:-1]+pf[1:])
#img -= pfc[np.argmax(ph)]

# Alternative bias correction via local median background
img -= medfilt(img,(25,25))

# Outlier in-painting
filt = medfilt(img,(3,3))
diff = np.abs(img-filt)
gdd = np.isfinite(diff)
fill = diff > np.percentile(diff[gdd],97.5)
img[fill] = filt[fill]

# Renormalization
img = np.asinh(img/0.01)

fig,ax = plt.subplots(1,1,figsize=(8,8))
plt.imshow(img.T,origin='lower',vmin=np.percentile(img[gd],2.5),vmax=np.percentile(img[gd],97.5),cmap='bone_r',
                 extent=[rabins[0],rabins[-1],decbins[0],decbins[-1]],aspect=1/np.cos(dec0*np.pi/180))
plt.scatter([ra0],[dec0],c='darkorange',marker='o',lw=1)
plt.xlim(ramax,ramin)
plt.ylim(decmin,decmax)
plt.title(f"SPHEREx $\lambda$={wv0:.2f}-{wv1:.2f} $\mu$m (Nimg={len(np.unique(np.asarray(pix['imageid'])[mw]))})")
plt.show()


# Three bands, color image

dxdec = 1.5/3600.0 # final pixel scale in arcsec
dxra = dxdec/np.cos(dec0*np.pi/180)
rabins = np.linspace(ra.min(),ra.max(),int(((ra.max()-ra.min())/dxra))+1)
decbins = np.linspace(dec.min(),dec.max(),int((dec.max()-dec.min())/dxdec)+1)

# Blue image
wv0 = 0.73
wv1 = 1.50
mw = (wave>wv0)&(wave<wv1)
bimg,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
bimg /= norm
gd = np.isfinite(bimg)
bimg -= np.median(bimg[gd])
bimg[~gd] = 0.0
# Bias-correct the image by finding the peak of the pixel histogram
#mid = (bimg[gd] > np.percentile(bimg[gd],15)) & (bimg[gd] < np.percentile(bimg[gd],85))
#ph,pf = np.histogram(bimg[gd][mid].flatten(),bins=200)
#pfc = 0.5*(pf[:-1]+pf[1:])
#bimg -= pfc[np.argmax(ph)]
bimg -= medfilt(bimg,(25,25))
# Outlier in-painting
filt = medfilt(bimg,(3,3))
diff = np.abs(bimg-filt)
gdd = np.isfinite(diff) & (diff > 0)
fill = diff > np.percentile(diff[gdd],97.5)
bimg[fill] = filt[fill]
# Renormalization
bimg = np.asinh(bimg/0.005)/6
bimg -= np.percentile(bimg[gdd],3.0)
bimg[bimg<0]=0
bimg[bimg>1]=1

# Green image
wv0 = 1.50
wv1 = 3.00
mw = (wave>wv0)&(wave<wv1)
gimg,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
gimg /= norm
gd = np.isfinite(gimg)
gimg -= np.median(gimg[gd])
gimg[~gd] = 0.0
#mid = (gimg[gd] > np.percentile(gimg[gd],15)) & (gimg[gd] < np.percentile(gimg[gd],85))
#ph,pf = np.histogram(gimg[gd][mid].flatten(),bins=200)
#pfc = 0.5*(pf[:-1]+pf[1:])
#gimg -= pfc[np.argmax(ph)]
gimg -= medfilt(gimg,(25,25))
# Outlier in-painting
filt = medfilt(gimg,(3,3))
diff = np.abs(gimg-filt)
gdd = np.isfinite(diff) & (diff > 0)
fill = diff > np.percentile(diff[gdd],97.5)
gimg[fill] = filt[fill]
# Renormalization
gimg = np.asinh(gimg/0.005)/7
gimg -= np.percentile(gimg[gdd],3.0)
gimg[gimg<0]=0
gimg[gimg>1]=1

# Red image
wv0 = 3.00
wv1 = 5.00
mw = (wave>wv0)&(wave<wv1)
rimg,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
rimg /= norm
gd = np.isfinite(rimg)
rimg -= np.median(rimg[gd])
rimg[~gd] = 0.0
#mid = (rimg[gd] > np.percentile(rimg[gd],15)) & (rimg[gd] < np.percentile(rimg[gd],85))
#ph,pf = np.histogram(rimg[gd][mid].flatten(),bins=200)
#pfc = 0.5*(pf[:-1]+pf[1:])
#rimg -= pfc[np.argmax(ph)]
rimg -= medfilt(rimg,(25,25))
# Outlier in-painting
filt = medfilt(rimg,(3,3))
diff = np.abs(rimg-filt)
gdd = np.isfinite(diff) & (diff > 0)
fill = diff > np.percentile(diff[gdd],97.5)
rimg[fill] = filt[fill]
# Renormalization
rimg = np.asinh(rimg/0.005)/6
rimg -= np.percentile(rimg[gdd],3.0)
rimg[rimg<0]=0
rimg[rimg>1]=1


fig,ax = plt.subplots(1,1,figsize=(9,9))
plt.imshow(np.array([rimg,gimg,bimg]).T,origin='lower',
           extent=[rabins[0],rabins[-1],decbins[0],decbins[-1]],aspect=1/np.cos(dec0*np.pi/180))
plt.xlim(ramax,ramin)
plt.ylim(decmin,decmax)
plt.title(r'EDF-N J1825+6520 (0.75$-$1.5 $\mu$m, 1.5$-$3 $\mu$m, 3$-$5 $\mu$m)',fontsize=20)
plt.tight_layout()
#plt.savefig('EDFN_RGB.png',dpi=150)
plt.show()


# NB-excess image

dxdec = 3.0/3600.0 # final pixel scale in arcsec
dxra = dxdec*np.cos(dec0*np.pi/180)
rabins = np.linspace(ra.min(),ra.max(),int(((ra.max()-ra.min())/dxdec))+1)
decbins = np.linspace(dec.min(),dec.max(),int((dec.max()-dec.min())/dxra)+1)

# Blue continuum
wv0 = 3.84
wv1 = 4.14
mw = (wave>wv0)&(wave<wv1)
bcimg,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
bcimg /= norm
gd = np.isfinite(bcimg)
bcimg -= np.median(bcimg[gd])
bcimg[~gd] = 0.0
# Bias-correct the image by finding the peak of the pixel histogram
#mid = (bcimg[gd] > np.percentile(bcimg[gd],15)) & (bcimg[gd] < np.percentile(bcimg[gd],85))
#ph,pf = np.histogram(bcimg[gd][mid].flatten(),bins=200)
#pfc = 0.5*(pf[:-1]+pf[1:])
#bcimg -= pfc[np.argmax(ph)]
bcimg -= medfilt(bcimg,(25,25))
# Outlier in-painting
filt = medfilt(bcimg,(3,3))
diff = np.abs(bcimg-filt)
gdd = np.isfinite(diff) & (diff > 0)
fill = diff > np.percentile(diff[gdd],97.5)
bcimg[fill] = filt[fill]

# NB image
wv0 = 4.14
wv1 = 4.24
mw = (wave>wv0)&(wave<wv1)
nbimg,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
nbimg /= norm
gd = np.isfinite(nbimg)
nbimg -= np.median(nbimg[gd])
nbimg[~gd] = 0.0
#mid = (nbimg[gd] > np.percentile(nbimg[gd],15)) & (nbimg[gd] < np.percentile(nbimg[gd],85))
#ph,pf = np.histogram(nbimg[gd][mid].flatten(),bins=200)
#pfc = 0.5*(pf[:-1]+pf[1:])
#nbimg -= pfc[np.argmax(ph)]
nbimg -= medfilt(nbimg,(25,25))
# Outlier in-painting
filt = medfilt(nbimg,(3,3))
diff = np.abs(nbimg-filt)
gdd = np.isfinite(diff) & (diff > 0)
fill = diff > np.percentile(diff[gdd],97.5)
nbimg[fill] = filt[fill]

# Red continuum
wv0 = 4.24
wv1 = 4.54
mw = (wave>wv0)&(wave<wv1)
rcimg,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
rcimg /= norm
gd = np.isfinite(rcimg)
rcimg -= np.median(rcimg[gd])
rcimg[~gd] = 0.0
#mid = (rcimg[gd] > np.percentile(rcimg[gd],15)) & (rcimg[gd] < np.percentile(rcimg[gd],85))
#ph,pf = np.histogram(rimg[gd][mid].flatten(),bins=200)
#pfc = 0.5*(pf[:-1]+pf[1:])
#rcimg -= pfc[np.argmax(ph)]
rcimg -= medfilt(rcimg,(25,25))
# Outlier in-painting
filt = medfilt(rcimg,(3,3))
diff = np.abs(rcimg-filt)
gdd = np.isfinite(diff) & (diff > 0)
fill = diff > np.percentile(diff[gdd],97.5)
rcimg[fill] = filt[fill]


fig,ax = plt.subplots(1,1,figsize=(9,9))
plt.imshow((nbimg-0.5*(rcimg+bcimg)).T,origin='lower',extent=[rabins[0],rabins[-1],decbins[0],decbins[-1]],aspect=1/np.cos(dec0*np.pi/180),cmap='managua',
            vmin=-0.05,vmax=0.05)
plt.xlim(ramax,ramin)
plt.ylim(decmin,decmax)
plt.title(r'EDF-N J1825+6520 H$\alpha$ NB excess',fontsize=20)
plt.tight_layout()
plt.savefig('EDFN_NBz54.png',dpi=150)
plt.show()


# Cube time?

ncube = 12

nwv1234 = 4*ncube
wvmin1234 = 0.733
wvmax1234 = 3.810

nwv56 = 2*ncube
wvmin56 = 3.810
wvmax56 = 5.015


dxdec = 3.0/3600.0 # final pixel scale in arcsec
dxra = dxdec/np.cos(dec0*np.pi/180)
rabins = np.linspace(ra.min(),ra.max(),int(((ra.max()-ra.min())/dxra))+1)
decbins = np.linspace(dec.min(),dec.max(),int((dec.max()-dec.min())/dxdec)+1)

cube = np.zeros((nwv1234+nwv56,len(rabins)-1,len(decbins)-1))

wvbins = 10**np.linspace(np.log10(wvmin1234),np.log10(wvmax1234),nwv1234+1)
waves1234 = 0.5*(wvbins[:-1]+wvbins[1:])
for ii in range(nwv1234):
    mw = (wave>wvbins[ii])&(wave<wvbins[ii+1])
    img,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
    norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1.0/var[~bm&mw])
    img /= norm
    gd = np.isfinite(img)
    img -= np.median(img[gd])
    img[~gd] = 0.0

    # Bias-correct the image by finding the peak of the pixel histogram
#    mid = (img[gd] > np.percentile(img[gd],15)) & (img[gd] < np.percentile(img[gd],85))
#    ph,pf = np.histogram(img[gd][mid].flatten(),bins=200)
#    pfc = 0.5*(pf[:-1]+pf[1:])
#    img -= pfc[np.argmax(ph)]

    img -= medfilt(img,(35,35))

    # Outlier in-painting
    filt = medfilt(img,(3,3))
    diff = np.abs(img-filt)
    gdd = np.isfinite(diff) & (diff > 0)
    fill = diff > np.percentile(diff[gdd],97.5)
    img[fill] = filt[fill]
    
    cube[ii] = np.copy(img)

wvbins = 10**np.linspace(np.log10(wvmin56),np.log10(wvmax56),nwv56+1)
waves56 = 0.5*(wvbins[:-1]+wvbins[1:])
for ii in range(nwv56):
    mw = (wave>wvbins[ii])&(wave<wvbins[ii+1])
    img,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
    norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1.0/var[~bm&mw])
    img /= norm
    gd = np.isfinite(img)
    img -= np.median(img[gd])
    img[~gd] = 0.0

    # Bias-correct the image by finding the peak of the pixel histogram
#    mid = (img[gd] > np.percentile(img[gd],15)) & (img[gd] < np.percentile(img[gd],85))
#    ph,pf = np.histogram(img[gd][mid].flatten(),bins=200)
#    pfc = 0.5*(pf[:-1]+pf[1:])
#    img -= pfc[np.argmax(ph)]

    img -= medfilt(img,(35,35))
    
    # Outlier in-painting
    filt = medfilt(img,(3,3))
    diff = np.abs(img-filt)
    gdd = np.isfinite(diff) & (diff > 0)
    fill = diff > np.percentile(diff[gdd],97.5)
    img[fill] = filt[fill]
    
    cube[ii+nwv1234] = np.copy(img)

waves = np.append(waves1234,waves56)

# Spit out a FITS cube

from astropy.wcs import WCS
wcs = WCS(naxis=2)
wcs.wcs.crpix = [0.5,0.5] # reference pixel
wcs.wcs.crval = [0.5*(rabins[0]+rabins[1]),0.5*(decbins[0]+decbins[1])]
wcs.wcs.cdelt = [-dxra,dxdec]
wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

header = fits.Header()
header.update(wcs.to_header())
header['BUNIT'] = 'MJr/sr'
primary_hdu = fits.PrimaryHDU(cube, header=header)

wave_hdu = fits.ImageHDU(waves, header=fits.Header())
wave_hdu.header['EXTNAME'] = 'WAVELENGTH'
wave_hdu.header['BUNIT'] = 'UM'
wave_hdu.header['TTYPE1'] = 'WAVELENGTH'

hdul = fits.HDUList([primary_hdu, wave_hdu])
hdul.writeto("spherex_cube_edfn.fits",overwrite=True)

# Write out a cube with the irregular wavelengths attached
# to be viewed with cubeviz/jdaviz (cannot get them to work properly though)

#from astropy import units as u
#from specutils import Spectrum, SpectralAxis
#
#cube_data = cube.T * u.MJy / u.sr * (6.15 * u.arcsec)**2
##cube_data[~np.isfinite(cube_data)] = 0.0
#wavelengths = waves * u.um
#spatial_wcs = WCS(naxis=2)
#spatial_wcs.wcs.ctype = ['RA---TAN','DEC--TAN']
#spatial_wcs.wcs.cunit = ['deg', 'deg']
#spatial_wcs.wcs.crval = [0.5*(rabins[0]+rabins[1]),0.5*(decbins[0]+decbins[1])]
#spatial_wcs.wcs.crpix = [0.5, 0.5]
#spatial_wcs.wcs.cdelt = [-dxdec, dxra]
#
#spec_cube = Spectrum(flux=cube_data.to('uJy'),
#                         spectral_axis=SpectralAxis(wavelengths),
#                         wcs=spatial_wcs)
#                         
#spec_cube.write('spherex_cube_spec.fits',overwrite=True)
#
#
