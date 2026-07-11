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
size = 0.1
nchunks = 3

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

flux = np.asarray(pix['flux'])
var = np.asarray(pix['variance'])
wave = np.asarray(pix['wavelength'])
imgids = np.asarray(pix['imageid'])

flags = np.asarray(pix['flags'])
bm = (flags.astype(np.uint32) & _BAD_BITS) != 0
bm = bm | ((flags.astype(np.uint32) & _BAD_BITS_MISC) != 0)

print("   Background subtraction...")
flux = background_subtract_pixels(flux,imgids,flags)
print("Done.")

ra,dec = healpy.pix2ang(2**22,np.asarray(pix['hphigh']),lonlat=True,nest=True)
dxdec = 4.0/3600.0 # final pixel scale in arcsec
dxra = dxdec*np.cos(dec0*np.pi/180)
rabins = np.linspace(ra.min(),ra.max(),int(((ra.max()-ra.min())/dxdec))+1)
decbins = np.linspace(dec.min(),dec.max(),int((dec.max()-dec.min())/dxra)+1)

nwv = 6
#wvbins = np.linspace(wave.min(),wave.max(),nwv+1)
wvbins = 10**np.linspace(np.log10(wave.min()),np.log10(wave.max()),nwv+1)

img = [None for ii in range(nwv)]
for ii in range(nwv):
    mw = (wave > wvbins[ii]) & (wave < wvbins[ii+1])
    mw = mw & ~((wave > 1.06) & (wave < 1.10)) # mask out the He I 1.08 micron line for now
    img[ii],_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
    norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
    img[ii] /= norm
    gd = np.isfinite(img[ii])
    img[ii] -= np.median(img[ii][gd])

    ph,pf = np.histogram(img[ii][gd].flatten(),bins=10000)
    pfc = 0.5*(pf[:-1]+pf[1:])
    img[ii] -= pfc[np.argmax(ph)]

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
    ax[ii].imshow(img[ii].T,vmin=lo,vmax=hi,extent=[ramax,ramin,decmin,decmax],
                  origin='lower',cmap='bone_r',aspect=1/np.cos(dec0*np.pi/180))
    ax[ii].scatter([ra0],[dec0],c='darkorange',marker='x',lw=1)
    ax[ii].set_xlim(ramax,ramin)
    ax[ii].set_title(f"{wvbins[ii]:.2f}-{wvbins[ii+1]:.2f}")
plt.tight_layout()
plt.show()



# One band at a time


wv0 = 1.25
wv1 = 5.00

dxdec = 3.0/3600.0 # final pixel scale in arcsec
dxra = dxdec*np.cos(dec0*np.pi/180)
rabins = np.linspace(ra.min(),ra.max(),int(((ra.max()-ra.min())/dxdec))+1)
decbins = np.linspace(dec.min(),dec.max(),int((dec.max()-dec.min())/dxra)+1)
rac = (len(rabins)-1)//2
decc = (len(decbins)-1)//2
npix = np.min([len(rabins)-1,len(decbins)-1])
ramin = rabins[rac-npix//2]; ramax = rabins[rac+npix//2]
decmin = decbins[decc-npix//2]; decmax = decbins[decc+npix//2]

# Create the image
mw = (wave>wv0)&(wave<wv1)&~((wave>1.02)&(wave<1.14))
img,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1.0/var[~bm&mw])
img /= norm
gd = np.isfinite(img)
img -= np.median(img[gd])

# Bias-correct the image by finding the peak of the pixel histogram
ph,pf = np.histogram(img[gd].flatten(),bins=10000)
pfc = 0.5*(pf[:-1]+pf[1:])
img -= pfc[np.argmax(ph)]

# Outlier in-painting
filt = medfilt(img,(3,3))
diff = np.abs(img-filt)
gdd = np.isfinite(diff)
fill = diff > np.percentile(diff[gdd],97.5)
img[fill] = filt[fill]

img = np.asinh(img/0.01)

fig,ax = plt.subplots(1,1,figsize=(8,8))
plt.imshow(img.T,origin='lower',vmin=np.percentile(img[gd],2.5),vmax=np.percentile(img[gd],97.5),cmap='bone_r',
             extent=[ramax,ramin,decmin,decmax],aspect=1/np.cos(dec0*np.pi/180))
plt.scatter([ra0],[dec0],c='darkorange',marker='o',lw=1)
#plt.xlim(npix,0)
plt.title(f"SPHEREx $\lambda$={wv0:.2f}-{wv1:.2f} $\mu$m (Nimg={len(np.unique(np.asarray(pix['imageid'])[mw]))})")
plt.show()


# Three bands, color image

dxdec = 4.0/3600.0 # final pixel scale in arcsec
dxra = dxdec*np.cos(dec0*np.pi/180)
rabins = np.linspace(ra.min(),ra.max(),int(((ra.max()-ra.min())/dxdec))+1)
decbins = np.linspace(dec.min(),dec.max(),int((dec.max()-dec.min())/dxra)+1)
rac = (len(rabins)-1)//2
decc = (len(decbins)-1)//2
npix = np.min([len(rabins)-1,len(decbins)-1])
ramin = rabins[rac-npix//2]; ramax = rabins[rac+npix//2]
decmin = decbins[decc-npix//2]; decmax = decbins[decc+npix//2]

# Blue image
wv0 = 3.0
wv1 = 4.12
mw = (wave>wv0)&(wave<wv1)
bimg,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
bimg /= norm
gd = np.isfinite(bimg)
bimg -= np.median(bimg[gd])
# Bias-correct the image by finding the peak of the pixel histogram
ph,pf = np.histogram(bimg[gd].flatten(),bins=10000)
pfc = 0.5*(pf[:-1]+pf[1:])
bimg -= pfc[np.argmax(ph)]
# Outlier in-painting
filt = medfilt(bimg,(3,3))
diff = np.abs(bimg-filt)
gdd = np.isfinite(diff)
fill = diff > np.percentile(diff[gdd],97.5)
bimg[fill] = filt[fill]
# Renormalization
bimg = np.asinh(bimg/0.005)/6
bimg -= np.percentile(bimg[gd],5)
bimg[bimg<0]=0
bimg[bimg>1]=1

# Green image
wv0 = 4.12
wv1 = 4.26
mw = (wave>wv0)&(wave<wv1)
gimg,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
gimg /= norm
gd = np.isfinite(gimg)
gimg -= np.median(gimg[gd])
ph,pf = np.histogram(gimg[gd].flatten(),bins=10000)
pfc = 0.5*(pf[:-1]+pf[1:])
gimg -= pfc[np.argmax(ph)]
# Outlier in-painting
filt = medfilt(gimg,(3,3))
diff = np.abs(gimg-filt)
gdd = np.isfinite(diff)
fill = diff > np.percentile(diff[gdd],97.5)
gimg[fill] = filt[fill]
# Renormalization
gimg = np.asinh(gimg/0.005)/6
gimg -= np.percentile(gimg[gd],10)
gimg[gimg<0]=0
gimg[gimg>1]=1

# Red image
wv0 = 4.26
wv1 = 5.00
mw = (wave>wv0)&(wave<wv1)
rimg,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=flux[~bm&mw]/var[~bm&mw])
norm,_,_ = np.histogram2d(ra[~bm&mw],dec[~bm&mw],bins=[rabins,decbins],weights=1/var[~bm&mw])
rimg /= norm
gd = np.isfinite(rimg)
rimg -= np.median(rimg[gd])
ph,pf = np.histogram(rimg[gd].flatten(),bins=10000)
pfc = 0.5*(pf[:-1]+pf[1:])
rimg -= pfc[np.argmax(ph)]
# Outlier in-painting
filt = medfilt(rimg,(3,3))
diff = np.abs(rimg-filt)
gdd = np.isfinite(diff)
fill = diff > np.percentile(diff[gdd],97.5)
rimg[fill] = filt[fill]
# Renormalization
rimg = np.asinh(rimg/0.005)/6
rimg -= np.percentile(rimg[gd],5)
rimg[rimg<0]=0
rimg[rimg>1]=1


fig,ax = plt.subplots(1,1,figsize=(8,8))
plt.imshow(np.array([rimg,gimg,bimg]).T,origin='lower',extent=[ramax,ramin,decmin,decmax],aspect='auto')
plt.tight_layout()
plt.show()




