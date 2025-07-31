
# project mass to a 2D 

# smotthing: read appending A of 
    # https://academic.oup.com/mnras/article/470/1/771/3807086

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from fnct import std_sim
import matplotlib.pyplot as plt
from get_gal_indexes import get_rnd_gal

import csv

filename = "particles_EAGLE.csv"
path     = "./EAGLE_prt_data/"

Gal = get_rnd_gal(sim=std_sim,check_prev=False)

Xstar,Ystar,Zstar = Gal.stars["coords"].T # in Mpc/h
Xgas,Ygas,Zgas    = Gal.gas["coords"].T   # in Mpc/h
Xdm,Ydm,Zdm       = Gal.dm["coords"].T    # in Mpc/h
Xbh,Ybh,Zbh       = Gal.bh["coords"].T    # in Mpc/h

Mstar = Gal.stars["mass"] # in Msun 
Mgas  = Gal.gas["mass"]  # in Msun 
Mdm   = Gal.dm["mass"] # in Msun 
Mbh   = Gal.bh["mass"] # in Msun 

# From https://academic.oup.com/mnras/article/470/1/771/3807086
# I think that 
# 1) smoothing make sense for hydrodym, maybe less so for lens modelling
# 2) we could test how important it is:
#    - w/o smoothing (all point particles)
#    - w. smoothing: - Gaussian
#                    - isoth-sphere
SmoothStar = Gal.stars["smooth"] # in Mpc/h
SmoothGas  = Gal.gas["smooth"] # in Mpc/h
SmoothDM   = np.zeros_like(Mdm) #Gal.dm["smooth"] # in Mpc/h -> no smoothing for DM
SmoothBH   = Gal.bh["smooth"] # in Mpc/h

# Concatenate particle properties
coords = np.concatenate([Gal.dm["coords"],Gal.stars["coords"],Gal.gas["coords"],Gal.bh["coords"]]) # shape: Nparts,3
x = np.concatenate([Xdm, Xstar, Xgas, Xbh])*u.Mpc
y = np.concatenate([Ydm, Ystar, Ygas, Ybh])*u.Mpc
z = np.concatenate([Zdm, Zstar, Zgas, Zbh])*u.Mpc
m = np.concatenate([Mdm, Mstar, Mgas, Mbh])*u.Msun
smooth = np.concatenate([SmoothDM,SmoothStar,SmoothGas,SmoothBH])

# first test: proj along z
from scipy.stats import gaussian_kde
X,Y = x,y
kde = gaussian_kde(np.array([X,Y]),weights=m)
xmin = X.min()
xmax = X.max()
ymin = Y.min()
ymax = Y.max()
Xg, Yg = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([Xg.ravel(), Yg.ravel()])
Z = np.reshape(kde(positions).T, Xg.shape)
fig, ax = plt.subplots()
ax.contour(Xg, Yg, Z)
namefig = "tmp/kde_massmap0.pdf"
plt.savefig(namefig)
print("Saved "+namefig)
"""
X,Y = x,y
kde = gaussian_kde(np.array([X,Y]),weights=m)
xmin = X.min()
xmax = X.max()
ymin = Y.min()
ymax = Y.max()
Xg, Yg = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([Xg.ravel(), Yg.ravel()])
Z = np.reshape(kde(positions).T, Xg.shape)
fig, ax = plt.subplots()
ax.contour(Xg, Yg, Z)
namefig = "tmp/kde_massmap.pdf"
plt.savefig(namefig)
print("Saved "+namefig)

"""

from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from astropy import units as u
from lenstronomy.Util import util

z_lens = Gal.z
# we should probably ignore galaxies too close to us
if Gal.z<0.1:
    print("WARNING: close by galaxy z="+str(Gal.z)+"\nIgnoring it and assuming it is at z=0.5")
    Gal.z = 0.5
print("z_lens",z_lens)
z_source = 1.5*z_lens
print("z_source",z_source)
default_cosmo = FlatLambdaCDM(H0=Gal.h*100, Om0=1-Gal.h)
arcXkpc = default_cosmo.arcsec_per_kpc_proper(Gal.z) # ''/kpc
# note: gal coords are in Mpc 
RAs  = x*arcXkpc # ''
RAs  = RAs.to("arcsec") 
DECs = y*arcXkpc # ''
DECs = DECs.to("arcsec")
print("RAs",np.mean(RAs))
print("mass",np.sum(m))
#np.concatenate([Ystar,Ygas,Ydm,Ybh]).to("kpc").value*arcXkpc

# fit the mass distribution w KDE
kde    = gaussian_kde(np.array([RAs.value,DECs.value]),weights=m.value)# Msun/''^2 (but dimensionless)
ramin  = RAs.min()
ramax  = RAs.max()
decmin = DECs.min()
decmax = DECs.max()
RAg, DECg = np.mgrid[ramin:ramax:100j, decmin:decmax:100j]
positions = np.vstack([RAg.ravel(), DECg.ravel()])
fit_kde = kde(positions).T*u.Msun/(u.arcsec**2)  # Msun/''^2
print("fit_kde sum",np.sum(fit_kde))
dens = np.reshape(fit_kde, RAg.shape) # Msun/''^2
fig, ax = plt.subplots(2)
print(np.shape(dens),"np.shape(dens)")
ax[0].contour(RAg, DECg, dens.value)
ax[1].imshow(np.rot90(dens.value), cmap=plt.cm.gist_earth_r)
namefig = "tmp/kde_densmap.pdf"
plt.savefig(namefig)
plt.close()
print("Saved "+namefig)


# create lensed image:
from lenstronomy.LensModel import convergence_integrals
# dPix has to be obtained from the physical size of the object
dP        = np.diff(RAs).mean()/u.pix # ''/pix 
cosmo_dd  = default_cosmo.angular_diameter_distance(z_lens).to("kpc")   #kpc
cosmo_ds  = default_cosmo.angular_diameter_distance(z_source).to("kpc") #kpc
cosmo_dds = default_cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source).to("kpc") #kpc
    
# Sigma_Crit = D_s*c^2/(4piG * D_d*D_ds)
Sigma_Crit  = (cosmo_ds*const.c**2)/(4*np.pi*const.G*cosmo_dds*cosmo_dd)
Sigma_Crit  = Sigma_Crit.to("Msun /(kpc kpc)")    
Sigma_Crit /= (arcXkpc**2)  # Msun / ''^2
kappa_grid = dens/Sigma_Crit # 1 
# add masking
# add padding

num_pot    = convergence_integrals.potential_from_kappa_grid(kappa_grid, dP) # ''^2 / pix^2
print("numpot,unit",num_pot.unit)
num_aRa,num_aDec = np.gradient(num_pot/dP)
#*u.arcsec

def sersic_brightness(x,y,n=4):
    # rotate the galaxy by the angle self.pa
    #x = np.cos(self.pa)*(x-self.ys1)+np.sin(self.pa)*(y2-self.ys2)
    #y = -np.sin(self.pa)*(y1-self.ys1)+np.cos(self.pa)*(y2-self.ys2)
    # include elliptical isophotes
    r = np.sqrt((x)**2+(y)**2)
    # brightness at distance r
    bn = 1.992*n - 0.3271
    re = 5.0
    brightness = np.exp(-bn*((r/re)**(1.0/n)-1.0))
    return brightness

#ra,dec = util.array2image(RAg),util.array2image(DECg)
ra  = RAg.reshape(num_aRa.shape)/u.pix  # not entirely sure this unit is correct, but anyway it's just book-keeping
dec = DECg.reshape(num_aDec.shape)/u.pix  

lensed_im = sersic_brightness(ra.value-num_aRa.value,dec.value-num_aDec.value)
plt.imshow(lensed_im.T)
plt.colorbar()
plt.title("Lensed Sersic image")
im_name = "tmp/lensed_im.pdf"
plt.savefig(im_name)
plt.close()
print("Saving "+im_name)