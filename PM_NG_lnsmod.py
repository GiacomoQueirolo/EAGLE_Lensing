# copy from PM_lensmodel_0.1.py

import pickle
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from python_tools.tools import mkdir
from fnct import std_sim

from remade_gal import get_rnd_NG,get_lens_dir
from remade_gal import get_z_source,get_dP #,get_radius
 
Gal = get_rnd_NG()
z_lens = Gal.z
default_cosmo   =  Gal.cosmo#FlatLambdaCDM(H0=Gal.h*100, Om0=1-Gal.h)
from NG_proj_part_hist import prep_Gal,get_dens_map_rotate_hist
Gal = prep_Gal(Gal)
# if this
dens_Ms_kpc2,radius,dP,dxdy,z_source,cosmo,proj_index = get_dens_map_rotate_hist(Gal,plot=False)


lens_dir = get_lens_dir(Gal)
#TODO: create link to original Gal
kw_lns = lens_dir+"/sim_kwlens.pkl"
kw_lensmodel_data = lens_dir+"/kwdata.pkl"
# cosmo from https://academic.oup.com/mnras/article/474/3/3391/4644836, Agnello 2017

# point mass theta_E (from eq.4.7 of Meneghetti's lecture note - and by memory)
# theta_E = \sqrt ( 4GM D_ls / (c^2 Ds Dl) )
# divide the computation such that it's done only once
def thetaE_PM_prefact(z_lens,z_source,cosmo=default_cosmo):
    # from eq 21 of Narayan "Lectures on GL" 2008
    # theta_E(PM) = sqrt(4GM Dds/(c^2 Dd Ds))   = sqrt(M) * sqrt(4G Dds/(c^2 Dd Ds))
    cosmo_ds  = cosmo.angular_diameter_distance(z_source)
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens)
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source)
    pref      = 4*const.G*cosmo_dds/(const.c*const.c*cosmo_ds*cosmo_dd)
    return np.sqrt(pref.to("1/g")) # 

@u.quantity_input
def thetaE_PM(M:u.g,theta_pref:u.g**-.5):
    thetaE_rad = np.sqrt(M)*theta_pref
    thetaE     = thetaE_rad.to("")*u.rad.to("arcsec")
    return thetaE.value #in arcsec

#while True:

print("Selected Gal:",Gal)
thetaE_pref = thetaE_PM_prefact(z_lens=Gal.z,z_source=z_source)
#thetaE_pref = thetaE_pref.to("1/Msun(1/2)").value #convert in 1/sqrt(Msun)

Mstar = Gal.stars["mass"] #should already be in Msun 
Mgas = Gal.gas["mass"]    #should already be in Msun 
Mdm = Gal.dm["mass"]      #should already be in Msun 
Mbh = Gal.bh["mass"]      #should already be in Msun 
Ms = np.concatenate([Mstar,Mgas,Mdm,Mbh])*u.Msun #Msun
"""
try:
    thetaEs = thetaE_PM(Ms,thetaE_pref)
except TypeError:
    Ms      = Ms*const.M_sun
    thetaEs = thetaE_PM(Ms,thetaE_pref)
"""
thetaEs = thetaE_PM(Ms,thetaE_pref)
# project along z axis 
# centered around 0
# convert in arcsec

arcXkpc = default_cosmo.arcsec_per_kpc_proper(Gal.z)


Xstar,Ystar,Zstar =  np.transpose(Gal.stars["coords"])# Mpc
#RAstar,DECstar = Xstar.to("kpc").value*arcXkpc,Ystar.to("kpc").value*arcXkpc
Xgas,Ygas,Zgas    =  np.transpose(Gal.gas["coords"]) # Mpc
#RAgas,DECgas = Xgas.to("kpc").value*arcXkpc,Ygas.to("kpc").value*arcXkpc
Xdm,Ydm,Zdm       =  np.transpose(Gal.dm["coords"]) # Mpc
#RAdm,DECdm = Xdm.to("kpc").value*arcXkpc,Ydm.to("kpc").value*arcXkpc
Xbh,Ybh,Zbh       =  np.transpose(Gal.bh["coords"]) # Mpc
#RAbh,DECbh = Xbh.to("kpc").value*arcXkpc,Ybh.to("kpc").value*arcXkpc
Xs = np.concatenate([Xstar,Xgas,Xdm,Xbh])*u.Mpc.to("kpc")
Ys = np.concatenate([Ystar,Ygas,Ydm,Ybh])*u.Mpc.to("kpc")
Zs = np.concatenate([Zstar,Zgas,Zdm,Zbh])*u.Mpc.to("kpc")
print("DEBUG - Npart")
print(len(Xs))

# projection along given indexes
# xy : ind=0
# xz : ind=1
# yz : ind=2
if proj_index==0:
    _=True # all as usual
elif proj_index==1:
    Ys = copy(Zs)
elif proj_index==2:
    Xs  = copy(Ys)
    Ys  = copy(Zs)
    
RAs  = Xs*arcXkpc.to('arcsec/kpc')
DECs = Ys*arcXkpc.to('arcsec/kpc')
RAs  = RAs.value
DECs = DECs.value
RA_cm  = np.sum(RAs* Ms.value)/np.sum(Ms.value)
DEC_cm = np.sum(DECs* Ms.value)/np.sum(Ms.value)


#lens_model_list = []
lens_model_list  = ["POINT_MASS"]*len(thetaEs)
print("DEBUG - Npart")
print(len(thetaEs),len(thetaEs)==len(Xs))
#use CGPT parallelisation:
from concurrent.futures import ThreadPoolExecutor

def build_kwargs_lens(args):
    tE, ra, dec = args
    return {
        "theta_E": tE,
        "center_x": ra,
        "center_y": dec
    }

# Parallel execution
with ThreadPoolExecutor() as executor:
    kwargs_lens = list(executor.map(build_kwargs_lens, zip(thetaEs, RAs, DECs)))
print("DEBUG - Npart")
print(len(kwargs_lens),len(kwargs_lens)==len(thetaEs))

# save the kwargs
print("Saving "+kw_lns)
with open(kw_lns,"wb") as f:
    pickle.dump(kwargs_lens,f)

## Simulate the lens image

import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

lens_model_class = LensModel(lens_model_list=lens_model_list)

# data specifics
background_rms = 0 #.5  # background noise per pixel
exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)

# Define a "sweet spot" for the numpix and the DeltaPix given the app. size of the lens
#Diam_arcsec = np.mean([np.max(RAs)-np.min(RAs),np.max(DECs)-np.min(DECs)])
#print("Size of the lens in arcsecs",Diam_arcsec)
rad = 70*u.kpc 
print(f"WRN: Consider an arbitrary radius of {rad} around the centre")
rad_arcsec  = rad*arcXkpc.to('arcsec/kpc')
Diam_arcsec = 2*rad_arcsec.value #diameter in arcsec

#numPix   = 100   # cutout pixel size
deltaPix = 0.04  # pixel size in arcsec (area per pixel = deltaPix**2)
numPix   = int(.7*Diam_arcsec/deltaPix)
if numPix >500:
    print("Numpix too high: ",numPix,", capping at 500")
    numPix = 500
    print("deltaPix:",deltaPix)
else:
    print("Resulting image size:", numPix)

#fwhm = 0.01  # full width half max of PSF -> very small, almost none


kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, 
                                             exp_time,background_rms,
                                             center_ra=RA_cm,center_dec=DEC_cm)
data_class = ImageData(**kwargs_data)
kwargs_psf = {'psf_type': 'NONE'} # no PSF for now
        #, 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 5}
psf_class = PSF(**kwargs_psf)

# Source Params
source_model_list = ['SERSIC_ELLIPSE']
ra_source, dec_source = RA_cm,DEC_cm
print("RA,DEC CMS:",RA_cm,DEC_cm)
kwargs_sersic_ellipse = {'amp': 4000., 'R_sersic': .1, 'n_sersic': 3, 
                         'center_x': ra_source,
                         'center_y': dec_source, 
                         'e1': -0.1, 'e2': 0.01}

kwargs_source      = [kwargs_sersic_ellipse]
source_model_class = LightModel(light_model_list=source_model_list)
kwargs_numerics    = {'supersampling_factor': 1, 'supersampling_convolution': False}

LensedimageModel = ImageModel(data_class, psf_class, lens_model_class,
                              source_model_class,lens_light_model_class=None,
                              point_source_class=None, kwargs_numerics=kwargs_numerics)
lensed_image_sim = LensedimageModel.image(kwargs_lens, kwargs_source, 
                                          kwargs_lens_light=None, kwargs_ps=None)
SourceimageModel = ImageModel(data_class, psf_class, lens_model_class=None, source_model_class=source_model_class,
                        lens_light_model_class=None,
                        point_source_class=None, kwargs_numerics=kwargs_numerics)
unlensed_image_sim = SourceimageModel.image(kwargs_lens=None, 
                                            kwargs_source=kwargs_source, 
                                            kwargs_lens_light=None,
                                            kwargs_ps=None)
#fg,ax=plt.subplots(2,2,figsize=(16,8))

# kappa is useless as it is always 0 -> point masses
#kappa_lnstr = lens_model_class.kappa(,kwargs_lens)
#ax[0][0].matshow(np.log10(kappa_lnstr), origin='lower')
#ax[0][0].set_title(r"$\kappa$ (lnstr)")

# to implement later
#ax[0][1].matshow(np.log10(SGl.density), origin='lower')
#ax[0][1].set_title("Denstiy (SimGal)")
#lnsd_SGl = SGl.source_class.image(SGl.ra_im,SGl.dec_im) 
#ax[1][1].matshow(np.log10(lnsd_SGl), origin='lower')
#ax[1][1].set_title("Lensed image (SimGal)")
extent = [-rad_arcsec.value,rad_arcsec.value,-rad_arcsec.value,rad_arcsec.value]

fg,ax=plt.subplots(1,2,figsize=(16,8))

ax[0].matshow(np.log10(unlensed_image_sim),extent=extent, origin='lower')
ax[0].contour(np.log10(unlensed_image_sim),cmap=plt.cm.inferno,extent=extent)

ax[0].set_title("Un-Lensed image (lnstr)")

ax[1].matshow(np.log10(lensed_image_sim),extent=extent, origin='lower')
ax[1].contour(np.log10(lensed_image_sim),cmap=plt.cm.inferno,extent=extent)
ax[1].set_title("Lensed image (lnstr)")

name_file = lens_dir+"/lensed_im_PM.pdf"
plt.savefig(name_file)
print("Saving "+name_file)
name_file = "./tmp/lensed_im_PM.pdf"
plt.savefig(name_file)
print("Saving "+name_file)


from lenstronomy.Util import util

x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)  
x,y = util.array2image(x_grid),util.array2image(y_grid)

# convergence map
"""_kappa_map = lens_model_class.kappa(x_grid, y_grid, kwargs_lens)  
kappa_map = util.array2image(_kappa_map)
print("any kappa map >1:",np.any(kappa_map>1))
fg,ax=plt.subplots(1,figsize=(16,8))
ax.imshow(np.log10(kappa_map),extent=extent,origin="lower")
ax.contour(np.log10(kappa_map),cmap=plt.cm.inferno,extent=extent)
ax.set_title(r"$\kappa$ map")
name_file = "./tmp/kappa_PM.pdf"
"""
# deflections
_alpha_x,_alpha_y = lens_model_class.alpha(x_grid, y_grid, kwargs_lens)  
alpha_x,alpha_y   = util.array2image(_alpha_x),util.array2image(_alpha_y)

fg,ax=plt.subplots(1,2,figsize=(16,8))
ax[0].imshow(np.log10(alpha_x),extent=extent,origin="lower")
ax[0].set_title(r"$\alpha_x$")
ax[1].imshow(np.log10(alpha_y),extent=extent,origin="lower")
#ax[0].contour(np.log10(alpha_x),cmap=plt.cm.inferno,extent=extent)
ax[1].set_title(r"$\alpha_y$")
name_file = "./tmp/alpha_PM.pdf"
plt.savefig(name_file)
print("Saving "+name_file)
"""
plt.matshow(np.log10(image_sim), origin='lower')
plt.savefig(lens_dir+"/lensed_im.pdf")

poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
bkg = image_util.add_background(image_sim, sigma_bkd=background_rms)
image_sim = image_sim + bkg + poisson

kwargs_data['image_data'] = image_sim
data_class.update_data(image_sim)

plt.matshow(np.log10(image_sim), origin='lower')
plt.savefig(lens_dir+"/lensed_im_noisy.pdf")
"""
print("Saving "+kw_lensmodel_data)
with open(kw_lensmodel_data,"wb") as f:
    pickle.dump(kwargs_data,f)