# project mass to a 2D 

# smotthing: read appending A of 
    # https://academic.oup.com/mnras/article/470/1/771/3807086

# to do: rewrite code in order to have a funct. to create mass distribution along all axes
# then complicate it by saving stuff

import os
import csv
import pickle
import numpy as np
from copy import copy
from scipy.stats import gaussian_kde

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

#from lenstronomy.Util import util

from fnct import std_sim
import matplotlib.pyplot as plt
from get_gal_indexes import get_rnd_gal

from python_tools.tools import mkdir,get_dir_basename
from python_tools.get_res import load_whatever


filename = "particles_EAGLE.csv"
path     = "./EAGLE_prt_data/"
dir_name = "proj_part"

Gal = get_rnd_gal(sim=std_sim,check_prev=False,reuse_previous=False)

Gal.proj_dir = Gal.gal_snap_dir+f"/{dir_name}/"
mkdir(Gal.proj_dir)
Gal.dens_res = f"{Gal.proj_dir}/dens_res.pkl"

rerun     = True
pixel_num = 100j
z_source_max = 5
verbose =True

    
if verbose:
    print("Assumptions: We are considering the maximum source redshift to be ",z_source_max)
    if int(pixel_num)<500:
        print("Warning: running test")
    elif int(pixel_num)>=1000:
        print("Warning: running very long")

def get_radius(RAs,DECs):
    # The following would not necessarily have the same pixelscale in the 2Dim
    # because RA and DEC might not be on the same range, but we create a grid with the same number of points for both
    # we rather would go as such: redefine the ranges such that the number of pixels 
    # and the ranges are the same (but there might be some empty pixels for either of the two dimensions)
    ramin  = RAs.min()
    ramax  = RAs.max()
    #rangeRa = ramax - ramin
    decmin = DECs.min()
    decmax = DECs.max()
    #rangeDec = decmax-decmin
    # we have the advantage that the center is set to 0 ->
    radius =  max([0-ramin,ramax-0,0-decmin,decmax-0])
    # verify:
    assert(ramin>=-radius)
    assert(decmin>=-radius)
    assert(ramax<=radius)
    assert(decmax<=radius)
    return radius

def get_z_source(cosmo,z_lens,dens_Ms_kpc2,z_source_max=z_source_max,verbose=verbose):
    # the lens has to be supercritical
    # dens>Sigma_crit = (c^2/4PiG D_d(z_lens) ) D_s(z_source)/D_ds(z_lens,z_source)
    # -> D_s(z_source)/D_ds(z_lens,z_source) < 4PiG D_d(z_lens) *dens/c^2
    # D_s(z_source)/D_ds(z_lens,z_source) is not easy to compute analytically, but we can sample it
    if z_lens>z_source_max:
        # to do : deal with this
        raise ValueError("The galaxy redshift is higher than the maximum allowed source redshift")
        #return 0
    max_DsDds = np.max(dens_Ms_kpc2)*4*np.pi*const.G*cosmo.angular_diameter_distance(z_lens)/(const.c**2)
    assert(max_DsDds.unit==u.dimensionless_unscaled)
    max_DsDds = max_DsDds.value # dimensionless
    #z_source_range = np.linspace(z_lens,z_source_max,100) # it's a very smooth funct->
    min_DsDds = cosmo.angular_diameter_distance(z_source_max)/cosmo.angular_diameter_distance_z1z2(z_lens,z_source_max) # this is the minimum
    if not np.any(min_DsDds<max_DsDds):
        # to do: deal with this kind of output
        if verbose:
            print("Warning: the minimum z_source needed to have a lens is higher than the maximum allowed z_source")
        return 0
    else:
        z_source_range = np.linspace(z_lens,z_source_max,100) # it's a very smooth funct->
        DsDds = np.array([cosmo.angular_diameter_distance(z_s).to("Mpc").value/cosmo.angular_diameter_distance_z1z2(z_lens,z_s).to("Mpc").value for z_s in z_source_range])
        minimise = np.abs(DsDds-max_DsDds) 
        z_source_min = z_source_range[np.argmin(minimise)]
        # select a random source within the range
        z_source = np.random.uniform(z_source_min,z_source_max,1)[0]
        if verbose:
            print("Minimum z_source:",np.round(z_source_min,2))
            print("Chosen z_source:", np.round(z_source,2))
        return z_source
try:
    if rerun:
        raise RuntimeError("Rerunning anyway")
    dens_Ms_kpc2, RAs,DECs,cosmo = load_whatever(Gal.dens_res)
except:
    
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
    # the unit is Mpc/h -> has to be converted to Mpc. Meaning that the value has to be divided by h
    x = np.concatenate([Xdm, Xstar, Xgas, Xbh])*u.Mpc/Gal.h # now in Mpc
    y = np.concatenate([Ydm, Ystar, Ygas, Ybh])*u.Mpc/Gal.h # now in Mpc
    z = np.concatenate([Zdm, Zstar, Zgas, Zbh])*u.Mpc/Gal.h # now in Mpc
    print("QUESTION: do we have to convert also the mass by h")
    m = np.concatenate([Mdm, Mstar, Mgas, Mbh])*u.Msun
    smooth = np.concatenate([SmoothDM,SmoothStar,SmoothGas,SmoothBH])
    
    # first test: proj along z
    X,Y = x,y
    # center around the center of the galaxy
    # correct from cMpc/h to Mpc/h
    # then from Mpc/h to Mpc
    Cx,Cy,_= Gal.centre*u.Mpc/(Gal.xy_propr2comov*Gal.h) # this should be now in Mpc
    X -=Cx
    Y -=Cy
    """
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
    namefig = f"{Gal.proj_dir}/kde_massmap0.pdf"
    plt.savefig(namefig)
    print("Saved "+namefig)
    """
    
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
    namefig = f"{Gal.proj_dir}/kde_massmap.pdf"
    plt.savefig(namefig)
    print("Saved "+namefig)
    
    """
    
    
    z_lens = Gal.z
    # we should probably ignore galaxies too close to us
    if Gal.z<0.1:
        print("WARNING: close by galaxy z="+str(Gal.z)+"\nIgnoring it and assuming it is at z=0.5")
        Gal.z = 0.5
    print("z_lens",z_lens)
    #z_source = 1.5*z_lens
    #print("z_source",z_source)
    cosmo = FlatLambdaCDM(H0=Gal.h*100, Om0=1-Gal.h)
    arcXkpc = cosmo.arcsec_per_kpc_proper(Gal.z) # ''/kpc
    # note: gal coords are in Mpc 
    RAs  = X*arcXkpc # ''
    RAs  = RAs.to("arcsec") 
    DECs = Y*arcXkpc # ''
    DECs = DECs.to("arcsec") 
    
    
    print("<RAs>",np.mean(RAs))
    print("tot mass",np.sum(m))
    #np.concatenate([Ystar,Ygas,Ydm,Ybh]).to("kpc").value*arcXkpc
    
    # fit the mass distribution w KDE
    kde    = gaussian_kde(np.array([RAs.value,DECs.value]),weights=m.value)# Msun/pix^2 (but dimensionless)

    radius = get_radius(RAs,DECs)
    dP     = 2*radius/(int(pixel_num)*u.pix) #''/pix

    RAg, DECg = np.mgrid[-radius:radius:pixel_num, -radius:radius:pixel_num]
    positions = np.vstack([RAg.ravel(), DECg.ravel()])
    fit_kde   = kde(positions)*u.Msun/(u.pix**2)  # Msun/pix^2 -> the kde give density as function of the pixel number, not the coordinates
    
    print("fit_kde sum",np.sum(fit_kde))
    #print("DEBUG shape fit_kde,RAg",np.shape(fit_kde),np.shape(RAg))
    dens    = np.reshape(fit_kde, RAg.shape).T # Msun/pix^2 # needed
    fig, ax = plt.subplots(2)
    #print(np.shape(dens),"np.shape(dens)")
    ax[0].contour(RAg, DECg, dens.value)
    ax[1].imshow(dens.value, cmap=plt.cm.gist_earth_r)
    namefig = f"{Gal.proj_dir}/kde_densmap.pdf"
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig)

    # define the z_source:
    dens_Ms_arcsec2 = dens/(dP**2) # 
    dens_Ms_kpc2    = dens_Ms_arcsec2*(arcXkpc**2)
    z_source        = get_z_source(cosmo,z_lens,dens_Ms_kpc2=dens_Ms_kpc2)
    if z_source==0:
        raise ValueError("Try rotating the galaxy")
    # save them
    #save_json([dens,RAs,DECs],Gal.dens_res)
    res = [dens_Ms_kpc2,RAs,DECs,cosmo]
    with open(Gal.dens_res,"wb") as f:
        pickle.dump(res,f)

    

# this is defined for the plot previously
if "RAg" not in locals():
    radius    = get_radius(RAs,DECs)
    RAg, DECg = np.mgrid[-radius:radius:pixel_num, -radius:radius:pixel_num]
    #arcXkpc   = cosmo.arcsec_per_kpc_proper(Gal.z) # ''/kpc
    dP        = 2*radius/(int(pixel_num)*u.pix) #''/pix

# create lensed image:
from lenstronomy.LensModel import convergence_integrals
# dPix has to be obtained from the physical size of the object
# -> no, it's given by the pixel_num 
"""
dPra        =  (RAs.max()-RAs.min())/(len(RAs)*u.pix) #np.diff(RAs).mean()/u.pix # ''/pix 
dPdec       =  (DECs.max()-DECs.min())/(len(DECs)*u.pix) # ''/pix
print("Pixel scale in 2 dim",dPra,dPdec,dPra/dPdec,)
print("dP has to be computed correctly from pixel_num")
"""
#print("dP.unit",dP.unit)
cosmo_dd  = cosmo.angular_diameter_distance(z_lens).to("kpc")   #kpc
cosmo_ds  = cosmo.angular_diameter_distance(z_source).to("kpc") #kpc
cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source).to("kpc") #kpc

# Sigma_Crit = D_s*c^2/(4piG * D_d*D_ds)
Sigma_Crit  = (cosmo_ds*const.c**2)/(4*np.pi*const.G*cosmo_dds*cosmo_dd)
Sigma_Crit  = Sigma_Crit.to("Msun /(kpc kpc)")    
#Sigma_Crit /= arcXkpc**2  # Msun / ''^2 
#Sigma_Crit *= dP**2       # Msun/pix^2
kappa_grid  = dens_Ms_kpc2/Sigma_Crit # 1 

# NOTE: a lens is such only if kappa_map > 1 at least at one point
# 1st assumption: we will realistically assume that such point is the center of the galaxy -> position the source there
# check that 
assert(np.any(kappa_grid>1))
# if not: check if there is a realistic (higher)  z_source  to make such that the 
# add masking
# add padding

num_pot    = convergence_integrals.potential_from_kappa_grid(kappa_grid, dP) # ''^2 / pix^2 -> this is odd but that's just how it is computed
print("numpot,unit",num_pot.unit)
numPot_div_dP = num_pot/dP # -> ''/pix 
raise("The latest unit problem is here: what exactly should it be the correct way to obtain alpha in arcsecs?")
print("numPot_div_dP,unit",numPot_div_dP.unit)
num_aRa,num_aDec = np.gradient(numPot_div_dP)/dP # this smh divide(?) by pix? and therefore the unit of the output must be only ''  -> set it by hand
print("num_aRa,unit",num_aRa.unit)
#*u.arcsec

def sersic_brightness(x,y,n=4):
    # rotate the galaxy by the angle self.pa
    #x = np.cos(self.pa)*(x-self.ys1)+np.sin(self.pa)*(y2-self.ys2)
    #y = -np.sin(self.pa)*(y1-self.ys1)+np.cos(self.pa)*(y2-self.ys2)
    # include elliptical isophotes
    try:
        # ugly but useful
        x=x.value
        y=y.value
    except:
        pass
    r = np.sqrt((x)**2+(y)**2)
    # brightness at distance r
    bn = 1.992*n - 0.3271
    re = 5.0
    brightness = np.exp(-bn*((r/re)**(1.0/n)-1.0))
    return brightness


#ra,dec = util.array2image(RAg),util.array2image(DECg)
print("DEBUG: shape RAg,num_aRa",np.shape(RAg),np.shape(num_aRa))

ra  = RAg.reshape(num_aRa.shape)/u.pix  # not entirely sure this unit is correct, but anyway it's just book-keeping
dec = DECg.reshape(num_aDec.shape)/u.pix  
print("DEBUG")
plt.imshow(ra.value)
plt.colorbar()
plt.title("Ra source")
im_name = f"{Gal.proj_dir}/ra_src.pdf"
plt.savefig(im_name)
plt.close()
plt.imshow(dec.value)
plt.colorbar()
plt.title("Dec source")
im_name = f"{Gal.proj_dir}/dec_src.pdf"
plt.savefig(im_name)
plt.close()
plt.imshow(np.log10(sersic_brightness(ra,dec)) )
plt.colorbar()
plt.title("log Source")
im_name = f"{Gal.proj_dir}/src.pdf"
plt.savefig(im_name)
plt.close()

plt.imshow(num_aRa.value)
plt.colorbar()
plt.title("Ra deflection")
im_name = f"{Gal.proj_dir}/alpha_ra.pdf"
plt.savefig(im_name)
plt.close()
plt.imshow(num_aDec.value)
plt.colorbar()
plt.title("Dec deflection")
im_name = f"{Gal.proj_dir}/alpha_dec.pdf"
plt.savefig(im_name)
plt.close()
print("DEBUG")

ra_im  = ra.value-num_aRa.value
dec_im = dec.value-num_aDec.value
lensed_im = sersic_brightness(ra_im,dec_im)
plt.imshow(np.log10(lensed_im))
plt.colorbar()
plt.title("Log Lensed Sersic image")
im_name = f"{Gal.proj_dir}/lensed_im.pdf"
plt.savefig(im_name)
plt.close()
print("Saving "+im_name)

# for convenience, I link the result to the tmp dir
os.unlink("./tmp/"+dir_name)
os.symlink(Gal.proj_dir[:-1],"./tmp/.")

print("Success")