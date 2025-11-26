# copied from create_mass_image.py (now old)
# we want to create iso-mass (and maybe iso-phot) contours of the galaxy

import os
import csv
import glob
import pickle
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from argparse import ArgumentParser 
from scipy.stats import gaussian_kde

from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel import convergence_integrals

#from lenstronomy.Util import util

from fnct import gal_dir,std_sim
#from get_gal_indexes import get_rnd_gal
from remade_gal import get_rnd_NG

from python_tools.tools import mkdir,get_dir_basename
from python_tools.get_res import load_whatever


from NG_proj_part_hist import z_source_max,pixel_num,prep_Gal
#z_source_max = 5
#pixel_num    = 150j
verbose      = True


from remade_gal import get_z_source,get_dP #get_radius



        
def plot_dens_map_hist(Gal,proj_index=0,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,save_res=True,plot=True):
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)

    # given a projection, plot the density map
    
    Xstar,Ystar,Zstar = Gal.stars["coords"].T # in Mpc/h
    Xgas,Ygas,Zgas    = Gal.gas["coords"].T   # in Mpc/h
    Xdm,Ydm,Zdm       = Gal.dm["coords"].T    # in Mpc/h
    Xbh,Ybh,Zbh       = Gal.bh["coords"].T    # in Mpc/h
    
    Mstar = Gal.stars["mass"] # in Msun 
    Mgas  = Gal.gas["mass"]  # in Msun 
    Mdm   = Gal.dm["mass"] # in Msun 
    Mbh   = Gal.bh["mass"] # in Msun 
    
    
    # Concatenate particle properties
    # the unit is Mpc/h -> has to be converted to Mpc. Meaning that the value has to be divided by h
    x = np.concatenate([Xdm, Xstar, Xgas, Xbh])*u.Mpc # now in Mpc
    y = np.concatenate([Ydm, Ystar, Ygas, Ybh])*u.Mpc # now in Mpc
    z = np.concatenate([Zdm, Zstar, Zgas, Zbh])*u.Mpc # now in Mpc
    m = np.concatenate([Mdm, Mstar, Mgas, Mbh])*u.Msun
    # center around the center of the galaxy
    # center of mass is given in Comiving coord 
    # see https://arxiv.org/pdf/1510.01320 D.23 
    # ->  it's given in cMpc (not cMpc/h) fsr
    Cx,Cy,Cz = Gal.centre*u.Mpc/(Gal.xy_propr2comov) # (now) Mpc
    x -= Cx
    y -= Cy
    z -= Cz
    # projection along given indexes
    # xy : ind=0
    # xz : ind=1
    # yz : ind=2
    if proj_index==0:
        _=True # all as usual
    elif proj_index==1:
        y  = copy(z)
        Cy = copy(Cz)
    elif proj_index==2:
        x  = copy(y)
        Cx = copy(Cy)
        y  = copy(z)
        Cz = copy(Cz)


    print("DEBUG\n",type(x))

    x = np.asarray(x.to("kpc"))
    y = np.asarray(y.to("kpc"))
    m = np.asarray(m.to("solMass"), dtype=float) 
    

    radius = 70 #kpc 
    print("NOTE: taking a small radius -",radius,"kpc")
    # Redshift: 
    z_lens = Gal.z
    if verbose:
        print("z_lens",z_lens)
    cosmo   = FlatLambdaCDM(H0=Gal.h*100, Om0=1-Gal.h)

    # X,Y already recentered around 0
    xmin = -radius
    ymin = -radius
    xmax = +radius
    ymax = +radius

    # numpy.histogram2d returns H with shape (nx_bins, ny_bins) where H[i,j]
    # counts x-bin i and y-bin j. We transpose to (ny, nx) so rows are y.
    H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny],
                                       range=[[xmin, xmax], [ymin, ymax]],
                                       weights=m,density=False)  
    #                           if density=True, it normalises it to the total density
    
    # H is then the distribution of mass for each bin, not the density
    mass_grid = H.T.copy() # Solar Masses
    # H shape: (nx, ny) -> transpose to (ny, nx)

    # area of the (dx/dy) bins:
    dx = (xmax - xmin) / nx #kpc
    dy = (ymax - ymin) / ny #kpc
    # density_ij = M_ij/(Area_bin_ij)
    density = mass_grid / (dx * dy)

    if verbose:
        print("dx,dy",dx,dy)
        print("area",dx*dy,"kpc^2")
        print("<mass>",np.mean(mass_grid))
        print("<density>",np.mean(density))


    extent = [xmin,xmax,ymin,ymax]
    #plt.imshow(np.log10(density),extent=extent, cmap=plt.cm.gist_earth_r,norm="log",alpha=.5)
    plt.contour(np.log10(density),extent=extent, cmap=plt.cm.gist_earth_r,norm="log")
    levels = np.logspace(-4,-1)
    #plt.contour(np.log10(density),extent=extent,levels=levels,norm="log",cmap=plt.cm.gist_earth_r)
    plt.contour(np.log10(density),extent=extent,cmap=plt.cm.inferno)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    #namefig = f"{Gal.proj_dir}/densmap_proj_{proj_index}.png"
    namefig = f"./tmp/cmi_densmap_proj_{proj_index}.png"
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig)
    


if __name__=="__main__":
    parser = ArgumentParser(description="Project particles into a mass sheet - histogram version")
    parser.add_argument("-dn","--dir_name",dest="dir_name",type=str, help="Directory name",default="proj_part_hist")
    parser.add_argument("-pxn","--pixel_num",dest="pixel_num",type=int, help="Pixel number",default=pixel_num.imag)
    parser.add_argument("-zsm","--z_source_max",dest="z_source_max",type=float, help="Maximum source redshift",default=z_source_max)
    parser.add_argument("-nrr", "--not_rerun", dest="rerun", 
                        default=True,action="store_false",help="if True, rerun code")

    parser.add_argument("-v", "--verbose", dest="verbose", 
                        default=False,action="store_true",help="verbose")
    args          = parser.parse_args()
    pixel_num     = args.pixel_num*1j
    rerun         = args.rerun
    dir_name      = args.dir_name
    verbose       = True#args.verbose
    z_source_max  = args.z_source_max

    Gal = get_rnd_NG()
    Gal = prep_Gal(Gal)
    plot_dens_map_hist(Gal=Gal,pixel_num=pixel_num, z_source_max=z_source_max,verbose=verbose)
    
    print("Success")