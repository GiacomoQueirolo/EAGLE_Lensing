# further simplification still, just plot 2D projection of the mass map

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
from get_gal_indexes import get_rnd_gal

from python_tools.tools import mkdir,get_dir_basename
from python_tools.get_res import load_whatever


z_source_max = 5
verbose      = True
pixel_num    = 150j
################################################
# debugging funct.

from proj_part import get_radius,get_z_source,get_dP



        
def plot_dens_map_hist(Gal,proj_index=0,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,save_res=True,plot=True):
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
    x = np.concatenate([Xdm, Xstar, Xgas, Xbh])*u.Mpc/Gal.h # now in Mpc
    y = np.concatenate([Ydm, Ystar, Ygas, Ybh])*u.Mpc/Gal.h # now in Mpc
    z = np.concatenate([Zdm, Zstar, Zgas, Zbh])*u.Mpc/Gal.h # now in Mpc
    #  print("QUESTION: do we have to convert also the mass by h") -> I think we have to
    m = np.concatenate([Mdm, Mstar, Mgas, Mbh])*u.Msun/Gal.h

    """
    # From https://academic.oup.com/mnras/article/470/1/771/3807086
    # I think that 
    # 1) smoothing make sense for hydrodym, maybe less so for lens modelling
    # 2) we could test how important it is:
    #    - w/o smoothing (all point particles)
    #    - w. smoothing: - Gaussian
    #                    - isoth-sphere
    SmoothStar = Gal.stars["smooth"]  # in Mpc/h
    SmoothGas  = Gal.gas["smooth"]    # in Mpc/h
    SmoothDM   = np.zeros_like(Mdm)   # in Mpc/h -> no smoothing for DM
    SmoothBH   = Gal.bh["smooth"]     # in Mpc/h
    smooth     = np.concatenate([SmoothDM,SmoothStar,SmoothGas,SmoothBH])
    """
    # DEBUG
    max_diam = np.max([np.max(x.value) - np.min(x.value),np.max(y.value) - np.min(y.value),np.max(z.value) - np.min(z.value)])*u.Mpc
    print("DEBUG","max_diam",max_diam)
    
    # center around the center of the galaxy
    # correct from cMpc/h to Mpc/h
    # then from Mpc/h to Mpc
    Cx,Cy,Cz= Gal.centre*u.Mpc/(Gal.xy_propr2comov*Gal.h) # this should be now in Mpc
    
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
    x -=Cx
    y -=Cy

    print("DEBUG\n",type(x))

    x = np.asarray(x.to("kpc"))
    y = np.asarray(y.to("kpc"))
    m = np.asarray(m.to("solMass")/1e8, dtype=float) #unit of 10^12 solar masses

    print("DEBUG")
    fig, ax = plt.subplots(3)
    ax[0].hist(x)
    ax[0].set_xlabel("X [kpc]")
    ax[1].hist(y)
    ax[1].set_xlabel("Y [kpc]")
    ax[2].hist(m)
    ax[2].set_xlabel("M [1e8 SolMass]")
    namefig = f"{Gal.proj_dir}/hist1D_{proj_index}.png"
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig)

    # Redshift: 
    z_lens = Gal.z
    if verbose:
        print("z_lens",z_lens)
    cosmo   = FlatLambdaCDM(H0=Gal.h*100, Om0=1-Gal.h)
    if verbose:
        print("<Xs>",np.mean(x))
        print("tot mass",np.sum(m))
    
    #x,y = x.to("kpc").value,y.to("kpc").value
    radius    = get_radius(x,y) #kpc
    """
    xmin = Cx.to("kpc").value-radius
    ymin = Cy.to("kpc").value-radius
    xmax = Cx.to("kpc").value+radius
    ymax = Cy.to("kpc").value+radius
    """
    # I think the following is wrong: it should be centered around 0 bc X,Y already recentered
    xmin = - radius
    ymin = - radius
    xmax = + radius
    ymax = + radius

    # numpy.histogram2d returns H with shape (nx_bins, ny_bins) where H[i,j]
    # counts x-bin i and y-bin j. We transpose to (ny, nx) so rows are y.
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)
    H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny],
                                       range=[[xmin, xmax], [ymin, ymax]],
                                       weights=m)
    # H shape: (nx, ny) -> transpose to (ny, nx)
    mass_grid = H.T.copy()

    dx      = (xmax - xmin) / nx
    dy      = (ymax - ymin) / ny
    density = mass_grid / (dx * dy)


    extent = [xmin,xmax,ymin,ymax]
    plt.imshow(np.log10(density),extent=extent, cmap=plt.cm.gist_earth_r,norm="log")
    plt.colorbar()
    #plt.scatter(x,y,c="w",marker=".")
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    namefig = f"{Gal.proj_dir}/hist_densmap_proj_{proj_index}.png"
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
    verbose       = args.verbose
    z_source_max  = args.z_source_max
    
    if rerun:
        Gal = get_rnd_gal(sim=std_sim,check_prev=False,reuse_previous=False,min_mass="1e13",max_z="1")
        Gal.proj_dir = Gal.gal_snap_dir+f"/{dir_name}_{Gal.Name}/"
        mkdir(Gal.proj_dir)
        Gal.dens_res = f"{Gal.proj_dir}/dens_res.pkl"
    else:
        # find an already "random" galaxy
        dens_res_path = glob.glob(gal_dir+"/snap_*/"+dir_name+"_*/dens_res.pkl")
        dens_res = np.random.choice(dens_res_path)
        class empty_class():
            def __init__(self):
                return None
        Gal = empty_class()
        Gal.dens_res = dens_res
        Gal.proj_dir = get_dir_basename(dens_res)[0]
        
    if verbose:
        print("Assumptions: We are considering the maximum source redshift to be ",z_source_max)
        if int(pixel_num.imag)<500:
            print("Warning: running test")
        elif int(pixel_num.imag)>=1000:
            print("Warning: running very long")
 
    plot_dens_map_hist(Gal=Gal,pixel_num=pixel_num, z_source_max=z_source_max,verbose=verbose)
    
    print("Success")