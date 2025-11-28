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


from NG_proj_part_hist import z_source_max,pixel_num,prep_Gal_projpath
#z_source_max = 5
#pixel_num    = 150j
verbose      = True

from Gen_PM_PLL import     cutoff_radius 

from remade_gal import get_z_source,get_dP,Gal2MXYZ #get_radius

from project_gal  import proj_parts,findDens

        
def plot_dens_map_hist(Gal,proj_index=0,pixel_num=pixel_num,cutoff_radius=cutoff_radius,verbose=verbose,save_res=True,plot=True,namefig=None):
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)
    if nx==0:
        nx,ny = int(pixel_num), int(pixel_num)
    # given a projection, plot the density map

    m,x,y,z = Gal2MXYZ(Gal)
    x,y = proj_parts({"Xs":x,"Ys":y,"Zs":z},proj_index)
    # recenter around 
    Xdns,Ydns = findDens(m,x,y,cutoff_radius)
    x -= Xdns
    y -= Ydns

    
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
    cosmo   = Gal.cosmo 
    # X,Y already recentered around densest point
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
    dx = np.diff(xedges) #kpc #==(xmax - xmin) / nx 
    dy = np.diff(yedges) #kpc
    # density_ij = M_ij/(Area_bin_ij)
    density = mass_grid / (dx * dy)

    if verbose:
        print("<dx,dy>",np.mean(dx),np.mean(dy))
        print("<area>",np.mean(dx*dy),"kpc^2")
        print("<mass>",np.mean(mass_grid),"Msun")
        print("<density>",np.mean(density),"Msun/kpc^2")

    extent = [xmin,xmax,ymin,ymax]
    #plt.imshow(np.log10(density),extent=extent, cmap=plt.cm.gist_earth_r,norm="log",alpha=.5)
    plt.close()
    plt.contour(np.log10(density),extent=extent, cmap=plt.cm.inferno,norm="log")
    levels = np.logspace(-4,-1)
    #plt.contour(np.log10(density),extent=extent,levels=levels,norm="log",cmap=plt.cm.gist_earth_r)
    #plt.contour(np.log10(density),extent=extent,cmap=plt.cm.inferno)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.xlabel("kpc")
    plt.ylabel("kpc")
    
    #namefig = f"{Gal.proj_dir}/densmap_proj_{proj_index}.png"
    if namefig is None:
        namefig = f"./tmp/cmi_densmap_proj_{proj_index}.png"
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig)
    



if __name__=="__main__":
    parser = ArgumentParser(description="Project particles into a mass sheet - histogram version")
    parser.add_argument("-dn","--dir_name",dest="dir_name",type=str, help="Directory name",default="proj_part_hist")
    parser.add_argument("-pxn","--pixel_num",dest="pixel_num",type=int, help="Pixel number",default=pixel_num.imag)
    #parser.add_argument("-zsm","--z_source_max",dest="z_source_max",type=float, help="Maximum source redshift",default=z_source_max)
    parser.add_argument("-nrr", "--not_rerun", dest="rerun", 
                        default=True,action="store_false",help="if True, rerun code")

    parser.add_argument("-v", "--verbose", dest="verbose", 
                        default=False,action="store_true",help="verbose")
    args          = parser.parse_args()
    pixel_num     = args.pixel_num*1j
    rerun         = args.rerun
    dir_name      = args.dir_name
    verbose       = True#args.verbose
    #z_source_max  = args.z_source_max

    Gal = get_rnd_NG()
    Gal = prep_Gal_projpath(Gal)
    plot_dens_map_hist(Gal=Gal,pixel_num=pixel_num,verbose=verbose,cutoff_radius=cutoff_radius)
    
    print("Success")