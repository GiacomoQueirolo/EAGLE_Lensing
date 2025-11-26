# Take Gal from remade_gal and does projection and similar calculations
# functions usefuls for lensing and later imported by Gen_PM_PLL.py


import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM

from python_tools.tools import mkdir,to_dimless,short_SciNot

from remade_gal import get_CM
# for now keep this and check if still needed
dir_name     = "proj_part_hist"
def prep_Gal_projpath(Gal,dir_name=dir_name):
    # impractical but easy to set up
    Gal.proj_dir = Gal.gal_snap_dir+f"/{dir_name}_{Gal.Name}/"
    mkdir(Gal.proj_dir)
    Gal.proj_zs_path = f"{Gal.proj_dir}/proj_zs.pkl"
    return Gal

from python_tools.get_res import load_whatever
from copy import copy,deepcopy

def proj_parts(kw_parts,proj_index,arcXkpc=None):    
    Xs,Ys,Zs = kw_parts["Xs"],kw_parts["Ys"],kw_parts["Zs"]
    if proj_index==0:
        _   = True  # all as usual
    elif proj_index==1:
        Ys  = copy(Zs)
    elif proj_index==2:
        Xs  = copy(Ys)
        Ys  = copy(Zs)    
    return Xs,Ys 

def findDens(M,X,Y,rad,nbins=200,XYCM=None):
    # loacte the coordinates of the densest bin
    if XYCM is None:
        XYCM = get_CM(M,X,Y)
    X_cm,Y_cm = XYCM
    bins = [np.linspace(X_cm - rad,X_cm+rad,nbins),
            np.linspace(Y_cm - rad,Y_cm+rad,nbins)]
    
    mass_grid, xedges, yedges   = np.histogram2d(X,Y,
                                       bins=bins,
                                       weights=M,
                                       density=False)
    # max density indexes
    ix, iy = np.unravel_index(np.argmax(mass_grid), mass_grid.shape)
    
    # Compute center coordinates
    X_dns  = 0.5 * (xedges[ix] + xedges[ix+1])
    Y_dns  = 0.5 * (yedges[iy] + yedges[iy+1])

    return X_dns,Y_dns

def get_minzsource_proj(Gal,kw_parts,cutoff_radius,pixel_num,z_source_max,verbose=True,save_res=True,reload=True):
    # try all projection in order to obtain a lens
    proj_index = 0
    kw_res     = None
    # if present and reload:
    try:
        assert reload
        kw_res = load_whatever(Gal.proj_zs_path)
        return kw_res
    except AssertionError:
        pass
    except Exception as e :
        if verbose:
            print("Failed to load because "+str(e))
            print("Recomputing min_z_source ...")
        pass
    # else compute it
    while proj_index<3:
        try:
            kw_res = get_min_z_source(Gal=Gal,kw_parts=deepcopy(kw_parts),cutoff_radius=cutoff_radius,
                                    proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose)
            break
        except AttributeError as Ae:
            print("Error : ")
            print(Ae)
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if kw_res is None:
        print("M(gal)",Gal.M_tot)
        raise RuntimeError("There is no projection of the galaxy that create a lens given the z_source_max")
    else:
            
        if save_res:
            with open(Gal.proj_zs_path,"wb") as f:
                pickle.dump(kw_res,f)
            print("Saved "+Gal.proj_zs_path)
        return kw_res

def get_min_z_source(Gal,kw_parts,proj_index,pixel_num,z_source_max,cutoff_radius,verbose=True):
    Ms,Xs,Ys,Zs = kw_parts["Ms"],kw_parts["Xs"],kw_parts["Ys"],kw_parts["Zs"]
    nx,ny       = int(pixel_num),int(pixel_num) # note:pixel_num now are real, integ. number
        
    # given a projection, return the minimal z_source
    # fails if it can't produce a supercritical lens w. z_source<z_source_max
     
    
    # project given the proj_index
    Xs,Ys = proj_parts({"Xs":Xs,"Ys":Ys,"Zs":Zs},proj_index)
    # recenter around 
    Xdns,Ydns = findDens(Ms,Xs,Ys,cutoff_radius)
    Xs -= Xdns
    Ys -= Ydns
    x  = np.asarray(Xs.to("kpc").value) #kpc
    y  = np.asarray(Ys.to("kpc").value) #kpc
    m  = np.asarray(Ms.to("solMass").value, dtype=float)  # M_sol

    # Redshift: 
    z_lens = Gal.z 
    cosmo  = Gal.cosmo 
    # X,Y already recentered around 0
    
    cutoff_radius = to_dimless(cutoff_radius) #kpc
    xmin = -cutoff_radius
    ymin = -cutoff_radius
    xmax = +cutoff_radius
    ymax = +cutoff_radius
    # numpy.histogram2d returns H with shape (nx_bins, ny_bins) where H[i,j]
    # counts x-bin i and y-bin j. We transpose to (ny, nx) so rows are y.
    H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny],
                                       range=[[xmin, xmax], [ymin, ymax]],
                                       weights=m,density=False)  
    # if density=True, it normalises it to the total density
    # H is then the distribution of mass for each bin, not the density
    mass_grid = H.T.copy() # Solar Masses
    # H shape: (nx, ny) -> transpose to (ny, nx)
    plt.close()
    plt.imshow(np.log10(mass_grid))
    plt.title("Gal Name:"+Gal.Name)
    plt.colorbar()
    plt.savefig("tmp/mass_"+str(proj_index)+".png")
    print("DEBUG:"+"tmp/mass_"+str(proj_index)+".png")
    plt.close()
    # area of the (dx/dy) bins:
    dx = np.diff(xedges) #kpc #==(xmax - xmin) / nx 
    dy = np.diff(yedges) #kpc
    # density_ij = M_ij/(Area_bin_ij)
    density = mass_grid / (dx * dy)
    print("<Area>",np.mean(dx*dy), "kpc^2") 
    
    # dens now is in Msun/kpc^2 
    dens_Ms_kpc2 = density*u.Msun/(u.kpc*u.kpc)
    # define the z_source:        
    z_source_min = _get_min_z_source(cosmo=cosmo,z_lens=z_lens,dens_Ms_kpc2=dens_Ms_kpc2,
                            z_source_max=z_source_max,verbose=verbose)
    if z_source_min==0:
        raise AttributeError("Rerun trying different projection")

    kw_res = {"z_source_min":z_source_min,
              "proj_index":proj_index}
    return kw_res


def _get_min_z_source(cosmo,z_lens,dens_Ms_kpc2,z_source_max,verbose=True):
    # the lens has to be supercritical
    # dens>Sigma_crit = (c^2/4PiG D_d(z_lens) ) D_s(z_source)/D_ds(z_lens,z_source)
    # -> D_s(z_source)/D_ds(z_lens,z_source) < 4PiG D_d(z_lens) *dens/c^2
    # D_s(z_source)/D_ds(z_lens,z_source) is not easy to compute analytically, but we can sample it
    if z_lens>z_source_max:
        raise ValueError("The galaxy redshift is higher than the maximum allowed source redshift")
        #return 0
    try:
        dens_Ms_kpc2.value
    except:
        # dens_Ms_kpc2 is already given in Msun/kpc^2
        dens_Ms_kpc2 *= u.Msun/(u.kpc**2)
    assert dens_Ms_kpc2.unit==u.Msun/(u.kpc**2)
    
    max_DsDds = np.max(dens_Ms_kpc2)*4*np.pi*const.G*cosmo.angular_diameter_distance(z_lens)/(const.c**2) 
    max_DsDds = max_DsDds.to("").value # assert(max_DsDds.unit==u.dimensionless_unscaled) -> equivalent

    min_DsDds = cosmo.angular_diameter_distance(z_source_max)/cosmo.angular_diameter_distance_z1z2(z_lens,z_source_max) # this is the minimum
    min_DsDds = min_DsDds.to("").value # dimensionless
    
    z_source_range = np.linspace(z_lens+0.1,z_source_max,100) # it's a very smooth funct->
    DsDds = np.array([cosmo.angular_diameter_distance(z_s).to("Mpc").value/cosmo.angular_diameter_distance_z1z2(z_lens,z_s).to("Mpc").value for z_s in z_source_range])
    if not min_DsDds<max_DsDds:
        # to do: deal with this kind of output
        if verbose:
            print("Warning: the minimum z_source needed to have a lens is higher than the maximum allowed z_source")
            plt.plot(z_source_range,DsDds,ls="-",c="k",label=r"D$_{\text{s}}$/D$_{\text{ds}}$(z$_{source}$)")
            plt.xlabel(r"z$_{\text{source}}$")
            plt.axhline(max_DsDds,ls="--",c="r",label=r"max(dens)*4$\pi$*G*$D_l$/c$^2$="+str( short_SciNot(max_DsDds)))
            plt.legend()
            name = "tmp/DsDds.pdf"
            plt.savefig(name)
            print("max density",np.max(dens_Ms_kpc2))
            print("Saved "+name)
        return 0
    else:
        # Note: successful test means only that there is AT LEAST 1 PIXEL that is supercritical
        minimise     = np.abs(DsDds-max_DsDds) 
        z_source_min = z_source_range[np.argmin(minimise)]
        return z_source_min




def get_rough_radius(cosmo,z_source,z_lens,kw_part_arc,scale=2,verbose=True):
    # -> this should only be used for plotting
    # the idea is simple:
    # we want a very approximate idea of the theta_E of the galaxy
    # to do that, we fit a SIS to its particle distribution 
    # basically in 1D, assuming (wrong but we don't care) spherical symmetry
    # then we scale that by the scale (default=2) and that is our aperture

    Dd      = cosmo.angular_diameter_distance(z_lens).to("Mpc")
    Ds      = cosmo.angular_diameter_distance(z_source).to("Mpc")
    Dds     = cosmo.angular_diameter_distance_z1z2(z_lens,z_source).to("Mpc") 
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc")

    
    Ms,RAs,DECs  = kw_part_arc["Ms"],kw_part_arc["RAs"],kw_part_arc["DECs"]
    RA_cm,DEC_cm = get_CM(Ms,RAs,DECs)
    # note: RA/DEC are given in arcsec, and Ms in Msun
    RA_centered,DEC_centered = RAs-RA_cm,DECs-DEC_cm
    kw_part_RADEC_cnt = {"Ms":Ms,"RA":RA_centered,"DEC":DEC_centered}
    kw_Ddds = {"Dd":Dd,"Dds":Dds,"Ds":Ds}
    return scale*theta_E_from_particles(verbose=verbose,**kw_part_RADEC_cnt,**kw_Ddds) # arcsec

from scipy.ndimage import gaussian_filter1d
def theta_E_from_particles(Ms, RA, DEC, Dd, Ds, Dds, nbins=100,verbose=True,sigma_smooth=2.):
    # Physical scale of 1 arcsec at Dd
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc") # arcsec/kpc (on the lens plane)
    # Critical density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    # Radii in kpc
    thetas = np.sqrt(RA**2 + DEC**2)
    r_kpc  = thetas / arcXkpc
    # Histogram Σ(R)
    t_max = np.max(thetas)/10
    i = np.arange(nbins + 1)
    t_edges = t_max * np.sqrt(i / nbins)
    r_edges = t_edges/arcXkpc 
    hist, edges = np.histogram(r_kpc, bins=r_edges, weights=Ms)
    rmid = 0.5*(edges[1:] + edges[:-1])

    # Smooth Σ(R)
    Sigma_R = hist / (2*np.pi*rmid*np.diff(edges))  # convert to Σ(R)
    Sigma_R_s = gaussian_filter1d(Sigma_R, sigma_smooth)*Sigma_R.unit
    # Enclosed Σ(<R)
    Menc = np.cumsum(Sigma_R_s * 2*np.pi*rmid*np.diff(edges))
    Sigma_encl = Menc / (np.pi*rmid**2)
    # Solve Σ=Σcrit by interpolation
    theta_E_kpc = np.interp(Sigma_crit, Sigma_encl[::-1], rmid[::-1])
    theta_E_arcsec = theta_E_kpc *arcXkpc

    print("--DEBUG")
    print("theta_E_arcsec found",theta_E_arcsec)
    plt.scatter(rmid,Sigma_encl)
    plt.axhline(Sigma_crit.value,ls="--",c="r",label=r"$\Sigma_{crit}$")
    plt.axvline(to_dimless(theta_E_kpc),label=r"$\theta_E$="+str(theta_E_kpc))
    plt.xlabel(r"''")
    plt.ylabel(r"$\Sigma$ ["+str(Sigma_encl.unit)+"]")
    plt.title(r"$\Sigma_{encl}$")
    plt.legend()
    nm_tmp = "tmp/Sigma.png"
    plt.savefig(nm_tmp)
    plt.close()

    #if np.all(Sigma_encl < Sigma_crit):
    #    raise RuntimeError()
    #    return None  # no Einstein radius

    
    
    return theta_E_arcsec

"""
# old - imperfect due to histogram resolution at small apertures
def theta_E_from_particles(Ms, RA, DEC, Dd, Ds, Dds, nbins=200,verbose=True):
   
    #Ms [Msun], RA, DEC [arcsec]
    #Dd,Ds,Dds angular diameter distances [Mpc]
    #Returns theta_E in arcsec.
    
    # big brain time: 
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc") # arcsec/kpc (on the lens plane)
    # critical surface density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2") # Msun/kpc^2
    # cylindrical radius
    thetas = np.sqrt(RA**2 + DEC**2)  # arcsec
    
    # mass in annuli -> not too slow to compute
    # we recompute it such that the area are the same for all bins AND skip the first 2 bins
    r_max = np.max(thetas)
    Sigma_encl,theta_mid = _get_Sigma_encl(thetas,Ms,r_max,nbins)
    Sigma_encl_kpc2 = Sigma_encl*arcXkpc**2  # Msun/kpc^2

    # find R where Sigma_encl = Sigma_crit
    idx = np.argmin(np.abs(Sigma_encl_kpc2 - Sigma_crit))
    theta_E_arcsec = theta_mid[idx] # arcsec
    while idx==0:
        # the resolution of the histogram is likely too small to 
        # get the required resolution -
        # rerun the histogram within this aperture
        r_max = 3*theta_E_arcsec
        Sigma_encl,theta_mid = _get_Sigma_encl(thetas,Ms,r_max,nbins)
        Sigma_encl_kpc2 = Sigma_encl*arcXkpc**2  # Msun/kpc^2
        idx = np.argmin(np.abs(Sigma_encl_kpc2 - Sigma_crit))
        theta_E_arcsec = theta_mid[idx] # arcsec

    print("--DEBUG")
    plt.scatter(theta_mid*arcXkpc,Sigma_encl_kpc2)
    plt.axhline(Sigma_crit.value,ls="--",c="r",label=r"$\Sigma_{crit}$")
    plt.xlabel(r"kpc")
    plt.axvline(to_dimless(theta_E_arcsec*arcXkpc),label=r"$\theta_E$ [kpc]")

    plt.ylabel(r"$\Sigma$ ["+str(Sigma_encl_kpc2.unit)+"]")
    plt.title(r"$\Sigma_{encl}$")
    plt.legend()
    nm_tmp = "tmp/Sigma.png"
    plt.savefig(nm_tmp)
    plt.close()
    print("SAVED "+nm_tmp)
    print("DEBUG--")
    
    # DEBUG 
    plt.scatter(theta_mid,hist)
    plt.axvline(to_dimless(theta_E_arcsec),label=r"$\theta_E$")
    plt.title("Mass x bin")
    plt.xlabel(r"$\theta$['']")
    plt.ylabel("M")
    plt.legend()
    nm_tmp = "tmp/masshist1D.png"
    plt.savefig(nm_tmp)
    plt.close()
    print("theta_E:",theta_E_arcsec)
    print("DEBUG\nSaving "+nm_tmp)
    
    return theta_E_arcsec
"""
    
    
def _get_Sigma_encl(thetas,Ms,r_max,nbins):
    i = np.arange(nbins + 1)
    R_edges = r_max * np.sqrt(i / nbins)
    
    hist, edges = np.histogram(thetas, bins=R_edges, weights=Ms)
    theta_mid   = 0.5 * (edges[1:] + edges[:-1])
    
    # enclosed mass
    M_encl = np.cumsum(hist)
    
    # area of circle
    area = np.pi * theta_mid**2 #arcsec^2
    # average Sigma(<R)
    Sigma_encl      = M_encl / area # Msun/arcsec^2
    return Sigma_encl,theta_mid

from remade_gal import Gal2kwMXYZ
def Gal2MRADEC(Gal,proj_index,arcXkpc):
    kw_parts = Gal2kwMXYZ(Gal)
    Xs,Ys    = proj_parts(kw_parts,proj_index)
    RAs,DECs = Xs.to("kpc")*arcXkpc,Ys.to("kpc")*arcXkpc
    return kw_parts["Ms"],RAs,DECs
