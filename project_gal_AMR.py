# copy from project_gal_alpha
# instead of a "dumb" optimised 2D histogram, run Adaptive mesh refinement
# this would give me easily the MD and as well we can get the first bin higher than sigma crit in order to get theta_E
import os
import glob
import pickle
import numpy as np
from time import time
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
    Gal.projection_path = f"{Gal.proj_dir}/projection.pkl"
    return Gal

from python_tools.get_res import load_whatever
from copy import copy,deepcopy

def proj_parts(kw_parts,proj_index):    
    Xs,Ys,Zs = kw_parts["Xs"],kw_parts["Ys"],kw_parts["Zs"]
    if proj_index==0:
        _   = True  # all as usual
    elif proj_index==1:
        Ys  = copy(Zs)
    elif proj_index==2:
        Xs  = copy(Ys)
        Ys  = copy(Zs)    
    else:
        raise RuntimeError("Projection index can only be 1,2 or 3, not "+str(proj_index))
    return Xs,Ys 

def project_kw_parts(kw_parts,proj_index):    
    Xs,Ys = proj_parts(kw_parts,proj_index) 
    kw_parts_proj = {"Xs":Xs,"Ys":Ys,"Ms":kw_parts["Ms"]} 
    return kw_parts_proj

def kwparts2arcsec(kw_parts,arcXkpc):
    if "Zs" in kw_parts:
        # have no sense to have 3D distr in arcsec
        raise RuntimeError("The kw_parts should already be projected")
    RAs = kw_parts["Xs"]*arcXkpc
    DECs = kw_parts["Ys"]*arcXkpc
    return {"RAs":RAs,"DECs":DECs,"Ms":kw_parts["Ms"]}

def projection_main_AMR(Gal,kw_parts,z_source_max,sample_z_source,
                    scale_radius=2,
                    arcXkpc=None,verbose=True,save_res=True,reload=True):
    # this is going to be the main function:
    # - for each projection:
    #       - find center and densest bin iteratively:
    #           - output:
    #               -  MD (mode/maximum density) coord
    #               -  MD value
    #               -  ~best 2D density histogram~ -> too large and easy to recover
    #                       - instead: nbins/pixel_num is fixed, give best cutout  
    #       - test if said densest bin is enough to be a SGL  
    #       - return:
    #              - projection
    #              - z_min
    #              - MD coord
    #              - theta_E (approx) centered around MD -
    # try all projection in order to obtain a lens
    proj_index = 0
    kw_res     = None
    # if present and reload: -> not sure if this is to be done
    try:
        assert reload
        kw_res = load_whatever(Gal.projection_path)
        return kw_res
    except AssertionError:
        pass
    except Exception as e :
        if verbose:
            print("Failed to load because "+str(e))
            print("Recomputing projection ...")
        pass
    # else compute it
    if arcXkpc is None:
        arcXkpc = Gal.cosmo.arcsec_per_kpc_proper(Gal.z)
    while proj_index<3:
        try:
            kw_proj = {"proj_index":proj_index}
            # iterate density histogram
            kw_parts_proj = project_kw_parts(kw_parts=kw_parts,proj_index=proj_index)
            #kw_parts_proj_arcsec = kwparts2arcsec(kw_parts_proj,arcXkpc)
            t0 = time()
            kw_2Ddens = dens_map_AMR(Gal=Gal,
                                      kw_parts_proj=kw_parts_proj,
                                      verbose=verbose)
            """t1 = time()-t0
            t0 = time()
            if verbose:
                print("info:\n",np.round(t1),"seconds for the dens_map_AMR")
            """
            kw_z_min = get_min_z_source(Gal=Gal,
                                      kw_2Ddens=kw_2Ddens,
                                      z_source_max=z_source_max,verbose=verbose)
            """
            t2 = time()-t0
            if verbose:
                print("info:\n",np.round(t2),"seconds for the get_min_z_source")
            """
            # sample z_source
            z_source = sample_z_source(z_source_min = kw_z_min["z_source_min"],z_source_max=z_source_max)
            kw_z_min["z_source"] = z_source
            # get an estimate of theta_E scaled
            #radius_kpc = get_rough_radius(Gal.cosmo,Gal.z,z_source,kw_2Ddens["AMR_cells"],kw_2Ddens["MD_coords"],scale=scale_radius,verbose=verbose)
            #t0 = time()
            radius = get_rough_radius_PLL(Gal.cosmo,Gal.z,z_source,kw_2Ddens["AMR_cells"],
                                              kw_2Ddens["MD_coords"],scale=scale_radius,path=Gal.proj_dir)
            """t3 = time()-t0
            if verbose:
                print("info:\n",np.round(t3),"seconds for the get_rough_radius_PLL")
            """
            del kw_2Ddens["AMR_cells"] 
            kw_radius = {"radius":radius} 
                        
            kw_res  = kw_proj|kw_2Ddens|kw_z_min|kw_radius
            break
        except AttributeError as Ae:
            print("Projection Error : "+str(Ae))
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if kw_res is None:
        print("M(gal)",short_SciNot(Gal.M_tot))
        print("z_gal",Gal.z)
        raise RuntimeError("There is no projection of the galaxy that create a lens given the z_source_max")
    else:
        if save_res:
            with open(Gal.projection_path,"wb") as f:
                pickle.dump(kw_res,f)
            print("Saved "+Gal.projection_path)
        return kw_res

#from AMR2D import AMR_density
from AMR2D_PLL import AMR_density_PLL
def dens_map_AMR(Gal,
                  kw_parts_proj,
                  max_particles=100,
                  min_size=0.1*u.kpc,
                  dens_thresh = 0.*u.Msun/(u.kpc**2),
                  verbose=True):
    # returns: kw_2Ddens["MD_value"][u.Msun/(u.kpc**2),1] 
    #          kw_2Ddens["MD_coord"][arcsec,2]
    #          kw_2Ddens["AMR_cells"][cells,N]
    
    Ms = np.asarray(kw_parts_proj["Ms"].to("Msun"))*u.Msun
    Xs = np.asarray(kw_parts_proj["Xs"].to("kpc"))*u.kpc
    Ys = np.asarray(kw_parts_proj["Ys"].to("kpc"))*u.kpc
    """
    # Get density map (dimensional)
    x  = np.asarray(Xs.to("kpc").value) #kpc
    y  = np.asarray(Ys.to("kpc").value) #kpc
    m  = np.asarray(Ms.to("solMass").value)  # M_sol
    """
    #AMR_cells = AMR_density(Xs,Ys,Ms,max_particles=max_particles,min_size=min_size)
    # units are stripped by numba - have to "reattach" them "by hand"
    AMR_cells = AMR_density_PLL(Xs,Ys,Ms, max_particles=max_particles, min_size=min_size,dens_thresh=dens_thresh)
    """
    print("DEBUG")
    tmp_amr_name = "tmp/del_amr_cells.pkl"
    print("Saving ",tmp_amr_name)
    with open(tmp_amr_name,"wb") as f:
        pickle.dump(AMR_cells,f)
    """
    #MD_coords_kpc,MD_value = get_MDfromAMRcells(AMR_cells)
    # use parallelised version - faster 
    MD_coords,MD_value = get_MDfromAMRcells_PLL(AMR_cells) 
    # Note: all inputs are still in kpc
    kw_2Ddens = {"MD_value":MD_value,"MD_coords":MD_coords,"AMR_cells":AMR_cells}
    return kw_2Ddens


def get_min_z_source(Gal,kw_2Ddens,z_source_max,verbose=True):
    # given a projection, return the minimal z_source
    # fails if it can't produce a supercritical lens w. z_source<z_source_max
    
    max_dens = kw_2Ddens["MD_value"]
    # define the z_source_min:        
    z_source_min = _get_min_z_source(cosmo=Gal.cosmo,z_lens=Gal.z,
                                    max_dens=max_dens,
                                    z_source_max=z_source_max,verbose=verbose)
    if z_source_min==0:
        raise AttributeError("Rerun trying different projection")

    kw_zs_min = {"z_source_min":z_source_min}
    return kw_zs_min


def _get_min_z_source(cosmo,z_lens,max_dens,z_source_max,verbose=True):
    # the lens has to be supercritical
    # dens>Sigma_crit = (c^2/4PiG D_d(z_lens) ) D_s(z_source)/D_ds(z_lens,z_source)
    # -> D_s(z_source)/D_ds(z_lens,z_source) < 4PiG D_d(z_lens) *dens/c^2
    # D_s(z_source)/D_ds(z_lens,z_source) is not easy to compute analytically, but we can sample it
    if z_lens>z_source_max:
        raise ValueError("The galaxy redshift is higher than the maximum allowed source redshift")
        #return 0
    try:
        max_dens.value
    except:
        # max_dens is already given in Msun/kpc^2
        max_dens *= u.Msun/(u.kpc**2)
    assert max_dens.unit==u.Msun/(u.kpc**2)
    
    max_DsDds = max_dens*4*np.pi*const.G*cosmo.angular_diameter_distance(z_lens)/(const.c**2) 
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
            print("max density",short_SciNot(max_dens.value))
            print("Saved "+name)
        return 0
    else:
        # Note: successful test means only that there is AT LEAST 1 PIXEL that is supercritical
        minimise     = np.abs(DsDds-max_DsDds) 
        z_source_min = z_source_range[np.argmin(minimise)]
        return z_source_min


def get_MDfromAMRcells(AMR_cells):
    try:
        dns_unit = AMR_cells[0].density.unit
        density = np.array([c.density.value for c in AMR_cells])*dns_unit
    except:
        density = np.array([c.density for c in AMR_cells])
    c_MD      = AMR_cells[np.argmax(density)]
    MD_coords = (c_MD.x0+c_MD.x1)/2.,(c_MD.y0+c_MD.y1)/2.
    try:
        MD_coords = np.array([mdc.value for mdc in MD_coords])*MD_coords[0].unit
    except:
        pass
    MD_value  = np.max(density)
    return MD_coords,MD_value
    
def get_MDfromAMRcells_PLL(AMR_cells):
    # for parallelised version
    try:
        dns_unit = AMR_cells[0][-1].unit
        density = np.array([c[-1].value for c in AMR_cells])*dns_unit
    except:
        density = np.array([c[-1] for c in AMR_cells])
    c_MD      = AMR_cells[np.argmax(density)]
    MD_coords = (c_MD[0]+c_MD[1])/2.,(c_MD[2]+c_MD[3])/2.
    try:
        MD_coords = np.array([mdc.value for mdc in MD_coords])*MD_coords[0].unit
    except:
        pass
    MD_value  = np.max(density)
    return MD_coords,MD_value

def get_rough_radius(cosmo,z_lens,z_source,AMR_cells,MD_coords=None,scale=2,nm_sigmaplot="tmp/Sigma_AMR.png"):
    # -> this should only be used for plotting
    # the idea is simple:
    # we want a very approximate idea of the theta_E of the galaxy
    # to do that, we fit a SIS to its particle distribution 
    # basically in 1D, assuming (wrong but we don't care) spherical symmetry
    # then we scale that by the scale (default=2) and that is our aperture
    if MD_coords is None:
        MD_coords,_ = get_MDfromAMRcells(AMR_cells)

    Dd      = cosmo.angular_diameter_distance(z_lens).to("Mpc")
    Ds      = cosmo.angular_diameter_distance(z_source).to("Mpc")
    Dds     = cosmo.angular_diameter_distance_z1z2(z_lens,z_source).to("Mpc") 
    
    return scale*theta_E_from_AMR_densitymap(AMR_cells,MD_coords, **kw_Ddds)

def get_rough_radius_PLL(cosmo,z_lens,z_source,AMR_cells,MD_coords=None,scale=2,nm_sigmaplot="Sigma_AMR.png",path="tmp/"):
    # -> this should only be used for plotting
    # the idea is simple:
    # we want a very approximate idea of the theta_E of the galaxy
    # to do that, we fit a SIS to its particle distribution 
    # basically in 1D, assuming (wrong but we don't care) spherical symmetry
    # then we scale that by the scale (default=2) and that is our aperture
    if MD_coords is None:
        MD_coords,_ = get_MDfromAMRcells(AMR_cells)

    Dd      = cosmo.angular_diameter_distance(z_lens).to("Mpc")
    Ds      = cosmo.angular_diameter_distance(z_source).to("Mpc")
    Dds     = cosmo.angular_diameter_distance_z1z2(z_lens,z_source).to("Mpc") 
    kw_Ddds = {"Dd":Dd,"Dds":Dds,"Ds":Ds}
    return scale*theta_E_from_AMR_densitymap_PLL(AMR_cells,MD_coords,nm_sigmaplot=nm_sigmaplot,path=path,**kw_Ddds)



def theta_E_from_AMR_densitymap(AMR_cells,MD_coords, Dd, Ds, Dds,nm_sigmaplot="Sigma.png",path="tmp/"):
    # Critical density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    # Physical scale of 1 arcsec at Dd
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc") # arcsec/kpc (on the lens plane)
    xc,yc = MD_coords*arcXkpc #kpc
    
    entries = []
    # Build list of (radius, mass)
    for c in AMR_cells:
        dx = .5*(c.x0+c.x1) - xc #kpc
        dy = .5*(c.y0+c.y1) - yc #kpc
        r = np.sqrt(dx*dx + dy*dy) #kpc
        mass = c.mass              #Msun
        entries.append((r, mass))

    # Sort by radius
    entries.sort(key=lambda x: x[0])

    # Cumulative sum
    cumulative_mass = 0.0
    Sigma_encl = []
    theta = []
    for r, m in entries:
        cumulative_mass += m # Msun
        Enc_Dens = cumulative_mass/(np.pi*r*r) # Msun/kpc^2 
        Sigma_encl.append(Enc_Dens.value)
        t = r*arcXkpc
        theta.append(t.value)
    Sigma_encl = np.array(Sigma_encl)*Sigma_encl.unit
    theta = np.array(theta)*t.unit
    thetaE = np.interp(Sigma_crit.value, Sigma_encl.value[::-1], theta[::-1].value)*theta.unit
    """
    for r, m in entries:
        cumulative_mass += m # Msun
        Enc_Dens = cumulative_mass/(np.pi*r*r) # Msun/kpc^2 
        if Enc_Dens >= Sigma_crit:
            theta = r*arcXkpc
            return theta
    # Threshold not reached
    return None
    """
    print("theta_E_arcsec found",short_SciNot(np.round(thetaE,2)))
    plt.scatter(theta,Sigma_encl)
    plt.axhline(Sigma_crit.value,ls="--",c="r",label=r"$\Sigma_{crit}$ ["+str(Sigma_crit.unit)+"]")
    plt.axvline(to_dimless(thetaE),label=r"$\theta_E$="+str(short_SciNot(thetaE)),ls="--",c="k")
    plt.xlabel(r"$\theta$ ['']")
    plt.ylabel(r"$\Sigma$ ["+str(Sigma_encl.unit)+"]")
    plt.title(r"$\Sigma_{encl}$")
    plt.legend()
    nm_savefig = path+"/"+nm_sigmaplot
    print("Saving "+nm_savefig)
    plt.savefig(nm_savefig)
    plt.close()

    return thetaE

def theta_E_from_AMR_densitymap_PLL(AMR_cells,MD_coords, Dd, Ds, Dds,nm_sigmaplot="Sigma.png",path="tmp/"):
    # Critical density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    # Physical scale of 1 arcsec at Dd
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc") # arcsec/kpc (on the lens plane)
    xc,yc = MD_coords #kpc
    # to speed up the code I need to vectorise it -
    # but then I need to ingore the units
    #x0,x1,y0,y1,mass  = np.array([[c[0].value,c[1].value,c[2].value,c[3].value,c[4].value] for c in AMR_cells]).T
    x0,x1,y0,y1,mass  = np.array([[cc.value for cc in c[:-1]] for c in AMR_cells]).T
    x0_unit,x1_unit,y0_unit,y1_unit,mass_unit  = [c.unit for c in AMR_cells[0][:-1]]
    # verify that the units are consistent
    assert x0_unit==x1_unit
    assert x0_unit==y0_unit
    assert x0_unit==y1_unit
    assert x0_unit==xc.unit
    assert x0_unit==yc.unit
    length_unit = x0_unit
    x0 *=length_unit
    x1 *=length_unit
    y0 *=length_unit
    y1 *=length_unit
    mass *=mass_unit

    # locate center of the cells
    xc_cell = 0.5 * (x0 + x1)
    yc_cell = 0.5 * (y0 + y1)

    # compute their radius wrt MD
    dx = xc_cell - xc
    dy = yc_cell - yc
    r = np.sqrt(dx*dx + dy*dy)
    
    # Sort by radius
    idx = np.argsort(r)
    r_sorted = r[idx]
    m_sorted = mass[idx]
    
    # Cumulative sum
    cumulative_mass = np.cumsum(m_sorted)
    
    # Compute enclosed density Sigma(<r)
    Sigma_encl = cumulative_mass/ (np.pi * r_sorted * r_sorted)

    # theta
    theta = r_sorted*arcXkpc

    """
    entries = []    
    # Build list of (radius, mass)
    for c in AMR_cells:
        dx = .5*(c[0]+c[1]) - xc #kpc
        dy = .5*(c[2]+c[3]) - yc #kpc
        r = np.sqrt(dx*dx + dy*dy) #kpc
        mass = c[-2]            #Msun
        entries.append((r, mass))
    # Sort by radius
    entries.sort(key=lambda x: x[0])

    # Cumulative sum
    cumulative_mass = 0.0
    Sigma_encl = []
    theta = []
    for r, m in entries:
        cumulative_mass += m # Msun
        Enc_Dens = cumulative_mass/(np.pi*r*r) # Msun/kpc^2 
        Sigma_encl.append(Enc_Dens.value)
        t = r*arcXkpc
        theta.append(t.value)
    for r, m in entries:
        cumulative_mass += m # Msun
        Enc_Dens = cumulative_mass/(np.pi*r*r) # Msun/kpc^2 
        if Enc_Dens >= Sigma_crit:
            theta = r*arcXkpc
            return theta
    # Threshold not reached
    return None
    Sigma_encl = np.array(Sigma_encl)*Enc_Dens.unit
    theta = np.array(theta)*t.unit
    """
    thetaE = np.interp(Sigma_crit.value, Sigma_encl.value[::-1], theta[::-1].value)*theta.unit
    print("theta_E_arcsec found",short_SciNot(np.round(thetaE,2)))
    plt.scatter(theta,Sigma_encl)
    plt.axhline(Sigma_crit.value,ls="--",c="r",label=r"$\Sigma_{crit}$ ["+str(Sigma_crit.unit)+"]")
    plt.axvline(to_dimless(thetaE),label=r"$\theta_E$="+str(short_SciNot(thetaE)),ls="--",c="k")
    plt.xlabel(r"$\theta$ ['']")
    plt.ylabel(r"$\Sigma$ ["+str(Sigma_encl.unit)+"]")
    plt.title(r"$\Sigma_{encl}$")
    plt.legend()
    nm_savefig = path+"/"+nm_sigmaplot
    print("Saving "+nm_savefig)
    plt.savefig(nm_savefig)
    plt.close()
    return thetaE
    
from remade_gal import Gal2kwMXYZ,get_CM
def Gal2MRADEC(Gal,proj_index,arcXkpc):
    kw_parts = Gal2kwMXYZ(Gal)
    Xs,Ys    = proj_parts(kw_parts,proj_index)
    RAs,DECs = Xs.to("kpc")*arcXkpc,Ys.to("kpc")*arcXkpc
    return kw_parts["Ms"],RAs,DECs

def Gal2kw_samples(Gal,proj_index,MD_coords,arcXkpc):
    # we scale the radius by a factor to be sure to include the center
    Ms,RAs,DECs = Gal2MRADEC(Gal,proj_index,arcXkpc=arcXkpc)
    # RA,DEC= arcsec, Ms = Msun
    #print("Some galaxy have a 'shifted' CM")
    RA_cm,DEC_cm = get_CM(Ms,RAs,DECs)
    print(f"We recenter around the densest point (MD) obtained iteratively (see iterate_dens_map)") 
    RA_MD,DEC_MD = MD_coords.to("kpc")*arcXkpc
    print("Info:  CM vs Densest ")
    print("CM:",np.round(RA_cm,2),np.round(DEC_cm,2))
    print("Dns:",np.round(RA_MD,2),np.round(DEC_MD,2))
    print("Dist:",np.round(np.sqrt((RA_cm-RA_MD)**2+(DEC_cm-DEC_MD)**2),2))

    kw_samples   = {}
    kw_samples["RAs"]  = RAs-RA_MD   #arcsec
    kw_samples["DECs"] = DECs-DEC_MD  #arcsec
    
    kw_samples["Ms"]   = Ms    #Msun
    kw_samples["cm"]   = RA_cm-RA_MD,DEC_cm-DEC_MD  # 
    return kw_samples
