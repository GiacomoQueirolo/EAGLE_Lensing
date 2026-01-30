# copy from project_gal_alpha
# instead of a "dumb" optimised 2D histogram, run Adaptive mesh refinement
# this would give me easily the MD and as well we can get the first bin higher than sigma crit in order to get theta_E
import dill
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import  Normalize
from matplotlib.cm import ScalarMappable

import astropy.units as u
import astropy.constants as const

from python_tools.tools import mkdir,to_dimless,ensure_unit,short_SciNot
from python_tools.get_res import load_whatever

from lib_cosmo import SigCrit
from ParticleGalaxy import Gal2kwMXYZ,get_CM

# for now keep this and check if still needed
dir_name     = "proj_part_hist"
def prep_Gal_projpath(Gal,dir_name=dir_name):
    # impractical but easy to set up
    Gal.proj_dir = Gal.gal_snap_dir+f"/{dir_name}_{Gal.Name}/"
    mkdir(Gal.proj_dir)
    Gal.projection_path = f"{Gal.proj_dir}/projection.pkl"
    return Gal


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

class ProjectionError(Exception):
    # very specific error: raise if there is no projection s.t. 
    def __init__(self, error):
        self.error   = str(error)
        self.message = "Projection Error: "+self.error
        super().__init__(self.message)
    def __str__(self):
        return self.message
        
def projection_main_AMR(Gal,kw_parts,z_source_max,sample_z_source,min_thetaE,
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
        print("Found and loaded projection from :"+Gal.projection_path)
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

    min_thetaE_kpc = min_thetaE/arcXkpc 

    while proj_index<3:
        try:
            kw_proj = {"proj_index":proj_index}
            # Project the particles 
            kw_parts_proj = project_kw_parts(kw_parts=kw_parts,proj_index=proj_index)
            
            kw_2Ddens = dens_map_AMR(Gal=Gal,
                                      kw_parts_proj=kw_parts_proj,
                                      verbose=verbose)

            savenameSigmaEnc =Gal.proj_dir+"/Sigma_enc_proj"+str(proj_index)+".png"
            kw_z_min = get_min_z_source(Gal=Gal,min_thetaE_kpc=min_thetaE_kpc,
                                      kw_2Ddens=kw_2Ddens,
                                      z_source_max=z_source_max,
                                      savenameSigmaEnc=savenameSigmaEnc,verbose=verbose)
            
            # sample z_source
            z_source = sample_z_source(z_source_min = kw_z_min["z_source_min"],z_source_max=z_source_max)
            kw_z_min["z_source"] = z_source
            
            # get an estimate of theta_E 
            thetaE = get_rough_thetaE_PLL(kw_2Ddens,Gal.cosmo,Gal.z,z_source,path=Gal.proj_dir,fig_Sig=kw_z_min["fig_Sig"])
            
            del kw_2Ddens["AMR_cells"] 
            del kw_z_min["fig_Sig"]
            kw_thetaE = {"thetaE":thetaE} 
                        
            kw_res  = kw_proj|kw_2Ddens|kw_z_min|kw_thetaE
            break
        except ProjectionError as PE:
            print(PE)
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if kw_res is None:
        print("M(gal)",short_SciNot(Gal.M_tot))
        print("z_gal",Gal.z)
        proj_message = "\nThere is no projection of the galaxy that create a lens given the z_source_max\n"
        Gal.update_is_lens(islens=False,message=proj_message)
        raise ProjectionError(proj_message)
    else:
        if save_res:
            with open(Gal.projection_path,"wb") as f:
                dill.dump(kw_res,f)
            print("Saved "+Gal.projection_path)
        return kw_res

#from AMR2D import AMR_density
from AMR2D_PLL import AMR_density_PLL
def dens_map_AMR(Gal,
                  kw_parts_proj,
                  max_particles=100,
                  min_area=0.1*u.kpc*u.kpc,
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
    #AMR_cells = AMR_density(Xs,Ys,Ms,max_particles=max_particles,min_area=min_area)
    # units are stripped by numba - have to "reattach" them "by hand"
    AMR_cells = AMR_density_PLL(Xs,Ys,Ms, max_particles=max_particles, min_area=min_area,dens_thresh=dens_thresh)
    """
    print("DEBUG")
    tmp_amr_name = "tmp/del_amr_cells.pkl"
    print("Saving ",tmp_amr_name)
    with open(tmp_amr_name,"wb") as f:
        dill.dump(AMR_cells,f)
    """
    #MD_coords_kpc,MD_value = get_MDfromAMRcells(AMR_cells)
    # use parallelised version - faster 
    MD_coords,MD_value = get_MDfromAMRcells_PLL(AMR_cells) 
    # Note: all inputs are still in kpc
    kw_2Ddens = {"MD_value":MD_value,"MD_coords":MD_coords,"AMR_cells":AMR_cells}
    return kw_2Ddens

def get_min_z_source(Gal,kw_2Ddens,z_source_max,min_thetaE_kpc,verbose=True,savenameSigmaEnc = "tmp/Sigma_enc.png"):
    # given a projection, return the minimal z_source
    # fails if it can't produce a supercritical lens w. z_source<z_source_max
        
    dens_at_thetamin = getDensAtRad(kw_2Ddens,min_thetaE_kpc)
    # add plot Sigma_encl vs theta
    Sigma_crit_min   = SigCrit(z_lens=Gal.z,z_source=z_source_max,cosmo=Gal.cosmo)
    r,Sigma_encl     = cells2SigRad(kw_2Ddens)
    arcXkpc = Gal.cosmo.arcsec_per_kpc_proper(Gal.z)
    theta = r*arcXkpc
    Sigma_encl_arc = Sigma_encl/(arcXkpc**2)
    Sigma_crit_min_arc = Sigma_crit_min/(arcXkpc**2)
    #plt.close()
    fig,ax = plt.subplots(1)
    ax.plot(theta,Sigma_encl_arc,color="k")
    ax.axhline(Sigma_crit_min_arc.value,ls="--",c="r",label=r"$\Sigma_{crit}^{min}=\Sigma_{crit}(z_{source,max}$="+str(z_source_max)+")="+str(short_SciNot(Sigma_crit_min_arc)))
    min_thetaE = min_thetaE_kpc*arcXkpc
    ax.axvline(to_dimless(min_thetaE),label=r"$\theta_{min}$="+str(short_SciNot(min_thetaE)),ls="-",c="grey")
    dens_at_thetamin_arc = dens_at_thetamin/(arcXkpc**2)
    ax.axhline(to_dimless(dens_at_thetamin_arc),label=r"$\Sigma(\theta_{min})$="+str(short_SciNot(dens_at_thetamin_arc)),ls="--",c="g")
    if np.any(Sigma_crit_min_arc<Sigma_encl_arc):
        theta_E_max = theta[np.argmin(np.abs(Sigma_crit_min_arc-Sigma_encl_arc))]
        ax.axvline(to_dimless(theta_E_max),label=r"$\theta_E(z_{s,max})$="+str(short_SciNot(theta_E_max)),ls="--",c="b")
    ax.set_xlabel(r'$\theta$ ["]')
    ax.set_ylabel(r"$\Sigma$ ["+str(Sigma_encl_arc.unit)+"]")
    ax.set_title(r"$\Sigma_{encl}$")
    ax.legend()
    #fig.savefig(savenameSigmaEnc)
    #plt.close(fig)
    #fig.savefig("tmp/Sigma_enc.png")
    #print("Saving "+savenameSigmaEnc)

    # define the z_source_min:        
    z_source_min = _get_min_z_source(cosmo=Gal.cosmo,z_lens=Gal.z,
                                    thresh_dens=dens_at_thetamin,
                                    z_source_max=z_source_max,verbose=verbose)
    if z_source_min==0:
        raise ProjectionError("This projection for this galaxy does not lead to a supercritical lens. Rerun trying different projection")

    kw_zs_min = {"z_source_min":z_source_min,"fig_Sig":fig}
    return kw_zs_min

def getDensAtRad(kw_2Ddens,rad):
    # get density within radius
    radii,Sigma_encl = cells2SigRad(kw_2Ddens)
    rad = ensure_unit(rad,radii.unit)
    i_r = np.argmin(np.abs(radii-rad))
    return Sigma_encl[i_r]
    

def _get_min_z_source(cosmo,z_lens,thresh_dens,z_source_max,verbose=True):
    # the lens has to be supercritical
    # dens>Sigma_crit = (c^2/4PiG D_d(z_lens) ) D_s(z_source)/D_ds(z_lens,z_source)
    # -> D_s(z_source)/D_ds(z_lens,z_source) < 4PiG D_d(z_lens) *dens/c^2
    # D_s(z_source)/D_ds(z_lens,z_source) is not easy to compute analytically, but we can sample it
    if z_lens>z_source_max:
        raise ValueError("The galaxy redshift is higher than the maximum allowed source redshift")
        #return 0 
    thresh_dens  = ensure_unit(thresh_dens,u.Msun/(u.kpc**2))
    
    thresh_DsDds = thresh_dens*4*np.pi*const.G*cosmo.angular_diameter_distance(z_lens)/(const.c**2) 
    thresh_DsDds = thresh_DsDds.to("").value # assert(max_DsDds.unit==u.dimensionless_unscaled) -> equivalent

    min_DsDds = cosmo.angular_diameter_distance(z_source_max)/cosmo.angular_diameter_distance_z1z2(z_lens,z_source_max) # this is the minimum
    min_DsDds = min_DsDds.to("").value # dimensionless
    
    z_source_range = np.linspace(z_lens+0.1,z_source_max,100) # it's a very smooth funct->
    DsDds = np.array([cosmo.angular_diameter_distance(z_s).to("Mpc").value/cosmo.angular_diameter_distance_z1z2(z_lens,z_s).to("Mpc").value for z_s in z_source_range])
    if not min_DsDds<thresh_DsDds:
        # to do: deal with this kind of output
        if verbose:
            print("Warning: the minimum z_source needed to have a lens is higher than the maximum allowed z_source")
            plt.close()
            fig_dsdds,ax = plt.subplots()
            ax.plot(z_source_range,DsDds,ls="-",c="k",label=r"D$_{\text{s}}$/D$_{\text{ds}}$(z$_{source}$)")
            ax.set_xlabel(r"z$_{\text{source}}$")
            ax.axhline(thresh_DsDds,ls="--",c="r",label=r"threshold(dens)*4$\pi$*G*$D_l$/c$^2$="+str( short_SciNot(thresh_DsDds)))
            ax.legend()
            name = "tmp/DsDds.pdf"
            fig_dsdds.savefig(name)
            plt.close(fig_dsdds)
            print("threshold density",short_SciNot(thresh_dens.value))
            print("Saved "+name)
        return 0
    else:
        # Note: successful test means that the threshold density = Sigmacrit given z_source =z_source_min
        minimise     = np.abs(DsDds-thresh_DsDds) 
        z_source_min = z_source_range[np.argmin(minimise)]
        return z_source_min
    
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

def get_rough_thetaE_PLL(kw_2Ddens,cosmo,z_lens,z_source,nm_sigmaplot="Sigma_AMR.png",path="tmp/",fig_Sig=None):
    # -> this should only be used for plotting
    # the idea is simple:
    # we want a very approximate idea of the theta_E of the galaxy
    # to do that, we fit a SIS to its particle distribution 
    # basically in 1D, assuming (wrong but we don't care) spherical symmetry
    # then we scale that by the scale (default=2) and that is our aperture

    Dd      = cosmo.angular_diameter_distance(z_lens).to("Mpc")
    Ds      = cosmo.angular_diameter_distance(z_source).to("Mpc")
    Dds     = cosmo.angular_diameter_distance_z1z2(z_lens,z_source).to("Mpc") 
    kw_Ddds = {"Dd":Dd,"Dds":Dds,"Ds":Ds}
    return theta_E_from_AMR_densitymap_PLL(kw_2Ddens=kw_2Ddens,nm_sigmaplot=nm_sigmaplot,path=path,fig_Sig=fig_Sig,**kw_Ddds)

def cells2SigRad(kw_2Ddens):    
    xc,yc = kw_2Ddens["MD_coords"] #kpc
    # to speed up the code I need to vectorise it -
    # but then I need to ingore the units
    #x0,x1,y0,y1,mass  = np.array([[c[0].value,c[1].value,c[2].value,c[3].value,c[4].value] for c in kw_2Ddens["AMR_cells"]]).T
    x0,x1,y0,y1,mass  = np.array([[cc.value for cc in c[:-1]] for c in kw_2Ddens["AMR_cells"]]).T
    x0_unit,x1_unit,y0_unit,y1_unit,mass_unit  = [c.unit for c in kw_2Ddens["AMR_cells"][0][:-1]]
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
    # area of the pixels
    area = (x1-x0)*(y1-y0)
    
    # Sort by radius
    idx = np.argsort(r)
    r_sorted = r[idx]
    m_sorted = mass[idx]
    area_sorted = area[idx] 
    # Cumulative sum
    cumulative_mass = np.cumsum(m_sorted)
    cumulative_area = np.cumsum(area_sorted)
    
    # Compute enclosed density Sigma(<r)
    Sigma_encl = cumulative_mass/cumulative_area
    return r_sorted,Sigma_encl
    
def theta_E_from_AMR_densitymap_PLL(kw_2Ddens, Dd, Ds, Dds,fig_Sig=None,nm_sigmaplot="Sigma.png",path="tmp/"):
    # Critical density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    # Physical scale of 1 arcsec at Dd
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc") # arcsec/kpc (on the lens plane)

    r_sorted,Sigma_encl = cells2SigRad(kw_2Ddens)
    # theta
    theta = r_sorted*arcXkpc

    thetaE = np.interp(Sigma_crit.value, Sigma_encl.value[::-1], theta[::-1].value)*theta.unit
    print("theta_E_arcsec found",short_SciNot(np.round(thetaE,2)))
    if fig_Sig is None:
        fig,ax = plt.subplots(1)
        ax.set_xlabel(r'$\theta$ ["]')
        ax.set_ylabel(r"$\Sigma$ ["+str(Sigma_encl.unit)+"]")
        ax.set_title(r"$\Sigma_{encl}$")
        ax.plot(theta,Sigma_encl,c="k")
    else:
        fig = fig_Sig
    fig.axes[0].axhline(Sigma_crit.value,ls="-.",c="r",label=r"$\Sigma_{crit}$ ["+str(Sigma_crit.unit)+"]")
    fig.axes[0].axvline(to_dimless(thetaE),label=r"$\theta_E$="+str(short_SciNot(thetaE)),ls="-",c="b")
    fig.axes[0].legend()
    nm_savefig = path+"/"+nm_sigmaplot
    print("Saving "+nm_savefig)
    fig.savefig(nm_savefig)
    fig.savefig("tmp/Sig_enc.png")
    plt.close(fig)
    return thetaE
    
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
    print(f"We recenter around the densest point (MD) obtained with AMR") 
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

def get_2Dkappa_map(Gal,proj_index,MD_coords,SigCrit,kwargs_extents,arcXkpc=None):
    if arcXkpc is None:
        arcXkpc = Gal.cosmo.arcsec_per_kpc_proper(Gal.z) 
    kw_samples = Gal2kw_samples(Gal=Gal,proj_index=proj_index,
                                MD_coords=MD_coords,arcXkpc=arcXkpc)
    Ms       = kw_samples["Ms"]
    RAs,DECs = kw_samples["RAs"],kw_samples["DECs"]
    
    mass_grid, xedges, yedges   = np.histogram2d(RAs,DECs,
                                       bins=kwargs_extents["bins_arcsec"],
                                       weights=Ms,
                                       density=False) 
    # mass_grid shape: (nx, ny) -> transpose to (ny, nx) -> given the circular simmetry, doesn't really matter
    Dra01,Ddec01 = kwargs_extents["DRaDec"]
    # density_ij = M_ij/(Area_bin_ij)
    density    = mass_grid.T / (Dra01*Ddec01/(arcXkpc**2)) # Msun/kpc^2
    kappa = density/SigCrit
    kappa = kappa.to("").value
    return kappa





def plot_amr_cells(kw_2Ddens):
    fig, ax = plt.subplots(figsize=(8,8))
    a,b= [],[]
    
    xc,yc = kw_2Ddens["MD_coords"] #kpc
    cells = kw_2Ddens["AMR_cells"]
    # to speed up the code I need to vectorise it -
    # but then I need to ingore the units
    #x0,x1,y0,y1,mass  = np.array([[c[0].value,c[1].value,c[2].value,c[3].value,c[4].value] for c in kw_2Ddens["AMR_cells"]]).T
    x0,x1,y0,y1,mass,dns  = np.array([[cc.value for cc in c] for c in cells]).T
    x0_unit,x1_unit,y0_unit,y1_unit,mass_unit,dns_unit  = [c.unit for c in cells[0]]
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
    dns  *=dns_unit
    
    vmax,vmin = np.max(dns.value),np.min(dns.value)
    cmap = plt.get_cmap("hot")
    norm = Normalize(vmin=vmin, vmax=vmax)
    _ = [ax.add_patch(patches.Rectangle((x0[i].value,y0[i].value),x1[i].value-x0[i].value,y1[i].value-y0[i].value,fill=True,linewidth=0.5,facecolor=cmap(norm(dns[i].value)))) for i in range(len(cells))]
    
    ax.set_xlim(np.min(x0.value),np.max(x0.value))
    ax.set_ylim(np.min(y0.value),np.max(y0.value))
    ax.set_aspect("equal")
    ax.set_xlabel("x ["+str(x0.unit)+"]")
    ax.set_ylabel("y ["+str(y0.unit)+"]")
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Density")
    return fig,ax