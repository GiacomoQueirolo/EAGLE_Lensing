# to run to test generate_particle_lens
import numpy as np
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from mpl_toolkits.axes_grid1 import make_axes_locatable

import stpsf,dill
import pandas as pd
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.ObservationConfig.JWST import JWST

from plot_PL import plot_all
from particle_galaxy import PartGal
from python_tools.tools import to_dimless
from python_tools.image_manipulation import plot_comp_two_images
from particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
from generate_particle_lens import wrapper_get_rnd_lens,get_extents,LoadLens,LensPart
from generate_particle_lens import pixel_num,kw_prior_z_source_minimal

import scipy.interpolate as interp



from lenstronomy.LensModel.LineOfSight.LOSModels.los import LOS

if __name__ == "__main__":

    #print("Loading specific gal for debugging")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_18//Gn5SGn0.pkl")
    profiler = Profiler()
    profiler.start()

    #mod_LP = wrapper_get_rnd_lens(reload=False)
    Gal    = PartGal(5,0,
                 z=None,snap="20",    # redshift or snap
                 M=None,Centre=None,
                 reload=True)

    kw_add_lenses = {"lens_model_list":["LOS"],
                    "kwargs_lens":[]}
    # load first parameters from analosis, golden sample
    path = "/pbs/home/g/gqueirolo/analosis/analosis/results/datasets/golden_sample_input_kwargs.csv"
    kw   = pd.read_csv(path)
    los_cols = ['kappa_os', 'gamma1_os', 'gamma2_os', 'omega_os',
    'kappa_od', 'gamma1_od', 'gamma2_od', 'omega_od',
    'kappa_ds', 'gamma1_ds', 'gamma2_ds', 'omega_ds',
    'kappa_los', 'gamma1_los', 'gamma2_los', 'omega_los']
    los  = kw.loc[:, los_cols]
    list_los = los.to_dict('records')
    kw_los = list_los[0]
    kw_add_lenses["kwargs_lens"] = [kw_los]

    # compute the LOS effect exactly (analytically)
    mod_LP_LOSanl = LensPart(Galaxy=Gal,
                      kwlens_part=kwlens_part_AS,
                      kw_prior_z_source=kw_prior_z_source_minimal,
                      pixel_num=pixel_num,
                      kw_add_lenses=kw_add_lenses,
                      reload=True,
                      savedir_sim="test_sim_lens_LOS")
    
    mod_LP_LOSanl.run()
    alpha_map_los_anl = mod_LP_LOSanl.alpha_map

    # compute LOS effects "a posteriori" (numerically)
    kw_prior_z_source = kw_prior_z_source_minimal | {"fixed_z_source":mod_LP_LOSanl.z_source}
    mod_LP_LOSnum = LensPart(Galaxy=Gal,
                      kwlens_part=kwlens_part_AS,
                      kw_prior_z_source=kw_prior_z_source,
                      pixel_num=pixel_num,
                      kw_add_lenses=None,
                      reload=True,
                      savedir_sim="test_sim_lens_LOSnum")

    mod_LP_LOSnum.run()
    #print("Verify same z_source: ",mod_LP_LOSanl.z_source,"==",mod_LP_LOSnum.z_source,mod_LP_LOSanl.z_source == mod_LP_LOSnum.z_source) #-> verified
    
    # fit the 2D alphamap
    ra,dec = mod_LP_LOSnum.get_RADEC()
    radec_points = np.stack([ra.ravel(),dec.ravel()],-1)

    alpha_map = mod_LP_LOSnum.alpha_map
    alpha_map_ra  = alpha_map[0].ravel()
    alpha_map_dec = alpha_map[1].ravel()

    # "sparse" the alpha map
    np.hypot np.gradient(alpha_map[0])
    alpha_ra_interp_flat = interp.RBFInterpolator(radec_points, alpha_map_ra,smoothing=0, kernel='cubic')
    alpha_dec_interp_flat = interp.RBFInterpolator(radec_points, alpha_map_dec,smoothing=0, kernel='cubic')
    print("too large to be stored")
    """
    # since it's a test, we do not store it in a standard way
    alpha_interp_name = "tmp/alpha_interp.pkl"
    print("Saving "+alpha_interp_name)
    kw_alpha_interp_flat = {"alpha_ra_interp_flat":alpha_ra_interp_flat,
                            "alpha_dec_interp_flat":alpha_dec_interp_flat}
    with open(alpha_interp_name,"wb") as f:
        dill.dump(kw_alpha_interp_flat,f)    
    """
    
    #initialise LOS class only used for the distort_vector function
    los = LOS() 
    # then follow the alpha function of lenstronomy.LensModel.LineOfSight.single_plane_los.SinglePlaneLOS
    
    # Angular position where the ray hits the deflector's plane
    ra_d,dec_d = los.distort_vector(
            ra,
            dec,
            kappa=kw_los["kappa_od"],
            omega=kw_los["omega_od"],
            gamma1=kw_los["gamma1_od"],
            gamma2=kw_los["gamma2_od"],
        )

    # Displacement due to the main lens only
    radec_d        = np.stack([ra_d.ravel(),dec_d.ravel()],-1)
    alpha_ra_flat  = alpha_ra_interp_flat(radec_d)
    alpha_dec_flat = alpha_dec_interp_flat(radec_d)
    alpha_ra  = alpha_ra_flat.reshape(alpha_map[0].shape)
    alpha_dec = alpha_dec_flat.reshape(alpha_map[1].shape)
    # Correction due to the background convergence, shear and rotation
    alpha_ra, alpha_dec = los.distort_vector(
        alpha_ra,
        alpha_dec,
        kappa=kw_los["kappa_ds"],
        omega=kw_los["omega_ds"],
        gamma1=kw_los["gamma1_ds"],
        gamma2=kw_los["gamma2_ds"],
    )
    # Perturbed position in the absence of the main lens
    theta_ra_os, theta_dec_os = los.distort_vector(
        ra,
        dec,
        kappa=kw_los["kappa_os"],
        omega=kw_los["omega_os"],
        gamma1=kw_los["gamma1_os"],
        gamma2=kw_los["gamma2_os"],
    )

    # Complete displacement
    alpha_ra += ra - theta_ra_os
    alpha_dec += dec - theta_dec_os
    alpha_map_los_num = (alpha_ra,alpha_dec)

    # since it's a test, we do not store it in a standard way
    alpha_los_name = "tmp/alpha_los.pkl"
    print("Saving "+alpha_los_name)
    with open(alpha_los_name,"wb") as f:
        dill.dump({"alpha_los_num":alpha_map_los_num,
                   "alpha_los_anl":alpha_map_los_anl,
                   },f)    

    kw_extents = get_extents(mod_LP_LOSanl.arcXkpc,mod_LP_LOSanl)
    extent_arcsec = kw_extents["extent_arcsec"]
    ttl1 = r"|$\alpha_{LOS}^{anl.}$|"
    ttl2 = r"|$\alpha_{LOS}^{num.}$|"
    xlbl = "RA ['']"
    ylbl = "DEC ['']"
    mod_alpha_los_num = np.sqrt(alpha_map_los_num[0]**2+alpha_map_los_num[1]**2)
    mod_alpha_los_anl = np.sqrt(alpha_map_los_anl[0]**2+alpha_map_los_anl[1]**2)
    fig = plot_comp_two_images(mod_alpha_los_num,mod_alpha_los_anl,
                         xlbl=xlbl,ylbl=ylbl,
                         ttl1=ttl1,ttl2=ttl2,
                         extent=extent_arcsec)
    nm = "tmp/comp_alpha_los_num_anl.pdf"
    fig.savefig(nm)
    print(f"Saving {nm}")


    profiler.stop()
    print(profiler.output_text(color=True,show_all=False))
    # have to move the source to exactly the same position
    kw_source = mod_LP_LOSanl.kwargs_source
    mod_LP_LOSnum.update_source_position(ra_source=kw_source["center_x"],
                                         dec_source=kw_source["center_y"])
    
    imsim_LOSanl = mod_LP_LOSanl.image_sim 
    _imsim_LOSanl = mod_LP_LOSanl.get_lensed_image(alpha_map=alpha_map_los_anl)
    print("verify correct computation\n mod_LP_LOSanl.image_sim == mod_LP_LOSanl.get_lensed_image(alpha_map=alpha_map_los_anl):",  imsim_LOSanl ==_imsim_LOSanl,np.mean(imsim_LOSanl),np.mean(_imsim_LOSanl))
    
    imsim_LOSnum = mod_LP_LOSnum.get_lensed_image(alpha_map=alpha_map_los_num)

    fg = plot_comp_two_images(imsim_LOSanl,imsim_LOSnum,
                         xlbl=xlbl,ylbl=ylbl,
                         ttl1="LOS (anl.)",ttl2="LOS (num.)",
                         extent=extent_arcsec)
    nm = "tmp/comp_im_LOS.png"
    fg.savefig(nm)
    print(f"Saving {nm}")

    exit()
    
    plot_all(mod_LP,skip_caustic=True)

    profiler.start()

    band_HST = HST(band='WFC3_F160W', psf_type="GAUSSIAN")
    #band = HST(band='WFC3_F160W', psf_type="PIXEL") #-> if pixel, we need to give kernel_point_source (and point_source_supersampling_factor etc)
    #band.obs["psf_type"] = "PIXEL"
    #del band.obs["seeing"]
    #band.obs["kernel_point_source"] = []
    SimObs_HST = mod_LP.get_SimObs(band_HST,kwargs_source_model=None)

    image_hst = mod_LP.sim_image(SimObs_HST)
    band_JWST = JWST(band='F444W', psf_type="GAUSSIAN")
    

    
    SimObs_jwst = mod_LP.get_SimObs(band_JWST,kwargs_source_model=None)
    image_jwst = mod_LP.sim_image(SimObs_jwst)


    # more realistic PSF for JWST
    nrc = stpsf.NIRCam()
    nrc.filter =  'F444W'
    
    pssf = 4
    psf = nrc.calc_psf(oversample=pssf)
    
    psf_data = psf[2].data
    band_JWST = JWST(band=nrc.filter, psf_type="PIXEL")
    kwb = band_JWST.kwargs_single_band()
    kwb["kernel_point_source"]               = psf_data
    kwb["point_source_supersampling_factor"] = pssf
    pixel_num   =  int(to_dimless(2*mod_LP.radius)/kwb["pixel_scale"])
    SimObs_JWST = SimAPI(numpix=pixel_num, 
                    kwargs_single_band=kwb,
                    kwargs_model=mod_LP.kwargs_source_model)
    
    image_jwst_real = mod_LP.sim_image(SimObs_JWST)


    plt.close()
    fig, axis = plt.subplots(1,4,figsize=(19,7))
    kw_extents = get_extents(mod_LP.arcXkpc,mod_LP)
    extent_arcsec = kw_extents["extent_arcsec"]
    ax  = axis[0]
    im0 = ax.matshow(mod_LP.image_sim,origin='lower',extent=extent_arcsec,cmap="hot")
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.set_title(r"Original sim. Image")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    ax  = axis[1]
    im0 = ax.matshow(image_hst,origin='lower',extent=extent_arcsec,cmap="hot")
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.set_title(r"HST sim. Image ")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    ax  = axis[2]
    im0 = ax.matshow(image_jwst,origin='lower',extent=extent_arcsec,cmap="hot")
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.set_title(r"JWST sim. Image ")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    ax  = axis[3]
    im0 = ax.matshow(image_jwst_real,origin='lower',extent=extent_arcsec,cmap="hot")
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.set_title(r"JWST sim. Image (real PSF) ")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    plt.suptitle(mod_LP.Gal.Name)
    nm = "tmp/comp_SimIm.png"
    plt.tight_layout()
    plt.savefig(nm)
    plt.close()
    print(f"Saving {nm}")
    profiler.stop()
