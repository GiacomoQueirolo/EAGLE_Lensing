# to run to test Gen_PM_PLL_AMR
from Gen_PM_PLL_AMR import wrapper_get_rnd_lens,get_extents,LoadLens
from plot_PL import plot_all
from pyinstrument import Profiler

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.ObservationConfig.JWST import JWST
if __name__ == "__main__":

    #print("Loading specific gal for debugging")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_18//Gn5SGn0.pkl")
    profiler = Profiler()
    profiler.start()
    
    mod_LP = wrapper_get_rnd_lens(reload=False)
    """
    mod_LP = LensPart(Galaxy=Gal,kwlens_part=kwlens_part_AS,
                       z_source_max=z_source_max, 
                       pixel_num=pixel_num,reload=False,savedir_sim="test_sim_lens_AMR")
    
    mod_LP = LoadLens("/pbs/home/g/gqueirolo/EAGLE/sim_lens//RefL0025N0752/snap22_G19.0//test_sim_lens_AMR/G19SGn0_Npix200_PartAS.pkl")
    """
    #sim_lens/RefL0025N0752/snap16_G8.0//test_sim_lens_AMR/G8SGn0_Npix200_PartAS.pkl")
    mod_LP.run()
    # mod_LP.psi_map -> works but doesn't store it correctly
    profiler.stop()
    print(profiler.output_text(color=True,show_all=False))
    plot_all(mod_LP,skip_caustic=True)



    profiler.start()
    band_HST = HST(band='WFC3_F160W', psf_type="GAUSSIAN")
    #band = HST(band='WFC3_F160W', psf_type="PIXEL") #-> if pixel, we need to give kernel_point_source (and point_source_supersampling_factor etc)
    #band.obs["psf_type"] = "PIXEL"
    #del band.obs["seeing"]
    #band.obs["kernel_point_source"] = []
    image_hst = mod_LP.sim(band_HST)
    band_JWST = JWST(band='F444W', psf_type="GAUSSIAN")
    image_jwst = mod_LP.sim(band_JWST)
    

    plt.close()
    fig, axis = plt.subplots(1,3,figsize=(15,7))
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

    plt.suptitle(mod_LP.Gal.Name)
    nm = "tmp/comp_SimIm.png"
    plt.tight_layout()
    plt.savefig(nm)
    plt.close()
    print("Saving "+nm)

    profiler.stop()
