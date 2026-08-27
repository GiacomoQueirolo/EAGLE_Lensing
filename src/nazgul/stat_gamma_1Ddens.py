# Study statistically core and power law index from the 1D mass distribution
# to add: compare with lens model results
import sys
import warnings
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import import_module
from scipy.optimize import curve_fit
from chainconsumer import Chain #, ChainConsumer
from chainconsumer.plotting import plot_contour,plot_dist #,plot_truths

from python_tools.get_res import load_whatever
from nazgul.mount_doom.lens_system import LensSystem
from plot_AMRxpart import get_kw_1D_density,get_savedir_plots
from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.combined_modelling_results import get_full_chain
# we select lenses the same way we select them for modelling:
from nazgul.Modelling.lib_models import get_lenses2model,get_model_res_dir
# thus we need to define which model to consider
# for now simNoShear:
std_lens_model = "simNoShear"
#from nazgul.Modelling.model_simNoShear import res_dir_base
#warnings.warn(f"Using lens cat. obtained from {res_dir_base}")
# note: this is for convenience, but we could implement it exnovo, indep. of modelling (at least for the study of the mass profile - if we compare to the lens model it's different)

def rho_cored(r,rho0,r_core,gamma):
    rho = rho0/(r**2+  r_core**2)**(gamma)
    return rho

def log_rho_cored(log_r,log_rho0,log_r_core,gamma):
    r      = 10**log_r
    r_core = 10**log_r_core
    rho0   = 10**log_rho0
    rho    = rho_cored(r,rho0,r_core,gamma)
    return np.log10(rho)

# to consider if not using the fit log_rho_cored

# should have a better setup for coloring and plots
matplotlib.use('Agg') 
warm         = ['#fdcc8a', '#fc8d59', '#d7301f']

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description=" Study statistically core and power law index from the 1D mass distribution")
    parser.add_argument('-nl','--n_lenses',type=int,dest="n_lenses",default=5,help=f"Number of lenses to model")
    parser.add_argument('-mtE','--min_thetaE',type=float,dest="min_thetaE",default=None,help=f"Min theta_E for the gal to be considered a lens")
    parser.add_argument('-snap','--snap',nargs="+",type=str,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")
    parser.add_argument('-lm','--lens_model',type=str,dest="lens_model",default=std_lens_model,help=f"Lens model to compare the power law to (and get the lens list from)")
    parser.add_argument('-nsp','--no_single_plot',dest="single_plot",
                        default=True,action="store_false",help=f"Do not plot each 1D density fit")

    args        = parser.parse_args()
    n_lenses    = args.n_lenses
    min_thetaE  = args.min_thetaE 
    snaps       = args.snaps #[25,26,27]
    sim         = args.sim
    subsim      = args.subsim
    simsuite    = args.simsuite
    lens_model  = args.lens_model
    single_plot = args.single_plot
    
    # picked by hand "bad" lenses ->
    lenses2skip = ["LS_Lens_Gn75SGn0_Prj1","LS_Lens_Gn4SGn0_Prj2","LS_Lens_Gn4SGn0_Prj0","LS_Lens_Gn14SGn0_Prj1",
                   "LS_Lens_Gn71SGn0_Prj2","LS_Lens_Gn7SGn1_Prj2","LS_Lens_Gn15SGn1_Prj0","LS_Lens_Gn15SGn1_Prj0",
                   "LS_Lens_Gn6SGn0_Prj2","LS_Lens_Gn1SGn2_Prj1","LS_Lens_Gn42SGn0_Prj1","LS_Lens_Gn18SGn0_Prj2",
                   "LS_Lens_Gn18SGn0_Prj0","LS_Lens_Gn18SGn0_Prj0","LS_Lens_Gn18SGn0_Prj2","LS_Lens_Gn22SGn1_Prj2",
                   "LS_Lens_Gn22SGn1_Prj1","LS_Lens_Gn66SGn0_Prj1","LS_Lens_Gn45SGn0_Prj0","LS_Lens_Gn33SGn0_Prj2"]

    warnings.warn(f"Using lens model: {lens_model}")
    module_path = f"nazgul.Modelling.model_{lens_model}"
    model_module = import_module(module_path)
    res_dir = getattr(model_module,"res_dir_base")
    
    kw_get_all_gallens = {"sim":sim,
                          "subsim":subsim,
                          "simsuite":simsuite,
                          "snaps":snaps}

    gal_lenses  = get_lenses2model(res_dir=res_dir,
                                   reload=True,
                                   n_lenses=np.nan, # has to load all of them anyway
                                   min_thetaE=min_thetaE,
                                   skip_lenses=lenses2skip,
                                   kw_get_all_gallens=kw_get_all_gallens)
    log_r_scaled,log_Sig_scaled = [],[]
    gammas,cores = [],[]
    gamma_lens   = []
    fig_overlap,ax_overlap = plt.subplots()
    nm_fig_overap = f"{res_dir}/1D_overlap.png"
    r_lbl = r"log$_{10}$r [kpc]"
    sig_lbl = r"log$_{10}\Sigma$ [10$^9$ M$_{\odot}$/kpc]"
    ax_overlap.set_xlabel(r_lbl)
    ax_overlap.set_ylabel(sig_lbl)
    ax_overlap.set_title("Overlap of scaled 1D density profiles")
    
    for i,gal_lens in enumerate(gal_lenses): 
        print("\nLoading gal lens "+gal_lens.name)
        
        gal_lens.unpack()
        kw_1d = get_kw_1D_density(gal_lens.Gal,gal_lens.proj_index)
                
        r = kw_1d["r_all"].value #kpc
        Sigma = kw_1d["Sigma_encl_all"].value
        Sigma_scaled = Sigma/1e9 # 1e9 SolMass/kpc
                
        fit_prm,fit_cov = curve_fit(rho_cored,r,Sigma_scaled)
        rho0    = fit_prm[0]
        core    = fit_prm[1]
        gamma   = fit_prm[2]
        print("Results from 1D density Cored Power Law fit")
        print("Core :",np.round(core,2))
        print("Gamma :",np.round(gamma,2))
        log_r,log_Sig = np.log10(r),np.log10(Sigma_scaled)
        savedir = get_savedir_plots(gal_lens.Gal)
        if single_plot:
            nm_fig = f"{savedir}/1D_Density_fit.png"
            # if plot doesn't exits (to implement)        
            plt.plot(log_r,log_Sig,c="b",label="data")
            plt.plot(log_r,np.log10(rho_cored(r,*fit_prm)),c="g",
                     ls="--",label="fit cored PL")
            plt.xlabel(r_lbl)
            plt.ylabel(sig_lbl)
            plt.axvline(np.log10(core),ls="-.",c="g",label="core")
            plt.legend()
            plt.savefig(nm_fig)
            print(f"Saving {nm_fig}")
            plt.close()
        # plot overlap of 1D density profiles (normalised by core and max dens) 
        _log_r_scaled   = log_r/np.log10(core)
        _log_Sig_scaled = log_Sig/np.log10(rho0)
        _lbl = None
        if i==0:
            _lbl = "N = "+str(len(gal_lenses))
        ax_overlap.plot(_log_r_scaled,_log_Sig_scaled,
                        alpha=.5,c="grey",label=_lbl)

        # get gamma from lens modelling and compare

        # the following is just used for convenience
        lens = LensSystem.from_GalLens(gal_lens)
        gal_lens.model_res_dir = get_model_res_dir(lens,res_dir=res_dir)
        try:
            full_chain = get_full_chain(gal_lens,model=lens_model)
            gamma_lns = full_chain["gamma_lens0"].to_numpy() 
            gamma_med,gamma_std = np.median(gamma_lns),np.std(gamma_lns)
            gamma_lens.append([gamma_med,gamma_std])
            del full_chain,gamma_lns
        except FileNotFoundError as e:
            
            warnings.warn(f"{e} \nWe ignore it and set it to NaN")
            gamma_lens.append([np.nan,np.nan])
        # store res
        cores.append(core)
        gammas.append(gamma)
        log_r_scaled.append(_log_r_scaled)
        log_Sig_scaled.append(_log_Sig_scaled)
        
        del gal_lens,lens
        
    fig_overlap.savefig(nm_fig_overap)
    print(f"Saving {nm_fig_overap}")
    plt.close(fig_overlap)
    
    # plot summ. stat of core and gamma
    chain4plotcont = Chain(samples=np.array([cores,gammas]), 
                           name="", 
                           shade=True, 
                           color=warm[1], 
                           shade_gradient = 0.8, linewidth=3.0)
    fig,ax = plt.subplots()
    plot_contour(ax, chain4plotcont, px="core", py=r"$\gamma$")
    nm_fig = f"{res_dir}/distr_coreVsGamma_1Ddens.png"
    plt.savefig(nm_fig)
    print(f"Saving {nm_fig}")
    plt.close(fig)
    
    # compare gamma vs core
    ind_sort = np.argsort(gammas)
    plt.scatter(gammas[ind_sort],cores[ind_sort],c="k")
    plt.title(r"1D dens. cored PL fit: $\gamma$ vs r$_{\rm{core}}$")
    plt.ylabel(r"log$_{10}$r$_{\rm{core}}$ [kpc]")
    plt.xlabel(r"$\gamma$")
    nm_fig = f"{res_dir}/dens_1dfit_gammaVs_core.png"
    plt.savefig()
    print(f"Saving {nm_fig}")
    
    # compare gammas
    plt.errorbar(gammas[ind_sort],gamma_lens[0][ind_sort],
                 yerr=gamma_lens[1][ind_sort])
    plt.scatter(gammas[ind_sort],gamma_lens[0][ind_sort],c="k")
    plt.title(r"$\gamma_{\rm{Density}}$ vs $\gamma_{\rm{PEMD}}$")
    plt.xlabel(r"$\gamma_{\rm{Density}}$")
    plt.ylabel(r"$\gamma_{\rm{Lens Model}}$")
    nm_fig = f"{res_dir}/gamma_dens_Vs_gamma_lens.png"
    plt.savefig()
    print(f"Saving {nm_fig}")
    