# I want to compute and plot the isondensity params for the lenses we have so far:

import glob
import numpy as np
import matplotlib.pyplot as plt

from fnct import std_sim
from remade_gal import sim_lens_path
from Gen_PM_PLL_AMR import LoadLens
from isophote4isodens_AMR import fit_isodens,plot_isodens

if __name__=="__main__":
    # this directory structure has to be rechecked
    savedir_sim="test_sim_lens_AMR"
    lenses = glob.glob(f"{sim_lens_path}/{std_sim}/snap*/{savedir_sim}/*.pkl")
    gamma_distr = [] 
    ellipt_distr = []
    PA_distr = []
    bxdk_distr = [] # boxydiskyness, b4
    fig,axis = plt.subplots(4,2,figsize=(10,17))
    for i,lens_pth in enumerate(lenses):
        
        lens    = LoadLens(lens_pth)
        if lens is False:
            continue
        try:
            kw_res  = fit_isodens(lens)
            #plot_isodens(lens,kw_res=kw_res)
        except Exception as e:
            print("Lens "+lens_pth+" failed:\n",e)
            print("skipping")
        logr    = kw_res["loglogfit"]["fitx"]
        isolist = kw_res["isodens"]["isolist"]
    
        y             = np.log10(isolist.intens[1:])
        gamma_der     = -np.gradient(y,logr)
        gamma_fit_fix = -kw_res["loglogfit"]["popt_log"][1]
        gamma_distr.append(gamma_fit_fix)

        eps    = isolist.eps[1:]
        pa     = isolist.pa[1:]/np.pi*180. # deg
        bxdk   = isolist.b4[1:] # boxy-diskyness
        
        ellipt_distr.append(np.median(eps))
        PA_distr.append(np.median(pa))
        bxdk_distr.append(np.median(bxdk))
        
        ax = axis[0][0]
        ax.plot(logr,gamma_der,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'log$_{10}$(Semimajor axis [kpc])')
            ax.set_ylabel(r"$\gamma$ []")
            ax.set_title(r"Fit $\gamma$(r)")
        ax = axis[1][0]
        ax.plot(logr,eps,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'log$_{10}$(Semimajor axis [kpc])')
            ax.set_ylabel(r"$\epsilon$ []")
            ax.set_title(r"Ellipticity (r)")
        ax = axis[2][0]
        ax.plot(logr,pa,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'log$_{10}$(Semimajor axis [kpc])')
            ax.set_ylabel(r"P.A. [$^o$]")
            ax.set_title(r"Pointing Angle (r)")
        ax = axis[3][0]
        ax.plot(logr,bxdk,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'log$_{10}$(Semimajor axis [kpc])')
            ax.set_ylabel(r"b4 []")
            ax.set_title(r"Boxy-diskiness (r)")
             
    ax = axis[0][1]
    ax.hist(gamma_distr)
    med_gamma = np.median(gamma_distr)
    ax.axvline(med_gamma,ls="--",c="r",label=r"median($\gamma$)="+str(np.round(med_gamma,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"$\gamma$(r)")
    ax.set_title(r"Distr. fit $\gamma$")
    ax.legend()

    ax = axis[1][1]
    ax.hist(ellipt_distr)
    med_ell = np.median(ellipt_distr)
    ax.axvline(med_ell,ls="--",c="r",label=r"median($\epsilon$)="+str(np.round(med_ell,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"$\epsilon$(r)")
    ax.set_title(r"Distr. ellipticity")
    ax.legend()

    ax = axis[2][1]
    ax.hist(PA_distr)
    med_pa = np.median(PA_distr)
    ax.axvline(med_pa,ls="--",c="r",label=r"median(P.A.)="+str(np.round(med_pa,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"P.A.(r)")
    ax.set_title(r"Distr. Pointing Angle")

    
    ax = axis[3][1]
    ax.hist(bxdk_distr)
    med_b4 = np.median(bxdk_distr)
    ax.axvline(med_b4,ls="--",c="r",label=r"median(b4)="+str(np.round(med_b4,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"b4(r)")
    ax.set_title(r"Distr. Boxy-Diskyness parameter")
    
    ax.legend()
    plt.tight_layout()
    nm = "tmp/distr_isoparams.png"
    print("Saving "+nm)
    plt.savefig(nm)
    