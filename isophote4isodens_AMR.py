# copy isophote4isodens_alpha.py 12/12/25
# adapted for AMR and small debugs
import dill
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lenstronomy.Data.imaging_data import ImageData
import lenstronomy.Util.simulation_util as sim_util
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model

from python_tools.get_res import load_whatever
from python_tools.tools import ensure_unit,to_dimless

from Gen_PM_PLL_AMR import LoadLens #LensPart,kwlens_part_AS,cutoff_radius,z_source_max,pixel_num
    
def linlaw(x, a, b) :
    return a + x * b

def get_radius2radecgrid(rad,pixel_num):
    deltaPix    = to_dimless(2*rad/pixel_num)
    kwargs_data = sim_util.data_configure_simple(pixel_num, deltaPix)
    dataclass   = ImageData(**kwargs_data)
    __radec     = dataclass.coordinate_grid(pixel_num,pixel_num)
    _radec      = __radec[0].flatten(),__radec[1].flatten()
    return _radec

def get_kwisodens(Lens,cutoff_rad=None,verbose=True):
    if cutoff_rad is None:
        cutoff_rad = get_iso_cutoff(Lens)
    cutoff_rad = ensure_unit(cutoff_rad,u.kpc)
    cutoff_rad = to_dimless(cutoff_rad)
    # kappa map is as of now, the same resolution as the alpha map -> could be recomputed taking _kappa_map(x,y)
    if cutoff_rad<=to_dimless(Lens.radius):
        # if it's smaller, we don't care and we take the whole image 
        # (with original resolution)
        kappa  = Lens.kappa_map()
    else:
        # if it's larger, we expand the grid to it (giving up resolution in the way)
        _radec = get_radius2radecgrid(cutoff_rad,Lens.pixel_num)
        kappa  = Lens._kappa_map(_radec=_radec)
    
    # x0, y0, sma(semimajor), eps(ellipticity=1-b/a), pa
    geom = EllipseGeometry(kappa.shape[0]/2., kappa.shape[1]/2., 10., 0.5, 0./180.*np.pi)
    geom.find_center(kappa)
    ellipse = Ellipse(kappa, geometry=geom)
    isolist = ellipse.fit_image()

    model_kappa = build_ellipse_model(kappa.shape, isolist)
    return {"isolist":isolist,"geom":geom,"kappa":kappa,"model_kappa":model_kappa,"cutoff_rad":cutoff_rad}



def fit_isodens(Lens,cutoff_rad=None,pixel_num=None,verbose=True,save=True,reload=True):
    
    res_path = f"{Lens.savedir}/kw_res_isodens.dll"
    if reload:
        try:
            kw_res = load_whatever(res_path)
            print("Previous isodensity fit results found: "+res_path)
            return kw_res
        except FileNotFoundError:
            print("Previous results not found, fitting isodensity")
            reload=False
            pass
    if cutoff_rad is None:
        cutoff_rad = get_iso_cutoff(Lens)
    if pixel_num is None:
       pixel_num  = Lens.pixel_num # here in case I need to change it
    cutoff_rad = ensure_unit(cutoff_rad,u.kpc)
    cutoff_rad = to_dimless(cutoff_rad)
    
    kw_isodens = get_kwisodens(Lens,cutoff_rad=cutoff_rad,verbose=verbose)
    isolist    = kw_isodens["isolist"]
    kpcPix     = cutoff_rad/pixel_num
    sma_kpc    = isolist.sma*kpcPix # semi-major axis in kcp

    # discard first point
    popt_log,pcov_log = curve_fit(linlaw,np.log10(sma_kpc[1:]),np.log10(isolist.intens[1:]))
    ydatafit          = linlaw(np.log10(sma_kpc[1:]), *popt_log)
    kw_loglogfit      = {"popt_log":popt_log,"pcov_log":pcov_log,"fity":ydatafit,"fitx":np.log10(sma_kpc[1:])}
    kw_res            = {"isodens":kw_isodens,"loglogfit":kw_loglogfit,"cutoff_rad":cutoff_rad}
    if save:
        print("Saving isodensity fit: "+res_path)
        with open(res_path,"wb") as f:
            dill.dump(kw_res,f)
    return kw_res

def plot_isodens(Lens,savedir=None,cutoff_rad=None,pixel_num=None,verbose=True,kw_res=None):
    if savedir is None:
        savedir = Lens.savedir
    if pixel_num is None:
        pixel_num  = Lens.pixel_num # here in case I need to change it
    if cutoff_rad is None:
        cutoff_rad = get_iso_cutoff(Lens)
    if kw_res is None:
        kw_res = fit_isodens(Lens=Lens,cutoff_rad=cutoff_rad,pixel_num=pixel_num,verbose=verbose)
    cutoff_rad = kw_res["cutoff_rad"]

    # assuming x,y centred around 0
    xmin = -cutoff_rad
    ymin = -cutoff_rad
    xmax = +cutoff_rad
    ymax = +cutoff_rad
    extent = [xmin,xmax,ymin,ymax] 
    kappa          = kw_res["isodens"]["kappa"]
    model_kappa    = kw_res["isodens"]["model_kappa"]
    residual_kappa = kappa - model_kappa 
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    im_i = ax1.imshow(np.log10(kappa),cmap=plt.cm.inferno,origin="lower",extent=extent)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_i, cax=cax, orientation='vertical',label=r"$\kappa_{sim}$")
    ax1.set_title(r"log$_{10}$($\kappa$)")
    
    im_i = ax2.imshow(np.log10(model_kappa),cmap=plt.cm.inferno,origin="lower",extent=extent)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_i, cax=cax, orientation='vertical',label=r"$\kappa_{iso}$")
    ax2.set_title(r"log$_{10}$($\kappa_{Model}$)")

    vm = np.median(residual_kappa) +2*np.std(residual_kappa)
    #print("testing vm residual:",vm)
    im_i = ax3.imshow(residual_kappa,cmap="bwr",extent=extent,origin="lower",vmin=-vm,vmax=vm)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_i, cax=cax, orientation='vertical',label=r"$\kappa_{sim}$-$\kappa_{iso}$")

    ax3.set_title("Residual")
    for ax in ax1,ax2,ax3:
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel("kpc")
        ax.set_ylabel("kpc")
    
    
    # overplot a few isophotes on the residual map
    isolist = kw_res["isodens"]["isolist"]
    iso1 = isolist.get_closest(10.)
    iso2 = isolist.get_closest(40.)
    iso3 = isolist.get_closest(100.)
    
    x, y, = iso1.sampled_coordinates()
    Nx,Ny = kappa.shape
    x_plot = xmin + (x / Nx) * (xmax - xmin)
    y_plot = ymin + (y / Ny) * (ymax - ymin)
    ax3.plot(x_plot, y_plot, color='black')
    x, y, = iso2.sampled_coordinates()
    x_plot = xmin + (x / Nx) * (xmax - xmin)
    y_plot = ymin + (y / Ny) * (ymax - ymin)
    ax3.plot(x_plot, y_plot, color='black')
    x, y, = iso3.sampled_coordinates()
    x_plot = xmin + (x / Nx) * (xmax - xmin)
    y_plot = ymin + (y / Ny) * (ymax - ymin)
    ax3.plot(x_plot, y_plot, color='black')
    name_plot = savedir+"/isodens_model.pdf"
    print("Saving "+name_plot)
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    kpcPix  = cutoff_rad/pixel_num
    sma_kpc = isolist.sma*kpcPix # semi-major axis in kcp
    
    geom    = kw_res["isodens"]["geom"]

    plt.figure(figsize=(10, 5))
    plt.figure(1)
    
    plt.subplot(221)
    plt.errorbar(sma_kpc, 1-isolist.eps, yerr=isolist.ellip_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('Axis Ratio')
    
    plt.subplot(222)
    plt.errorbar(sma_kpc, isolist.pa/np.pi*180., yerr=isolist.pa_err/np.pi* 80., fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('PA (deg)')
    plt.subplot(223)
    plt.errorbar(sma_kpc, (isolist.x0-geom.x0)*kpcPix, yerr=isolist.x0_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('X0-Xcnt [kpc]')
    
    plt.subplot(224)
    plt.errorbar(sma_kpc, (isolist.y0-geom.y0)*kpcPix, yerr=isolist.y0_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('Y0-Ycnt [kpc]')
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
    name_plot = savedir+"/isodens_prms1.pdf"
    print("Saving "+name_plot)
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.figure(1)
    #limits = [0., 100., -0.1, 0.1]
    
    plt.subplot(221)
    #plt.axis(limits)
    plt.errorbar(sma_kpc, isolist.a3, yerr=isolist.a3_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('A3')
    
    plt.subplot(222)
    #plt.axis(limits)
    plt.errorbar(sma_kpc, isolist.b3, yerr=isolist.b3_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('B3')
    
    plt.subplot(223)
    #plt.axis(limits)
    plt.errorbar(sma_kpc, isolist.a4, yerr=isolist.a4_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('A4')
    
    plt.subplot(224)
    #plt.axis(limits)
    plt.errorbar(sma_kpc, isolist.b4, fmt='o', yerr=isolist.b4_err, markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('B4')
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
    
    
    name_plot = savedir+"/isodens_prms2.pdf"
    print("Saving "+name_plot)
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    fig,axis = plt.subplots(2,figsize=(13,8))
    ax = axis[0]
    ax.set_title(r"Plot of $\kappa$")
    
    ax.scatter(sma_kpc,isolist.intens,c="k")
    #ax.legend()
    ax.set_xlabel(r'Semimajor axis [kpc])')
    ax.set_ylabel(r'$\kappa$')
    #name_plot = savedir+"/isodens_kappa.pdf"
    #print("Saving "+name_plot)
    #plt.tight_layout()
    #plt.savefig(name_plot)
    #plt.close()

    # fit as linear in log
    popt_log,pcov_log = kw_res["loglogfit"]["popt_log"],kw_res["loglogfit"]["pcov_log"]
    ydatafit          = kw_res["loglogfit"]["fity"]
    ax = axis[1]
    ax.set_title(r"LogLog plot of $\kappa$")
    str_fit = "log10(kappa) ="+str(np.round(popt_log[0],2))+"log10(sma)^"+str(np.round(popt_log[1],2))
    ax.plot(np.log10(sma_kpc[1:]),ydatafit, c="b",ls="--",label="Fit:"+str_fit)
    
    ax.scatter(np.log10(sma_kpc),np.log10(isolist.intens),c="k")
    ax.legend()
    ax.set_xlabel(r'log$_{10}$(Semimajor axis [kpc])')
    ax.set_ylabel(r'log$_{10}$($\kappa$)')
    name_plot = savedir+"/isodens_kappa.pdf"
    print("Saving "+name_plot)
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    # Plot gamma(r)
    
    #kw_loglogfit = kw_res["loglogfit"]
    #fity    = kw_loglogfit["fity"]
    logr    = kw_res["loglogfit"]["fitx"]
    #gamma_fit_r = -fity/logr
    #kw_isodens = kw_res["isodens"]
    #isolist    = kw_isodens["isolist"]
    y             = np.log10(isolist.intens[1:])
    #gamma_r    = -y/logr
    gamma_der     = -np.gradient(y,logr)
    gamma_fit_fix = -kw_res["loglogfit"]["popt_log"][1]
    #gamma_der_fit  = -np.gradient(fity,np.median(np.diff(logr))) 1:1 with gamma const
    #print("this doesn't make sense because gamma is constant as given by the fit")
    #plt.plot(logr,gamma_fit_r,c="k",label=r"fit $\gamma$=-\frac{\gamma_{opt.}}{log(r)}$") #ill behave at log(r)=0 by construction
    #plt.plot(logr,gamma_r,c="r",label=r" $\gamma$=-\frac{\gamma log(r)}{log(r)}$") 
    plt.plot(logr,gamma_der,c="g",label=r"$\gamma=-\frac{\mathrm{d isodensity}}{\mathrm{d log(r)}}$") 
    plt.plot(logr,gamma_fit_fix*np.ones_like(logr),c="b",ls="--",label=r"$\gamma_{opt.}$="+str(np.round(gamma_fit_fix,2))) 
    plt.title(r"$\gamma$(r)")
    plt.xlabel(r"log$_{10}$r")
    plt.ylabel(r"$\gamma$(r)")
    plt.legend()
    namefig = f"{savedir}/gamma_r.pdf"
    print("Saving "+namefig)
    plt.savefig(namefig)
    plt.close()
    return kw_res

def get_iso_cutoff(Lens,scale_cutoff=3):
    cutoff_rad = Lens.thetaE*scale_cutoff/Lens.arcXkpc
    print("Cutting plot at "+str(np.round(cutoff_rad,3))+", "+str(scale_cutoff)+" times the approx. theta_E")
    return cutoff_rad

if __name__=="__main__":
    # for now applied to a "known" lens galaxy
    #Lens = LoadLens("sim_lens/RefL0025N0752/snap23_G3.0//lensing/G3SGn0_Npix200_PartAS.pkl")
    #Lens = LoadLens("sim_lens/RefL0025N0752/snap18_G5.0//lensing/G5SGn0_Npix200_PartAS.pkl")
    Lens = LoadLens("sim_lens/RefL0025N0752/snap19_G6.0/test_sim_lens_AMR/G6SGn0_Npix200_PartAS.pkl")
    #name_proj = Lens.savedir+"/proj_mass.pdf"
    savedir = "tmp/"
    scale_cutoff = 3
    cutoff_rad = get_iso_cutoff(Lens,scale_cutoff)
    kw_res = plot_isodens(Lens,savedir,cutoff_rad=cutoff_rad)
