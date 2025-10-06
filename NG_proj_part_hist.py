# copied from proj_part_hist but now using the "NewGalaxy" class

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

# from lenstronomy.Util import util
from remade_gal import get_rnd_NG
from remade_gal import get_z_source,get_dP #,get_radius
#from get_gal_indexes import get_rnd_gal

from python_tools.tools import mkdir,get_dir_basename
from python_tools.get_res import load_whatever


z_source_max = 5
pixel_num    = 200j
verbose      = True
plot_dnsmap  = True

dir_name     = "proj_part_hist"

def get_dens_map_rotate_hist(Gal,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,plot=plot_dnsmap):
    # try all projection in order to obtain a lens
    proj_index = 0
    """
    #DEBUG
    res = None
    res = get_dens_map_hist(Gal=Gal,proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose,plot=plot)
    raise RuntimeError("DEBUG--Arrived here")
    """
    while proj_index<3:
        try:
            res = get_dens_map_hist(Gal=Gal,proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose)
            break
        except AttributeError as Ae:
            print("Error : ")
            print(Ae)
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if res is None:
        raise RuntimeError("There is no projection of the galaxy that create a lens given the z_source_max")
    else:
        return res

def get_dens_map_hist(Gal,proj_index=0,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,save_res=True,plot=True,DEBUG=False):
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)

    # given a projection, produce the density map
    # fails if it can't produce a supercritical lens w. z_source<z_source_max
    
    Xstar,Ystar,Zstar = Gal.stars["coords"].T # in Mpc
    Xgas,Ygas,Zgas    = Gal.gas["coords"].T   # in Mpc
    Xdm,Ydm,Zdm       = Gal.dm["coords"].T    # in Mpc
    Xbh,Ybh,Zbh       = Gal.bh["coords"].T    # in Mpc
    
    Mstar = Gal.stars["mass"] # in Msun 
    Mgas  = Gal.gas["mass"]  # in Msun 
    Mdm   = Gal.dm["mass"] # in Msun 
    Mbh   = Gal.bh["mass"] # in Msun 
    
    # center around the center of the galaxy
    # center of mass is given in Comiving coord 
    # see https://arxiv.org/pdf/1510.01320 D.23 
    # ->  it's given in cMpc (not cMpc/h) fsr
    Cx,Cy,Cz = Gal.centre*u.Mpc/(Gal.xy_propr2comov) # (now) Mpc
    if DEBUG:
        print("DEBUG")
        # mass factor for particles
        fact_M = 5e4
        fig, ax = plt.subplots(3)
        for XX,YY,MM,name in zip([Xstar,Xgas,Xdm,Xbh],[Ystar,Ygas,Ydm,Ybh],[Mstar,Mgas,Mdm,Mbh],["star","gas","dm","bh"]):
            xx,yy = XX*u.Mpc-Cx,YY*u.Mpc-Cy
            mm    = MM/fact_M
            ax[0].hist(xx,bins=nx,alpha=.5,label=name)
            ax[1].hist(yy,bins=ny,alpha=.5,label=name)
            ax[2].hist(mm,alpha=.5,label=name)
        ax[0].axvline(0,ls="--",label="centre")
        ax[1].axvline(0,ls="--",label="centre")
        ax[0].set_xlabel("X [kpc]")
        ax[1].set_xlabel("Y [kpc]")
        ax[2].set_xlabel(f"M [{str(fact_M)} SolMass]")
        ax[2].legend()
        namefig = f"./tmp/NG_hist1D_{proj_index}_part.png"
        plt.tight_layout()
        plt.savefig(namefig)
        plt.close()
        print("Saved "+namefig)
    
    # Concatenate particle properties
    # already in proper/physical units (corrected for h as well)
    x = np.concatenate([Xdm, Xstar, Xgas, Xbh])*u.Mpc#/Gal.h # now in Mpc
    y = np.concatenate([Ydm, Ystar, Ygas, Ybh])*u.Mpc#/Gal.h # now in Mpc
    z = np.concatenate([Zdm, Zstar, Zgas, Zbh])*u.Mpc#/Gal.h # now in Mpc
    m = np.concatenate([Mdm, Mstar, Mgas, Mbh])*u.Msun #/Gal.h

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
    if DEBUG:
        max_diam = np.max([np.max(x.value) - np.min(x.value),np.max(y.value) - np.min(y.value),np.max(z.value) - np.min(z.value)])*u.Mpc
        print("DEBUG","max_diam",np.round(max_diam,2))
        print(Cx,Cy,Cz)
        print(np.mean(x),np.mean(y),np.mean(z))
        print(np.sum(x*m)/np.sum(m),np.sum(y*m)/np.sum(m),np.sum(y*m)/np.sum(m))
    x -= Cx
    y -= Cy
    z -= Cz
    if DEBUG:
        print("DEBUG np.mean(x),Cx",np.mean(x),Cx)
        print("DEBUG np.mean(y),Cy",np.mean(y),Cy)
        print("DEBUG np.median(y),Cy",np.median(y))
        print("DEBUG np.std(y)",np.std(y))
        print("DEBUG np.mean(z),Cz",np.mean(z),Cz)
    # projection along given indexes
    # xy : ind=0
    # xz : ind=1
    # yz : ind=2
    if proj_index==0:
        _=True # all as usual
    elif proj_index==1:
        y  = copy(z)
    elif proj_index==2:
        x  = copy(y)
        y  = copy(z)
    x  = np.asarray(x.to("kpc").value) #kpc
    y  = np.asarray(y.to("kpc").value) #kpc
    m  = np.asarray(m.to("solMass").value, dtype=float)  # M_sol
    #radius = get_radius(x,y)               #kpc
    radius = 70 #kpc 
    print("NOTE: taking a small radius -",radius,"kpc")
    # Redshift: 
    z_lens = Gal.z
    if verbose:
        print("z_lens",z_lens)
    cosmo   = FlatLambdaCDM(H0=Gal.h*100, Om0=1-Gal.h)
    if DEBUG:
        print("DEBUG")
        print("cosmo",cosmo)
        print("H0",Gal.h*100, "Om0",1-Gal.h)
    
    if DEBUG:
        print("<X> [kpc]",np.round(np.mean(x),3))
        print("<Y> [kpc]",np.round(np.mean(y),3))
        print("radius [kpc]",np.round(radius,3))
        print("tot mass [1e8 M_sol]",np.round(np.sum(m)/1e8,3))
        
    # X,Y already recentered around 0
    xmin = -radius
    ymin = -radius
    xmax = +radius
    ymax = +radius

    if DEBUG:    
        print("DEBUG")
        fig, ax = plt.subplots(3)
        ax[0].hist(x,bins=nx,range=[xmin, xmax])
        ax[0].set_xlabel("X [kpc]")
        #ax[0].set_xlim([xmin,xmax])
        ax[1].hist(y,bins=ny,range=[ymin, ymax])
        ax[1].set_xlabel("Y [kpc]")
        #ax[1].set_xlim([ymin,ymax])
        ax[2].hist(m/1e8,bins=nx)
        ax[2].set_xlabel("M [1e8 SolMass]")
        #namefig = f"{Gal.proj_dir}/hist1D_{proj_index}.png"
        namefig = f"./tmp/NG_hist1D_{proj_index}.png"
        plt.tight_layout()
        plt.savefig(namefig)
        plt.close()
        print("Saved "+namefig)
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
    dx = (xmax - xmin) / nx #kpc
    dy = (ymax - ymin) / ny #kpc
    # density_ij = M_ij/(Area_bin_ij)
    density = mass_grid / (dx * dy)

    if plot:
        extent = [xmin,xmax,ymin,ymax]
        plt.imshow(np.log10(density),extent=extent, cmap=plt.cm.gist_earth_r,norm="log",label="Density [Msun/kpc^2]")
        plt.colorbar()
        #plt.scatter(x,y,c="w",marker=".")
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        #
        if DEBUG:
            namefig = f"tmp/NG_proj_hist_densmap_{proj_index}.png"
        else:
            namefig = f"{Gal.proj_dir}/hist_densmap_proj_{proj_index}.png"
        plt.savefig(namefig)
        plt.close()
        print("Saved "+namefig)
    # define the z_source:
    # dens now is already in Msun/kpc^2
    """
    dens_Ms_arcsec2 = dens/(dP**2)  # Msun /''^2 
    dens_Ms_kpc2    = dens_Ms_arcsec2*(arcXkpc**2) # Msun/kpc^2
    """ 
    dens_Ms_kpc2    = density*u.Msun/(u.kpc*u.kpc)
    if DEBUG:
        print("DEBUG ")
        print("M(density)",np.sum(mass_grid))
        print("M(m)",np.sum(m))
        print("M(gal)",Gal.M_tot)
        print("M(gal2)",Gal.M)
        print("dx,dy",dx,dy)
        print("bin area",dx*dy,"kpc^2")
        print("<density>",np.mean(dens_Ms_kpc2))
        print("max(density)",np.max(dens_Ms_kpc2))
    z_source = get_z_source(cosmo=cosmo,z_lens=z_lens,dens_Ms_kpc2=dens_Ms_kpc2,
                            z_source_max=z_source_max,verbose=verbose)
    if z_source==0:
        raise AttributeError("Rerun trying different projection")
        
    # dP to convert from kpc/pix to ''/pix
    dP = get_dP(radius*u.kpc,pixel_num,cosmo=cosmo,Gal=Gal) # ''/pix -> to double check that this is correct
    # store the results
    res = [dens_Ms_kpc2,radius,dP,[dx,dy],z_source,cosmo]
    if save_res:
        with open(Gal.dens_res,"wb") as f:
            pickle.dump(res,f)
        print("Saved "+Gal.dens_res)
    return res

from fnct import Sersic


def prep_Gal(Gal,dir_name=dir_name):
    # impractical but easy to set up
    Gal.proj_dir = Gal.gal_snap_dir+f"/{dir_name}_{Gal.Name}/"
    mkdir(Gal.proj_dir)
    Gal.dens_res = f"{Gal.proj_dir}/dens_res.pkl"
    return Gal

if __name__=="__main__":
    parser = ArgumentParser(description="Project particles into a mass sheet - histogram version")
    parser.add_argument("-dn","--dir_name",dest="dir_name",type=str, help="Directory name",default=dir_name)
    parser.add_argument("-pxn","--pixel_num",dest="pixel_num",type=int, help="Pixel number",default=pixel_num.imag)
    parser.add_argument("-zsm","--z_source_max",dest="z_source_max",type=float, help="Maximum source redshift",default=z_source_max)
    parser.add_argument("-nrr", "--not_rerun", dest="rerun", 
                        default=True,action="store_false",help="if True, rerun code")
    parser.add_argument("-pl", "--plot", dest="plot", 
                        default=False,action="store_true",help="Plot dens map")
    parser.add_argument("-v", "--verbose", dest="verbose", 
                        default=False,action="store_true",help="verbose")
    args          = parser.parse_args()
    pixel_num     = args.pixel_num*1j
    rerun         = args.rerun
    dir_name      = args.dir_name
    verbose       = args.verbose
    z_source_max  = args.z_source_max
    plot          = args.plot
    """
    if rerun:
        #print("DEBUG -- USING test sym")
        #Gal = get_rnd_gal(sim=test_sim,check_prev=False,reuse_previous=False,min_mass="1e13",max_z="1")
        Gal = get_rnd_NG()#sim=std_sim,check_prev=False,reuse_previous=False,min_mass="1e13",max_z="1")
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
    """    
    Gal = get_rnd_NG()
    z_lens = Gal.z
    """
    print("DEBUG")
    NG = Gal
    fig, ax = plt.subplots(3)
    nx = 100
    for name,part in zip(["stars","dm","gas"],[NG.stars,NG.dm,NG.gas]):
        coords = part["coords"]
        x,y,z  = coords.T
        print(np.std(coords,axis=0))
        ax[0].hist(x,bins=nx,alpha=.5,label=name)#,range=[xmin, xmax])
        ax[1].hist(y,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
        ax[2].hist(z,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
    ax[0].set_xlabel("X [kpc]")
    ax[2].set_xlabel("Z [kpc]")
    ax[1].set_xlabel("Y [kpc]")
    ax[2].legend()
    namefig = f"tmp/hist_by_hand_parts.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig) 
    print("DEBUG")
    """
    Gal = prep_Gal(Gal)
    if verbose:
        print("Assumptions: We are considering the maximum source redshift to be ",z_source_max)
        if int(pixel_num.imag)<500:
            print("Warning: running test")
        elif int(pixel_num.imag)>=1000:
            print("Warning: running very long")
    dens_Ms_kpc2,radius,dP,dxdy,z_source,cosmo = get_dens_map_rotate_hist(Gal=Gal,pixel_num=pixel_num,
                                                    z_source_max=z_source_max,verbose=True)#plot=plot,verbose=verbose)
    dx,dy = dxdy
    """
    try:
        if rerun:
            raise RuntimeError("Rerunning anyway")
        dens_Ms_kpc2,radius,dP,cosmo = load_whatever(Gal.dens_res)
        
        if len(dens_Ms_kpc2)!=int(pixel_num.imag):
            print("DEBUG")
            print(len(dens_Ms_kpc2),int(pixel_num.imag))
            print("Num pixel != of the wanted number of pixel, Rerunning")
            raise RuntimeError()
    except:
        dens_Ms_kpc2,radius,dP,cosmo = get_dens_map_rotate_hist(Gal=Gal,pixel_num=pixel_num,
                                                           z_source_max=z_source_max,verbose=True,plot=plot)#verbose=verbose)
    """
    Xg, Yg    = np.mgrid[-radius:radius:pixel_num, -radius:radius:pixel_num] # kpc
    arcXkpc   = cosmo.arcsec_per_kpc_proper(z_lens) # ''/kpc
    
    # create lensed image:
    # dPix it's given by the pixel_num  
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens).to("kpc")   #kpc
    cosmo_ds  = cosmo.angular_diameter_distance(z_source).to("kpc") #kpc
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source).to("kpc") #kpc
    
    # Sigma_Crit = D_s*c^2/(4piG * D_d*D_ds)
    Sigma_Crit  = (cosmo_ds*const.c**2)/(4*np.pi*const.G*cosmo_dds*cosmo_dd)
    Sigma_Crit  = Sigma_Crit.to("Msun /(kpc kpc)")    
    #Sigma_Crit /= arcXkpc**2  # Msun / ''^2 
    #Sigma_Crit *= dP**2       # Msun/pix^2
    kappa_grid  = dens_Ms_kpc2/Sigma_Crit # 1 
    
    # NOTE: a lens is such only if kappa_map > 1 at least at one point
    # 1st assumption: we will realistically assume that such point is the center of the galaxy -> position the source there
    # check that 
    assert(np.any(kappa_grid>1))#this should be true given our get_z_source function
    # if not: check if there is a realistic (higher)  z_source  to make such that the 
    
    # add masking
    # add padding
    
    # I don't have to go trough the potential->deflection_from_kappa_grid ->later needed when we'll have to add the LOS effects
    """
        #num_pot    = convergence_integrals.potential_from_kappa_grid(kappa_grid, dP) # ''^2 / pix^2 -> this is odd but that's just how it is computed
        #print("numpot,unit",num_pot.unit)
        #numPot_div_dP = num_pot/dP # -> ''/pix 
        #raise("The latest unit problem is here: what exactly should it be the correct way to obtain alpha in arcsecs?")
        #print("numPot_div_dP,unit",numPot_div_dP.unit)
        #num_aRa,num_aDec = np.gradient(numPot_div_dP)/dP # this smh divide(?) by pix? and therefore the unit of the output must be only ''  -> set it by hand
        -> the function doesn't respect the units
    """
    ##
    """
    Deflection angle :math:`\\vec {\\alpha }}` from a convergence grid :math:`\\kappa`.
    
    .. math::
        {\\vec {\\alpha }}({\\vec {\\theta }})={\\frac {1}{\\pi }}
        \\int d^{2}\\theta ^{\\prime }{\\frac {({\\vec {\\theta }}-{\\vec {\\theta }}^{\\prime })
        \\kappa ({\\vec {\\theta }}^{\\prime })}{|{\\vec {\\theta }}-{\\vec {\\theta }}^{\\prime }|^{2}}}
    
    The computation is performed as a convolution of the Green's function with the convergence map using FFT.
    
    :param kappa: convergence values for each pixel (2-d array)
    :param grid_spacing: scale of an individual pixel (per axis) of grid
    :return: numerical deflection angles in x- and y- direction over the convergence grid points    
    """
    num_aRa,num_aDec = convergence_integrals.deflection_from_kappa_grid(kappa_grid,dP.value) # this function does not respect dimensions
    num_aRa  *=u.arcsec
    num_aDec *=u.arcsec
    #print("num_aRa,unit",num_aRa.unit) #*u.arcsec
    
    #ra,dec = util.array2image(RAg),util.array2image(DECg)
    #if DEBUG:
    #    print("DEBUG: shape Xg,num_aRa",np.shape(Xg),np.shape(num_aRa))

    ra  = Xg.reshape(num_aRa.shape)*arcXkpc/u.pix  # not entirely sure this unit is correct, but anyway it's just book-keeping
    dec = Yg.reshape(num_aDec.shape)*arcXkpc/u.pix  
    """
    print("DEBUG")
    plt.imshow(ra.value)
    plt.colorbar()
    plt.title("Ra source")
    #im_name = f"{Gal.proj_dir}/ra_src.pdf"
    im_name = f"tmp/ra_src.pdf"
    plt.savefig(im_name)
    plt.close()
    plt.imshow(dec.value)
    plt.colorbar()
    plt.title("Dec source")
    #im_name = f"{Gal.proj_dir}/dec_src.pdf"
    im_name = f"tmp/dec_src.pdf"
    plt.savefig(im_name)
    plt.close()
    plt.imshow(np.log10(sersic_brightness(ra,dec)) )
    plt.colorbar()
    plt.title("log Source")
    #im_name = f"{Gal.proj_dir}/src.pdf"
    im_name = f"tmp/src.pdf"
    plt.savefig(im_name)
    plt.close()
    
    plt.imshow(num_aRa.value)
    plt.colorbar()
    plt.title("Ra deflection")
    #im_name = f"{Gal.proj_dir}/alpha_ra.pdf"
    im_name = f"tmp/alpha_ra.pdf"
    plt.savefig(im_name)
    plt.close()
    plt.imshow(num_aDec.value)
    plt.colorbar()
    plt.title("Dec deflection")
    #im_name = f"{Gal.proj_dir}/alpha_dec.pdf"
    im_name = f"tmp/alpha_dec.pdf"
    plt.savefig(im_name)
    plt.close()
    print("DEBUG")
    
    ra_im  = ra.value-num_aRa.value
    dec_im = dec.value-num_aDec.value
    lensed_im = sersic_brightness(ra_im,dec_im)
    plt.imshow(np.log10(lensed_im))
    plt.colorbar()
    plt.title("Log Lensed Sersic image")
    #im_name = f"tmp/lensed_im.pdf"
    im_name = f"{Gal.proj_dir}/lensed_im.pdf"
    plt.savefig(im_name)
    plt.close()
    print("Saving "+im_name)
    """
    # define the source to be behind the most dense pixel (not necessarily==CMS)
    # -> could consider not exactly behind but at least within a small radius of it    
    # find the "center" of the galaxy -> most dens pixel
    index_maxk = np.where(kappa_grid==np.max(kappa_grid))
    cntx_meas  = -((index_maxk[1]+.5) -int((len(kappa_grid[1]))/2.))*dx
    cnty_meas  = -((index_maxk[0]+.5) -int((len(kappa_grid))/2.))*dy
    cntx_meas  = cntx_meas[0]
    cnty_meas  = cnty_meas[0]
    
    cntx_meas_arcsec = cntx_meas*arcXkpc.to("arcsec/kpc").value
    cnty_meas_arcsec = cnty_meas*arcXkpc.to("arcsec/kpc").value
    print("Center measured")
    print(cntx_meas,cnty_meas,"[kpc]")
    print(cntx_meas_arcsec,cnty_meas_arcsec,"['']")
    

    print("WARNING - INVERTING X AND Y FOR THE SOURCE")
    source = Sersic(I=10,cnty=cntx_meas_arcsec,cntx=cnty_meas_arcsec,pa=45,q=.65,n=4)
    fg,ax=plt.subplots(2,3,figsize=(16,8))
    #ax[0][0].imshow(ra.value)
    #ax[0][0].set_title("Ra source")
    #ax[0][1].imshow(dec.value)
    #ax[0][1].set_title("Dec source")
    #ax[0][0].imshow(np.log10(Gal.image),extent=[np.min(ra),np.max(ra),np.min(dec),np.max(dec)])
    #ax[0][0].set_title(r"Particle Distrib.")
    ax[0][1].imshow(np.log10(kappa_grid))
    ax[0][1].scatter(cnty_meas,cntx_meas,marker="x",c="r",label="x,y meas:"+str(cnty_meas)+","+str(cntx_meas))
    ax[0][1].set_title(r"log $\kappa$")
    ax[0][1].imshow(np.log10(kappa_grid))
    ax[0][1].set_title(r"log $\kappa$")
    Src = np.log10(source.image(ra,dec)) 
    ax[0][2].imshow(Src)
    ax[0][2].contour(Src,c="w")
    ax[0][2].set_title("log Source")
    ax[1][0].imshow(num_aRa.value)
    ax[1][0].set_title("Ra deflection")
    ax[1][1].imshow(num_aDec.value)
    ax[1][1].set_title("Dec deflection")
    ra_im  = ra.value-num_aRa.value
    dec_im = dec.value-num_aDec.value
    lensed_im =  source.image(ra_im,dec_im)
    Lsnd      = np.log10(lensed_im)
    ax[1][2].imshow(Lsnd)
    ax[1][2].contour(Lsnd,c="w")
    ax[1][2].set_title("Log Lensed Sersic image")
    im_name = f"tmp/lensed_im.pdf"
    # just to be sure
    os.remove(im_name)
    plt.savefig(im_name)
    plt.show()
    plt.close()
    print("Saving "+im_name)

    # for convenience, I link the result to the tmp dir
    #os.unlink("./tmp/"+dir_name)
    #os.symlink(Gal.proj_dir[:-1],"./tmp/.")
    
    print("Success")
