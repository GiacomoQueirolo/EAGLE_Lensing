# copied and adapted from my machine [tmp_dir]/alpha_map.py
# calling dens_plot and maxminDsDds
from lenstronomy.LensModel import convergence_integrals
import numpy as np
import matplotlib.pyplot as plt


#from dens_plot import get_density,plot_density #,x,y,m
import astropy.units as u
import astropy.constants as const

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
def sersic_brightness(x,y,n=4,I=10,cntx=0,cnty=0,pa=0):
    try:
        # ugly but useful
        x=x.value
        y=y.value
    except:
        pass
    try:
        # ugly but useful
        cntx=cntx.value
        cnty=cnty.value
    except:
        pass
        
    # rotate the galaxy by the angle self.pa
    x = np.cos(pa)*(x-cntx)+np.sin(pa)*(y-cnty)
    y = -np.sin(pa)*(x-cntx)+np.cos(pa)*(y-cnty)
    # include elliptical isophotes
    r = np.sqrt((x-cntx)**2+(y-cnty)**2)
    # brightness at distance r
    bn = 1.992*n - 0.3271
    re = 5.0
    brightness = I*np.exp(-bn*((r/re)**(1.0/n)-1.0))
    return brightness

z_lens = .75
cosmo  = FlatLambdaCDM(H0=67.7,Om0=0.3)

def get_density(x,y,m,rng,pxn,ret_extent=False):
    nx,ny = pxn,pxn
    xmin,xmax,ymin,ymax = -rng,rng,-rng,rng
    # counts x-bin i and y-bin j. We transpose to (ny, nx) so rows are y.
    H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny],
                                       range=[[xmin, xmax], [ymin, ymax]],
                                       weights=m,density=False)  # if density=True, it normalises it to the total density
    # H is then the distribution of mass for each bin, not the density
    mass_grid = H.T.copy() # Solar Masses
    # H shape: (nx, ny) -> transpose to (ny, nx)

    # area of the (dx/dy) edges of bins:
    dx = (xmax - xmin) / nx #kpc
    dy = (ymax - ymin) / ny #kpc
    # density_ij = M_ij/(Area_bin_ij)
    density = mass_grid / (dx * dy)
    extent = [xmin,xmax,ymin,ymax]
    if ret_extent:
        return density,extent
    return density
    
def plot_density(density,extent):
    plt.imshow(np.log10(density),extent=extent, cmap=plt.cm.gist_earth_r,norm="log",label="Density [Msun/kpc^2]")
    plt.colorbar()
    plt.xlim([extent[0],extent[1]])
    plt.ylim([extent[2],extent[3]])
    plt.show()


#radius: how "large" we take the central cutout of the lens
radius    = 10        # kpc
# resolution 
pixel_num = 130  # n*pix

# coords
A = np.random.rand(2,2)
B = np.dot(A, A.transpose())
#x,y = np.random.multivariate_normal([0,0],B,(500000)).T
x,y = np.random.normal([0,0],3,(500000,2)).T #
rad = np.sqrt(x**2+y**2)
# Mass
#m   = 1e6*np.exp(-rad/3) #M_sun 
m = np.random.normal(1e6,1e2,len(x))
print("Tot Mass",np.round(m.sum(),0),r"M$_{\odot}]$")
plt.hist(m,bins=100)
plt.xlabel(r"M [M$_{\odot}]$")
plt.title("Mass distribution")
plt.show()
plt.close()
density,extent = get_density(x,y,m,rng=radius,pxn=pixel_num,ret_extent=True)
plot_density(density,extent)

print("np.max(density)",np.max(density))
#from maxminDsDds import plot_DsDds
def plot_DsDds(z_lens,dens_max,cosmo=cosmo,z_source_range=None,ret_source=False,verbose=True):
    if z_source_range is None:
        z_source_range = np.linspace(z_lens+0.09,z_source_max,100) # it's a very smooth funct->
    dens_Ms_kpc2 =np.array([dens_max,0])*u.Msun/(u.kpc**2)
    max_DsDds = np.max(dens_Ms_kpc2)*4*np.pi*const.G*cosmo.angular_diameter_distance(z_lens)/(const.c**2) 
    if verbose:
        print("DEBUG NOTE: the approx MW surf.dens. is 2*1e9Msun/kpc^2")
        print("DEBUG\n","np.max(dens_Ms_kpc2)",np.max(dens_Ms_kpc2).to("1e9Msun/kpc^2"))
        print("DEBUG\n","max_DsDds",max_DsDds)
    max_DsDds = max_DsDds.to("") 
    max_DsDds = max_DsDds.value # dimensionless
    if verbose:
        print("max_DsDds",max_DsDds)
    min_DsDds = cosmo.angular_diameter_distance(z_source_max)/cosmo.angular_diameter_distance_z1z2(z_lens,z_source_max) # this is the minimum
    min_DsDds = min_DsDds.to("") # dimensionless
    min_DsDds = min_DsDds.value
    if verbose:
        print("min_DsDds",min_DsDds)
    DsDds = np.array([cosmo.angular_diameter_distance(z_s).to("Mpc").value/cosmo.angular_diameter_distance_z1z2(z_lens,z_s).to("Mpc").value for z_s in z_source_range])
    plt.axhline(max_DsDds,ls="--",c="r",label=r"max(dens)*4$\pi$*G*$D_l$/c$^2$")
    plt.plot(z_source_range,DsDds,ls="-",c="k",label=r"D$_{\text{s}}$/D$_{\text{ds}}$(z$_{source}$)")
    plt.legend()
    plt.show()
    if ret_source:
        # note that the successful test means only that there is AT LEAST 1 PIXEL that is supercritical
        minimise     = np.abs(DsDds-max_DsDds) 
        z_source_min = z_source_range[np.argmin(minimise)]
        # select a random source within the range
        z_source = np.random.uniform(z_source_min,z_source_max,1)[0]
        if verbose:
            print("Minimum z_source:",np.round(z_source_min,2))
            print("Chosen z_source:", np.round(z_source,2))
        return z_source

z_source_max =5
z_source = plot_DsDds(z_lens,dens_max=np.max(density),cosmo=cosmo,ret_source=True,verbose=False)
#exit()

Xg, Yg    = np.mgrid[-radius:radius:pixel_num*1j, -radius:radius:pixel_num*1j] # kpc
arcXkpc   = cosmo.arcsec_per_kpc_proper(z_lens) # ''/kpc
    
cosmo_dd  = cosmo.angular_diameter_distance(z_lens).to("kpc")   #kpc
cosmo_ds  = cosmo.angular_diameter_distance(z_source).to("kpc") #kpc
cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source).to("kpc") #kpc

    # Sigma_Crit = D_s*c^2/(4piG * D_d*D_ds)
Sigma_Crit  = (cosmo_ds*const.c**2)/(4*np.pi*const.G*cosmo_dds*cosmo_dd)
Sigma_Crit  = Sigma_Crit.to("Msun /(kpc kpc)")    
#Sigma_Crit /= arcXkpc**2  # Msun / ''^2 
#Sigma_Crit *= dP**2       # Msun/pix^2
kappa_grid  = density/Sigma_Crit.value #dens_Ms_kpc2/Sigma_Crit # 1 

#SigmaCrit  = np.mean(density)/5
#kappa_grid = density/SigmaCrit

#dP  = .1 # ''/pix 
dP = 2*radius*u.kpc*arcXkpc.to("arcsec/kpc")/(int(pixel_num)*u.pix) #''/pix
num_aRa,num_aDec = convergence_integrals.deflection_from_kappa_grid(kappa_grid,dP.value) 

ra  = Xg.reshape(num_aRa.shape)*arcXkpc/u.pix  # not entirely sure this unit is correct, but anyway it's just book-keeping
dec = Yg.reshape(num_aDec.shape)*arcXkpc/u.pix  

fg,ax=plt.subplots(2,3,figsize=(16,8))
ax[0][0].imshow(ra.value)
#ax[0].colorbar()
ax[0][0].set_title("Ra source")
#im_name = f"{Gal.proj_dir}/ra_src.pdf"
#im_name = f"tmp/ra_src.pdf"
#plt.savefig(im_name)
#plt.close()
ax[0][1].imshow(dec.value)
#ax[1].colorbar()
ax[0][1].set_title("Dec source")
#im_name = f"{Gal.proj_dir}/dec_src.pdf"
#im_name = f"tmp/dec_src.pdf"
#.savefig(im_name)
#plt.close()
ax[0][2].imshow(np.log10(sersic_brightness(ra,dec)) )
#ax[2].colorbar()
ax[0][2].set_title("log Source")
#im_name = f"{Gal.proj_dir}/src.pdf"
"""
im_name = f"src.pdf"
plt.savefig(im_name)
plt.show()
plt.close()

fg,ax=plt.subplots(1,3,figsize=(16,8))
"""

ax[1][0].imshow(num_aRa)
#plt.colorbar()
ax[1][0].set_title("Ra deflection")
ax[1][1].imshow(num_aDec)
ax[1][1].set_title("Dec deflection")
ra_im  = ra.value-num_aRa
dec_im = dec.value-num_aDec
lensed_im = sersic_brightness(ra_im,dec_im)
ax[1][2].imshow(np.log10(lensed_im))
ax[1][2].set_title("Log Lensed Sersic image")
#im_name = f"tmp/lensed_im.pdf"
im_name = f"lensed_im.pdf"
plt.savefig(im_name)
plt.show()
plt.close()
print("Saving "+im_name)

