from   astropy.cosmology import FlatLambdaCDM
import astropy.constants as const
from numpy import pi

default_cosmo       = FlatLambdaCDM(H0=67.7,Om0=0.3)
def SigCrit(z_lens,z_source,cosmo=default_cosmo):
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens).to("kpc")   #kpc
    cosmo_ds  = cosmo.angular_diameter_distance(z_source).to("kpc") #kpc
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source).to("kpc") #kpc

    Sigma_Crit        = (cosmo_ds*const.c**2)/(4*pi*const.G*cosmo_dds*cosmo_dd) #
    return Sigma_Crit.to("Msun /(kpc kpc)")

def SigCrArc2(z_lens,z_source,cosmo=default_cosmo):
    arcXkpc           = cosmo.arcsec_per_kpc_proper(z_lens) # ''/kpc
    Sigma_Crit        = SigCrit(cosmo=cosmo,z_lens=z_lens,z_source=z_source)
    Sigma_Crit_arcs2  = Sigma_Crit.to("Msun /(kpc kpc)")/(arcXkpc*arcXkpc)
    return Sigma_Crit_arcs2

"""
# big brain time: 

cosmo= FlatLambdaCDM(H0=70, Om0=0.3)

arcXkpc = cosmo.arcsec_per_kpc_proper(2)


cosmo.angular_diameter_distance(2)
Out[6]: <Quantity 1726.62069147 Mpc>

1/arcXkpc.to("rad/Mpc")
Out[7]: <Quantity 1726.62069147 Mpc / rad>
"""