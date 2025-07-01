# try to model a lens with point masses for the particles:
import pickle
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from tools import *
from fnct import std_sim
from get_gal_indexes import get_rnd_gal

# Sim source:
z_source = 2.1
default_cosmo= FlatLambdaCDM(H0=70, Om0=0.3)
Gal = get_rnd_gal(sim=std_sim,check_prev=False,reuse_previous=False)
