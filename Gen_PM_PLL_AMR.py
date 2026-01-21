# copy from Gen_PM_PLL_AMR_01 12/12/25
# "slim" the saved components - delete everything which is 1)heavy 2) easy to recompute
# and re-compute it automatically when loading the class

# lens_model_PART: is already present in imageModel.LensModel
# Gal : can be re-loaded
# PMLens : can be re-computed
# cosmo : present in Gal, can be re-computed
#imageModel: very heavy, can be re-computed
# all outputs of setup_lenses are very fast to compute and heavy - mainly lens_model_PART
import os,dill
import numpy as np
from copy import copy,deepcopy
import matplotlib.pyplot as plt
from functools import cached_property 
from scipy.interpolate import splprep, splev, RectBivariateSpline

from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.units as u
import astropy.constants as const

from lenstronomy.Util import util
from lenstronomy.Plots import plot_util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.simulation_util as sim_util

from lenstronomy.Data.psf import PSF
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel  #this has been changed - we are taking my branch of lenstronomy
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


# My libs
from python_tools.fwhm import get_fwhm
from python_tools.get_res import load_whatever
from python_tools.tools import mkdir,short_SciNot,to_dimless,ensure_unit
from remade_gal import get_rnd_NG,get_lens_dir,Gal2kwMXYZ
from project_gal_AMR import get_2Dkappa_map,prep_Gal_projpath,projection_main_AMR
from project_gal_AMR import Gal2kw_samples,project_kw_parts,kwparts2arcsec,ProjectionError
# cosmol. params.
from lib_cosmo import SigCrit

pixel_num     = 200 # pix for image
pixel_dens    = 100 # pixel for mass density
verbose       = True
# for z_source computation:
z_source_max  = 4
# cutoff to define the 2D density hist
# from there the maximum density pixel
# and from there min_z_source

#minimum theta_E
min_thetaE = .3*u.arcsec #arcsec


##########################
##########################
# Sampling of the Profiles
##########################
##########################
#
# Particle functions
#
# cosmo from https://academic.oup.com/mnras/article/474/3/3391/4644836, Agnello 2017
# point mass theta_E (from eq.4.7 of Meneghetti's lecture note - and by memory)
# theta_E = \sqrt ( 4GM D_ls / (c^2 Ds Dl) )
# divide the computation such that it's done only once
def thetaE_PM_prefact(z_lens,z_source,cosmo):    
    cosmo_ds  = cosmo.angular_diameter_distance(z_source)
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens)
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source)
    pref      = 4*const.G*cosmo_dds/(const.c*const.c*cosmo_ds*cosmo_dd)
    return np.sqrt(pref) # 

@u.quantity_input
def thetaE_PM(M:u.g,theta_pref:u.g**-.5):
    thetaE_rad = np.sqrt(M)*theta_pref
    thetaE     = thetaE_rad.to("")*u.rad.to("arcsec")
    return thetaE.value #in arcsec
# ARCSINH thetaE is actually the same as PM
def thetaE_AS_prefact(z_lens,z_source,cosmo):    
    # is actually the same of PM, but it could be in principle different
    return thetaE_PM_prefact(z_lens,z_source,cosmo)
@u.quantity_input
def thetaE_AS(M:u.g,theta_pref:u.g**-.5):
    # is actually the same of PM, but it could be in principle different
    return thetaE_PM(M,theta_pref)

# maybe useful funct:
def MfromtE(tE,theta_pref:u.g**-.5):
    tErad  = tE*u.arcsec.to("rad")
    M = (tErad/theta_pref)**2
    return M.to("Msun")

def _build_kwargs_lens_AS(args):
        tE,tCAS, ra, dec = args
        return {
            "theta_E": tE,
            "theta_c": tCAS,
            "center_x": ra,
            "center_y": dec
        }
        
def _build_kwargs_lens_PM(args):
        tE, ra, dec = args
        return {
            "theta_E": tE,
            "center_x": ra,
            "center_y": dec
        }


# From EAGLE simulation

# Helper funct - create the kwargs_lens given the part. parameters, ie theta_E,x,y,(core if needed) 
# and the lens_model
#
def get_lens_model_PM(thetaEs,samples):
    kwargs_lens_PM  = [_build_kwargs_lens_PM((thetaEs, samples[0],samples[1]))]
    lens_model_list = ["POINT_MASS_PARALL"]
    lens_model_PM   = LensModel(lens_model_list=lens_model_list)
    return kwargs_lens_PM,lens_model_PM 

def get_lens_model_AS(theta_cAS,thetaEs,samples):
    try:
        len(theta_cAS)
    except TypeError:
        theta_cAS *= np.ones_like(thetaEs)
    kwargs_lens_AS = [_build_kwargs_lens_AS((thetaEs,theta_cAS, samples[0],samples[1]))]
    lens_model_list = ["ARSINH_PARALL"]
    lens_model_AS   = LensModel(lens_model_list=lens_model_list)
    return kwargs_lens_AS,lens_model_AS
    
#
# Particle functions
# -> not needed anymore due to structural changes
    
"""
def get_kwrg_PM(samples,Ms,
                    z_lens,z_source,
                    theta_E):     
    theta_pref = thetaE_PM_prefact(z_lens=z_lens,z_source=z_source)
    thetaEs    = thetaE_PM(M=Ms,theta_pref = theta_pref)

    kwargs_lens_PM,lens_model_PM = get_lens_model_PM(thetaEs,samples)
    return {"kwargs_lens_PART":kwargs_lens_PM,"lens_model_PART":lens_model_PM}
                
def get_kwrg_AS(samples,Ms,theta_cAS
                    z_lens,z_source,
                    theta_E):
                    
    theta_pref = thetaE_AS_prefact(z_lens=z_lens,z_source=z_source)
    thetaEs    = thetaE_AS(M=Ms,theta_pref=theta_pref)
 
    kwargs_lens_AS,lens_model_AS = get_lens_model_AS(theta_cAS,thetaEs,samples)

    return {"kwargs_lens_PART":kwargs_lens_AS,"lens_model_PART":lens_model_AS}
"""
#
# naming functions
#

def _get_tcAS_str(kwargs_lens):
    try:
        tcAS = kwargs_lens["theta_cAS"].value
    except AttributeError:
        tcAS = kwargs_lens["theta_cAS"]
    tcAS_str = short_SciNot(tcAS)
    return tcAS_str 
    
def get_name_PM(kw_lens=None):
    return f"PM"
def get_name_AS(kwargs_lens):
    tcAS_str   = _get_tcAS_str(kwargs_lens)
    return f"AS_tc{tcAS_str}"

# Lens modelling 
#################

#
# Class wrapper for Particle Lens computation
#
class PMLens():
    def __init__(self,kwargs_lens_part):
        self.kwargs_lens = kwargs_lens_part
        type_part = kwargs_lens_part["type"]
        self.name = type_part
        
        if type_part=="PM":
            self.thetaE_prefact = thetaE_PM_prefact
            self.thetaE         = thetaE_PM
            self.get_lens_model = get_lens_model_PM

        elif type_part=="ARCSINH" or type_part=="AS":
            self.thetaE_prefact = thetaE_AS_prefact
            self.thetaE         = thetaE_AS
            self.get_lens_model = get_lens_model_AS 
        else:
            raise TypeError("This particle model is not known: "+type_part)

    
    def setup(self,Mod):
         self.z_lens   = Mod.z_lens
         self.z_source = Mod.z_source
         self.cosmo    = Mod.cosmo
                                          
    def get_lens_PART(self,samples,Ms):
        theta_pref = self.thetaE_prefact(z_lens=self.z_lens,z_source=self.z_source,cosmo=self.cosmo)
        thetaEs    = self.thetaE(M=Ms,theta_pref = theta_pref)
        kw_lns_mod = {}
        if self.name =="ARSINH"  or self.name =="AS":
            kw_lns_mod = {"theta_cAS":self.kwargs_lens["theta_cAS"]}
        kwargs_lens_PART,lens_model_PART = self.get_lens_model(thetaEs=thetaEs,samples=samples,**kw_lns_mod)
        return kwargs_lens_PART,lens_model_PART

    
    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        # convert kwargs in immuatable tuple to be hashable
        Id = (self.name,
              tuple(sorted(self.kwargs_lens.items())))
        return Id
    
    def __hash__(self):
        # simplify the hash method
        return hash(self._identity())

    def __eq__(self, other):
        if not isinstance(other, PMLens):
            return NotImplemented
        return self._identity() == other._identity()

    def __str__(self):
        if not getattr(self,"name",False):
            self._setup_names()
        return self.name
########################
########################

def get_LensSystem_kwrgs(deltaPix,pixel_num=pixel_num,background_rms=None,exp_time=None,ra_source=0.,dec_source = 0.):
    # data specifics
    # background noise per pixel
    # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    print("Pixel_num: ",  pixel_num)
    print("DeltaPix: ",  np.round(deltaPix,3))
    deltaPix = to_dimless(deltaPix,True) # if dimensional, convert to dimensionless
    kwargs_data = sim_util.data_configure_simple(pixel_num, deltaPix, exp_time, background_rms)
    data_class  = ImageData(**kwargs_data)
    kwargs_psf  = {'psf_type': 'NONE'}  
    psf_class   = PSF(**kwargs_psf)
    # Source Params
    source_model_class,kwargs_source = get_model_source(ra_source,dec_source)
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    # for modelling later:
    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 
                     'multi_band_type': 'single-band'}
    return data_class, psf_class, source_model_class, kwargs_numerics, kwargs_source,kwargs_data_joint

def get_model_source(ra_source=0,dec_source=0):
    source_model_list = ['SERSIC_ELLIPSE']
    
    kwargs_sersic_ellipse = {'amp': 4000., 'R_sersic': .1, 'n_sersic': 3, 
                            'center_x': ra_source,
                             'center_y': dec_source, 
                             'e1': 0.0, 'e2': 0.0} #purely circular
                             #'e1': -0.1, 'e2': 0.01} #mildly elliptical
    kwargs_source = [kwargs_sersic_ellipse]
    source_model_class = LightModel(light_model_list=source_model_list)
    return source_model_class,kwargs_source
    
##########################
# Model class for parts. #
##########################
theta_c_AS     = 5e-3 
kwlens_part_AS = {"type":"AS","theta_cAS":theta_c_AS}
kwlens_part_PM = {"type":"PM"}

class LensPart(): 
    def __init__(self,
                    Galaxy,
                    kwlens_part, # if PM or AS, and if so size of the core
                    pixel_num=pixel_num, # sim prms 
                    z_source_max  = z_source_max,     # for z_source sampling
                    min_thetaE = min_thetaE,
                    #z_lens=z_lens,z_source=z_source,cosmo=cosmo, # cosmo prms -> obtained from Galaxy 
                    #exp_time=1500, #sec. (~<exp_time> for J1433 opt. HST obs ) -> will be defined later
                    #bckg_rms=0.006, #count (~ from f814w) observation parameters
                    savedir_sim="lensing",
                    reload=True # reload previos lens
                    ):
        Galaxy             = prep_Gal_projpath(Galaxy) # just set up some directories
        # setup of data
        self.Gal           = Galaxy
        self.Gal_path      = Galaxy.get_pkl_path()
        # if reload, check if Gal is already a lens - if not, raise error
        if reload:
            if not self.Gal.is_lens:
                print(load_whatever(self.Gal.islens_file)["message"])
                raise RuntimeError("Previously defined as not a lens")
        
        self.kwlens_part   = kwlens_part
        lens_dir           = get_lens_dir(self.Gal)
        self.savedir_sim   = savedir_sim
        mkdir(savedir_sim)
        self.savedir       = f"{lens_dir}/{savedir_sim}"
        self.reload        = reload
        self.pixel_num     = pixel_num      
        mkdir(self.savedir)
        # cosmo prms
        self.z_lens        = self.Gal.z
        self.cosmo         = self.Gal.cosmo
        self.arcXkpc       = self.cosmo.arcsec_per_kpc_proper(self.z_lens)

        # To obtain the z_source and projection index 
        #-> computed only once in the run function
        self.z_source_max  = z_source_max        
        self.min_thetaE    = ensure_unit(min_thetaE,u.arcsec) #arcsec
        # observational params - defined a posteriori
        self.exp_time  =  None #exp_time
        self.bckg_rms  =  None #bckg_rms
        # kwargs_lens for the particles :
        # type:"AS" or "PM"
        # if AS: param:{"thetacAS"}
        self.PMLens      = PMLens(kwlens_part)
        self._setup_names()

    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        Id = (self.Gal._identity(),
              self.PMLens._identity(),
              self.pixel_num,
              self.z_source_max)
              #,self.exp_time,self.bckg_rms)
        return Id
    
    def __hash__(self):
        # simplify the hash method
        return hash(self._identity())

    def __eq__(self, other):
        if not isinstance(other, LensPart):
            return NotImplemented
        return self._identity() == other._identity()

    def __str__(self):
        if not getattr(self,"name",False):
            self._setup_names()
        return self.name
    # the following struct. is more clear and allow a slimmer stored class
    def __getstate__(self):
        state = self.__dict__.copy()
        # remove large but recomputable attributes (if present)
        state.pop('kwargs_lens_PART', None)
        state.pop('lens_model_PART', None)
        state.pop('kw_shear',None)
        state.pop('imageModel',None)
        # Gal should not be stored again, but reloaded afterwards
        state.pop('Gal',None)
        # PMLens is easy to recompute (careful to set it up as well)
        state.pop('PMLens',None)
        # cosmo can be obtain by Gal
        state.pop('cosmo',None)
        # could also remove image_sim if it's heavy
        #try:
        #    state['imageModel'].ImageNumerics._numerics_subframe._grid = None
        #except Exception:
        #    pass
        return state

    def __setstate__(self, state):
        # Optional: restore defaults or trigger rebuild of heavy attributes
        self.__dict__.update(state)

    def store(self):
        if not hasattr(self,"pkl_path"):
            self._setup_names()
        with open(self.pkl_path, "wb") as f:
            dill.dump(self, f)
        # Assuming that if we got here, everything worked out fine:
        self.Gal.update_is_lens(islens=True,message="No issues")
        print("Saved", self.pkl_path)
        
    def _unpack(self):
        # this function recover the parts deleted before storing
        # to save space
        # reload Galaxy and cosmo
        Galaxy = LoadClass(self.Gal_path)
        Galaxy = prep_Gal_projpath(Galaxy)
        self.Gal   = Galaxy
        self.cosmo = Galaxy.cosmo
        # re-define PMLens
        self.PMLens = PMLens(self.kwlens_part)
        self.PMLens.setup(self)
        # recover kwargs_lens_PART and lens_model_PART
        if not hasattr(self,"lens_model_PART"):
            self.setup_lenses() 
        # recover imageModel
        self.get_imageModel()
        """ -> the whole imageModel is deleted
        # recover the grid
        gridClassType = getattr(self.kwargs_numerics,"compute_mode","regular")
        if gridClassType =="regular":
            from lenstronomy.ImSim.Numerics.grid import RegularGrid as Grid
        elif gridClassType  == "adaptive":
            from lenstronomy.ImSim.Numerics.grid import AdaptiveGrid as Grid
        recomputed_grid = Grid(nx=self.pixel_num,ny=self.pixel_num,
                        transform_pix2angle=self.imageModel.Data.transform_pix2angle,
                        ra_at_xy_0=self.imageModel.Data.radec_at_xy_0[0],
                        dec_at_xy_0=self.imageModel.Data.radec_at_xy_0[1])
        
        self.imageModel.ImageNumerics._numerics_subframe._grid = recomputed_grid 
        """
        return True
        
    def unpack(self):
        #if self.imageModel.ImageNumerics._numerics_subframe._grid is None:
        if getattr(self,"Gal",False) is False:
            self._unpack()
        return True
    ########################
    ########################

    def _get_name(self):
        # define name and path of savefile
        self.name       = f"{self.Gal.Name}_Npix{self.pixel_num}_Part{self.PMLens.name}"
        
    def _setup_names(self):
        if not getattr(self,"name",False):
            self._get_name()
        self.pkl_path = f"{self.savedir}/{self.name}.pkl"
        
    def upload_prev(self):
        if not self.reload:
            return False
        prev_mod = ReadLens(self)
        if prev_mod is False:
            return False
        # we have now a good way to define equality
        if prev_mod==self:
            for attr, value in prev_mod.__dict__.items():
                setattr(self, attr, value)
            return True
        return False
            
    def run(self,read_prev=True):
        upload_successful = False
        if read_prev:
            upload_successful = self.upload_prev()
        if not upload_successful:
            # Read particles ONLY ONCE
            kw_parts         = Gal2kwMXYZ(self.Gal) # kwargs of Msun, XYZ in kpc (explicitely) centered around Centre of Mass (CM)
            kwres_proj       = projection_main_AMR(Gal=self.Gal,kw_parts=kw_parts,
                            z_source_max=self.z_source_max,sample_z_source=self.sample_z_source,min_thetaE=self.min_thetaE,
                            arcXkpc=self.arcXkpc,verbose=True,save_res=True,reload=self.reload)

            self.proj_index   = kwres_proj["proj_index"]
            self.z_source_min = kwres_proj["z_source_min"]
            self.z_source     = kwres_proj["z_source"]
            self.MD_coords    = kwres_proj["MD_coords"]
            print("Z source sampled:",self.z_source)
            self.thetaE    = kwres_proj["thetaE"]
            print("Approx. thetaE:",np.round(self.thetaE,3))

            # the following 2 can only be computed once we know the z_source:
            self.SigCrit       = SigCrit(cosmo=self.cosmo,z_lens=self.z_lens,z_source=self.z_source) # Msun/kpc^2

            self.PMLens.setup(self) # only run now bc it needs z_source 
            # Then define the radius based on ~ theta_E
            scale_tE       = 2 
            self.radius    = self.thetaE*scale_tE
            print("Image radius:",np.round(self.radius,3))
    
            Diam_arcsec      = 2*self.radius #diameter in arcsec
            self.deltaPix    = Diam_arcsec/self.pixel_num # ''/pix
            # setup dataclasses (dataclass,psf_class,sourcemodel and some helper kwargs):
            self.setup_dataclasses()
            # setup imageModel:
            self.get_imageModel()
            # setup lenses 
            self.setup_lenses()
            # this is the most computationally intense function:
            self.image_sim  = self.get_lensed_image()
            self.store()

    def setup_dataclasses(self,ra_source=0,dec_source=0):
        self.data_class,self.psf_class,self.source_model_class,\
        self.kwargs_numerics, self.kwargs_source,self.kwargs_data_joint = \
                get_LensSystem_kwrgs(self.deltaPix,self.pixel_num,background_rms=self.bckg_rms,exp_time=self.exp_time,
                                     ra_source =ra_source,dec_source =dec_source)
        return 0
    def update_source_position(self,ra_source,dec_source):
        # useful if we want to put it in the center of the caustic
        self.source_model_class,self.kwargs_source = get_model_source(ra_source,dec_source)
        
    # the following is meant to be rerun every time we load the class to save space
    # -> computationally not intense 
    def setup_lenses(self):
        print("Setting up lensing parameters")
        # Convert x,y,z in samples and get masses
        kw_samples = Gal2kw_samples(Gal=self.Gal,proj_index=self.proj_index,
                                    MD_coords=self.MD_coords,arcXkpc=self.arcXkpc)
        samples    = kw_samples["RAs"],kw_samples["DECs"]
        Ms         = kw_samples["Ms"]
        # Convert in lenses parameters 
        kwLns_PART,LnsMod_PART = self.PMLens.get_lens_PART(samples=samples,Ms=Ms)
        self.kwargs_lens_PART  = kwLns_PART
        self.lens_model_PART   = LnsMod_PART
        return 0
        
    def sample_z_source(self,z_source_min,z_source_max):
        # this is here to allow modularity 
        # for now a simple uniform sample, but we could define something more fancy
        z_source = np.random.uniform(z_source_min,z_source_max,1)[0]
        return z_source
        
    def get_lensed_image(self):
        self.unpack()
        sourceModel  = self.source_model_class
        imageModel   = self.imageModel
        x_source,y_source = self.sample_source_pos(update=True)
        
        x_source_plane,y_source_plane = self.get_xy_source_plane()
        source_light = sourceModel.surface_brightness(x_source_plane, y_source_plane, self.kwargs_source, k=None)
        image_sim    = imageModel.ImageNumerics.re_size_convolve(source_light, unconvolved=False)
        # imageModel.image(self.kwargs_lens_PART, self.kwargs_source, kwargs_lens_light=None, kwargs_ps=None)
        return image_sim

    def sample_source_pos(self,update=True):
        kw_caustics  = self.critical_curve_caustics
        ra_ct,dec_ct = kw_caustics["caustics"]["tangential"]
        """
        # old simplified
        print("Centering source in the center of the tangential caustic")
        if update:
            self.update_source_position(np.mean(ra_ct),np.mean(dec_ct))
        """
        print("Sampling source position within tangential caustic")
        # the tangential caustic is approximated to circular
        # and we sample uniformily within that 
        ra0_ct,dec0_ct = np.mean(ra_ct),np.mean(dec_ct)
        rads_ct = np.hypot(ra_ct-ra0_ct,dec_ct-dec0_ct)
        rad0_ct = np.std(rads_ct)
        rad_source = np.random.uniform(0,rad0_ct)
        phi_source = np.random.uniform(0,2*np.pi)
        ra_source  = rad_source*np.cos(phi_source) 
        dec_source = rad_source*np.sin(phi_source) 
        if update:
            self.update_source_position(ra_source,dec_source)
        return ra_source,dec_source

    def get_xy_source_plane(self):
        RA,DEC       = self.get_RADEC()
        # if not already, compute alpha_map
        alpha_x,alpha_y = self.alpha_map        
        x_source_plane, y_source_plane = RA-alpha_x,DEC-alpha_y
        # the coords have to be given as flat
        x_source_plane = util.image2array(x_source_plane)
        y_source_plane = util.image2array(y_source_plane)
        return x_source_plane,y_source_plane
        
    def get_imageModel(self):
        if not hasattr(self,"setup_dataclasses"):
            self.setup_dataclasses()
        if not hasattr(self,"lens_model_PART"):
            self.setup_lenses()
        self.imageModel = ImageModel(self.data_class, self.psf_class, 
                        self.lens_model_PART, 
                        self.source_model_class,lens_light_model_class=None,point_source_class=None, 
                        kwargs_numerics=self.kwargs_numerics)
        return self.imageModel
    
    @cached_property
    def alpha_map(self):
        return self._alpha_map(_radec=None)
    @cached_property
    def psi_map(self):
        return self._psi_map(_radec=None)
    @cached_property
    def kappa_map(self):
        return self._kappa_map(_radec=None)
    @cached_property
    def hessian(self):
        return self._hessian()
    
    def _psi_map(self,_radec=None):
        print("Computing lensing PM potential...")
        self.unpack()
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        psi = self.lens_model_PART.potential(_ra, _dec, self.kwargs_lens_PART)
        psi = util.array2image(psi)
        return psi
        
    def _alpha_map(self,_radec=None):
        print("Computing lensing PM deflection...")
        self.unpack()
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        alpha_x,alpha_y = self.lens_model_PART.alpha(_ra, _dec, self.kwargs_lens_PART)
        alpha_x,alpha_y = util.array2image(alpha_x),util.array2image(alpha_y)
        return alpha_x,alpha_y
        
    def _kappa_map_from_lens(self,_radec=None,exact=False):
        # compute analytically from the particles -> actually should not be the way to do it !
        print("Computing kappa map from PM...")
        self.unpack()
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        kappa = self.lens_model_PART.kappa(_ra, _dec, self.kwargs_lens_PART)
        kappa = util.array2image(kappa)
        return kappa
        
    def _kappa_map(self,_radec=None):
        # compute from density map
        # actually better bc does not depend on the particle profile
        print("Computing kappa map from density map...")
        if _radec is None:
            kw_extents = self.kw_extents
        else:
            kw_extents = get_extents(arcXkpc=self.arcXkpc,Model=self,_radec=_radec)
        kappa = get_2Dkappa_map(Gal=self.Gal,proj_index=self.proj_index,MD_coords=self.MD_coords,kwargs_extents=kw_extents,
                                SigCrit=self.SigCrit,arcXkpc=self.arcXkpc)
        return kappa
    @property
    def kw_extents(self):
        _radec = self.imageModel.ImageNumerics.coordinates_evaluate
        kw_extents = get_extents(arcXkpc=self.arcXkpc,Model=self,_radec=_radec)
        return kw_extents
        
    def get_RADEC(self):
        self.unpack()
        _ra,_dec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        RA,DEC   = util.array2image(_ra),util.array2image(_dec)
        return RA,DEC

    def update_kwargs_data_joint(self,add_noise=False):
        # to be reconsidered in face of GPPA_realistic_images
        raise RuntimeError("To re-implement better - use create_realistic_image from GPPA_realistic_images")
        # updating it with the PM image
        if add_noise:
            image   = self.image_sim
            # rememeber that now exp_time and bckg_rms have to be defined
            poisson = image_util.add_poisson(image, exp_time=self.exp_time)
            bkg     = image_util.add_background(image, sigma_bkd=self.bckg_rms)
            
            self.image_sim = image + poisson + bkg

            #other realisation of the bkg for the error map
            bkg       = image_util.add_background(image, sigma_bkd=self.bckg_rms)
            noise_map =  np.sqrt(poisson**2+bkg**2)
            self.kwargs_data_joint["multi_band_list"][0][0]["noise_map"] = noise_map
        self.kwargs_data_joint["multi_band_list"][0][0]["image_data"] = self.image_sim

    # Shear components and caustics/CL
    #def hessian_old(self):
    def _hessian(self):
         # Note: this hessian only consider the contribution of the alpha map within the cutout!
        alpha_x,alpha_y = self.alpha_map
        # taking the non-dimensional pixel scale for the gradient
        dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, to_dimless(self.deltaPix))
        dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, to_dimless(self.deltaPix))
        #print("Note: Taking the average of dalpha_x_dy and dalpha_y_dx for fxy")
        f_xx,f_xy,f_yx,f_yy  = dalpha_x_dx,dalpha_x_dy,dalpha_y_dx,dalpha_y_dy
        return f_xx,f_xy,f_yx,f_yy

    """def _hessian(self,_radec=None):
        print("Computing lensing PM hessian matrix...")
        self.unpack()
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        f_xx, f_xy, f_yx, f_yy = self.lens_model_PART.hessian(_ra, _dec, self.kwargs_lens_PART)
        f_xx, f_xy, f_yx, f_yy= util.array2image(f_xx),util.array2image(f_xy),util.array2image(f_yx),util.array2image(f_yy)
        return f_xx, f_xy,f_yx, f_yy
    """ 
    def get_kw_shear(self):
        f_xx,f_xy,f_yx,f_yy = self.hessian
        # derived kappa, shear1,shear2 and shear
        """
        # this is not almost equal up to 2 decimal in ~40% of the cases -> investigate
        # -> might make sense bc the one I was comparing it to was the ANALYTICAL exact kappa
        # to an approximate one
        kappa  = (f_xx + f_yy)/2.
        np.testing.assert_almost_equal(kappa,self.kappa_map,decimal=2)
        """
        shear1 = 1./2 * (f_xx - f_yy)
        shear2 = f_xy
        shear  = np.hypot(shear1,shear2)
        self.kw_shear = {"shear1":shear1,"shear2":shear2,"shear":shear}
        return self.kw_shear
        
    @property
    def shear_map(self):
        if hasattr(self,"kw_shear"):
            return self.kw_shear["shear"]
        return self.get_kw_shear()["shear"]

    @cached_property
    def critical_curve_caustics(self):
        return self.get_kw_critical_curve_caustics()

    def get_kw_critical_curve_caustics(self):
        alpha_x,alpha_y = self.alpha_map        
        fxx,fxy,fyx,fyy = self.hessian
        kappa  = (fxx + fyy)/2.
        shear1 = 1./2 * (fxx - fyy)
        shear2 = fxy
        shear  = np.hypot(shear1,shear2)
        
        eigen_rad = 1 - kappa + shear
        eigen_tan = 1 - kappa - shear
        """
        ###DEBUG####
        kw_extents   = self.kw_extents
        extent_arcsec = kw_extents["extent_arcsec"]
        kw_plot = {"cmap":"bwr","extent":extent_arcsec,"origin":"lower"}
        fig, axs = plt.subplots(1,2, figsize=(10, 5))
        im0 = axs[0].imshow(np.abs(eigen_rad),**kw_plot)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   
        axs[0].set_title("| Radial Eigenvalue |") 
        
        im0 = axs[1].imshow(np.abs(eigen_tan),**kw_plot)
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')
        axs[1].set_title("| Tangential Eigenvalue |")
        plt.suptitle(self.name)
        nm = "tmp/abs_eigenv_"+self.name+".png"
        print("DEBUG - plotting Eigenvalues in \n"+nm)
        plt.savefig(nm)
        ##############
        """
        # have to find when those are ~0
        mintv = np.min(np.abs(eigen_rad))
        Dv    = np.max(np.abs(eigen_rad)) - mintv
        maxtv = mintv + 0.1*Dv
        test_values = np.linspace(mintv,maxtv,20)
    
        ierx,ietx = [],[] #placeholders 
        # radial
        for tv in test_values:
            if len(ierx)/(self.pixel_num**2) >0.001 :
                break
            else:
                iery,ierx = np.where(MAD_mask(np.abs(eigen_rad),0,tv)) 
        # tangential
        mintv = np.min(np.abs(eigen_tan))
        Dv = np.max(np.abs(eigen_tan)) - mintv
        maxtv = mintv + 0.1*Dv
        test_values = np.linspace(mintv,maxtv,20)
        for tv in test_values:
            if len(ietx)/(self.pixel_num**2) >0.001:
                break
            else:
                iety,ietx = np.where(MAD_mask(np.abs(eigen_tan),0,tv))
        # coords
        RA,DEC      = self.get_RADEC()
        ra0,dec0    = RA[0],DEC.T[0]
        # critical and caustics divided in tangential and radial
        cl_rad_x_noisy,cl_rad_y_noisy   = ra0[ierx],dec0[iery]    
        cl_tan_x_noisy,cl_tan_y_noisy   = ra0[ietx],dec0[iety]
        # fit them with splines  
        cl_rad_x,cl_rad_y = fit_xy_spline(cl_rad_x_noisy,cl_rad_y_noisy)
        cl_tan_x,cl_tan_y = fit_xy_spline(cl_tan_x_noisy,cl_tan_y_noisy)
        
        # fit alpha in 2D
        # TODO: verify that it is indeed dec0,ra0 and not the other way out ->correct, see TEST
        alpha_x_spline = RectBivariateSpline(dec0,ra0, alpha_x)
        alpha_y_spline = RectBivariateSpline(dec0,ra0, alpha_y)

        """
        #TEST: Passed
        i_dec = np.random.choice(np.arange(0,len(dec0)-1))
        i_ra  = np.random.choice(np.arange(0,len(ra0)-1))
        # Direct grid value
        v_grid = alpha_x[i_dec, i_ra]
        
        # Interpolated value at exact grid point
        v_spline = alpha_x_spline.ev(dec0[i_dec], ra0[i_ra])
        
        print(v_grid, v_spline)
        """
        cc_rad_x,cc_rad_y   = cl_rad_x-alpha_x_spline.ev(cl_rad_y,cl_rad_x),\
                              cl_rad_y-alpha_y_spline.ev(cl_rad_y,cl_rad_x)

        cc_tan_x,cc_tan_y   = cl_tan_x-alpha_x_spline.ev(cl_tan_y,cl_tan_x),\
                              cl_tan_y-alpha_y_spline.ev(cl_tan_y,cl_tan_x)

        kw_crit = {"caustics":{"radial":[cc_rad_x,cc_rad_y],
                               "tangential":[cc_tan_x,cc_tan_y]},
                   "critical_lines":{"radial":[cl_rad_x,cl_rad_y],
                                     "tangential":[cl_tan_x,cl_tan_y]}
                  }
        """
        # DEBUG
        fig,ax = plt.subplots(figsize=(8,8))
        ax.scatter(cc_rad_x,cc_rad_y,c="b",marker=".",label="Radial Caustics")
        ax.scatter(cc_tan_x,cc_tan_y,c="r",marker=".",label="Tangential Caustics")
        ax.scatter(cl_rad_x,cl_rad_y,c="cyan",marker=".",label="Radial Crit. Curve")
        ax.scatter(cl_tan_x,cl_tan_y,c="darkorange",marker=".",label="Tangential Crit. Curve")
        
        #ax.scatter(cc_rad_x_noisy,cc_rad_y_noisy,c="gold",marker=".",label="Radial Caustics noisy")
        #ax.scatter(cc_tan_x_noisy,cc_tan_y_noisy,c="purple",marker=".",label="Tangential Caustics noisy")
        ax.scatter(cl_rad_x_noisy,cl_rad_y_noisy,c="lime",marker=".",label="Radial Crit. Curve noisy")
        ax.scatter(cl_tan_x_noisy,cl_tan_y_noisy,c="peru",marker=".",label="Tangential Crit. Curve noisy")
        
        
        
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        plt.gca().set_aspect('equal')
        ax.set_xlabel("RA ['']")
        ax.set_ylabel("DEC ['']")
        ax.legend()
        ax.set_title("Caustics and Critical Curves") 
        
        plt.tight_layout()
        nm = "tmp/del2.png"
        print("Saving "+nm)
        plt.savefig(nm)
        """
        
        return kw_crit

    def old_critical_curve_caustics(self):
        # no need to save this - as long as we have {hessian,alpha}_map,
        # which are cached once computed,
        # should be fast to compute
        alpha_x,alpha_y = self.alpha_map        
        """
        # this should be correct, but I prefer to do it easy
        hessian = self.hessian
        H       = np.array(hessian).reshape(2,2,self.pixel_num,self.pixel_num)
        #A     = Id - H
        A       = -H
        A[0,0] +=1
        A[1,1] +=1
        
        l1,l2     = np.linalg.eig(A.T).eigenvalues.T
        # radial and tangential eigenvalues
        eigen_tan = np.min([l1,l2],axis=0) 
        eigen_rad = np.max([l1,l2],axis=0)
        """
        fxx,fxy,fyx,fyy = self.hessian
        kappa  = (fxx + fyy)/2.
        shear1 = 1./2 * (fxx - fyy)
        shear2 = fxy
        shear  = np.hypot(shear1,shear2)
        
        eigen_rad = 1 - kappa + shear
        eigen_tan = 1 - kappa - shear

        # have to find when those are ~0
        mask_rad = MAD_mask(eigen_rad)
        mask_tan = MAD_mask(eigen_tan)
        ierx,iery = np.where(mask_rad.T==1)
        ietx,iety = np.where(mask_tan.T==1)
        # coords
        RA,DEC      = self.get_RADEC()
        ra0,dec0    = RA[0],DEC.T[0]
        # critical and caustics divided in tangential and radial
        cl_rad_x,cl_rad_y   = ra0[ierx],dec0[iery]    
        cc_rad_x,cc_rad_y   = ra0[ierx]-alpha_x[iery,ierx],dec0[iery]-alpha_y[iery,ierx]  
        cl_tan_x,cl_tan_y   = ra0[ietx],dec0[iety]    
        cc_tan_x,cc_tan_y   = ra0[ietx]-alpha_x[iety,ietx],dec0[iety]-alpha_y[iety,ietx]  
        kw_crit = {"caustics":{"radial":[cc_rad_x,cc_rad_y],"tangential":[cc_tan_x,cc_tan_y]},
                   "critical_lines":{"radial":[cl_rad_x,cl_rad_y],"tangential":[cl_tan_x,cl_tan_y]}
                  }
        return kw_crit

    
    def oldold_critical_curve_caustics(self):
        # no need to save this - as long as we have {kappa,shear,alpha}_map,
        # it's fast to compute
        kappa = self.kappa_map
        shear = self.shear_map
        alpha_x,alpha_y = self.alpha_map
        # radial and tangential eigenvalues
        eigen_rad = 1 - kappa + shear
        eigen_tan = 1 - kappa - shear
        # have to find when those are ~0
        mask_rad = MAD_mask(eigen_rad)
        mask_tan = MAD_mask(eigen_tan)
        ierx,iery = np.where(mask_rad.T==1)
        ietx,iety = np.where(mask_tan.T==1)
        # coords
        RA,DEC      = self.get_RADEC()
        ra0,dec0    = RA[0],DEC.T[0]
        # critical and caustics divided in tangential and radial
        cl_rad_x,cl_rad_y   = ra0[ierx],dec0[iery]    
        cc_rad_x,cc_rad_y   = ra0[ierx]-alpha_x[iery,ierx],dec0[iery]-alpha_y[iery,ierx]  
        cl_tan_x,cl_tan_y   = ra0[ietx],dec0[iety]    
        cc_tan_x,cc_tan_y   = ra0[ietx]-alpha_x[iety,ietx],dec0[iety]-alpha_y[iety,ietx]  
        kw_crit = {"caustics":{"radial":[cc_rad_x,cc_rad_y],"tangential":[cc_tan_x,cc_tan_y]},
                   "critical_lines":{"radial":[cl_rad_x,cl_rad_y],"tangential":[cl_tan_x,cl_tan_y]}
                  }
        return kw_crit
                
def MAD_mask(values,v0=0,sigma_scale=3):
    # robust estimator of noise: Median Absolute Deviation    
    mad = np.median(np.abs(values - np.median(values)))

    sigma = 1.4826 * mad

    mask = np.abs(values-v0) < sigma_scale*sigma   # ~99.7% Gaussian confidence
    return mask

def slow_fit_xy_spline(x,y,u=np.linspace(0, 1, 200)):
    xc = np.median(x)
    yc = np.median(y)
    theta = np.arctan2(y - yc, x - xc)
    order = np.argsort(theta)
    
    x_ord = x[order]
    y_ord = y[order]

    ## optimise scale:
    diffss = [] 
    sss = np.linspace(-5,-1,50)
    lnx = len(x)
    # for fitting
    u_fit=np.linspace(0, 1, len(x_ord))
    for s_scale in sss:
        tck, _ = splprep([x_ord, y_ord],\
        s=s_scale*lnx,per=True)
        xs, ys = splev(u_fit, tck)
        diffs = np.hypot(xs-x_ord,ys-y_ord)
        diffss.append(diffs.sum())
    s_scale = sss[np.argmin(diffss)]
    tck, _ = splprep([x_ord, y_ord],\
    s=s_scale*len(x_ord),per=True)
    xs, ys = splev(u, tck)
    return xs,ys
    
# optimised w. CGPT:
def fit_xy_spline(x, y,
    u=np.linspace(0, 1, 200),
    n_eval=150,           # points used for error estimation
    ):
    # --- angular ordering ---
    xc = np.median(x)
    yc = np.median(y)
    theta = np.arctan2(y - yc, x - xc)
    order = np.argsort(theta)
    
    x_ord = x[order]
    y_ord = y[order]
    n = len(x_ord)

    # fixed parameter grid
    u_fit = np.linspace(0, 1, n, endpoint=False)

    # subsampling indices for error evaluation
    idx = np.linspace(0, n - 1, n_eval).astype(int)
    u_sub = u_fit[idx]
    x_sub = x_ord[idx]
    y_sub = y_ord[idx]

    # --- coarse search ---
    s_vals = np.logspace(-5,-1, 12)
    errs = np.empty(len(s_vals))

    for i, s in enumerate(s_vals):
        # DEBUG
        try:
            tck, _ = splprep(
                [x_ord, y_ord],
                s=s * n,
                per=True,
                quiet=True
            )
        except TypeError as e:
            print(x_ord,y_ord,s,n)
            raise TypeError(e)
        xs, ys = splev(u_sub, tck)
        errs[i] = np.sum(np.hypot(xs - x_sub, ys - y_sub))

    # --- refine around minimum ---
    i0 = np.argmin(errs)
    lo = max(i0 - 1, 0)
    hi = min(i0 + 1, len(s_vals) - 1)

    s_refined = np.logspace(
        np.log10(s_vals[lo]),
        np.log10(s_vals[hi]),
        10
    )

    best_err = np.inf
    best_tck = None

    for s in s_refined:
        tck, _ = splprep(
            [x_ord, y_ord],
            s=s * n,
            per=True,
            quiet=True
        )
        xs, ys = splev(u_sub, tck)
        err = np.sum(np.hypot(xs - x_sub, ys - y_sub))
        if err < best_err:
            best_err = err
            best_tck = tck

    # --- final evaluation ---
    xs, ys = splev(u, best_tck)
    return xs, ys

#
# helper funct
#

# this function is a wrapper for convenience - it takes the class itself as input
from python_tools.get_res import LoadClass

# we override the standard loading function to recompute some large dataset that
# are deleted to save space
def LoadLens(LnsCl,verbose=True):
    LnsCl = LoadClass(LnsCl,verbose=verbose)
    # has to consider the possibility it failed to load
    if LnsCl: 
        # recompute deleted components
        LnsCl.unpack()
    return LnsCl
    
def ReadLens(aClass,verbose=True):
    return LoadLens(aClass.pkl_path,verbose=verbose)




def _plot_caustics(LPClass,
                   lensModelExt,
                   kwargs_lens,
                   kw_extents=None,
                   fast_caustic = True,
                   savename="test_caustics.png"):
    if kw_extents is None:
        kw_extents = get_extents(LPClass.arcXkpc,LPClass) 
    xmin,xmax,ymin,ymax = kw_extents["extent_arcsec"]
    kw_crit             = LPClass.critical_curve_caustics
    cl_rad_x,cl_rad_y   = kw_crit["critical_lines"]["radial"]
    cc_rad_x,cc_rad_y   = kw_crit["caustics"]["radial"]
    cl_tan_x,cl_tan_y   = kw_crit["critical_lines"]["tangential"]
    cc_tan_x,cc_tan_y   = kw_crit["caustics"]["tangential"]

    cent_caust_tan = np.mean(cc_tan_x),np.mean(cc_tan_y)
    fig,ax = plt.subplots()
    ax.scatter(cc_rad_x,cc_rad_y,c="b",marker=".",label="Radial Caustics")
    ax.scatter(cc_tan_x,cc_tan_y,c="r",marker=".",label="Tangential Caustics")
    ax.scatter(cl_rad_x,cl_rad_y,c="cyan",marker=".",label="Radial Crit. Curve")
    ax.scatter(cl_tan_x,cl_tan_y,c="darkorange",marker=".",label="Tangential Crit. Curve")
    # not used anymore
    #ax.scatter(*cent_caust_tan,c="k",marker="x",label="Tang. Center Caustic")
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    plt.gca().set_aspect('equal')
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.legend()
    ax.set_title("Caustics and Critical Curves") 
    plt.tight_layout()
    print("Saving "+savename) 
    plt.savefig(savename)
    plt.close()
    
def plot_caustics(Model,fast_caustic = True,savename="test_caustics.png",kw_extents=None):
    lensModelExt = LensModelExtensions(Model.lens_model_PART)
    kwargs_lens  = Model.kwargs_lens_PART

    return _plot_caustics(Model,
                          lensModelExt,kwargs_lens,
                          fast_caustic=fast_caustic,savename=savename,kw_extents=kw_extents)

def get_extents(arcXkpc,Model=None,_radec=None):
    if _radec is None:
        _radec = Model.imageModel.ImageNumerics.coordinates_evaluate #arcsecs 
    _ra,_dec = _radec
    RA,DEC   = util.array2image(_ra),util.array2image(_dec)
    ra0,dec0 = RA[0]*u.arcsec,DEC.T[0]*u.arcsec # center of the bin

    Dra0  = np.diff(ra0)  
    Ddec0 = np.diff(dec0)

    ra_edges  = np.hstack([ra0[0]-(Dra0[0]/2.),ra0[1:]-(Dra0/2.),ra0[-1]+.5*Dra0[-1]])
    dec_edges = np.hstack([dec0[0]-(Ddec0[0]/2.),dec0[1:]-(Ddec0/2.),dec0[-1]+.5*Ddec0[-1]])
    
    Dra01   = np.diff(ra_edges) 
    # ugly, could be cleaner-> but after all it's constant so 
    Ddec01  = np.diff(dec_edges)

    # in kpc:
    xmin = ra_edges[0]/arcXkpc
    xmax = ra_edges[-1]/arcXkpc
    ymin = dec_edges[0]/arcXkpc
    ymax = dec_edges[-1]/arcXkpc

    extent_kpc    = [xmin.value, xmax.value,ymin.value, ymax.value] #kpc
    extent_arcsec = [ra_edges[0].value, ra_edges[-1].value,dec_edges[0].value, dec_edges[-1].value] #arcsec
    bins_arcsec   = [ra_edges,dec_edges]
    kw_extents = {"extent_kpc":extent_kpc,
              "extent_arcsec":extent_arcsec,
              "bins_arcsec":bins_arcsec,
              "DRaDec":[Dra01,Ddec01]}
    return kw_extents

def plot_kappamap(kappa1,extent_kpc,title1="",savename="kappa.png",skip_show=False):
    fig,axes = plt.subplots(2,figsize=(8,16))

    ax  = axes[0]
    im0 = ax.imshow(kappa1,origin="lower",extent=extent_kpc)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(title1) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')


    # take advantage of the circular simmetry and obtain the projection
    k1_proj = kappa1[int(len(kappa1)/2)]
    x = np.linspace(extent_kpc[0],extent_kpc[1],len(k1_proj))
    _xcnt = np.median(x)
    ax = axes[1]
    ax.plot(x,k1_proj,c="k")
    fwhm_k1 = get_fwhm(k1_proj,x) 
    hmax  = max(k1_proj)/2.
    ax.axvline(_xcnt,c="g",alpha=.5)

    ax.plot([_xcnt-fwhm_k1/2,_xcnt+fwhm_k1/2],[hmax,hmax],ls="-.",c="r",label="FWHM="+str(np.round(fwhm_k1,3)))
    ax.legend()
    ax.set_title(title1 +" projection at x=0")
    plt.suptitle("Density distribution")
    print("Saving "+savename)
    plt.savefig(savename)
    if not skip_show:
        plt.show()
    plt.close()


def plot_lensed_im_and_kappa(Model,savename="lensed_im.pdf",kw_extents=None):
    #kappa,kw_extents = get_kappa(Model,plot=False)
    kappa = Model.kappa_map
    if kw_extents is None:
        kw_extents = get_extents(Model.arcXkpc,Model)
    fg,axes = plt.subplots(1,2,figsize=(10,5))
    ax = axes[0]

    extent_kpc    = kw_extents["extent_kpc"]
    extent_arcsec = kw_extents["extent_arcsec"]
    
    im0   = ax.matshow(kappa,origin='lower',extent=extent_kpc,cmap="hot")
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(r"Convergence "+Model.Gal.Name)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label=r"$\kappa$")

    lnsd_im  = Model.image_sim 
    ax = axes[1]
    im0   = ax.matshow(np.log10(lnsd_im),origin='lower',extent=extent_arcsec)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    
    ax.set_title("Lensed image "+Model.Gal.Name)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label=r"log$_{10}$ flux [arbitrary]")
    plt.suptitle(r"With z$_{\text{lens}}$="+str(np.round(Model.z_lens,2))+" z$_{\text{source}}$="+str(np.round(Model.z_source,2)))
    plt.tight_layout()
    print("Saving "+savename) 
    plt.savefig(savename)
    plt.close("all")
    
def plot_all(Model,savename_lensed="lensed_im.pdf",savename_kappa="kappa.png",savename_caustics="caustics.png",fast_caustic=True,skip_caustic=False):
    
    #plot_lensed_im(Model,savename=Model.savedir+"/"+savename_lensed,skip_show=skip_show)
    #get_kappa(Model,savename=Model.savedir+"/"+savename_kappa,skip_show=skip_show)
    kw_extents = get_extents(Model.arcXkpc,Model)
    plot_lensed_im_and_kappa(Model,savename=Model.savedir+"/"+savename_lensed,kw_extents=kw_extents)
    if not skip_caustic:
        plot_caustics(Model,fast_caustic=fast_caustic,savename=Model.savedir+"/"+savename_caustics,kw_extents=kw_extents)
    plt.close("all")
    return 0
# monkey-patch for compatibility reason with previous versions:
Lens_PART = LensPart

# get a lens no matter what:
def wrapper_get_rnd_lens(reload=True):
    while True:
        Gal = get_rnd_NG()
        mod_LP = LensPart(Galaxy=Gal,kwlens_part=kwlens_part_AS,
                           z_source_max=z_source_max, 
                           pixel_num=pixel_num,reload=reload,savedir_sim="test_sim_lens_AMR")
        try:
            mod_LP.run()
            break
        except ProjectionError as PE:
            print("This galaxy failed: ",PE,"\n","Trying different galaxy")
            pass
    return mod_LP


if __name__ == "__main__":
    print("Do not run this script, but test_Gen_PM_PLL_AMR.py")
    exit()