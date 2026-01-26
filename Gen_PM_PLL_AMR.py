"""
From randomly selected galaxies, read particles and produces lenses
Revolves around the LensPart class, plus several helper functions
"""
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
from functools import cached_property 
from scipy.interpolate import splprep, splev, RectBivariateSpline
import astropy.units as u
import astropy.constants as const

from lenstronomy.Util import util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

from lenstronomy.SimulationAPI.sim_api import SimAPI

# My libs
from python_tools.fwhm import get_fwhm
from python_tools.get_res import load_whatever,LoadClass
from python_tools.tools import mkdir,short_SciNot,to_dimless,ensure_unit
from particle_lenses import PMLens 
from ParticleGalaxy import get_rnd_PG,Gal2kwMXYZ
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

# Path definition:

# define where to store the obtained lenses classes
sim_lens_path = "/pbs/home/g/gqueirolo/EAGLE/sim_lens/"
def get_lens_dir(Gal):
    lens_dir = f"{sim_lens_path}/{Gal.sim}/snap{Gal.snap}_G{Gal.Gn}.{Gal.SGn}/"
    mkdir(lens_dir)
    Gal.lens_dir = lens_dir
    return lens_dir



##########################
##########################
# Sampling of the Profiles
##########################
##########################

# source parametrisation
kwargs_sersic_ellipse_basic = {'R_sersic': .1, 'n_sersic': 3, 
                            'center_x': 0,
                             'center_y': 0, 
                             'e1': 0.0, 'e2': 0.0}
kwargs_sersic_ellipse = {'amp': 4000.}|kwargs_sersic_ellipse_basic
kwargs_sersic_ellipse_mag = {'mag':25.}|kwargs_sersic_ellipse_basic
source_model_list = ['SERSIC_ELLIPSE']

def get_LensSystem_kwrgs(Sim):
    # data specifics
    print("Pixel_num: ",  Sim.numpix)
    print("DeltaPix: ",   np.round(Sim.pixel_scale,3))
    kwargs_data = Sim.kwargs_data
    kwargs_psf  = Sim.kwargs_psf
    data_class  = Sim.data_class
    psf_class   = Sim.psf_class
    # Source Params
    source_model_class = Sim.source_model_class
    kwargs_source      = get_kwargs_sourceSim(Sim)
    kwargs_numerics    = {'supersampling_factor': 1, 'supersampling_convolution': False}

    return data_class, psf_class, source_model_class, kwargs_numerics, kwargs_source
    
def get_LensSystem_kwrgs_old(deltaPix,pixel_num=pixel_num,
                         background_rms=None,exp_time=None,
                         ra_source=0.,dec_source = 0.,source_model_list=source_model_list):
    # data specifics
    # background noise per pixel
    # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    print("Pixel_num: ",  pixel_num)
    print("DeltaPix: ",  np.round(deltaPix,3))
    deltaPix    = to_dimless(deltaPix,True) # if dimensional, convert to dimensionless
    kwargs_data = sim_util.data_configure_simple(pixel_num, deltaPix, exp_time, background_rms)
    data_class  = ImageData(**kwargs_data)
    kwargs_psf  = {'psf_type': 'NONE'}  
    psf_class   = PSF(**kwargs_psf)
    # Source Params
    source_model_class,kwargs_source = get_model_source(ra_source,dec_source,source_model_list=source_model_list)
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    # for modelling later:
    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 
                     'multi_band_type': 'single-band'}
    return data_class, psf_class, source_model_class, kwargs_numerics, kwargs_source,kwargs_data_joint


def get_model_source(ra_source=0,dec_source=0,
                     source_model_list=source_model_list,_kwargs_source=kwargs_sersic_ellipse):
    if ra_source!=0:
        _kwargs_source["center_x"] = ra_source
    if dec_source!=0:
        _kwargs_source["center_y"] = dec_source
    kwargs_source = [_kwargs_source]
    source_model_class = LightModel(light_model_list=source_model_list)
    return source_model_class,kwargs_source

from lenstronomy.SimulationAPI.mag_amp_conversion import MagAmpConversion

def get_kwargs_sourceSim(Sim,_kwargs_source_mag=kwargs_sersic_ellipse_mag):
    # the following only depends on -kwargs_source_params, -magnitude_0_point -source_model_list
    _, kwargs_source, _ = Sim.magnitude2amplitude(kwargs_source_mag = _kwargs_source_mag)
    return kwargs_source

def get_dataclasses(Sim):
        print("Pixel_num: ",  Sim.numpix)
        print("DeltaPix: ",   np.round(Sim.pixel_scale,3))
        data_class  = Sim.data_class
        psf_class   = Sim.psf_class
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
        # Source Params
        source_model_class = Sim.source_model_class
        kwargs_source      = get_kwargs_sourceSim(Sim)    
        return data_class,psf_class,source_model_class,kwargs_numerics,kwargs_source
##########################
# Model class for parts. #
##########################
# particle lens params.
#from particle_lenses import kwlens_part_PM
from particle_lenses import default_kwlens_part_AS  as kwlens_part_AS

# kwargs of ultra-performing band for default simulated images
kwargs_band_sim = {'read_noise': 0, # no RN noise
 'pixel_scale': None,               # to update depending on the lens
 'ccd_gain': 2.5,             # standard gain for HST
 'exposure_time': 5400.0,     # standard exp time for HST
 'sky_brightness': 35,        #"dark" sky
 'magnitude_zero_point': 30,  # very deep 
 'num_exposures': 1,          # standard HST n exp.
 'psf_type': 'NONE'}          # "infinite" psf resolution 

class LensPart(): 
    def __init__(self,
                    Galaxy,
                    kwlens_part, # if PM or AS, and if so size of the core
                    pixel_num=pixel_num, # sim prms 
                    z_source_max  = z_source_max,     # for z_source sampling
                    min_thetaE = min_thetaE,
                    source_model_list=source_model_list, # this might not be the most efficient way to do it..
                    kwargs_band_sim=kwargs_band_sim,
                    #z_lens=z_lens,z_source=z_source,cosmo=cosmo, # cosmo prms -> obtained from Galaxy 
                    #exp_time=1500, #sec. (~<exp_time> for J1433 opt. HST obs ) -> will be defined later
                    #bckg_rms=0.006, #count (~ from f814w) observation parameters
                    savedir_sim="lensing",
                    reload=True # reload previous lens
                    ):
        Galaxy             = prep_Gal_projpath(Galaxy) # just set up some directories
        # setup of data
        self.Gal           = Galaxy
        self.Gal_path      = Galaxy.get_pkl_path()
        # if reload, check if Gal is already a lens - if it isn't, raise error
        if reload:
            if not self.Gal.is_lens:
                print(load_whatever(self.Gal.islens_file)["message"])
                raise RuntimeError("Previously defined as not a lens")
        
        lens_dir           = get_lens_dir(self.Gal)
        self.savedir_sim   = savedir_sim
        self.savedir       = f"{lens_dir}/{savedir_sim}"
        self.reload        = reload
        mkdir(self.savedir)
        ######
        # lensing params
        self.pixel_num     = pixel_num      
        self.kwlens_part   = kwlens_part
        self.PMLens        = PMLens(kwlens_part)
        # cosmo prms
        self.z_lens        = self.Gal.z
        self.cosmo         = self.Gal.cosmo
        self.arcXkpc       = self.cosmo.arcsec_per_kpc_proper(self.z_lens)

        # source model - probably to improve
        self.source_model_list = source_model_list

        # To obtain the z_source and projection index 
        #-> computed only once in the run function
        self.z_source_max  = z_source_max        
        self.min_thetaE    = ensure_unit(min_thetaE,u.arcsec) #arcsec
        
        # observational params
        self.kwargs_band_sim     = kwargs_band_sim
        self.kwargs_source_model = {"source_light_model_list":self.source_model_list}
        self.Sim = None # will be initialised with kwargs_band_sim once we have the correct deltaPix
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
        # store the instance of the class
        if not hasattr(self,"pkl_path"):
            self._setup_names()
        with open(self.pkl_path, "wb") as f:
            dill.dump(self, f)
        print("Saved", self.pkl_path)
        return 0
        
    def _unpack(self):
        # this function recover the parts deleted before 
        # storing to save space
        Galaxy = LoadClass(self.Gal_path)
        Galaxy = prep_Gal_projpath(Galaxy)
        self.Gal   = Galaxy
        self.cosmo = Galaxy.cosmo
        # re-define PMLens
        self.PMLens = PMLens(self.kwlens_part)
        self.PMLens.setup(self)
        # recover kwargs_lens_PART and lens_model_PART
        if not "lens_model_PART" in self.__dict__:
            self.setup_lenses() 
        # recover imageModel
        self.get_imageModel()
        return True
        
    def unpack(self):
        # wrapper for the unpack function, 
        # so it runs only when necessary
        if not "Gal" in self.__dict__:
            self._unpack()
        return True
    ########################
    ########################

    def _get_name(self):
        # define name and path of savefile
        self.name = f"{self.Gal.Name}_Npix{self.pixel_num}_Part{self.PMLens.name}"
        
    def _setup_names(self):
        if not "name" in self.__dict__:
            self._get_name()
        self.pkl_path = f"{self.savedir}/{self.name}.pkl"
    #############################
    # Run:
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
            # update kwargs_band_sim:
            self.kwargs_band_sim["pixel_scale"] = to_dimless(self.deltaPix)
            # update self.Sim
            # self.Sim DOES NOT contain lensing information, it's used only partially and carefully
            self.Sim = SimAPI(numpix=self.pixel_num,kwargs_single_band=self.kwargs_band_sim,kwargs_model=self.kwargs_source_model)
            # in a similar way a posterior SimObs can be used to create images given different telescopes
            # the following should not matter, but for cleanliness:
            self.Sim._cosmo = self.cosmo
            
            # setup dataclasses (dataclass,psf_class,sourcemodel and some helper kwargs):
            self.setup_dataclasses()
            # setup imageModel:
            self.get_imageModel()
            # setup lenses 
            self.setup_lenses()
            # this is the most computationally intense function:
            self.image_sim  = self.get_lensed_image()
            self.store()
            # Assuming that if we got here, everything worked out fine:
            self.Gal.update_is_lens(islens=True,message="No issues")

    def setup_dataclasses(self):
        self.data_class,self.psf_class,self.source_model_class,self.kwargs_numerics,self.kwargs_source = get_dataclasses(self.Sim)
        return 0
        
    def update_source_position(self,ra_source,dec_source):
        # useful if we want to put it in the center of the caustic
        self.kwargs_source["center_x"]= ra_source
        self.kwargs_source["center_y"]= dec_source
        return 0
        
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
        image_sim    = imageModel.ImageNumerics.re_size_convolve(source_light, unconvolved=True)
        # imageModel.image(self.kwargs_lens_PART, self.kwargs_source, kwargs_lens_light=None, kwargs_ps=None)
        return image_sim

    def sample_source_pos(self,update=True):
        """Sample the source position within the tangential
        critical caustic
        """
        kw_caustics  = self.critical_curve_caustics
        ra_ct,dec_ct = kw_caustics["caustics"]["tangential"]
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
        """Map the x,y grid into the source plane
        (used to fit the light of the source to the image)
        """
        RA,DEC       = self.get_RADEC()
        # if not already, compute alpha_map
        alpha_x,alpha_y = self.alpha_map        
        x_source_plane, y_source_plane = RA-alpha_x,DEC-alpha_y
        # the coords have to be given as flat
        x_source_plane = util.image2array(x_source_plane)
        y_source_plane = util.image2array(y_source_plane)
        return x_source_plane,y_source_plane
        
    def get_imageModel(self,Sim):
        if not "data_class" in self.__dict__ :
            self.setup_dataclasses()
        if not "lens_model_PART" in self.__dict__:
            self.setup_lenses()
        self.imageModel = ImageModel(self.data_class, self.psf_class, 
                            self.lens_model_PART, 
                            self.source_model_class,
                            lens_light_model_class=None,point_source_class=None, 
                            kwargs_numerics=self.kwargs_numerics)
        return self.imageModel
    
    @cached_property
    def alpha_map(self):
        return self._alpha_map(_radec=None)
    @cached_property 
    def _psi_map(self):
        # this is not meant to be computed in the run function, so we have yet
        # another wrapper function around it to ensure its results are stored
        # after computation (see @property def psi_map)
        return self.compute_psi_map(_radec=None)
    @cached_property
    def kappa_map(self):
        return self._kappa_map(_radec=None)
    @cached_property
    def hessian(self):
        return self._hessian()
    
    def compute_psi_map(self,_radec=None):
        print("Computing lensing PM potential...")
        self.unpack()
        if _radec is None:
            # equivalent to np.reshape(np.array(self.data_class.pixel_coordinates),(2,self.pixel_num*self.pixel_num))
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
    def psi_map(self):
        # if psi_map has not been computed yet, computes it
        # and stores it
        # if it has, just returns the values
        if not "_psi_map" in self.__dict__:
            _ = self._psi_map
            self.store()
        return self._psi_map
    
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
    ################
    ################
    # Simulating observations: 

    # band will have to have also psf information!
    def set_SimObs(band=None):
        kwargs_model = {"z_source":self.z_source,
                        "source_light_model_list":self.source_model_list
                       }
        self.band = band # so we know which one it is running
        if band is not None:
            # realistic observation
            kwargs_band = band.kwargs_single_band()
            # must recompute pixel_num in order to covert the same aperture,
            # but with the new resolution 
            # -> round down to be sure we are within the bounds
            pixel_num = int(to_dimless(2*self.radius)/kwargs_band["pixel_scale"])
        else:
            # this is the ultra-refined band -> default sim with highest resolution
            kwargs_band = self.kwargs_band_sim
            pixel_num   = self.pixel_num
        #call the simulation API class
        self.SimObs = SimAPI(numpix = pixel_num, # number of pixels we want in our image
                     kwargs_single_band = kwargs_band, # give the SimAPI class the keyword arguments for HST that we got above
                     kwargs_model = kwargs_model)
        
        return self.SimObs

    
    def get_imageModelObs(self):
        # contrary to the stnd. get_imageModel (used to compute lensing params and sim image),
        # this is used ONLY to create realistic observations from it (no lensing recomputation)
        #if not "SimObs" in self.__dict__:
            
        if not "data_classObs" in self.__dict__ :
            self.setup_dataclassesObs()
        self.imageModelObs = ImageModel(data_class= self.data_classObs, 
                                        psf_class = psf_self.psf_classObs, 
                        source_model=self.source_model_class,
                        lens_model=None, #this is only used to replot the image, alpha map already computed 
                        lens_light_model_class=None,point_source_class=None, 
                        kwargs_numerics=self.kwargs_numerics) # for now numerics stay the same
        return self.imageModelObs
        
    def setup_dataclassesObs(self,ra_source=0,dec_source=0):
        self.data_classObs,self.psf_classObs,self.source_model_classObs,_, self.kwargs_sourceObs = get_dataclasses(self.SimObs)
        return 0
        
    # deal with SIM API by considering
    # input: band and PSF
    # used to produce realistic images
    def get_lensed_imageObs(self):
        self.unpack()
        sourceModel  = self.source_model_classObs
        imageModel   = self.imageModelObs
        x_source_plane,y_source_plane = self.get_xy_source_plane()
        source_light = sourceModel.surface_brightness(x_source_plane, y_source_plane, self.kwargs_sourceObs, k=None)
        image_simObs = imageModel.ImageNumerics.re_size_convolve(source_light, unconvolved=False)
        return image_simObs

    def sim(band):
        self.set_SimObs(band)
        self.get_imageModelObs()
        image_simObs = self.get_lensed_imageObs()
        
    
    ################
    ################

    # Shear components and caustics/CL
    def _hessian(self):
        """Computes the hessian matrix on the grid by taking the gradient 
        of the alpha map
        """
         # Note: this hessian only consider the contribution of the alpha map within the cutout!
        alpha_x,alpha_y = self.alpha_map
        # taking the non-dimensional pixel scale for the gradient
        dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, to_dimless(self.deltaPix))
        dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, to_dimless(self.deltaPix))
        #print("Note: Taking the average of dalpha_x_dy and dalpha_y_dx for fxy")
        f_xx,f_xy,f_yx,f_yy  = dalpha_x_dx,dalpha_x_dy,dalpha_y_dx,dalpha_y_dy
        return f_xx,f_xy,f_yx,f_yy

    def get_kw_shear(self):
        """From the hessian matrix compute the shear components
        """
        f_xx,f_xy,f_yx,f_yy = self.hessian
        # derived shear1,shear2 and shear
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
        """ Fit the critical curve and map it to the caustic
        """
        # note: alpha is computed from the particle (and shear from alpha)
        # thus depends on the particle lens model chosen, while kappa
        # is obtained directly as a density map + cosmological scaling
        alpha_x,alpha_y = self.alpha_map
        kappa           = self.kappa_map
        shear           = self.shear_map
        
        eigen_rad = 1 - kappa + shear
        eigen_tan = 1 - kappa - shear
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

                
def MAD_mask(values,v0=0,sigma_scale=3):
    # robust estimator of noise: Median Absolute Deviation    
    mad = np.median(np.abs(values - np.median(values)))

    sigma = 1.4826 * mad

    mask = np.abs(values-v0) < sigma_scale*sigma   # ~99.7% Gaussian confidence
    return mask

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


# monkey-patch for compatibility reason with previous versions:
Lens_PART = LensPart

def get_extents(arcXkpc,Model=None,_radec=None):
    """Returns the extent of the image in various units
    """
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

# get a lens no matter what:
def wrapper_get_rnd_lens(reload=True):
    """Try to get a lens from random galaxies, repeat until finds one
    which is an actual lens (i.e. supercritical)
    """
    while True:
        Gal = get_rnd_PG()
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
    print("Do not run this script, but tests/test_Gen_PM_PLL_AMR.py")
    exit()