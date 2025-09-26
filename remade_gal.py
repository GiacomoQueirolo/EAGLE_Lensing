# copied from tmp_print_xyz, we want to 
# obtain a new gal class that works
import glob
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
from fnct import part_data_path,std_sim,get_z_snap,prepend_str,get_snap
from get_gal_indexes import get_gals

#part_data_path = '/pbs/home/g/gqueirolo/EAGLE/data/'
# combination btw get_rnd_gal and _get_rnd_gal
def get_rnd_gal_indexes(sim=std_sim,min_mass = "1e12",min_z="0",max_z="2",pkl_name="massive_gals.pkl",check_prev=True,save_pkl=True):
    data  = get_gals(sim=sim,min_mass=min_mass,max_z=max_z,min_z=min_z,pkl_name=pkl_name,check_prev=check_prev,plot=False,save_pkl=save_pkl)
    index = np.arange(len(data["z"]))
    rnd_i = np.random.choice(index)
    kw = {}
    for k in data.keys():
        if k=="query" or k=="sim":
            kw[k] = data[k]
        else:
            kw[k] = data[k][rnd_i]
    return kw
#itype    = 1 #dm
# gas,dm, stars,bh : 0,1,4,5 
class NewGal:
    def __init__(self, Gn, SGn, sim=std_sim,z=None,snap=None): #,query="",CMx,CMy,CMz,M=None):
        self.sim    = sim
        z,snap      = get_z_snap(z,snap)
        self.snap   = snap
        self.z      = z
        #self.centre = np.array([CMx,CMy,CMz]) # cMpc
        self.Gn     = Gn
        self.SGn    = SGn
        self.Name   = f"G{Gn}SGn{SGn}" #note this is unique only within the snap
        #self.M      = M
        #self.query  = query # query from which this gal 
        self._init_path_snap()
        self._initialise_parts()
        
    def _init_path_snap(self):
        z_str          = prepend_str(str(int(self.z)),ln_str=3,fill="0")
        str_snap       = get_snap(self.z,3)
        fullsnap_str   = f"_{str_snap}_z{z_str}p???"
        self.path_snap = f"{part_data_path}/{self.sim}/snapshot{fullsnap_str}/snap{fullsnap_str}"
        return 0
        
    def _initialise_parts(self):

        self.gas   = self.read_part(0)
        self.dm    = self.read_part(1)
        self.stars = self.read_part(4)
        self.bh    = self.read_part(5)
        return 0
    
    def read_part(self,itype):
        kw   = {}
        atts = ["GroupNumber","SubGroupNumber","Coordinates"]
        if itype!=1:
            atts.append("Mass")
            atts.append("SmoothingLength")
        else:
            """Special case for the mass of dark matter particles."""   
            fl =  glob.glob(f"{self.path_snap}.0.hdf5")
            if len(fl)!=1:
                raise RuntimeError(f"{fl} found to be not of lenght 1 from glob({self.path_snap}.0.hdf5)")
            fl = fl[0]
            with h5py.File(fl, 'r') as f:
                h = f['Header'].attrs.get('HubbleParam')
                a = f['Header'].attrs.get('Time')
                dm_mass = f['Header'].attrs.get('MassTable')[1]
                n_particles = f['Header'].attrs.get('NumPart_Total')[1]
                # Create an array of lenght n_particles each set to dm_mass
                m = np.ones(n_particles, dtype='f8') *dm_mass
                # Use the conversion factors from the mass entry in the gas particles.
                cgs = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
                aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
                hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')
            # Convert to proper/physical mass
            kw["Mass"] = np.multiply(m, cgs*(a**aexp)*(h**hexp), dtype='f8')
            
        for att in atts:
            data = []
            nfiles = 16
            for i in range(nfiles):
                fl =  glob.glob(f"{self.path_snap}.{i}.hdf5")
                if len(fl)!=1:
                    raise RuntimeError(f"{fl} found to be not of lenght 1 from glob({self.path_snap}.{i}.hdf5)")
                fl = fl[0]
                with h5py.File(fl,'r') as f:
                    tmp = f['PartType%i/%s'%(itype,att)][...]
                    data.append(tmp)
                    # Get conversion factors
                    cgs  = f['PartType%i/%s'%(itype,att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype,att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype,att)].attrs.get('h-scale-exponent')
                    # Get expansion factor and Hubble parameter from the header
                    a = f['Header'].attrs.get('Time')
                    h = f['Header'].attrs.get('HubbleParam')
            if len(tmp.shape) > 1:
                data = np.vstack(data)
            else:
                data = np.concatenate(data)
            # Convert to physical
            if data.dtype!=np.int32 and data.dtype!=np.int64:
                data = np.multiply(data,cgs*a**aexp*h**hexp,dtype='f8')
            kw[att] = data
        mask   = np.logical_and(kw["GroupNumber"]==Gn,kw["SubGroupNumber"]==SGn)
        output = {}
        output["coords"] = kw["Coordinates"][mask]*u.cm.to(u.Mpc)
        output["mass"]   = kw["Mass"][mask] * u.g.to(u.Msun)
        if itype!=1:
            output["smooth"] = kw["SmoothingLength"][mask]*u.cm.to(u.Mpc)
        return output


kw_gal   = get_rnd_gal_indexes()
z        = kw_gal["z"] #0 
Gn,SGn   = kw_gal["Gn"],kw_gal["SGn"]

NG = NewGal(Gn,SGn,std_sim,z)


coords = NG.stars["coords"]
x,y,z = coords.T
name = "stars"
plt.close()
fig, ax = plt.subplots(3)
nx = 100
ax[0].hist(x,bins=nx,alpha=.5,label=name)#,range=[xmin, xmax])
ax[0].set_xlabel("X [kpc]")
ax[1].hist(y,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
ax[1].set_xlabel("Y [kpc]")
ax[2].hist(z,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
ax[2].set_xlabel("Z [kpc]")
ax[2].legend()

print(np.std(coords,axis=0))
namefig = f"tmp/hist_by_hand_stars.png"
plt.tight_layout()
plt.savefig(namefig)
plt.close()
print("Saved "+namefig) 
