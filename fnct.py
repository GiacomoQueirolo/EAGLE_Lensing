import glob
import h5py
import pickle
import numpy as np
from python_tools.get_res import load_whatever
from python_tools.tools import mkdir
####

# Setup and General Structure
##############################

# data dir structure: data_path 
#                        |_ Sim
#                            |_snapshots_of_particles
#                            |_Gals
#                                |_snaphots_of_gals (obtained from particles)

# data path
part_data_path = "/pbs/home/g/gqueirolo/EAGLE/data/"
# "Standard" simulation
# use the following only as test case
#std_sim  = "RefL0012N0188"
std_sim  = "RefL0025N0752"
test_sim = "RefL0012N0188"
sim_path = part_data_path+std_sim+"/"
# Where to store the galaxies
gal_dir = sim_path+"/Gals"
mkdir(gal_dir)
# from https://dataweb.cosma.dur.ac.uk:8443/eagle-snapshots/
# valid fo all sims apart the variable IMF runs
kw_snap_z = {"28":0, "27":0.1, "26":0.18, "25":0.27, "24":0.37, "23":0.5, "22":0.62, "21":0.74, "20":0.87, "19":1, "18":1.26, "17":1.49, "16":1.74, "15":2.01, "14":2.24, "13":2.48, "12":3.02, "11":3.53, "10":3.98, "9":4.49, "8":5.04, "7":5.49, "6":5.97, "5":7.05, "4":8.07, "3":8.99, "2":9.99, "1":15.13, "0":20}
#inverted kw
kw_z_snap = {}
for k in kw_snap_z:
    kw_z_snap[kw_snap_z[k]] = k
# z indexes
z_index = np.array([float(f) for f in list(kw_z_snap.keys())])

# Useful functions:
###################

def get_z(snap):
    snap = str(snap)
    while snap[0]=="0" and snap!="0":
        snap = snap[1:]
    return kw_snap_z[snap]

def get_snap(z,_ln_snap=None):
    # consider a continous z instead of the discreet version
    # works for discreet z as well
    key_z = min(kw_z_snap.keys(),key=lambda k:np.abs(k-float(z)))
    snap  = str(kw_z_snap[key_z])
    snap  = prepend_str(snap,ln_str=_ln_snap,fill=0)
    return snap

def get_z_snap(z=None,snap=None):
    if z is None and snap is None:
        raise UserWarning("Give either z or snap")
    if z is None:
        z = get_z(snap)
    else:
        snap = get_snap(z)
    return z,snap
    
def prepend_str(str_i,ln_str,fill="0"):
    if ln_str is None:
        return str_i
    str_i = str(str_i)
    fill  = str(fill) 
    while len(str_i)<ln_str:
        str_i=fill+str_i
    return str_i

def get_files(sim,z=None,snap=None,_i_="*"):
    """
    Find the files 
    If _i_ is specified, only that specific subsection of the snapshot (useful for DM)
    If no redshift/snapshots are defined, take all of them
    """
    
    sim_path = part_data_path+"/"+sim
    # find the files
    _i_ = str(_i_)
    pstring = "???"
    suffix = "p"+pstring+"."+_i_+".hdf5"
    prefix = sim_path+"/snapshot_"
    if z is None and snap is None:
        # take all snapshots/all redshifts
        snap ="0??"
        zstr = "???"
    else:
        if z is not None and snap is not None:
            # verify that they are compatible:
            assert int(get_snap(z))==int(snap)
        if z is not None:
            zstr = str(int(z))
            snap = get_snap(z)
        elif snap is not None:
            #zstr = str(get_z(snap))
            pth   = prefix+prepend_str(snap,ln_str=3,fill="0")+"_z*"
            _zstr = glob.glob(pth)
            assert len(_zstr)==1
            zstr  = _zstr[0].split("_z")[1].split("p")[0]
        snap = prepend_str(snap,ln_str=3,fill="0")
        zstr = prepend_str(zstr,ln_str=3,fill="0")
    
    fix  = f"{snap}_z{zstr}p{pstring}/snap_{snap}_z{zstr}"
    file_string = prefix+fix+suffix
    files = glob.glob(file_string)
    # checking that the files are not empty
    try:
        assert files != []
    except:
        print("#DEBUG")
        print(file_string)
        print("If snap==17, might be bc that snap was not downloaded fsr - and we fail to do so")
    return files



def read_snap_header_simple(z=None,snap=None,sim=std_sim):
    """ Read various attributes from the header group.  -> simplified"""
    file    = get_files(sim,z,snap,_i_=0)
    #print("#DEBUG")
    #print(sim,z,snap)
    if len(file)!=1:
        raise RuntimeError("Warning: define only one snapshot")
        print("file=",file)
    file      = file[0]
    aexp,hexp = {},{}
    with h5py.File(file, 'r') as f:
        a       = f['Header'].attrs.get('Time')                # Scale factor.
        h       = f['Header'].attrs.get('HubbleParam')         # h = H0/(100km/s/Mpc)
        boxsize = f['Header'].attrs.get('BoxSize')             # L [cMph/h].
        """
        # aexp and hexp are different for the diff. variable (mainly coord and mass)
        # but should be the same between different type of particles and redshift bins
        atts = "GroupNumber","SubGroupNumber","Mass","Coordinates","SmoothingLength"
        for att in atts:
            aexp[att] = f[f"PartType0/{att}"].attrs["aexp-scale-exponent"]  # Exponent of Scale factor.
            hexp[att] = f[f"PartType0/{att}"].attrs["h-scale-exponent"]     # Exponent of h
    #return a,aexp,h,hexp,boxsize
    """
    return a,h,boxsize

def _count_part(part):
    return len(part["mass"])

def _mass_part(part):
    return np.sum(part["mass"])


def get_gal_path(Gal,ret_snap_dir=False):
    try:
        gal_path,gal_snap_dir =  Gal.gal_path,Gal.gal_snap_dir
    except:
        gal_snap_dir = f"{gal_dir}/snap_{Gal.snap}/"
        gal_path = f"{gal_snap_dir}/{Gal.Name}.pkl"
    if ret_snap_dir:
        return gal_path,gal_snap_dir
    return gal_path

################################
def sersic_brightness(x,y,n=4,I=10,cntx=0,cnty=0,pa=0,q=1):
    # x,y : N,M grid of coordinates
    # n : Sersic index
    # I : amplitude
    # cntx,cnty : center coordinates
    # pa: pointing angle
    # q : axis ratio
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
    if q==1:
        # for some reason pa has effects even if q is 1 
        pa=0
    paRad = pa*np.pi/180
    # rotate the galaxy by the angle paRad
    x = np.cos(paRad)*(x-cntx)+np.sin(paRad)*(y-cnty)
    y = -np.sin(paRad)*(x-cntx)+np.cos(paRad)*(y-cnty)
    # include elliptical isophotes
    #r = np.sqrt(x**2+y**2) #-> if q==1
    xt2difq2 = y/(q*q)
    r = np.sqrt(x**2+y*xt2difq2)
    # brightness at distance r
    bn = 1.992*n - 0.3271
    re = 5.0
    brightness = I*np.exp(-bn*((r/re)**(1.0/n)-1.0))
    return brightness
    
class Sersic():
    # useful to lens an image:
    def __init__(self,I=10,cntx=0,cnty=0,pa=0,q=1,n=4):
        self.cntx = cntx
        self.cnty = cnty
        self.pa   = pa
        self.q    = q
        self.n    = n
        self.I    = I
    def image(self,x,y):
        return sersic_brightness(x,y,**self.__dict__)
