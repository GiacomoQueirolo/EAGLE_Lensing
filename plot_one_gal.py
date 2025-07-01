import numpy as np
from fnct import Galaxy,std_sim,sim_path
import matplotlib.pyplot as plt

from get_gal_indexes import get_rnd_gal
# obtained from http://virgodb.dur.ac.uk:8080/Eagle/MyDB
# with the followin command:
"""
SELECT   
        gal.Redshift as z,   
        gal.Image_Face as face, 
        gal.CentreOfMass_x as x, 
        gal.CentreOfMass_y as y, 
        gal.CentreOfMass_z as z,
        gal.GroupNumber as Gn,
        gal.SubGroupNumber as SGn
   FROM
        RefL0025N0752_SubHalo as gal,   
        RefL0025N0752_SubHalo as ref   
   WHERE   
        ref.GalaxyID=1848116 and -- GalaxyID at z=1   
        ((gal.SnapNum > ref.SnapNum and ref.GalaxyID   
        between gal.GalaxyID and gal.TopLeafID) or    
        (gal.SnapNum <= ref.SnapNum and gal.GalaxyID    
        between ref.GalaxyID and ref.TopLeafID))   
   ORDER BY   
        gal.Redshift
"""

#centre3,gn,sgn = np.array([14.434582,24.12927,19.225077]),22,0

gl3 = get_rnd_gal(sim=std_sim,min_z=1.9,max_z=2.02,reuse_previous=True)
# Galaxy(Gn=gn,SGn=sgn,CntX=cntre3[0],CntY=centre3[1],CntZ=centre3[2],z=0)

xyz_dm  = gl3.dm["coords"].T
xyz_str = gl3.stars["coords"].T
xyz_gas = gl3.gas["coords"].T
xyz_bh  = gl3.bh["coords"].T

x_dm  = xyz_dm[0]
x_str = xyz_str[0]
x_gas = xyz_gas[0]
x_bh  = xyz_bh[0]


m_str = gl3.stars["mass"]
m_dm  =  gl3.dm["mass"]
m_gas = gl3.gas["mass"]
m_bh  = gl3.bh["mass"]

# bh can be ignored

b = 40
plt.style.use('classic')
plt.hist(x_str,bins=b,weights=m_str, color="yellow",label="stars",alpha=.3)
plt.hist(x_gas,bins=b,weights=m_gas, color="violet",label="gas",alpha=.3)
plt.hist(x_dm,bins=b,weights=m_dm,color="grey",label="dm",alpha=.3)
plt.yscale("log")
plt.ylabel(r"M [M$_\odot$]")
plt.xlabel("X coord [Mpc]")
plt.legend()
plt.title("Mass Histogram For Different Particles of 1 EAGLE Gal.")
nm = sim_path+"/mHistGal1.pdf"
print("Saving "+nm)
plt.savefig(nm)
plt.close()


dxy=.06
xy_str = xyz_str[:-1]
plt.scatter(*xy_str,c=np.log(m_str),alpha=.2,cmap="coolwarm_r",marker=".")
plt.colorbar(label="log(Star Mass)")
xm,ym=np.mean(xy_str,axis=1)
plt.xlim(xm-dxy,xm+dxy)
plt.ylim(ym-dxy,ym+dxy)
plt.title("Stars particles for 1 EAGLE Gal")
plt.xlabel("X [Mpc]")
plt.ylabel("Y [Mpc]")
nm = sim_path+"/strDistrGal1.pdf"
print("Saving "+nm)
plt.savefig(nm)
plt.close()


