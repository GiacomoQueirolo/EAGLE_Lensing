# copy from plot_GSMF
# adapted to plot N of galaxies at different redshifts
# then output a table of indexes to use to get them, with 
# - coordinates
# - mass
# - redshift
# - Group and Subgroup (not really useful but anyway)
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os,copy

from python_tools.get_res import load_whatever
from sql_connect import exec_query
from fnct import std_sim,gal_dir

def get_gals(sim=std_sim,min_mass = "1e12",min_z="0",max_z="2",save_pkl=True,pkl_name="massive_gals.pkl",plot=True,check_prev=True):
    pkl_path = f"{gal_dir}/{pkl_name}" 
     # select higher masses bc 1) lenses 2) else we have too many points
    myQuery = "SELECT \
        gal.GroupNumber as Gn, \
        gal.SubGroupNumber as SGn, \
        gal.Redshift as z, \
        gal.Mass as M, \
        gal.CentreOfMass_x as CMx, \
        gal.CentreOfMass_y as CMy, \
        gal.CentreOfMass_z as CMz  \
    FROM \
        %s_Subhalo as gal \
    WHERE \
        (gal.Redshift between %s and %s) and \
        gal.Mass > %s \
    ORDER BY \
        gal.Redshift"%(sim,min_z,max_z,min_mass)
    
    # NOTE: center of mass is in comoving coord.(cMpc)
    # Execute
    #print("DEBUG - check that the pickling doesn't mess w. the data")
    check_prev = False
    save_pkl = False
    if check_prev:
        try:
            myData = load_whatever(pkl_path)
            #formatting might be slightly diff.
            if myData["query"].replace(" ","") != myQuery.replace(" ",""):
                raise UserWarning("Loaded previous results doesn't have the same query - rerunning and overwriting")
        except:
            print("Tried and failed to load previous results :"+pkl_path+"\nRerunning SQL query")
            check_prev = False
    if not check_prev:
        myData = exec_query(myQuery)
    if plot:
        logMass = np.log(myData["M"])
        str_logMass = r'log$_{10}$M${_*}$[M$_{\odot}$]'
        zGal   = myData["z"]
        plt.hist(logMass)
        plt.title("Mass of Galaxies selected")
        plt.xlabel(str_logMass)
        plt.savefig("hist_gal_mass.png")
        plt.close()
        plt.hist(zGal)
        plt.title("Redshift of Galaxies selected")
        plt.xlabel(r'z')
        plt.savefig("hist_gal_z.png")
        plt.close()
        
        
        plt.scatter(zGal,logMass,marker=".")
        plt.title("Mass at redshift")
        plt.xlabel(r'z')
        plt.ylabel(str_logMass)
        plt.savefig("gal_mvsz.png")
    """
    if save_pkl and not check_prev:
        with open(pkl_path,"wb") as f:
            pickle.dump(myData,f)
        print("Saving "+pkl_path)
    """
    return myData


