from python_tools.get_res import LoadClass
#from python_tools.read_fits import load_fits, load_fitshead,get_transf_matrix
#from python_tools.conversion import get_pixscale
from Gen_PM_PLL_AMR import LensPart,plot_all,LoadLens
from Gen_PM_PLL_AMR import kwlens_part_AS,z_source_max,pixel_num
#from pyinstrument import Profiler

gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_24/Gn27SGn0.pkl")
lens = LensPart(Galaxy=gal,kwlens_part=kwlens_part_AS,
                           z_source_max=z_source_max, 
                           pixel_num=pixel_num,reload=True,savedir_sim="test_sim_lens_AMR")
lens.run()

